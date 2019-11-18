import os
import numpy as np
import scipy.stats
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
import pyro
import pyro.distributions as dist

from collections import defaultdict
from torch.distributions import constraints
from matplotlib import pyplot
from matplotlib.patches import Ellipse
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, \
    config_enumerate, infer_discrete, EmpiricalMarginal
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
from pyro.infer import Predictive

warnings.filterwarnings('ignore')
smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('0.5.1')
pyro.enable_validation(True)
pd.set_option('display.max_columns', None)
pyplot.style.use('ggplot')

def get_train_test_split(movies_metadata):
    """Get train test split of obs and new data"""
    movies_metadata = normalize_data(movies_metadata)
    features = get_features(movies_metadata)
    data = torch.tensor(features, dtype=torch.float32)
    N = 500
    data_new = data[N:2*N].clone().detach()
    data = data[:N].clone().detach()
    return data, data_new


def plot_svi_convergence(losses, gradient_norms):
    """Plot SVI convergence and gradient"""
    fig = pyplot.figure(figsize=(16, 4))
    ax1 = fig.add_subplot(121, xlabel="iters", ylabel="loss",
                          yscale="log", title="Convergence of SVI")
    ax1.plot(losses)
    ax2 = fig.add_subplot(122, xlabel="iters", ylabel="gradient norm",
                          yscale="log", title="Gradient norm SVI")
    for name, grad_norms in gradient_norms.items():
        ax2.plot(grad_norms, label=name)
    ax2.legend()
    fig.show()


def plot(data, mus=None, sigmas=None, colors='black', K=None, d=None, ax=None):
    """Plot 2D GMMs"""
    x = data[:, 0]
    y = data[:, 1]
    pyplot.scatter(x, y, 24, c=colors)

    if mus is not None:
        x = [float(m[0]) for m in mus]
        y = [float(m[1]) for m in mus]
        pyplot.scatter(x, y, 99, c='red')

    if sigmas is not None:
        for sig_ix in range(K):
            try:
                cov = (torch.eye(d) * sigmas[sig_ix]).detach().numpy()
            except TypeError:
                cov = np.array(sigmas[sig_ix])
            lam, v = np.linalg.eig(cov)
            lam = np.sqrt(lam)
            ell = Ellipse(xy=(x[sig_ix], y[sig_ix]),
                          width=lam[0]*4, height=lam[1]*4,
                          angle=np.rad2deg(np.arccos(v[0, 0])),
                          color='blue')
            ell.set_facecolor('none')
            ax.add_artist(ell)


def normalize_data(movies_metadata):
    """Return normalized movies_metadata"""
    movies_metadata['revenue'] = normalize(movies_metadata['revenue'])
    movies_metadata['budget'] = normalize(movies_metadata['budget'])
    movies_metadata['vote_average'] = normalize(movies_metadata['vote_average'])
    movies_metadata['vote_count'] = normalize(movies_metadata['vote_count'])
    movies_metadata['popularity'] = normalize(movies_metadata['popularity'])
    movies_metadata['runtime'] = normalize(movies_metadata['runtime'])
    return movies_metadata


def normalize(np_arr):
    """Normalize an array"""
    return (np_arr - np.mean(np_arr)) / np.std(np_arr)


def get_features(movies_metadata):
    """Extract features"""
    features = np.stack((
        movies_metadata['budget'],
        movies_metadata['revenue'],
        # movies_metadata['vote_count'],
        # movies_metadata['vote_average'],
        # movies_metadata['popularity']
    ), axis=1)
    return features


def get_Sigma_samples(posterior_samples):
    """Compute Sigma based on theta and L_omega"""
    Sigma_samples = []
    for i in range(len(posterior_samples["theta"])):
        L_Omega = torch.mm(
            torch.diag(posterior_samples["theta"][i].sqrt()),
            posterior_samples["L_omega"][i])
        Sigma = torch.mm(L_Omega, L_Omega.t()).tolist()
        Sigma_samples.append(Sigma)
    Sigma_samples = torch.tensor(Sigma_samples)
    return Sigma_samples


def get_bayes_estimate_cov(Sigma_samples, K):
    """Compute covariance"""
    Sigma_bayes_est = [[torch.mean(Sigma_samples[:, i, j]).item()
                        for j in range(len(Sigma_samples[0]))] for i in range(len(Sigma_samples[0]))]
    cov = [Sigma_bayes_est for num_clusters in range(K)]
    cov = torch.tensor(cov)
    return cov


def get_bayes_estimate_mu(posterior_samples):
    """Compute bayes estimate of mu"""
    mu = [[torch.mean(posterior_samples["locs"][:, i, j]).item() for j in range(len(posterior_samples["locs"][0][0]))] for i in range(len(posterior_samples['locs'][0]))]
    mu = torch.tensor(mu)
    return mu


def get_bayes_estimate_pi(posterior_samples):
    pi = [torch.mean(posterior_samples["weights"][:, i]).item() for i in range(len(posterior_samples['weights'][0]))]
    return pi


def plot_mcmc_mu(posterior_samples, K, d):
    """Plot posterior of mu"""
    for i in range(K):
        trace = posterior_samples["locs"][:, i, :]
        fig = pyplot.figure(figsize=(16, 2))
        for j in range(d):
            ax1 = fig.add_subplot(
                121, xlabel="x", ylabel="Density", title="mu"+str(i))
            ax1.hist(trace[:, j], 50, density=True)
            ax2 = fig.add_subplot(
                122, xlabel="Steps", ylabel="Sample Values", title="mu"+str(i))
            ax2.plot((trace[:, j]))
        fig.show()


def plot_mcmc_pi(posterior_samples, K, d):
    """Plot posterior of pi"""
    fig = pyplot.figure(figsize=(16, 2))
    ax1 = fig.add_subplot(
        121, xlabel="x", ylabel="Density", title="pi")
    ax2 = fig.add_subplot(
        122, xlabel="Steps", ylabel="Sample Values", title="pi")
    for i in range(K):
        ax1.hist(posterior_samples["weights"][:, i])
        ax2.plot(posterior_samples["weights"][:, i])
    fig.show()


def plot_mcmc_theta(posterior_samples, K, d):
    """Plot posterior of theta"""
    fig = pyplot.figure(figsize=(16, 2))
    for j in range(d):
        ax1 = fig.add_subplot(
            121, xlabel="x", ylabel="Density", title="theta")
        ax1.hist(posterior_samples["theta"][:, j], 50, density=True)
        ax2 = fig.add_subplot(
            122, xlabel="Steps", ylabel="Sample Values", title="theta")
        ax2.plot(posterior_samples["theta"][:, j])
    fig.show()


def plot_mcmc_Sigma(Sigma_samples, K, d):
    """Plot posterior of Sigma"""
    fig = pyplot.figure(figsize=(16, 2))
    ax1 = fig.add_subplot(
        121, xlabel="x", ylabel="Density", title="Sigma")
    ax2 = fig.add_subplot(
        122, xlabel="Steps", ylabel="Sample Values", title="Sigma")
    for i in range(d):
        for j in range(d):
            ax1.hist(Sigma_samples[:, i, j], 30, density=True)
            ax2.plot(Sigma_samples[:, i, j])
    fig.show()


def get_members(data, assignment, group):
    data_df = pd.DataFrame(
        data.detach().numpy(), columns=['budget', 'revenue'])
    data_df['assignment'] = pd.DataFrame(assignment, columns=['assignment'])
    return data_df[data_df['assignment'] == group].to_numpy()[:, 0:2]


def compute_log_likelihood(data, mu, cov, pi):
    log_likelihood = 0
    for i in range(len(data)):
        for j in range(len(data[0])):
            log_likelihood += np.log(pi[j]) + \
                dist.MultivariateNormal(mu[j], cov[j]).log_prob(data[i])
    return log_likelihood
