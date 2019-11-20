import numpy as np
import torch
import pandas as pd
import pyro
import pyro.distributions as dist

from matplotlib import pyplot
from matplotlib.patches import Ellipse


def get_train_test_split(movies_metadata):
    """Get train test split of obs and new data"""
    movies_metadata = normalize_data(movies_metadata)
    features = get_features(movies_metadata)
    data = torch.tensor(features, dtype=torch.float32)
    N = 500
    data_new = data[N:2 * N].clone().detach()
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
                          width=lam[0] * 4, height=lam[1] * 4,
                          angle=np.rad2deg(np.arccos(v[0, 0])),
                          color='blue')
            ell.set_facecolor('none')
            ax.add_artist(ell)


def normalize_data(movies_metadata):
    """Return normalized movies_metadata"""
    movies_metadata['revenue'] = normalize(movies_metadata['revenue'])
    movies_metadata['budget'] = normalize(movies_metadata['budget'])
    movies_metadata['vote_average'] = normalize(
        movies_metadata['vote_average'])
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
        movies_metadata['vote_count'],
        movies_metadata['vote_average'],
        movies_metadata['popularity']
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
    Sigma_bayes_est = [
        [torch.mean(Sigma_samples[:, i, j]).item()
         for j in range(len(Sigma_samples[0]))]
        for i in range(len(Sigma_samples[0]))]
    cov = [Sigma_bayes_est for num_clusters in range(K)]
    cov = torch.tensor(cov)
    return cov


def get_bayes_estimate_mu(posterior_samples):
    """Compute bayes estimate of mu"""
    mu = [[torch.mean(posterior_samples["locs"][:, i, j]).item()
           for j in range(len(posterior_samples["locs"][0][0]))]
          for i in range(len(posterior_samples['locs'][0]))]
    mu = torch.tensor(mu)
    return mu


def get_bayes_estimate_pi(posterior_samples):
    pi = [torch.mean(posterior_samples["weights"][:, i]).item()
          for i in range(len(posterior_samples['weights'][0]))]
    return pi


def plot_mcmc_mu(posterior_samples, K, d):
    """Plot posterior of mu"""
    for i in range(K):
        trace = posterior_samples["locs"][:, i, :]
        fig = pyplot.figure(figsize=(16, 2))
        for j in range(d):
            ax1 = fig.add_subplot(
                121, xlabel="x", ylabel="Frequency", title="mu" + str(i))
            ax1.hist(trace[:, j], 50, density=True)
            ax2 = fig.add_subplot(
                122,
                xlabel="Steps",
                ylabel="Sample Values",
                title="mu" + str(i))
            ax2.plot((trace[:, j]))
        fig.show()


def plot_mcmc_pi(posterior_samples, K, d):
    """Plot posterior of pi"""
    fig = pyplot.figure(figsize=(16, 2))
    ax1 = fig.add_subplot(
        121, xlabel="x", ylabel="Frequency", title="pi")
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
            121, xlabel="x", ylabel="Frequency", title="theta")
        ax1.hist(posterior_samples["theta"][:, j], 50, density=True)
        ax2 = fig.add_subplot(
            122, xlabel="Steps", ylabel="Sample Values", title="theta")
        ax2.plot(posterior_samples["theta"][:, j])
    fig.show()


def plot_mcmc_Sigma(Sigma_samples, K, d):
    """Plot posterior of Sigma"""
    fig = pyplot.figure(figsize=(16, 2))
    ax1 = fig.add_subplot(
        121, xlabel="x", ylabel="Frequency", title="Sigma")
    ax2 = fig.add_subplot(
        122, xlabel="Steps", ylabel="Sample Values", title="Sigma")
    for i in range(d):
        for j in range(d):
            ax1.hist(Sigma_samples[:, i, j], 30, density=True)
            ax2.plot(Sigma_samples[:, i, j])
    fig.show()

def plot_gmm_results(data, mu, cov, K, d):
    """Plot how clusters look like"""
    fig = pyplot.figure(figsize=(16, 4))
    ax = fig.add_subplot(121, xlabel="revenue", ylabel="budget")
    plot(data[:, 0:2], mu[:, 0:2], cov, K=K, d=d, ax=ax)
    ax = fig.add_subplot(122, xlabel="vote_average", ylabel="vote_count")
    plot(data[:, 2:4], mu[:, 2:4], cov, K=K, d=d, ax=ax)
    fig.show()


def plot_assignments(assignment, K):
    pyplot.figure(figsize=(8, 4))
    pyplot.hist(assignment, bins=K, ec="k")
    pyplot.xlabel("pi")
    pyplot.ylabel("Frequency")
    pyplot.title("Components")
    pyplot.show()


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


def get_replicated_data(data, mu, cov, pi):
    data_rep = []
    for i in range(len(data)):
        cluster = pyro.sample('category', dist.Categorical(torch.tensor(pi)))
        idx = cluster.item()
        sample = pyro.sample("obs", dist.MultivariateNormal(mu[idx], cov[idx]))
        while sample[0] < min(data[:, 0]) or sample[1] < min(data[:, 1]):
            # Only sample valid points
            sample = pyro.sample("obs", dist.MultivariateNormal(mu[idx], cov[idx]))
        data_rep.append(sample.tolist())
    data_rep = torch.tensor(data_rep)
    return data_rep


def plot_rep_obs_new_data(data, data_rep, data_new):
    fig = pyplot.figure(figsize=(16, 4))
    ax1 = fig.add_subplot(
        121, xlabel="budget", ylabel="revenue",
        title="PPC", ylim=(-1, 6), xlim=(-1, 4))
    ax1.scatter(data_rep[:, 0], data_rep[:, 1], label="replicated data")
    ax1.scatter(data[:, 0], data[:, 1], label="observed data")
    ax1.legend()
    ax2 = fig.add_subplot(
        122, xlabel="budget", ylabel="revenue",
        title="POP-PC", ylim=(-1, 6), xlim=(-1, 4))
    ax2.scatter(data_rep[:, 0], data_rep[:, 1], label="replicated data")
    ax2.scatter(data_new[:, 0], data_new[:, 1], label="new data")
    ax2.legend()
    fig.show()


def plot_ppc_vs_poppc():
    K_variable = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    ppc_response = [1028425, 499302, 2828440, 15175103, 15331, 30597838,
                    4952241, 17231320, 6284324, 5327910, 4547856, 23852158,
                    17231514]

    pop_pc_response = [2522314, 833008, 1790494, 77044920, 3299700,
                       158990000, 8688746, 129820000, 1125285, 19079732,
                       30745118, 72734376, 94568328]

    fig = pyplot.figure(figsize=(16, 4))
    ax = fig.add_subplot(
        111, xlabel="K number of clusters", ylabel="Discrepancy", title="POP-PC vs PPC")
    ax.plot(K_variable, np.sqrt(ppc_response), label="PPC")
    ax.plot(K_variable, np.sqrt(pop_pc_response), label="POP-PC")
    ax.legend()
    fig.show()
