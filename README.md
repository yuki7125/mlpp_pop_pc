# mlpp_pop_pc
- This repository is for the Machine Learning for Probabilistic Programming Project.
- We focus on the criticism aspect of the Box's loop, and will analyze criticism frameworks such as the Posterior Predictive Check (PPC) and Population Predictive Check (POP-PC), and test them on different types of probabilistic models that were not mentioned in the original paper.

## Directory
    .
    ├── data                    # store data here
    ├── mlpp_pop_pc             # Github source code here
    │   ├── final-project
    │   ├── requirements.txt
    └── └── README.md


## Installation
Create a Python 3 (Preferably 3.7) virtual environment (make sure you have Python 3 in /usr/bin/)
```bash
python3 -m venv venv
```
then
```bash
source venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name venv --display-name "CHOOSENAME"
```
Install requirements via pip
```bash
pip install -r requirements.txt
```

## Usage
```python
jupyter notebook
```

## About the Data
- Since the data is too big to place on Github, we have placed the required data on a Google Drive in the following link:
https://drive.google.com/open?id=1yBBesBV5kQj5aPwDz4hq6uu6uJ9AgAAM

Place the data as specified in the directory.

## References
- Ranganath & Blei 2019 Population Predictive Checks
- Salakhutdinov & Mnih 2008 Probabilistic Matrix Factorization
- Salakhutdinov & Mnih 2008 Bayesian Probabilistic Matrix Factorization using Markov Chain Monte Carlo
- Gopalan & Hofman & Blei 2013 Scalable Recommendation with Poisson Factorization
- Gelman et al. 2013 Bayesian Data Analysis
- Barnard & McCulloch & Meng 2000 Modeling Covariance Matrices in Terms of Standard Deviations and Correlations, with Application to Shrinkage

## Notes
PEP8 guidelines: https://realpython.com/python-pep8/
