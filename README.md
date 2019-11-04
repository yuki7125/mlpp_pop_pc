# mlpp_pop_pc
- This repository is for the Machine Learning for Probabilistic Programming Project.
- We focus on the criticism aspect of the Box's loop, and will analyze criticism frameworks such as the Posterior Predictive Check (PPC) and Population Predictive Check (POP-PC), and test them on different types of probabilistic models that were not mentioned in the original paper.

## Directory
    .
    ├── data                    # store data here
    ├── mlpp_pop_pc             # Github source code here
    │   ├── GMM
    │   ├── PMF
    └── └── README.md


## Installation
Create a Python 3.7 virtual environment (make sure you have Python 3.7 in /usr/bin/)
```bash
virtualenv -p /usr/bin/python3.7 --no-site-packages venv
```
then
```bash
source venv/bin/activate
```

Install requirements via pip
```bash
pip install -r requirements.txt
```

## Usage
```python
jupyter notebook
```

## Notes
PEP8 guidelines: https://realpython.com/python-pep8/
