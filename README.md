# Machine Learning Model Testing Framework

A framework for testing machine learning models performance on the UCI Wine Quality dataset.

## Features

- Tests multiple ML models (LinearRegression, SVR)
- Automated performance evaluation with R2 score
- GitHub Actions CI integration
- Supports both pip and Poetry dependency management

## Installation
### Using pip:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Using poetry:

```bash
poetry install
```

### Usage
#### Run tests:

```bash
# using pip
python3 test/model_testing.py

# using poetry
poetry run python3 test/model_testing.py
```

### Testing
The framework tests models against a reference R2 score of 0.30 (±0.05). Tests will pass if:
- Model achieves R2 score within 0.05 of reference
- Model achieves R2 score above 0.35

### Dependencies
- Python >=3.12
- numpy >=2.1
- pandas >=2.2
- scikit-learn >=1.5
- ucimlrepo >=0.0.7

### CI/CD
Includes GitHub Actions workflows for automated testing with:
- Dependency caching
- Multiple Python versions
- Both pip and Poetry environments