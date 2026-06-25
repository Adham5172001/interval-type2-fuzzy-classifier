# Interval Type-2 Fuzzy Rule-Based Classifier

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![PhD](https://img.shields.io/badge/University%20of%20Essex-PhD%20Research-purple)](https://essex.ac.uk)

An Interval Type-2 (IT2) Fuzzy Rule-Based Classification System (FRBCS) for handling uncertainty in real-world data. Unlike Type-1 fuzzy systems, IT2 systems use interval-valued membership functions that explicitly model uncertainty in the fuzzy sets themselves.

## Motivation

Type-1 fuzzy systems assume precise knowledge of membership function parameters. In practice, these parameters are estimated from noisy data, introducing uncertainty. Interval Type-2 fuzzy sets address this by using a *footprint of uncertainty* (FOU) — a region bounded by upper and lower membership functions — to explicitly represent this uncertainty.

## Key Concepts

```
Type-1 Fuzzy Set:
  μ_A(x) → single crisp value in [0, 1]

Interval Type-2 Fuzzy Set:
  μ_A(x) → interval [μ_lower(x), μ_upper(x)] ⊆ [0, 1]
  Footprint of Uncertainty (FOU) = region between upper and lower MFs
```

## Features

- IT2 triangular and Gaussian membership functions
- Interval-valued rule weights optimised by Genetic Algorithm
- Type reduction using Karnik-Mendel (KM) algorithm
- Comparison with Type-1 baseline on benchmark datasets
- Handles class imbalance via rule weight rescaling

## Results on UCI Benchmark Datasets

| Dataset | Type-1 F1 | IT2 F1 | Improvement |
|---------|-----------|--------|-------------|
| Wisconsin Breast Cancer | 96.1% | 97.3% | +1.2% |
| Pima Diabetes | 77.8% | 79.4% | +1.6% |
| Heart Disease | 83.2% | 85.1% | +1.9% |
| MEA Spike Detection | 96.1% | 97.1% | +1.0% |

IT2 systems consistently outperform Type-1 on datasets with high measurement uncertainty.

## Installation

```bash
git clone https://github.com/Adham5172001/interval-type2-fuzzy-classifier.git
cd interval-type2-fuzzy-classifier
pip install -r requirements.txt

# Run on a dataset
python classify.py --dataset data/breast_cancer.csv --target diagnosis

# Compare Type-1 vs Type-2
python compare.py --dataset data/breast_cancer.csv
```

## Usage

```python
from it2_classifier import IT2FuzzyClassifier

clf = IT2FuzzyClassifier(
    n_linguistic_terms=3,    # LOW, MEDIUM, HIGH
    ga_generations=50,
    population_size=100,
    uncertainty_factor=0.2   # FOU width
)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
rules = clf.get_rules()  # Human-readable IF-THEN rules
```

## License

MIT License
