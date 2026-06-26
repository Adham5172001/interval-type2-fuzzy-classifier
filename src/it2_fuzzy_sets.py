"""Interval Type-2 Fuzzy Sets — Author: Adham Aboulkheir | University of Essex"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class IT2FuzzySet:
    name: str
    upper: Tuple[float, float, float]
    lower: Tuple[float, float, float]

    def upper_mf(self, x):
        a, b, c = self.upper
        x = np.atleast_1d(np.array(x, dtype=float))
        return np.clip(np.minimum((x-a)/(b-a+1e-9), (c-x)/(c-b+1e-9)), 0, 1)

    def lower_mf(self, x):
        a, b, c = self.lower
        x = np.atleast_1d(np.array(x, dtype=float))
        return np.clip(np.minimum((x-a)/(b-a+1e-9), (c-x)/(c-b+1e-9)), 0, 1)

    def interval_membership(self, x):
        return self.lower_mf(x), self.upper_mf(x)

def create_it2_sets(n_terms=3, uncertainty=0.15):
    centers = np.linspace(0, 1, n_terms)
    width = 1.0 / (n_terms - 1) if n_terms > 1 else 0.5
    names = ["LOW", "MEDIUM", "HIGH"] if n_terms == 3 else [f"TERM_{i}" for i in range(n_terms)]
    sets = []
    for center, name in zip(centers, names):
        upper = (max(0, center - width*(1+uncertainty)), center, min(1, center + width*(1+uncertainty)))
        lower = (max(0, center - width*(1-uncertainty)), center, min(1, center + width*(1-uncertainty)))
        sets.append(IT2FuzzySet(name=name, upper=upper, lower=lower))
    return sets

if __name__ == "__main__":
    sets = create_it2_sets(n_terms=3, uncertainty=0.15)
    print("IT2 Fuzzy Sets Demo")
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        l, u = sets[1].interval_membership(np.array([x]))
        print(f"  x={x:.2f}: MEDIUM=[{l[0]:.3f}, {u[0]:.3f}] (FOU width={u[0]-l[0]:.3f})")
