"""Type-1 vs IT2 Comparison — Author: Adham Aboulkheir | University of Essex"""
import numpy as np, matplotlib, os, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from src.it2_fuzzy_sets import create_it2_sets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier

def main():
    print("Type-1 vs IT2 Fuzzy Classifier Comparison")
    os.makedirs("outputs", exist_ok=True)
    X, y = load_breast_cancer(return_X_y=True)
    scaler = MinMaxScaler()
    X_s = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.3, stratify=y, random_state=42)
    t1 = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    t1.fit(X_train, y_train)
    t1_f1 = f1_score(y_test, t1.predict(X_test), average="weighted")
    sets = create_it2_sets(n_terms=3, uncertainty=0.15)
    def augment(X_data):
        n, d = X_data.shape
        extra = np.zeros((n, d*3))
        for j in range(d):
            for k, s in enumerate(sets):
                l, u = s.interval_membership(X_data[:, j])
                extra[:, j*3+k] = (l+u)/2
        return np.hstack([X_data, extra])
    it2 = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    it2.fit(augment(X_train), y_train)
    it2_f1 = f1_score(y_test, it2.predict(augment(X_test)), average="weighted")
    print(f"  Type-1 F1: {t1_f1:.4f}")
    print(f"  IT2 F1:    {it2_f1:.4f} (+{(it2_f1-t1_f1)*100:.2f}%)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1117")
    for ax in axes: ax.set_facecolor("#161b22")
    x = np.linspace(0, 1, 200)
    colors = ["#ff7b72", "#f4a261", "#3fb950"]
    for s, color in zip(sets, colors):
        l, u = s.interval_membership(x)
        axes[0].fill_between(x, l, u, alpha=0.3, color=color, label=f"{s.name} FOU")
        axes[0].plot(x, u, color=color, linewidth=2)
        axes[0].plot(x, l, color=color, linewidth=1, linestyle="--")
    axes[0].set_title("IT2 Membership Functions (with FOU)", color="white")
    axes[0].set_xlabel("Feature Value", color="white"); axes[0].set_ylabel("Membership", color="white")
    axes[0].legend(facecolor="#161b22", labelcolor="white", fontsize=8); axes[0].tick_params(colors="white"); axes[0].grid(alpha=0.3, color="#21262d")
    axes[1].bar(["Type-1", "IT2 (Ours)"], [t1_f1, it2_f1], color=["#58a6ff", "#00c9b1"], alpha=0.85)
    axes[1].set_title("F1-Score Comparison (Breast Cancer)", color="white")
    axes[1].set_ylabel("F1-Score", color="white"); axes[1].tick_params(colors="white"); axes[1].grid(axis="y", alpha=0.3, color="#21262d"); axes[1].set_ylim(0.92, 0.99)
    for i, (name, score) in enumerate(zip(["Type-1", "IT2"], [t1_f1, it2_f1])):
        axes[1].text(i, score+0.001, f"{score:.4f}", ha="center", color="white", fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/it2_fuzzy_results.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Saved: outputs/it2_fuzzy_results.png")

if __name__ == "__main__":
    main()
