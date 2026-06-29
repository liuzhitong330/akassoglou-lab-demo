"""
Generate real figures from the GSE71083 re-analysis (no synthetic data).
- PCA of the 4 real samples (all annotated probes)
- Volcano plot of the real log2FC / p-value results from analyze.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from io import StringIO

with open("series_matrix.txt") as f:
    lines = f.readlines()
start = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_begin")) + 1
end = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_end"))
expr = pd.read_csv(StringIO("".join(lines[start:end])), sep="\t").dropna()
expr.columns = [c.strip('"') for c in expr.columns]
expr = expr.set_index("ID_REF")
expr.columns = ["Control_1", "Control_2", "Fibrin_1", "Fibrin_2"]

# --- PCA ---
X = expr.T.values  # 4 samples x probes
pca = PCA(n_components=2)
coords = pca.fit_transform(X)
groups = ["Control", "Control", "Fibrin", "Fibrin"]
colors = {"Control": "#2255aa", "Fibrin": "#c03030"}

fig, ax = plt.subplots(figsize=(7, 6))
for i, (x, y) in enumerate(coords):
    ax.scatter(x, y, s=160, color=colors[groups[i]], edgecolor="white", linewidth=1.2, zorder=3)
    ax.annotate(expr.columns[i], (x, y), textcoords="offset points", xytext=(8, 8), fontsize=12, color="#333")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=13)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=13)
ax.set_title("GSE71083: rat microglia, all probes", fontsize=14)
ax.tick_params(labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("pca.svg", format="svg")
plt.close(fig)

# --- Volcano ---
res = pd.read_csv("de_results.csv")
res["neglog10p"] = -np.log10(res["pvalue"].clip(lower=1e-300))

candidates = ["Il6", "Tnf", "Il1b", "Ccl2", "Cxcl10", "Nos2", "Ptgs2", "Il1a",
              "Ccl5", "Cxcl1", "Itgam", "Nfkb1", "Tlr4", "C3"]

fig, ax = plt.subplots(figsize=(9, 7.5))
sig_up = (res.log2FC > 1) & (res.pvalue < 0.05)
sig_dn = (res.log2FC < -1) & (res.pvalue < 0.05)
ns = ~(sig_up | sig_dn)
# rasterize the dense background cloud so the SVG stays small and crisp;
# axes/text/labels remain vector for sharpness at any zoom level
ax.scatter(res.log2FC[ns], res.neglog10p[ns], s=14, color="#ccc", alpha=0.6, linewidth=0, label="NS", rasterized=True)
ax.scatter(res.log2FC[sig_up], res.neglog10p[sig_up], s=16, color="#c03030", alpha=0.75, linewidth=0, label="Up in fibrin", rasterized=True)
ax.scatter(res.log2FC[sig_dn], res.neglog10p[sig_dn], s=16, color="#2255aa", alpha=0.75, linewidth=0, label="Down in fibrin", rasterized=True)

lab = res[res.gene_symbol.isin(candidates)].copy()
lab = lab.sort_values("log2FC")
# spread labels vertically so they don't overlap each other or the points
used_y = []
for _, row in lab.iterrows():
    x, y = row.log2FC, row.neglog10p
    ty = y
    while any(abs(ty - uy) < 0.35 for uy in used_y):
        ty += 0.35
    used_y.append(ty)
    ax.scatter([x], [y], s=55, color="black", zorder=4, edgecolor="white", linewidth=0.6)
    ax.annotate(row.gene_symbol, (x, y), xytext=(x + 0.15, ty),
                fontsize=13, fontweight="bold", color="#111", zorder=5,
                arrowprops=dict(arrowstyle="-", color="#999", linewidth=0.7, shrinkA=0, shrinkB=4),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

ax.axvline(0, color="#ddd", linewidth=0.8)
ax.set_xlabel("log2 fold change (fibrin vs. unstimulated)", fontsize=13)
ax.set_ylabel("-log10(p-value)", fontsize=13)
ax.set_title("GSE71083: fibrin-stimulated vs. unstimulated rat microglia", fontsize=14)
ax.tick_params(labelsize=11)
ax.legend(fontsize=11, frameon=False, loc="upper left", markerscale=1.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("volcano.svg", format="svg", dpi=150)
plt.close(fig)

print("Wrote pca.svg and volcano.svg")
