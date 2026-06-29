"""
Real differential expression analysis on GSE71083 (Akassoglou lab, GEO).
Rat microglia: unstimulated (n=2) vs fibrin-stimulated for 6h (n=2).
Affymetrix Rat Gene 1.0 ST array, RMA-normalized log2 values (as deposited by the lab).

This is an independent re-analysis using a simple t-test (NOT the original
limma moderated-t pipeline used in the published paper). With n=2 per group,
results are illustrative/exploratory, not a replication of the paper's
statistics. Output is used only for an honest, clearly-labeled personal demo.
"""
import pandas as pd
import numpy as np
from scipy import stats

# --- Load expression matrix ---
# series_matrix_table starts after the line "!series_matrix_table_begin"
with open("series_matrix.txt") as f:
    lines = f.readlines()
start = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_begin")) + 1
end = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_end"))
table_lines = lines[start:end]

from io import StringIO
expr = pd.read_csv(StringIO("".join(table_lines)), sep="\t")
expr.columns = [c.strip('"') for c in expr.columns]
expr["ID_REF"] = expr["ID_REF"].astype(str)
expr = expr.dropna()
expr = expr.set_index("ID_REF")
expr.columns = ["Control_1", "Control_2", "Fibrin_1", "Fibrin_2"]
print("Expression matrix shape (probes with values in all 4 samples):", expr.shape)

# --- Load platform annotation: probe ID -> gene symbol ---
with open("platform.annot") as f:
    plines = f.readlines()
pstart = next(i for i, l in enumerate(plines) if l.startswith("!platform_table_begin")) + 1
pend = next(i for i, l in enumerate(plines) if l.startswith("!platform_table_end"))
annot = pd.read_csv(StringIO("".join(plines[pstart:pend])), sep="\t", dtype=str)
annot["ID"] = annot["ID"].astype(str)
id2sym = dict(zip(annot["ID"], annot["Gene symbol"]))

def first_symbol(s):
    if pd.isna(s) or s == "":
        return None
    return str(s).split("///")[0]

expr["gene_symbol"] = [first_symbol(id2sym.get(i)) for i in expr.index]
expr = expr.dropna(subset=["gene_symbol"])
print("Probes with annotated gene symbol:", expr.shape)

# --- Differential expression: fibrin-stimulated vs unstimulated ---
ctrl_cols = ["Control_1", "Control_2"]
fib_cols = ["Fibrin_1", "Fibrin_2"]

log2fc = expr[fib_cols].mean(axis=1) - expr[ctrl_cols].mean(axis=1)
tstat, pval = stats.ttest_ind(expr[fib_cols].values, expr[ctrl_cols].values, axis=1)

results = pd.DataFrame({
    "gene_symbol": expr["gene_symbol"],
    "log2FC": log2fc,
    "pvalue": pval,
}).dropna()

# collapse duplicate probes per gene by taking the most significant probe
results = results.sort_values("pvalue").drop_duplicates("gene_symbol", keep="first")

# FDR (Benjamini-Hochberg)
results = results.sort_values("pvalue").reset_index(drop=True)
n = len(results)
results["rank"] = np.arange(1, n + 1)
results["padj"] = results["pvalue"] * n / results["rank"]
results["padj"] = results["padj"][::-1].cummin()[::-1].clip(upper=1.0)
results = results.drop(columns="rank")

results.to_csv("de_results.csv", index=False)
print(results.head(20))
print("\nTotal genes tested:", len(results))
print("Genes with nominal p<0.05:", (results.pvalue < 0.05).sum())

# Genes of known relevance to fibrin/neuroinflammation biology, if present
candidates = ["Il6", "Tnf", "Il1b", "Ccl2", "Cxcl10", "Nos2", "Ptgs2", "Il1a",
              "Ccl5", "Cxcl1", "Itgam", "Nfkb1", "Tlr4", "C3", "Cd68"]
present = results[results.gene_symbol.isin(candidates)]
print("\nCandidate innate-immune genes found in this array/probe set:")
print(present)
