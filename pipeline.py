"""Leak-free evaluation of drug-response prediction.

Each example = (cell-line expression + drug fingerprint) -> ln(IC50).
Three regimes: random split, leave-cell-lines-out, leave-drugs-out.
Gene selection (variance filter) is fitted inside each training fold only.

Usage:
  python src/pipeline.py --synthetic        # smoke test on fake data
  python src/pipeline.py --data data        # real GDSC/CCLE files
"""
import argparse, numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

N_FP = 256

class TopVarGenes(BaseEstimator, TransformerMixin):
    """Keep the k highest-variance gene columns (first n_genes columns), fitted on TRAIN fold only."""
    def __init__(self, n_genes, k=200): self.n_genes, self.k = n_genes, k
    def fit(self, X, y=None):
        v = X[:, :self.n_genes].var(axis=0)
        self.idx_ = np.argsort(v)[::-1][:self.k]; return self
    def transform(self, X):
        return np.hstack([X[:, :self.n_genes][:, self.idx_], X[:, self.n_genes:]])

def fingerprint(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(smiles)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=N_FP)) if m else np.zeros(N_FP)

def synthetic(n_cells=80, n_drugs=25, n_genes=300, seed=0):
    r = np.random.default_rng(seed)
    cells = [f"CL{i}" for i in range(n_cells)]; drugs = [f"D{j}" for j in range(n_drugs)]
    expr = pd.DataFrame(r.normal(size=(n_cells, n_genes)), index=cells)
    fp = pd.DataFrame(r.integers(0, 2, size=(n_drugs, N_FP)), index=drugs)
    w = r.normal(size=(n_genes, n_drugs)) * (r.random((n_genes, n_drugs)) < 0.02)
    rows = [(c, d, expr.loc[c].values @ w[:, j] + r.normal(scale=0.5))
            for c in cells for j, d in enumerate(drugs)]
    return expr, fp, pd.DataFrame(rows, columns=["cell_line", "drug", "ln_ic50"])

def load_real(folder):
    expr = pd.read_csv(f"{folder}/expression.csv", index_col=0)
    resp = pd.read_csv(f"{folder}/drug_response.csv")
    smi = pd.read_csv(f"{folder}/drug_smiles.csv").drop_duplicates("drug").set_index("drug")
    fp = pd.DataFrame({d: fingerprint(s) for d, s in smi["smiles"].items()}).T
    resp = resp[resp.cell_line.isin(expr.index) & resp.drug.isin(fp.index)]
    print(f"Cell lines surviving join: {resp.cell_line.nunique()} | drugs: {resp.drug.nunique()} | pairs: {len(resp)}")
    return expr, fp, resp

def build(expr, fp, resp):
    X = np.hstack([expr.loc[resp.cell_line].values, fp.loc[resp.drug].values]).astype(float)
    return X, resp.ln_ic50.values, resp.cell_line.values, resp.drug.values

def models(n_genes, k):
    sel = lambda: TopVarGenes(n_genes, k)
    return {
      "ElasticNet": Pipeline([("sel", sel()), ("m", ElasticNet(alpha=0.1, max_iter=5000))]),
      "XGBoost": Pipeline([("sel", sel()), ("m", XGBRegressor(n_estimators=200, max_depth=4,
                    learning_rate=0.1, subsample=0.8, n_jobs=2, verbosity=0))]),
    }

def splits(regime, groups_cell, groups_drug, n, seed, folds=5):
    if regime == "random": return KFold(folds, shuffle=True, random_state=seed).split(np.arange(n))
    g = groups_cell if regime == "leave_cell_lines_out" else groups_drug
    # shuffle group labels with seed so repeats differ
    u = np.unique(g); r = np.random.default_rng(seed); r.shuffle(u)
    gm = {k: i for i, k in enumerate(u)}; gi = np.array([gm[x] for x in g])
    return GroupKFold(folds).split(np.arange(n), groups=gi)

def run(expr, fp, resp, k, seeds):
    X, y, gc, gd = build(expr, fp, resp); n_genes = expr.shape[1]; out = []
    for regime in ["random", "leave_cell_lines_out", "leave_drugs_out"]:
        for name, _ in models(n_genes, k).items():
            for seed in seeds:
                P, T = [], []
                for tr, te in splits(regime, gc, gd, len(y), seed):
                    mdl = models(n_genes, k)[name].fit(X[tr], y[tr])
                    P.append(mdl.predict(X[te])); T.append(y[te])
                P, T = np.concatenate(P), np.concatenate(T)
                out.append(dict(regime=regime, model=name, seed=seed,
                                RMSE=mean_squared_error(T, P) ** 0.5, R2=r2_score(T, P)))
                print(out[-1])
    return pd.DataFrame(out)

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--synthetic", action="store_true"); a.add_argument("--data", default="data")
    a.add_argument("--k", type=int, default=200); a.add_argument("--seeds", type=int, default=3)
    a = a.parse_args()
    expr, fp, resp = synthetic() if a.synthetic else load_real(a.data)
    df = run(expr, fp, resp, a.k, range(a.seeds))
    df.to_csv("results/synthetic_results.csv" if a.synthetic else "results/results.csv", index=False)
    print(df.groupby(["regime", "model"])[["RMSE", "R2"]].agg(["mean", "std"]).round(3))
