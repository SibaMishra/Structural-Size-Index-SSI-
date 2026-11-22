"""Per language PCA, supervised weights, and a node type histogram baseline.
Usage: python weighting_study.py --csb-repo PATH --results DIR   (reads CSVs from run_revision_analysis.py)"""
import argparse, json, os, warnings
from collections import Counter
import numpy as np, pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from ssi_pipeline import FEATURES, ASTVisitor, parse
from run_revision_analysis import kendall_w, cv_rows, r4

warnings.filterwarnings('ignore')
LANGS = ['java', 'python', 'javascript', 'go']


def per_language_ssi(ref, target):
    """Fit scaler and PCA on each language separately, score that language."""
    out = pd.Series(index=target.index, dtype=float)
    for lg in LANGS:
        tr = ref[ref.language == lg]
        sc = StandardScaler().fit(tr[FEATURES]); p = PCA().fit(sc.transform(tr[FEATURES]))
        w = p.components_[0]
        if w[FEATURES.index('ast_nodes')] < 0: w = -w
        z_tr = sc.transform(tr[FEATURES]) @ w
        c0 = float((w * sc.mean_ / sc.scale_).sum())
        m = (target.language == lg).values
        out[m] = sc.transform(target.loc[m, FEATURES]) @ w + c0
    return out


def cv_spearman(X, y, seed=42):
    rs, r2 = [], []
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed).split(X):
        sc = StandardScaler().fit(X[tr]); m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(X[tr]), y[tr])
        pr = m.predict(sc.transform(X[te]))
        rs.append(stats.spearmanr(pr, y[te])[0] if np.std(pr) > 0 else 0.0)
        r2.append(1 - ((y[te] - pr) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
    return r4(np.mean(rs)), r4(np.mean(r2))


def histogram(codes):
    rows = []
    for c in codes:
        v = ASTVisitor(); v.visit(parse(c, 'python').root_node)
        rows.append(v.counter)
    return pd.DataFrame(rows).fillna(0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--csb-repo', required=True); ap.add_argument('--results', required=True)
    a = ap.parse_args(); R = {}
    ref = pd.read_csv(os.path.join(a.results, 'reference_with_metrics.csv'))
    hx = pd.read_csv(os.path.join(a.results, 'humaneval_x_with_metrics.csv'))
    # per language PCA
    ref['SSI_lang'] = per_language_ssi(ref, ref)
    R['per_language_vs_pooled_spearman'] = {lg: r4(stats.spearmanr(ref.SSI[ref.language == lg], ref.SSI_lang[ref.language == lg])[0]) for lg in LANGS}
    hx['SSI_lang'] = per_language_ssi(ref, hx)
    cvs = {k: cv_rows(hx, c) for k, c in {'SSI_lang': 'SSI_lang', 'SSI0': 'SSI0', 'LOC': 'phys_loc', 'Node count': 'ast_nodes', 'Token count': 'token_count'}.items()}
    mats = hx.pivot(index='task_idx', columns='language', values='SSI_lang')[LANGS].values
    R['hx_per_language'] = dict(cv_mean=r4(cvs['SSI_lang'].mean()), cv_median=r4(cvs['SSI_lang'].median()), W=r4(kendall_w(mats)),
                                lower_than_loc=int((cvs['SSI_lang'] < cvs['LOC']).sum()), lower_than_nodes=int((cvs['SSI_lang'] < cvs['Node count']).sum()),
                                lower_than_tokens=int((cvs['SSI_lang'] < cvs['Token count']).sum()))
    # supervised weights on the 32 features, and node type histogram, on CodeSizeBERT labels
    R['supervised'] = {}
    for ds, fn, lab in (('cosmic', 'code_cosmic.csv', 'cfp'), ('microm', 'code_microm.csv', 'microm')):
        d = pd.read_csv(os.path.join(a.results, f'codesizebert_{ds}_with_metrics.csv'))
        src = pd.read_csv(os.path.join(a.csb_repo, 'data', fn))
        y = d[lab].values.astype(float)
        out = {}
        out['LOC'] = cv_spearman(d[['phys_loc']].values.astype(float), y)
        out['SSI only'] = cv_spearman(d[['SSI']].values, y)
        out['32 features, supervised ridge'] = cv_spearman(d[FEATURES].values.astype(float), y)
        out['32 features + LOC'] = cv_spearman(np.c_[d[FEATURES].values.astype(float), d.phys_loc.values], y)
        H = histogram(src.code)
        keep = H.columns[(H > 0).sum() >= 20]
        out['node type histogram (log1p), ridge'] = cv_spearman(np.log1p(H[keep].values), y)
        out['n_node_types'] = int(len(keep))
        R['supervised'][ds] = out
    json.dump(R, open(os.path.join(a.results, 'weighting_results.json'), 'w'), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()
