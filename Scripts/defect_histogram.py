"""Node type histogram model for the Defects4J and BugsInPy functions (needs results/benchmark_functions.csv with a hist column).
Logistic regression with an L2 penalty on log1p node type counts, folds grouped by bug, 5 repeats, same folds as defect_analysis.py.
Usage: python defect_histogram.py --results DIR     writes defect_histogram_results.json"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from ssi_pipeline import FEATURES

warnings.filterwarnings('ignore')
r4 = lambda x: round(float(x), 4)


def cv(X, y, g0, C, seed=42, reps=5):
    out = []
    for rep in range(reps):
        rng = np.random.RandomState(seed + rep); g = rng.permutation(g0.max() + 1)[g0]; pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
            pred[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        out.append(roc_auc_score(y, pred))
    return r4(np.mean(out))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', required=True); a = ap.parse_args()
    d = pd.read_csv(os.path.join(a.results, 'benchmark_functions.csv')).drop_duplicates(['code_hash', 'label']).reset_index(drop=True)
    H = pd.DataFrame([{f'{lg}:{k}': v for k, v in json.loads(s).items()} for s, lg in zip(d['hist'], d.language)]).fillna(0)
    H = H.loc[:, (H > 0).sum() >= 20]; Xh = np.log1p(H.values)
    y = d.label.values; g0 = d.commit.astype('category').cat.codes.values
    loc = np.log1p(d[['phys_loc']].values); nodes = np.log1p(d[['ast_nodes']].values)
    f32 = d[FEATURES].values.astype(float)
    R = {'n': len(d), 'n_node_types': int(H.shape[1]), 'cv_auc': {}}
    for C in (0.01, 0.1, 1.0):
        R['cv_auc'][str(C)] = {'LOC': cv(loc, y, g0, C), 'LOC + node count': cv(np.hstack([loc, nodes]), y, g0, C),
                               '32 features': cv(f32, y, g0, C), 'node type histogram': cv(Xh, y, g0, C),
                               'LOC + node type histogram': cv(np.hstack([loc, Xh]), y, g0, C)}
        print(C, R['cv_auc'][str(C)], flush=True)
    R['by_language'] = {}
    for lg in ('java', 'python'):
        m = (d.language == lg).values; gl = d.commit[m].astype('category').cat.codes.values
        R['by_language'][lg] = {'LOC': cv(loc[m], y[m], gl, 0.1), 'node type histogram': cv(Xh[m], y[m], gl, 0.1), '32 features': cv(f32[m], y[m], gl, 0.1)}
    json.dump(R, open(os.path.join(a.results, 'defect_histogram_results.json'), 'w'), indent=1); print(json.dumps(R['by_language']))


if __name__ == '__main__':
    main()
