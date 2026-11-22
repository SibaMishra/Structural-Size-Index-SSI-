"""Defect proneness test on the mined bug fix functions.
Usage: python defect_analysis.py --results DIR [--input FILE --tag _suffix]   (needs mined_functions.csv and frozen_parameters.csv)"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from ssi_pipeline import FEATURES, ssi

warnings.filterwarnings('ignore')
r4 = lambda x: round(float(x), 4)
METRICS = {'LOC': 'phys_loc', 'Node count': 'ast_nodes', 'Token count': 'token_count', 'Halstead volume': 'halstead_volume',
           'Cyclomatic (text)': 'cc_text', 'Cyclomatic (Lizard)': 'liz_ccn', 'Cognitive complexity': 'cog', 'SSI': 'SSI'}


def cv_auc(d, cols, seed=42, reps=5):
    out = []
    for rep in range(reps):
        g = d.commit.astype('category').cat.codes.values
        rng = np.random.RandomState(seed + rep); perm = rng.permutation(g.max() + 1); g = perm[g]
        pred = np.zeros(len(d)); X = d[cols].values.astype(float); y = d.label.values
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            pred[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        out.append(roc_auc_score(y, pred))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--results', required=True); ap.add_argument('--input', default='mined_functions.csv'); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    d = pd.read_csv(os.path.join(a.results, a.input))
    params = pd.read_csv(os.path.join(a.results, 'frozen_parameters.csv'))
    d['SSI'] = ssi(d, params)
    d = d.drop_duplicates(['code_hash', 'label']).reset_index(drop=True)
    for c in ('liz_ccn',): d[c] = d[c].fillna(d[c].median())
    for c in ('phys_loc', 'ast_nodes', 'token_count', 'halstead_volume', 'cc_text', 'liz_ccn', 'cog'):
        d['log_' + c] = np.log1p(d[c])
    R = {'n': len(d), 'n_bug': int(d.label.sum()), 'by_language': d.groupby('language').label.agg(['size', 'sum']).to_dict('index'),
         'by_repo': d.groupby('repo').label.agg(['size', 'sum']).to_dict('index'), 'parse_error_share': r4(d.has_error.mean())}
    # AUC of each metric, pooled and by language
    R['auc'] = {k: {'all': r4(roc_auc_score(d.label, d[c]))} for k, c in METRICS.items()}
    for lg, g in d.groupby('language'):
        for k, c in METRICS.items():
            R['auc'][k][lg] = r4(roc_auc_score(g.label, g[c]))
    # AUC inside LOC strata (size held roughly fixed)
    d['loc_bin'] = pd.qcut(d.phys_loc, 5, duplicates='drop', labels=False)
    R['auc_within_loc_strata'] = {}
    for k, c in METRICS.items():
        aucs = [roc_auc_score(g.label, g[c]) for _, g in d.groupby('loc_bin') if g.label.nunique() == 2]
        R['auc_within_loc_strata'][k] = r4(np.mean(aucs))
    # incremental value, cross validated by commit
    base = ['log_phys_loc']
    models = {'LOC': base, 'LOC + node count': base + ['log_ast_nodes'], 'LOC + node count + SSI': base + ['log_ast_nodes', 'SSI'],
              'LOC + Halstead + cognitive': base + ['log_halstead_volume', 'log_cog'],
              'LOC + Halstead + cognitive + SSI': base + ['log_halstead_volume', 'log_cog', 'SSI'],
              'LOC + Lizard CCN': base + ['log_liz_ccn'], 'LOC + Lizard CCN + SSI': base + ['log_liz_ccn', 'SSI'],
              'SSI': ['SSI'], '32 features': FEATURES, 'LOC + 32 features': base + FEATURES}
    R['cv_auc'] = {k: r4(cv_auc(d, c)) for k, c in models.items()}
    # bootstrap interval for the gain of SSI over LOC + node count (resampling commits)
    rng = np.random.RandomState(1); commits = d.commit.unique(); gains = []
    for _ in range(100):
        pick = rng.choice(commits, len(commits)); b = pd.concat([d[d.commit == c] for c in pick[:len(commits)]], ignore_index=True)
        b['commit'] = np.repeat(np.arange(len(pick)), [ (d.commit == c).sum() for c in pick])[:len(b)]
        try: gains.append(cv_auc(b, models['LOC + node count + SSI'], reps=1) - cv_auc(b, models['LOC + node count'], reps=1))
        except Exception: pass
    R['gain_ssi_over_loc_nodes'] = dict(mean=r4(np.mean(gains)), ci=[r4(np.percentile(gains, 2.5)), r4(np.percentile(gains, 97.5))], n=len(gains))
    d.to_csv(os.path.join(a.results, 'mined_functions_scored%s.csv' % a.tag), index=False)
    json.dump(R, open(os.path.join(a.results, 'defect_results%s.json' % a.tag), 'w'), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()
