"""Analyses added for the revision of the SSI paper.

Usage:
    python run_revision_analysis.py --ssi-repo PATH --csb-repo PATH --hx-dir PATH --out PATH

--ssi-repo  clone of the SSI repository (needs Datasets/ and Results/)
--csb-repo  clone of smtnkc/CodeSizeBERT (needs data/code_cosmic.csv, data/code_microm.csv)
--hx-dir    folder with h_python.jsonl.gz, h_java.jsonl.gz, h_js.jsonl.gz, h_go.jsonl.gz
            from THUDM/CodeGeeX (codegeex/benchmark/humaneval-x/<lang>/data/humaneval_<lang>.jsonl.gz)
--out       folder for results.json and the CSV files
Everything uses fixed seeds, so a rerun gives the same numbers.
"""
import argparse
import gzip
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from extra_metrics import analyze
from ssi_pipeline import FEATURES, frozen_parameters, ssi

warnings.filterwarnings('ignore')
SEED = 42
LANGS = ['java', 'python', 'javascript', 'go']
COUNTS = ['function_count', 'parameter_count', 'variable_count', 'declaration_count', 'if_count', 'for_count',
          'while_count', 'switch_count', 'return_count', 'call_count', 'argument_count', 'assignment_count',
          'binary_expression_count', 'identifier_count', 'block_count', 'expression_count', 'literal_count',
          'ast_nodes', 'leaf_nodes', 'internal_nodes']
LOG_COLS = COUNTS + ['max_depth', 'cyclomatic_estimate']
DEAD_FOR_PYTHON = ['parameter_count', 'expression_count', 'expression_density', 'statement_density', 'avg_parameters']
REDUNDANT = ['internal_nodes', 'internal_ratio', 'statement_density']


def r4(x):
    return None if x is None else float(np.round(x, 4))


def spearman(a, b):
    return float(stats.spearmanr(a, b)[0])


def pearson(a, b):
    r, p = stats.pearsonr(a, b)
    return float(r), float(p)


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a, b):
        b1 = np.c_[np.ones(len(b)), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


def tucker(a, b):
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def kendall_w(mat):
    """mat: objects x raters. Kendall's W with the tie correction."""
    n, m = mat.shape
    ranks = np.column_stack([stats.rankdata(mat[:, j]) for j in range(m)])
    s = ((ranks.sum(axis=1) - ranks.sum(axis=1).mean()) ** 2).sum()
    t = 0.0
    for j in range(m):
        _, c = np.unique(mat[:, j], return_counts=True)
        t += ((c ** 3) - c).sum()
    return float(12 * s / (m ** 2 * (n ** 3 - n) - m * t))


def mean_pairwise_spearman(mat):
    m = mat.shape[1]
    vals = [spearman(mat[:, i], mat[:, j]) for i in range(m) for j in range(i + 1, m)]
    return float(np.mean(vals))


def cv_rows(df, col):
    g = df.pivot(index='task_idx', columns='language', values=col)[LANGS]
    return g.std(axis=1, ddof=1) / g.mean(axis=1)


def sign_fix(w, cols):
    return -w if w[cols.index('ast_nodes')] < 0 else w


# ------------------------------------------------------------------ weightings

def fit_weighting(ref, name):
    """Return a function that scores a feature table with weights frozen on ref."""
    cols = FEATURES
    if name == 'pca':
        sc = StandardScaler().fit(ref[cols]); p = PCA().fit(sc.transform(ref[cols]))
        w = sign_fix(p.components_[0], cols)
        return lambda d: sc.transform(d[cols]) @ w
    if name == 'equal':
        sc = StandardScaler().fit(ref[cols])
        return lambda d: sc.transform(d[cols]).mean(axis=1)
    if name == 'factor':
        sc = StandardScaler().fit(ref[cols]); fa = FactorAnalysis(n_components=1, random_state=SEED).fit(sc.transform(ref[cols]))
        w = sign_fix(fa.components_[0], cols)
        return lambda d: sc.transform(d[cols]) @ w
    if name == 'log':
        def tr(d):
            x = d[cols].copy()
            x[LOG_COLS] = np.log1p(x[LOG_COLS])
            return x
        sc = StandardScaler().fit(tr(ref)); p = PCA().fit(sc.transform(tr(ref)))
        w = sign_fix(p.components_[0], cols)
        return lambda d: sc.transform(tr(d)) @ w
    if name == 'nonredundant':
        c = [x for x in cols if x not in REDUNDANT]
        sc = StandardScaler().fit(ref[c]); p = PCA().fit(sc.transform(ref[c]))
        w = p.components_[0]; w = -w if w[c.index('ast_nodes')] < 0 else w
        return lambda d: sc.transform(d[c]) @ w
    if name == 'no_dead':
        c = [x for x in cols if x not in DEAD_FOR_PYTHON]
        sc = StandardScaler().fit(ref[c]); p = PCA().fit(sc.transform(ref[c]))
        w = p.components_[0]; w = -w if w[c.index('ast_nodes')] < 0 else w
        return lambda d: sc.transform(d[c]) @ w
    raise ValueError(name)


WEIGHTINGS = ['pca', 'equal', 'factor', 'log', 'nonredundant', 'no_dead']


def build_table(codes, langs):
    rows = []
    for code, lang in zip(codes, langs):
        a = analyze(code, lang)
        f = dict(a['features'])
        f.update(language=lang, phys_loc=a['loc'], logical_loc=a['logical_loc'], token_count=a['token_count'],
                 halstead_volume=a['halstead_volume'], mi=a['mi'], cog=a['cog'], cc_text=a['cc_text'],
                 handlers=a['handlers'], raises=a['raises'], has_error=a['has_error'])
        f.update({'named_' + k: v for k, v in a['named'].items()})
        rows.append(f)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ssi-repo', required=True)
    ap.add_argument('--csb-repo', required=True)
    ap.add_argument('--hx-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fix-python', action='store_true', help='count Python parameters and expression statements that the repository mapping misses')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from ssi_pipeline import set_fix_python
    set_fix_python(a.fix_python)
    R = {'fix_python': bool(a.fix_python)}

    # ---------------------------------------------------------------- reference corpus
    cs = pd.read_csv(os.path.join(a.ssi_repo, 'Datasets/codesearchnet_sample.csv'))
    repo_feat = pd.read_csv(os.path.join(a.ssi_repo, 'Datasets/ast_features_final.csv'))
    repo_ssi = pd.read_csv(os.path.join(a.ssi_repo, 'Results/dataset_with_ssi.csv'))
    ref = build_table(cs.whole_func_string, cs.language)
    ref['function'] = cs.func_name.values
    if a.fix_python:
        nonpy = (cs.language != 'python').values
        assert np.abs(ref[FEATURES].values[nonpy] - repo_feat[FEATURES].values[nonpy]).max() < 1e-9, 'non Python features differ'
    else:
        assert np.abs(ref[FEATURES].values - repo_feat[FEATURES].values).max() < 1e-9, 'features differ from repository'
    params, pca = frozen_parameters(ref)
    params.to_csv(os.path.join(a.out, 'frozen_parameters.csv'), index=False)
    ref['SSI'] = ssi(ref, params)
    if not a.fix_python:
        assert np.abs(ref.SSI - repo_ssi.SSI).max() < 1e-9, 'SSI differs from repository'
    w = params.loading.values
    c0 = float((w * params['mean'].values / params['std'].values).sum())
    ref['SSI0'] = ref.SSI + c0
    R['reference'] = dict(
        n=len(ref), pc1_var=r4(pca.explained_variance_ratio_[0]), pc2_var=r4(pca.explained_variance_ratio_[1]),
        ssi_sd=r4(ref.SSI.std(ddof=1)), c0=r4(c0), ssi0_min=r4(ref.SSI0.min()),
        neg_loadings={k: r4(v) for k, v in zip(FEATURES, w) if v < 0},
        parse_errors=ref.groupby('language').has_error.sum().astype(int).to_dict(),
        parse_errors_total=int(ref.has_error.sum()),
        features_reproduced_exactly=True)
    pc2 = pd.Series(pca.components_[1], index=FEATURES)
    R['reference']['pc2_top'] = {k: r4(pc2[k]) for k in pc2.abs().sort_values(ascending=False).index[:6]}
    ref.to_csv(os.path.join(a.out, 'reference_with_metrics.csv'), index=False)

    # ---------------------------------------------------------------- RQ1
    base = {'Physical LOC': 'phys_loc', 'Cyclomatic Complexity': 'cc_text', 'AST Node Count': 'ast_nodes',
            'Maximum AST Depth': 'max_depth', 'Logical LOC': 'logical_loc', 'Token count': 'token_count',
            'Halstead volume': 'halstead_volume', 'Maintainability Index': 'mi', 'Cognitive complexity': 'cog'}
    R['rq1'] = {}
    for k, c in base.items():
        r, p = pearson(ref.SSI, ref[c])
        R['rq1'][k] = dict(r=r4(r), p=float(p), rho=r4(spearman(ref.SSI, ref[c])))
    R['rq1_by_language'] = {}
    for lg in LANGS:
        d = ref[ref.language == lg]
        R['rq1_by_language'][lg] = {k: dict(r=r4(pearson(d.SSI, d[c])[0]), rho=r4(spearman(d.SSI, d[c])))
                                    for k, c in [('LOC', 'phys_loc'), ('AST Node Count', 'ast_nodes'),
                                                 ('Cyclomatic Complexity', 'cc_text')]}
    # divergence pair
    ia = ref.index[ref.function.str.contains('_plot_debug_slopes_directions', regex=False)][0]
    ib = ref.index[ref.function.str.contains('IrafPar.pretty', regex=False)][0]
    z = (ref[FEATURES].values - params['mean'].values) / params['std'].values
    contrib = pd.Series(w * (z[ib] - z[ia]), index=FEATURES)
    R['divergence_python'] = dict(
        ssi_a=r4(ref.SSI[ia]), ssi_b=r4(ref.SSI[ib]), nodes_a=int(ref.ast_nodes[ia]), nodes_b=int(ref.ast_nodes[ib]),
        gap=r4(ref.SSI[ib] - ref.SSI[ia]), sum_of_terms=r4(contrib.sum()),
        top_terms={k: r4(contrib[k]) for k in contrib.abs().sort_values(ascending=False).index[:5]},
        if_count=[int(ref.if_count[ia]), int(ref.if_count[ib])],
        for_count=[int(ref.for_count[ia]), int(ref.for_count[ib])],
        while_count=[int(ref.while_count[ia]), int(ref.while_count[ib])])
    # residual analysis on all 2,000
    X1 = np.c_[np.ones(len(ref)), ref.ast_nodes]
    res = ref.SSI.values - X1 @ np.linalg.lstsq(X1, ref.SSI.values, rcond=None)[0]
    def resid(v):
        return v - X1 @ np.linalg.lstsq(X1, v, rcond=None)[0]
    ref['loops'] = ref.for_count + ref.while_count
    R['residual'] = dict(
        r2_nodes=r4(1 - res.var() / ref.SSI.var()), resid_sd=r4(res.std(ddof=1)),
        partial_r_if=r4(np.corrcoef(res, resid(ref.if_count.values.astype(float)))[0, 1]),
        partial_r_loops=r4(np.corrcoef(res, resid(ref.loops.values.astype(float)))[0, 1]),
        partial_r_cyclomatic=r4(np.corrcoef(res, resid(ref.cyclomatic_estimate.values.astype(float)))[0, 1]))

    # ---------------------------------------------------------------- RQ2 extras
    zc = pd.Series(w ** 2, index=FEATURES)
    groups = {'Topology': ['max_depth', 'ast_nodes', 'leaf_nodes', 'internal_nodes', 'leaf_ratio', 'internal_ratio', 'nesting_factor'],
              'Control Flow': ['if_count', 'for_count', 'while_count', 'switch_count', 'return_count', 'branch_density', 'cyclomatic_estimate'],
              'Declarations': ['function_count', 'parameter_count', 'variable_count', 'declaration_count', 'avg_parameters']}
    groups['Expressions'] = [f for f in FEATURES if f not in sum(groups.values(), [])]
    R['group_share'] = {k: dict(share=r4(100 * zc[v].sum()), n=len(v), per_feature=r4(100 * zc[v].sum() / len(v)))
                        for k, v in groups.items()}
    # bootstrap of the PCA fit
    rng = np.random.default_rng(SEED)
    var1, ldg = [], []
    for _ in range(1000):
        idx = rng.integers(0, len(ref), len(ref))
        sc = StandardScaler().fit(ref[FEATURES].iloc[idx]); p = PCA().fit(sc.transform(ref[FEATURES].iloc[idx]))
        var1.append(p.explained_variance_ratio_[0]); ldg.append(sign_fix(p.components_[0], FEATURES))
    ldg = np.array(ldg)
    R['bootstrap'] = dict(
        pc1_var_ci=[r4(np.percentile(var1, 2.5)), r4(np.percentile(var1, 97.5))],
        loading_ci={f: [r4(np.percentile(ldg[:, i], 2.5)), r4(np.percentile(ldg[:, i], 97.5))]
                    for i, f in enumerate(FEATURES) if f in ('ast_nodes', 'identifier_count', 'cyclomatic_estimate', 'if_count', 'nesting_factor')},
        tucker_min=r4(min(tucker(l, w) for l in ldg)))
    # alternative weightings on the reference corpus
    alt_ref = {}
    for nm in WEIGHTINGS:
        alt_ref[nm] = fit_weighting(ref, nm)(ref)
    R['alt_vs_ssi'] = {nm: r4(spearman(ref.SSI, alt_ref[nm])) for nm in WEIGHTINGS}
    # named nodes and exception features, refit
    named_cols = ['named_' + f for f in FEATURES]
    nm_in = ref[named_cols].rename(columns=lambda c: c[6:])
    sc = StandardScaler().fit(nm_in); p = PCA().fit(sc.transform(nm_in))
    s_named = sc.transform(nm_in) @ sign_fix(p.components_[0], FEATURES)
    R['named_nodes'] = dict(spearman_with_ssi=r4(spearman(ref.SSI, s_named)), pearson_with_ssi=r4(pearson(ref.SSI, s_named)[0]),
                            pc1_var=r4(p.explained_variance_ratio_[0]))
    f34 = ref[FEATURES + ['handlers', 'raises']]
    sc = StandardScaler().fit(f34); p = PCA().fit(sc.transform(f34))
    w34 = sign_fix(p.components_[0], FEATURES + ['handlers', 'raises'])
    s34 = sc.transform(f34) @ w34
    top = pd.Series(w34, index=FEATURES + ['handlers', 'raises']).abs().sort_values(ascending=False)
    R['exception_features'] = dict(
        spearman_with_ssi=r4(spearman(ref.SSI, s34)), pearson_with_ssi=r4(pearson(ref.SSI, s34)[0]),
        pc1_var=r4(p.explained_variance_ratio_[0]),
        loading_handlers=r4(w34[-2]), loading_raises=r4(w34[-1]),
        top5=[k for k in top.index[:5]],
        functions_with_handlers=int((ref.handlers > 0).sum()), functions_with_raises=int((ref.raises > 0).sum()),
        by_language=ref.groupby('language')[['handlers', 'raises']].sum().astype(int).to_dict('index'))
    # additive variant
    sca = StandardScaler().fit(ref[COUNTS]); pa = PCA().fit(sca.transform(ref[COUNTS]))
    wa = pa.components_[0]; wa = -wa if wa.sum() < 0 else wa
    ssi_a = ref[COUNTS].values @ (wa / sca.scale_)
    R['ssi_a'] = dict(n_features=len(COUNTS), pc1_var=r4(pa.explained_variance_ratio_[0]), min_loading=r4(wa.min()),
                      all_loadings_nonnegative=bool((wa >= 0).all()), negative=[c for c, v in zip(COUNTS, wa) if v < 0],
                      spearman_with_ssi=r4(spearman(ref.SSI, ssi_a)), pearson_with_ssi=r4(pearson(ref.SSI, ssi_a)[0]),
                      min_value=r4(ssi_a.min()))
    # leave one language out loading stability
    R['loading_stability'] = {}
    for lg in LANGS:
        tr, te = ref[ref.language != lg], ref[ref.language == lg]
        sc = StandardScaler().fit(tr[FEATURES]); p = PCA().fit(sc.transform(tr[FEATURES]))
        wl = sign_fix(p.components_[0], FEATURES)
        s_refit = sc.transform(te[FEATURES]) @ wl
        R['loading_stability'][lg] = dict(tucker=r4(tucker(wl, w)), spearman=r4(spearman(s_refit, te.SSI)),
                                          pc1_var=r4(p.explained_variance_ratio_[0]))

    # ---------------------------------------------------------------- RQ3 HumanEval-X
    rows = []
    for key, lang in [('python', 'python'), ('java', 'java'), ('js', 'javascript'), ('go', 'go')]:
        with gzip.open(os.path.join(a.hx_dir, f'h_{key}.jsonl.gz'), 'rt') as fh:
            for line in fh:
                t = json.loads(line)
                rows.append(dict(language=lang, task_id=t['task_id'], task_idx=int(t['task_id'].split('/')[1]),
                                 code=t['declaration'] + t['canonical_solution']))
    hx_in = pd.DataFrame(rows)

    def strip_boilerplate(code, lang):
        code = re.sub(r'^\s*import\s*\([^)]*\)\s*', '', code, flags=re.M)
        return '\n'.join(l for l in code.split('\n') if not re.match(r'\s*(import\s|package\s|from\s+\S+\s+import)', l))

    hx = build_table(hx_in.code, hx_in.language)
    for c in ('task_id', 'task_idx'):
        hx[c] = hx_in[c].values
    hx['SSI'] = ssi(hx, params)
    hx['SSI0'] = hx.SSI + c0
    repo_hx = pd.read_csv(os.path.join(a.ssi_repo, 'Results/humaneval_x_with_ssi.csv'))
    m = hx.merge(repo_hx[['task_id', 'language', 'SSI', 'LOC']], on=['task_id', 'language'], suffixes=('', '_repo'))
    R['hx_reproduction'] = dict(n=len(m), max_ssi_diff=float(np.abs(m.SSI - m.SSI_repo).max()), max_loc_diff=float(np.abs(m['phys_loc'] - m.LOC).max()))
    hx.to_csv(os.path.join(a.out, 'humaneval_x_with_metrics.csv'), index=False)

    def hx_block(h, tag):
        out = {}
        for sh in (5, 10, 20):
            h[f'SSI+{sh}'] = h.SSI + sh
        cols = {'LOC': 'phys_loc', 'SSI+5': 'SSI+5', 'SSI+10': 'SSI+10', 'SSI+20': 'SSI+20', 'SSI0': 'SSI0',
                'Logical LOC': 'logical_loc', 'AST Node Count': 'ast_nodes', 'Token count': 'token_count',
                'Halstead volume': 'halstead_volume'}
        cv = {k: cv_rows(h, c) for k, c in cols.items()}
        out['cv'] = {k: dict(mean=r4(v.mean()), median=r4(v.median())) for k, v in cv.items()}
        out['tasks_ssi_lower_than'] = {}
        for var in ('SSI+5', 'SSI+10', 'SSI+20', 'SSI0'):
            out['tasks_ssi_lower_than'][var] = {k: int((cv[var] < cv[k]).sum()) for k in
                                                ('LOC', 'Logical LOC', 'AST Node Count', 'Token count', 'Halstead volume')}
        agree = {}
        for k, c in {'LOC': 'phys_loc', 'Logical LOC': 'logical_loc', 'Token count': 'token_count', 'AST Node Count': 'ast_nodes',
                     'Halstead volume': 'halstead_volume', 'SSI': 'SSI'}.items():
            mat = h.pivot(index='task_idx', columns='language', values=c)[LANGS].values.astype(float)
            agree[k] = dict(W=r4(kendall_w(mat)), rho=r4(mean_pairwise_spearman(mat)))
        out['rank_agreement'] = agree
        return out

    R['hx'] = hx_block(hx.copy(), 'full')
    h2 = hx_in.copy()
    h2['code'] = [strip_boilerplate(c, l) for c, l in zip(h2.code, h2.language)]
    hxt = build_table(h2.code, h2.language)
    for c in ('task_id', 'task_idx'):
        hxt[c] = h2[c].values
    hxt['SSI'] = ssi(hxt, params); hxt['SSI0'] = hxt.SSI + c0
    R['hx_trimmed'] = hx_block(hxt.copy(), 'trimmed')
    # variant without the five features that are zero for Python
    nd = fit_weighting(ref, 'no_dead')
    hxn = hx.copy(); hxn['SSI'] = nd(hx)
    mats = hxn.pivot(index='task_idx', columns='language', values='SSI')[LANGS].values
    R['hx_no_dead_features'] = dict(W=r4(kendall_w(mats)), rho=r4(mean_pairwise_spearman(mats)))
    # sanity: does the number of tasks where Python is the smallest SSI change?
    piv = hx.pivot(index='task_idx', columns='language', values='SSI')[LANGS]
    R['hx_ssi_by_language_mean'] = {l: r4(piv[l].mean()) for l in LANGS}
    R['hx_loc_by_language_mean'] = {l: r4(hx.pivot(index='task_idx', columns='language', values='phys_loc')[l].mean()) for l in LANGS}

    # ---------------------------------------------------------------- RQ4 CodeSizeBERT
    cos = pd.read_csv(os.path.join(a.csb_repo, 'data/code_cosmic.csv'))
    mic = pd.read_csv(os.path.join(a.csb_repo, 'data/code_microm.csv'))
    cos['lang'] = 'python'; mic['lang'] = 'python'
    tc, tm_ = build_table(cos.code, cos.lang), build_table(mic.code, mic.lang)
    for d, src in ((tc, cos), (tm_, mic)):
        d['SSI'] = ssi(d, params)
        for nm in WEIGHTINGS:
            d['w_' + nm] = fit_weighting(ref, nm)(d)
    tc = pd.concat([tc, cos.drop(columns=['code', 'docstring', 'lang'])], axis=1)
    tm_ = pd.concat([tm_, mic.drop(columns=['code', 'docstring', 'lang'])], axis=1)
    tc.to_csv(os.path.join(a.out, 'codesizebert_cosmic_with_metrics.csv'), index=False)
    tm_.to_csv(os.path.join(a.out, 'codesizebert_microm_with_metrics.csv'), index=False)
    norm = lambda s: re.sub(r'\s+', '', s)
    py_sample = set(norm(c) for c in cs.whole_func_string[cs.language == 'python'])
    R['csb'] = dict(
        n_cosmic=len(tc), n_microm=len(tm_), cfp_max=int(tc.cfp.max()), microm_max=int(tm_.microm.max()),
        cfp_median=float(tc.cfp.median()), microm_median=float(tm_.microm.median()),
        cfp_zero=int((tc.cfp == 0).sum()), parse_errors=dict(cosmic=int(tc.has_error.sum()), microm=int(tm_.has_error.sum())),
        overlap_with_sample=dict(cosmic=int(sum(norm(c) in py_sample for c in cos.code)),
                                 microm=int(sum(norm(c) in py_sample for c in mic.code))),
        ssi_mean=dict(cosmic=r4(tc.SSI.mean()), microm=r4(tm_.SSI.mean()), reference_python=r4(ref.SSI[ref.language == 'python'].mean())))
    metrics = {'SSI': 'SSI', 'LOC': 'phys_loc', 'AST Node Count': 'ast_nodes', 'Halstead volume': 'halstead_volume',
               'Maintainability Index': 'mi', 'Cognitive complexity': 'cog', 'Logical LOC': 'logical_loc', 'Token count': 'token_count'}
    labels = {'cosmic': (tc, ['cfp', 'entry', 'read', 'write', 'exit']),
              'microm': (tm_, ['microm', 'interaction', 'communication', 'process'])}
    R['csb_corr'] = {}
    for ds, (d, labs) in labels.items():
        R['csb_corr'][ds] = {}
        for lab in labs:
            R['csb_corr'][ds][lab] = {k: r4(spearman(d[c], d[lab])) for k, c in metrics.items()}
            R['csb_corr'][ds][lab]['SSI given LOC'] = r4(partial_spearman(d.SSI, d[lab], d.phys_loc))
            R['csb_corr'][ds][lab]['SSI given AST Node Count'] = r4(partial_spearman(d.SSI, d[lab], d.ast_nodes))
            R['csb_corr'][ds][lab]['p_SSI'] = float(stats.spearmanr(d.SSI, d[lab])[1])
    # sensitivity: only functions that Tree-sitter parses without an error node
    R['csb_clean_subset'] = {}
    for ds, (d, tot) in {'cosmic': (tc, 'cfp'), 'microm': (tm_, 'microm')}.items():
        dd = d[~d.has_error]
        R['csb_clean_subset'][ds] = dict(
            n=len(dd), **{k: r4(spearman(dd[c], dd[tot])) for k, c in metrics.items()},
            ssi_given_loc=r4(partial_spearman(dd.SSI, dd[tot], dd.phys_loc)))
    # indentation check: share of functions in which a block body sits at the same indent as its header
    R['csb_alt'] = {ds: {nm: r4(spearman(d['w_' + nm], d[tot])) for nm in WEIGHTINGS}
                    for ds, (d, tot) in {'cosmic': (tc, 'cfp'), 'microm': (tm_, 'microm')}.items()}
    # AUC for zero CFP
    pos = (tc.cfp > 0).astype(int)
    R['csb_auc_cfp_positive'] = dict(SSI=r4(roc_auc_score(pos, tc.SSI)), LOC=r4(roc_auc_score(pos, tc['phys_loc'])),
                                     AST_nodes=r4(roc_auc_score(pos, tc.ast_nodes)))
    # cross validated regression
    models = {'LOC': ['phys_loc'], 'LOC + nodes': ['phys_loc', 'ast_nodes'], 'LOC + nodes + SSI': ['phys_loc', 'ast_nodes', 'SSI'],
              'LOC + Halstead + cognitive': ['phys_loc', 'halstead_volume', 'cog'],
              'LOC + Halstead + cognitive + SSI': ['phys_loc', 'halstead_volume', 'cog', 'SSI']}

    def oof(d, y, cols):
        pred = np.zeros(len(d)); cnt = np.zeros(len(d))
        for tr, te in RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED).split(d):
            sc = StandardScaler().fit(d.iloc[tr][cols])
            lr = LinearRegression().fit(sc.transform(d.iloc[tr][cols]), y.iloc[tr])
            pred[te] += lr.predict(sc.transform(d.iloc[te][cols])); cnt[te] += 1
        return pred / cnt

    R['csb_reg'] = {}
    rng = np.random.default_rng(SEED)
    for ds, (d, tot) in {'cosmic': (tc, 'cfp'), 'microm': (tm_, 'microm')}.items():
        y = d[tot].reset_index(drop=True); d = d.reset_index(drop=True)
        P = {k: oof(d, y, c) for k, c in models.items()}
        R['csb_reg'][ds] = {k: dict(r2=r4(r2_score(y, P[k])), mae=r4(np.abs(y - P[k]).mean())) for k in models}
        for a_, b_ in [('LOC + nodes', 'LOC + nodes + SSI'), ('LOC + Halstead + cognitive', 'LOC + Halstead + cognitive + SSI'),
                       ('LOC', 'LOC + nodes')]:
            diffs = []
            for _ in range(2000):
                i = rng.integers(0, len(y), len(y))
                diffs.append(r2_score(y.iloc[i], P[b_][i]) - r2_score(y.iloc[i], P[a_][i]))
            R['csb_reg'][ds][f'delta {a_} -> {b_}'] = dict(delta=r4(r2_score(y, P[b_]) - r2_score(y, P[a_])),
                                                           ci=[r4(np.percentile(diffs, 2.5)), r4(np.percentile(diffs, 97.5))])

    with open(os.path.join(a.out, 'results.json'), 'w') as fh:
        json.dump(R, fh, indent=1)
    print('done; results in', os.path.join(a.out, 'results.json'))


if __name__ == '__main__':
    main()
