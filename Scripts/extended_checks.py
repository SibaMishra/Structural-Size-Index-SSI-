"""Extension checks: held out C++, cross corpus loading stability, measurement properties, runtime.
Usage: python extended_checks.py --ssi-repo PATH --hx-dir PATH --results DIR"""
import argparse, gzip, json, os, re, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import lizard
from radon.complexity import cc_visit
from complexipy import code_complexity
from ssi_pipeline import FEATURES, ssi, extract_features
from extra_metrics import analyze
from run_revision_analysis import kendall_w, cv_rows, tucker, COUNTS, r4

warnings.filterwarnings('ignore')
LANGS5 = ['java', 'python', 'javascript', 'go', 'cpp']


def build(codes, langs):
    rows = []
    for c, l in zip(codes, langs):
        a = analyze(c, l); f = dict(a['features'])
        f.update(language=l, phys_loc=a['loc'], logical_loc=a['logical_loc'], token_count=a['token_count'],
                 halstead_volume=a['halstead_volume'], cog=a['cog'], cc_text=a['cc_text'], has_error=a['has_error'])
        rows.append(f)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    for k in ('ssi-repo', 'hx-dir', 'results'): ap.add_argument('--' + k, required=True)
    a = ap.parse_args(); R = {}
    params = pd.read_csv(os.path.join(a.results, 'frozen_parameters.csv'))
    c0 = float((params.loading * params['mean'] / params['std']).sum())
    ref = pd.read_csv(os.path.join(a.results, 'reference_with_metrics.csv'))
    hx = pd.read_csv(os.path.join(a.results, 'humaneval_x_with_metrics.csv'))

    # ---- (a) held out C++ on HumanEval-X
    rows = []
    with gzip.open(os.path.join(a.hx_dir, 'h_cpp.jsonl.gz'), 'rt') as fh:
        for line in fh:
            t = json.loads(line)
            code = t['declaration'] + t['canonical_solution']
            code = '\n'.join(l for l in code.split('\n') if not re.match(r'\s*(#include|using\s+namespace)', l))
            rows.append(dict(task_idx=int(t['task_id'].split('/')[1]), code=code))
    cp = pd.DataFrame(rows)
    cpp = build(cp.code, ['cpp'] * len(cp)); cpp['task_idx'] = cp.task_idx.values
    cpp['SSI'] = ssi(cpp, params); cpp['SSI0'] = cpp.SSI + c0
    all5 = pd.concat([hx[['language', 'task_idx', 'phys_loc', 'logical_loc', 'token_count', 'halstead_volume', 'ast_nodes', 'SSI', 'SSI0']],
                      cpp[['language', 'task_idx', 'phys_loc', 'logical_loc', 'token_count', 'halstead_volume', 'ast_nodes', 'SSI', 'SSI0']]])
    cols = {'LOC': 'phys_loc', 'Token count': 'token_count', 'Node count': 'ast_nodes', 'Halstead volume': 'halstead_volume', 'SSI0': 'SSI0', 'SSI': 'SSI'}
    out = {}; cvs = {}
    for name, c in cols.items():
        g = all5.pivot(index='task_idx', columns='language', values=c)[LANGS5]
        if name != 'SSI':
            cvs[name] = g.std(axis=1, ddof=1) / g.mean(axis=1)
            out[name] = dict(cv_mean=r4(cvs[name].mean()), cv_median=r4(cvs[name].median()))
        out.setdefault(name, {})['W'] = r4(kendall_w(g.values.astype(float)))
    R['cpp_five_languages'] = out
    R['cpp_ssi_lower_than'] = {k: int((cvs['SSI0'] < cvs[k]).sum()) for k in ('LOC', 'Node count', 'Token count', 'Halstead volume')}
    R['cpp_mean_ssi'] = r4(cpp.SSI.mean()); R['cpp_mean_loc'] = r4(cpp.phys_loc.mean()); R['cpp_parse_errors'] = int(cpp.has_error.sum())
    # C++ against the other four on the same tasks, rank correlation of SSI with node count
    R['cpp_spearman_ssi_nodes'] = r4(stats.spearmanr(cpp.SSI, cpp.ast_nodes)[0])
    cpp.to_csv(os.path.join(a.results, 'humaneval_x_cpp_with_metrics.csv'), index=False)

    # ---- (b) cross corpus loading stability on the mined functions
    m = pd.read_csv(os.path.join(a.results, 'mined_functions_scored.csv'))
    sc = StandardScaler().fit(m[FEATURES]); p = PCA().fit(sc.transform(m[FEATURES])); w = p.components_[0]
    if w[FEATURES.index('ast_nodes')] < 0: w = -w
    wref = params.loading.values
    s_refit = sc.transform(m[FEATURES]) @ w
    R['cross_corpus'] = dict(n=len(m), tucker_with_reference=r4(tucker(w, wref)), pc1_var_mined=r4(p.explained_variance_ratio_[0]),
                             spearman_refit_vs_frozen=r4(stats.spearmanr(s_refit, m.SSI)[0]), by_repo={})
    for repo, g in m.groupby('repo'):
        tr = m[m.repo != repo]; sc2 = StandardScaler().fit(tr[FEATURES]); p2 = PCA().fit(sc2.transform(tr[FEATURES])); w2 = p2.components_[0]
        if w2[FEATURES.index('ast_nodes')] < 0: w2 = -w2
        R['cross_corpus']['by_repo'][repo] = r4(tucker(w2, w))
    R['cross_corpus']['min_leave_one_repo_out_tucker'] = min(R['cross_corpus']['by_repo'].values())

    # ---- (c) measurement properties on Python reference functions
    py = ref[(ref.language == 'python') & (~ref.has_error)]
    cs = pd.read_csv(os.path.join(a.ssi_repo, 'Datasets/codesearchnet_sample.csv'))
    codes = cs.whole_func_string[py.index].tolist()
    sca = StandardScaler().fit(ref[COUNTS]); pa = PCA().fit(sca.transform(ref[COUNTS])); wa = pa.components_[0]
    wa = -wa if wa.sum() < 0 else wa
    def metrics(code):
        d = build([code], ['python']); f = d.iloc[0]
        ssi_v = float(ssi(d, params)[0])
        return dict(LOC=f.phys_loc, Nodes=f.ast_nodes, Tokens=f.token_count, Halstead=f.halstead_volume, Cognitive=f.cog,
                    CCN_text=f.cc_text, SSI=ssi_v, SSI0=ssi_v + c0, SSI_A=float(d[COUNTS].values[0] @ (wa / sca.scale_)))
    rng = np.random.RandomState(42); idx = rng.permutation(len(codes))
    pairs = [(codes[idx[2 * i]], codes[idx[2 * i + 1]]) for i in range(min(250, len(codes) // 2))]
    dev = {k: [] for k in metrics(pairs[0][0])}; mono = {k: 0 for k in dev}
    for A, B in pairs:
        mA, mB, mAB = metrics(A), metrics(B), metrics(A.rstrip() + '\n\n' + B.lstrip())
        for k in dev:
            denom = abs(mA[k] + mB[k])
            dev[k].append(abs(mAB[k] - mA[k] - mB[k]) / denom if denom else np.nan)
            mono[k] += mAB[k] > max(mA[k], mB[k])
    R['properties'] = {k: dict(additivity_deviation_median=r4(np.nanmedian(v)), additivity_deviation_mean=r4(np.nanmean(v)),
                               share_union_larger_than_parts=r4(mono[k] / len(pairs))) for k, v in dev.items()}
    empty = metrics('')
    R['properties_empty_module'] = {k: r4(v) for k, v in empty.items()}
    R['properties_n_pairs'] = len(pairs)
    R['negative_share'] = dict(SSI=r4((ref.SSI < 0).mean()), SSI0=r4((ref.SSI0 < 0).mean()))

    # ---- (d) runtime per function (ms), reference corpus
    def timeit(fn, items):
        t = time.perf_counter(); ok = 0
        for x in items:
            try: fn(*x); ok += 1
            except Exception: pass
        return (time.perf_counter() - t) / max(len(items), 1) * 1000, ok
    items = list(zip(cs.whole_func_string, cs.language)); pyitems = [(c,) for c, l in items if l == 'python']
    ext = {'java': 'x.java', 'python': 'x.py', 'javascript': 'x.js', 'go': 'x.go'}
    liz = lambda c, l: lizard.analyze_file.analyze_source_code(ext[l], c)
    R['runtime_ms_per_function'] = {
        'SSI features (parse and 32 features)': timeit(extract_features, items),
        'All baselines of this paper (one parse)': timeit(analyze, items),
        'Lizard': timeit(liz, items),
        'Radon cyclomatic (Python only)': timeit(cc_visit, pyitems),
        'Complexipy cognitive (Python only)': timeit(code_complexity, pyitems)}
    R['runtime_ms_per_function'] = {k: dict(ms=r4(v[0]), succeeded=v[1], of=len(items) if 'Python' not in k else len(pyitems))
                                    for k, v in R['runtime_ms_per_function'].items()}
    json.dump(R, open(os.path.join(a.results, 'extended_results.json'), 'w'), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()
