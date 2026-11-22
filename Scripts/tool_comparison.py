"""Compares SSI and the metrics of this paper with output of existing analysis tools.

Tools: Lizard (Java, Python, JavaScript, Go), Radon and Complexipy (Python only).
Usage:
    python tool_comparison.py --ssi-repo PATH --csb-repo PATH --hx-dir PATH --results DIR
reads the CSV files written by run_revision_analysis.py in --results and writes tool_results.json
and tool_metrics_*.csv in the same folder.
"""
import argparse, gzip, json, os, warnings
import numpy as np, pandas as pd
from scipy import stats
import lizard
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze as raw_analyze
from complexipy import code_complexity

warnings.filterwarnings('ignore')
LANGS = ['java', 'python', 'javascript', 'go']
EXT = {'java': 'x.java', 'python': 'x.py', 'javascript': 'x.js', 'go': 'x.go'}


def r4(x): return round(float(x), 4)
def sp(a, b):
    m = ~(pd.isna(a) | pd.isna(b))
    return r4(stats.spearmanr(a[m], b[m])[0])


def lizard_metrics(code, lang):
    src = 'class X {\n' + code + '\n}\n' if lang == 'java' and 'class ' not in code else code
    try:
        r = lizard.analyze_file.analyze_source_code(EXT[lang], src)
    except Exception:
        return dict(liz_ccn=np.nan, liz_nloc=np.nan, liz_tokens=np.nan, liz_found=False)
    fl = r.function_list
    if fl:
        f = max(fl, key=lambda g: g.nloc)
        return dict(liz_ccn=f.cyclomatic_complexity, liz_nloc=f.nloc, liz_tokens=f.token_count, liz_found=True)
    return dict(liz_ccn=np.nan, liz_nloc=r.nloc, liz_tokens=r.token_count, liz_found=False)


def python_tools(code):
    out = dict(rad_cc=np.nan, rad_vol=np.nan, rad_mi=np.nan, rad_sloc=np.nan, cpx_cog=np.nan)
    try:
        blocks = cc_visit(code)
        out['rad_cc'] = max(b.complexity for b in blocks) if blocks else np.nan
        out['rad_vol'] = h_visit(code).total.volume
        out['rad_mi'] = mi_visit(code, True)
        out['rad_sloc'] = raw_analyze(code).sloc
    except Exception:
        pass
    try:
        out['cpx_cog'] = code_complexity(code).complexity
    except Exception:
        pass
    return out


def run(codes, langs):
    rows = []
    for c, l in zip(codes, langs):
        d = lizard_metrics(c, l)
        d.update(python_tools(c) if l == 'python' else {})
        rows.append(d)
    return pd.DataFrame(rows)


def cv(mat):
    mat = np.asarray(mat, float)
    m = mat.mean(1); s = mat.std(1, ddof=1)
    return np.where(m != 0, s / np.abs(m), np.nan)


from run_revision_analysis import kendall_w


def main():
    ap = argparse.ArgumentParser()
    for k in ('ssi-repo', 'csb-repo', 'hx-dir', 'results'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args()
    R = {'versions': dict(lizard=lizard.version, radon=__import__('radon').__version__)}

    # reference corpus
    cs = pd.read_csv(os.path.join(a.ssi_repo, 'Datasets/codesearchnet_sample.csv'))
    ref = pd.read_csv(os.path.join(a.results, 'reference_with_metrics.csv'))
    T = run(cs.whole_func_string, cs.language)
    ref = pd.concat([ref, T], axis=1)
    T.assign(language=cs.language.values).to_csv(os.path.join(a.results, 'tool_metrics_reference.csv'), index=False)
    R['reference_found'] = {l: int(T.liz_found[cs.language == l].sum()) for l in LANGS}
    R['reference_n'] = {l: int((cs.language == l).sum()) for l in LANGS}
    py = ref[ref.language == 'python']
    R['agreement_with_own'] = {
        'cyclomatic_text_vs_lizard_ccn': sp(ref.cc_text, ref.liz_ccn),
        'tokens_vs_lizard_tokens': sp(ref.token_count, ref.liz_tokens),
        'logical_loc_vs_lizard_nloc': sp(ref.logical_loc, ref.liz_nloc),
        'python_halstead_vs_radon': sp(py.halstead_volume, py.rad_vol),
        'python_mi_vs_radon': sp(py.mi, py.rad_mi),
        'python_cognitive_vs_complexipy': sp(py.cog, py.cpx_cog),
        'python_lizard_ccn_vs_radon_cc': sp(py.liz_ccn, py.rad_cc)}
    R['ssi_vs_tool'] = {
        'lizard_ccn': {l: sp(ref.SSI[ref.language == l], ref.liz_ccn[ref.language == l]) for l in LANGS},
        'lizard_nloc': {l: sp(ref.SSI[ref.language == l], ref.liz_nloc[ref.language == l]) for l in LANGS},
        'lizard_tokens': {l: sp(ref.SSI[ref.language == l], ref.liz_tokens[ref.language == l]) for l in LANGS},
        'pooled': {'lizard_ccn': sp(ref.SSI, ref.liz_ccn), 'lizard_nloc': sp(ref.SSI, ref.liz_nloc),
                   'lizard_tokens': sp(ref.SSI, ref.liz_tokens)},
        'python': {'radon_cc': sp(py.SSI, py.rad_cc), 'radon_volume': sp(py.SSI, py.rad_vol),
                   'radon_mi': sp(py.SSI, py.rad_mi), 'complexipy_cognitive': sp(py.SSI, py.cpx_cog)}}

    # HumanEval-X
    rows = []
    for lang, f in (('python', 'h_python'), ('java', 'h_java'), ('javascript', 'h_js'), ('go', 'h_go')):
        with gzip.open(os.path.join(a.hx_dir, f + '.jsonl.gz'), 'rt') as fh:
            for line in fh:
                t = json.loads(line)
                rows.append(dict(language=lang, task_idx=int(t['task_id'].split('/')[1]), code=t['declaration'] + t['canonical_solution']))
    hx_in = pd.DataFrame(rows)
    hx = pd.read_csv(os.path.join(a.results, 'humaneval_x_with_metrics.csv'))
    TH = run(hx_in.code, hx_in.language)
    hx = pd.concat([hx.reset_index(drop=True), TH], axis=1)
    cols = {'Lizard CCN': 'liz_ccn', 'Lizard NLOC': 'liz_nloc', 'Lizard tokens': 'liz_tokens', 'SSI0': 'SSI0',
            'Node count': 'ast_nodes', 'LOC': 'phys_loc'}
    R['hx_found'] = {l: int(TH.liz_found[hx_in.language == l].sum()) for l in LANGS}
    out = {}
    cvs = {}
    for name, c in cols.items():
        piv = hx.pivot(index='task_idx', columns='language', values=c)[LANGS]
        piv = piv.dropna()
        cvs[name] = cv(piv.values)
        out[name] = dict(n_tasks=len(piv), cv_mean=r4(np.nanmean(cvs[name])), cv_median=r4(np.nanmedian(cvs[name])),
                         W=r4(kendall_w(piv.values)))
    R['hx'] = out
    R['hx_ssi0_lower_than'] = {k: int((cvs['SSI0'] < cvs[k]).sum()) for k in ('Lizard CCN', 'Lizard NLOC', 'Lizard tokens')
                               if len(cvs[k]) == len(cvs['SSI0'])}

    # CodeSizeBERT
    res = {}
    for ds, fn, lab in (('cosmic', 'code_cosmic.csv', 'cfp'), ('microm', 'code_microm.csv', 'microm')):
        src = pd.read_csv(os.path.join(a.csb_repo, 'data', fn))
        m = pd.read_csv(os.path.join(a.results, f'codesizebert_{ds}_with_metrics.csv'))
        assert len(src) == len(m), (ds, len(src), len(m))
        Tc = run(src.code, ['python'] * len(src))
        d = pd.concat([m.reset_index(drop=True), Tc], axis=1)
        Tc.to_csv(os.path.join(a.results, f'tool_metrics_{ds}.csv'), index=False)
        res[ds] = {k: sp(d[c], d[lab]) for k, c in (('SSI', 'SSI'), ('Radon CC', 'rad_cc'), ('Radon volume', 'rad_vol'),
                   ('Radon MI', 'rad_mi'), ('Complexipy cognitive', 'cpx_cog'), ('Lizard CCN', 'liz_ccn'),
                   ('Lizard NLOC', 'liz_nloc'), ('Lizard tokens', 'liz_tokens'))}
        res[ds]['n_radon_failed'] = int(d.rad_cc.isna().sum())
        res[ds]['n_lizard_failed'] = int(d.liz_ccn.isna().sum())
        res[ds]['n'] = len(d)
    R['csb'] = res
    json.dump(R, open(os.path.join(a.results, 'tool_results.json'), 'w'), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()
