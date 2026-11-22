"""Compression based size baseline after Bergmans et al.: compressed size of the code tokens (comments removed).
Usage: python compression_baseline.py --ssi-repo PATH --csb-repo PATH --hx-dir PATH --results DIR"""
import argparse, gzip, json, lzma, os
import numpy as np, pandas as pd
from scipy import stats
from ssi_pipeline import parse
from extra_metrics import tokens
from run_revision_analysis import kendall_w, cv_rows, r4

FILT = [{'id': lzma.FILTER_LZMA2, 'preset': 9}]


def csize(code, lang):
    toks = tokens(parse(code, lang).root_node, lang)
    toks.sort(key=lambda t: (t[2], 0))
    text = ' '.join((t[1].decode('utf8', 'replace') if isinstance(t[1], bytes) else str(t[1])) for t in toks)
    return len(lzma.compress(text.encode('utf8'), format=lzma.FORMAT_RAW, filters=FILT))


def sp(a, b): return r4(stats.spearmanr(a, b)[0])


def main():
    ap = argparse.ArgumentParser()
    for k in ('ssi-repo', 'csb-repo', 'hx-dir', 'results'): ap.add_argument('--' + k, required=True)
    a = ap.parse_args(); R = {}
    cs = pd.read_csv(os.path.join(a.ssi_repo, 'Datasets/codesearchnet_sample.csv'))
    ref = pd.read_csv(os.path.join(a.results, 'reference_with_metrics.csv'))
    ref['comp'] = [csize(c, l) for c, l in zip(cs.whole_func_string, cs.language)]
    R['reference_spearman_with'] = {'SSI': sp(ref.comp, ref.SSI), 'AST nodes': sp(ref.comp, ref.ast_nodes), 'LOC': sp(ref.comp, ref.phys_loc),
                                    'tokens': sp(ref.comp, ref.token_count)}
    R['reference_by_language_with_ssi'] = {l: sp(g.comp, g.SSI) for l, g in ref.groupby('language')}
    rows = []
    for lang, f in (('python', 'h_python'), ('java', 'h_java'), ('javascript', 'h_js'), ('go', 'h_go')):
        for line in gzip.open(os.path.join(a.hx_dir, f + '.jsonl.gz'), 'rt'):
            t = json.loads(line)
            rows.append(dict(language=lang, task_idx=int(t['task_id'].split('/')[1]), code=t['declaration'] + t['canonical_solution']))
    hxi = pd.DataFrame(rows)
    hxi['comp'] = [csize(c, l) for c, l in zip(hxi.code, hxi.language)]
    hx = pd.read_csv(os.path.join(a.results, 'humaneval_x_with_metrics.csv'))
    hx = hx.merge(hxi[['language', 'task_idx', 'comp']], on=['language', 'task_idx'])
    cvs = {k: cv_rows(hx, c) for k, c in {'comp': 'comp', 'SSI0': 'SSI0', 'LOC': 'phys_loc', 'Node count': 'ast_nodes', 'Token count': 'token_count'}.items()}
    piv = hx.pivot(index='task_idx', columns='language', values='comp')[['java', 'python', 'javascript', 'go']]
    R['hx'] = dict(cv_mean=r4(cvs['comp'].mean()), cv_median=r4(cvs['comp'].median()), W=r4(kendall_w(piv.values.astype(float))),
                   ssi0_lower_than_comp=int((cvs['SSI0'] < cvs['comp']).sum()), comp_lower_than_loc=int((cvs['comp'] < cvs['LOC']).sum()),
                   comp_lower_than_nodes=int((cvs['comp'] < cvs['Node count']).sum()))
    for ds, fn, lab in (('cosmic', 'code_cosmic.csv', 'cfp'), ('microm', 'code_microm.csv', 'microm')):
        src = pd.read_csv(os.path.join(a.csb_repo, 'data', fn))
        m = pd.read_csv(os.path.join(a.results, f'codesizebert_{ds}_with_metrics.csv'))
        c = np.array([csize(x, 'python') for x in src.code])
        R[ds] = dict(spearman_compressed=sp(c, m[lab]), spearman_ssi=sp(m.SSI, m[lab]), spearman_loc=sp(m.phys_loc, m[lab]))
    json.dump(R, open(os.path.join(a.results, 'compression_results.json'), 'w'), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()
