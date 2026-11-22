"""Builds a function level bug fix dataset from git history and computes metrics for each function.

For every bug fix commit (message matches a keyword rule, at most MAX_FILES source files changed), the pre-fix version
of each changed source file is parsed. A function is labeled 1 if a line deleted or replaced by the fix lies inside
it, and 0 otherwise (clean functions of the same files, at most CLEAN_PER_FILE sampled per file). This is a keyword
based heuristic with noisy labels, and the paper states so.
Usage: python mine_bugfix_functions.py --repos DIR --out FILE.csv [--max-commits 300]
"""
import argparse, os, re, subprocess, random, textwrap, hashlib
import numpy as np, pandas as pd
import lizard
from ssi_pipeline import parse
from extra_metrics import analyze

REPOS = {'requests': 'python', 'flask': 'python', 'click': 'python', 'commons-lang': 'java', 'gson': 'java', 'commons-io': 'java',
         'express': 'javascript', 'axios': 'javascript', 'gin': 'go', 'chi': 'go'}
EXT = {'python': ('.py',), 'java': ('.java',), 'javascript': ('.js',), 'go': ('.go',)}
FUNC = {'python': {'function_definition'}, 'java': {'method_declaration', 'constructor_declaration'},
        'javascript': {'function_declaration', 'method_definition'}, 'go': {'function_declaration', 'method_declaration'}}
FIX = re.compile(r'\b(fix(es|ed)?|bug(fix)?|defect|regression)\b', re.I)
NOT_FIX = re.compile(r'\b(typo|doc(s|umentation)?|readme|changelog|ci|lint|format(ting)?|style|release|bump)\b', re.I)
SKIP_DIR = re.compile(r'(^|/)(tests?|spec|docs?|examples?|benchmarks?|vendor|node_modules|test)(/|$)|_test\.go$|\.test\.js$|Test\.java$')
MAX_FILES, CLEAN_PER_FILE, MIN_LINES, MAX_LINES = 3, 4, 3, 400


def git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True, errors='replace').stdout


def functions(src, lang):
    tree = parse(src, lang); out = []; stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type in FUNC[lang]:
            code = n.text.decode('utf8', 'replace')
            if lang == 'python':
                code = textwrap.dedent(' ' * n.start_point[1] + code)
            out.append((n.start_point[0] + 1, n.end_point[0] + 1, code))
        stack.extend(n.children)
    return out


def hunks(repo, parent, commit, path):
    d = git(repo, 'diff', '-U0', '--no-color', parent, commit, '--', path); r = []
    for m in re.finditer(r'^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@', d, re.M):
        a = int(m.group(1)); b = int(m.group(2)) if m.group(2) is not None else 1
        r.append((a, a + b - 1) if b > 0 else (a, a + 1))
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--repos', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--max-commits', type=int, default=300); a = ap.parse_args()
    rng = random.Random(42); rows = []
    for name, lang in REPOS.items():
        repo = os.path.join(a.repos, name)
        log = git(repo, 'log', '--no-merges', '--format=%H%x09%P%x09%s').strip().split('\n')
        fixes = []
        for l in log:
            p = l.split('\t')
            if len(p) == 3 and p[1] and FIX.search(p[2]) and not NOT_FIX.search(p[2]):
                fixes.append((p[0], p[1].split()[0], p[2]))
        rng.shuffle(fixes); used = 0; n_before = len(rows)
        for commit, parent, msg in fixes:
            if used >= a.max_commits: break
            files = [f for f in git(repo, 'diff', '--name-only', parent, commit).split('\n')
                     if f.endswith(EXT[lang]) and not SKIP_DIR.search(f)]
            if not files or len(files) > MAX_FILES: continue
            used += 1
            for f in files:
                src = git(repo, 'show', f'{parent}:{f}')
                if not src or src.count('\n') > 5000: continue
                hs = hunks(repo, parent, commit, f)
                try: fs = functions(src, lang)
                except Exception: continue
                fs = [x for x in fs if MIN_LINES <= x[1] - x[0] + 1 <= MAX_LINES]
                bug = [x for x in fs if any(s <= x[1] and e >= x[0] for s, e in hs)]
                clean = [x for x in fs if x not in bug]; rng.shuffle(clean)
                for lab, grp in ((1, bug), (0, clean[:CLEAN_PER_FILE])):
                    for s0, e0, code in grp:
                        try: a_ = analyze(code, lang)
                        except Exception: continue
                        lz = lizard.analyze_file.analyze_source_code('x.' + EXT[lang][0].strip('.'), 'class X {\n' + code + '\n}\n' if lang == 'java' else code).function_list
                        f_ = dict(a_['features']); f_.update(repo=name, language=lang, commit=commit, file=f, start=s0, label=lab,
                                  phys_loc=a_['loc'], logical_loc=a_['logical_loc'], token_count=a_['token_count'], halstead_volume=a_['halstead_volume'],
                                  mi=a_['mi'], cog=a_['cog'], cc_text=a_['cc_text'], has_error=a_['has_error'],
                                  liz_ccn=max((g.cyclomatic_complexity for g in lz), default=np.nan),
                                  code_hash=hashlib.md5(code.encode()).hexdigest())
                        rows.append(f_)
        print(name, lang, 'fix commits used', used, 'functions', len(rows) - n_before, flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print('total', len(rows))


if __name__ == '__main__':
    main()
