"""Builds a function level data set from Defects4J (Java) and BugsInPy (Python) and computes the metrics of every function.

Bug and fix revisions, and the fixing patches, are taken from the two benchmarks. For each bug, the changed source files are
read at the buggy revision. A function is labeled 1 if a line deleted or replaced by the fix patch lies inside it, and 0 otherwise
(clean functions of the same files, at most CLEAN_PER_FILE per file). The group id of every row is the bug id.
Defects4J: needs the framework repository (rjust/defects4j) and upstream clones named as in D4J_REPOS. Project Chart is skipped
because its history is not in a public git repository. BugsInPy: needs soarsmu/BugsInPy and clones of the project repositories.
Usage: python mine_benchmarks.py --d4j DIR --d4j-repos DIR --bip DIR --bip-repos DIR --out FILE.csv
"""
import argparse, csv, json, os, re, random, hashlib
import numpy as np, pandas as pd
import lizard
from extra_metrics import analyze
from mine_bugfix_functions import git, functions, EXT, SKIP_DIR, MIN_LINES, MAX_LINES, CLEAN_PER_FILE

D4J_REPOS = {'Lang': 'commons-lang', 'Csv': 'commons-csv', 'Gson': 'gson', 'Jsoup': 'jsoup', 'Mockito': 'mockito', 'Math': 'commons-math',
             'Cli': 'commons-cli', 'Codec': 'commons-codec', 'Compress': 'commons-compress', 'Collections': 'commons-collections',
             'JacksonCore': 'jackson-core', 'JacksonDatabind': 'jackson-databind', 'JacksonXml': 'jackson-dataformat-xml',
             'Closure': 'closure-compiler', 'JxPath': 'commons-jxpath', 'Time': 'joda-time'}


def patch_hunks(text):
    out = {}; cur = None
    for l in text.split('\n'):
        m = re.match(r'--- a/(.+?)\s*$', l)
        if m: cur = m.group(1); out.setdefault(cur, []); continue
        m = re.match(r'@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@', l)
        if m and cur:
            a = int(m.group(1)); b = int(m.group(2)) if m.group(2) is not None else 1
            out[cur].append((a, a + b - 1) if b > 0 else (a, a + 1))
    return out


def rows_for(repo, rev, hs, lang, name, bug, rng):
    rows = []
    for f, h in hs.items():
        if not f.endswith(EXT[lang]) or SKIP_DIR.search(f): continue
        src = git(repo, 'show', f'{rev}:{f}')
        if not src or src.count('\n') > 5000: continue
        try: fs = functions(src, lang)
        except Exception: continue
        fs = [x for x in fs if MIN_LINES <= x[1] - x[0] + 1 <= MAX_LINES]
        bug_f = [x for x in fs if any(s <= x[1] and e >= x[0] for s, e in h)]
        clean = [x for x in fs if x not in bug_f]; rng.shuffle(clean)
        for lab, grp in ((1, bug_f), (0, clean[:CLEAN_PER_FILE])):
            for s0, e0, code in grp:
                try: a_ = analyze(code, lang)
                except Exception: continue
                lz = lizard.analyze_file.analyze_source_code('x' + EXT[lang][0], 'class X {\n' + code + '\n}\n' if lang == 'java' else code).function_list
                f_ = dict(a_['features']); f_.update(repo=name, language=lang, commit=bug, file=f, start=s0, label=lab,
                          phys_loc=a_['loc'], logical_loc=a_['logical_loc'], token_count=a_['token_count'], halstead_volume=a_['halstead_volume'],
                          mi=a_['mi'], cog=a_['cog'], cc_text=a_['cc_text'], has_error=a_['has_error'], hist=json.dumps(a_['counter']),
                          liz_ccn=max((g.cyclomatic_complexity for g in lz), default=np.nan), code_hash=hashlib.md5(code.encode()).hexdigest())
                rows.append(f_)
    return rows


def main():
    ap = argparse.ArgumentParser()
    for k in ('--d4j', '--d4j-repos', '--bip', '--bip-repos', '--out'): ap.add_argument(k, required=True)
    a = ap.parse_args(); rng = random.Random(42); rows = []; skipped = []
    for proj, rn in D4J_REPOS.items():
        repo = os.path.join(a.d4j_repos, rn); n0 = len(rows)
        for x in csv.DictReader(open(os.path.join(a.d4j, 'framework/projects', proj, 'active-bugs.csv'))):
            p = os.path.join(a.d4j, 'framework/projects', proj, 'patches', x['bug.id'] + '.src.patch')
            if not os.path.exists(p): skipped.append((proj, x['bug.id'])); continue
            rows += rows_for(repo, x['revision.id.buggy'], patch_hunks(open(p, errors='replace').read()), 'java', 'D4J-' + proj, f"{proj}-{x['bug.id']}", rng)
        print('Defects4J', proj, len(rows) - n0, flush=True)
    for proj in sorted(os.listdir(os.path.join(a.bip, 'projects'))):
        info = open(os.path.join(a.bip, 'projects', proj, 'project.info')).read()
        repo = os.path.join(a.bip_repos, re.search(r'github_url="(.+?)"', info).group(1).rstrip('/').split('/')[-1].lower()); n0 = len(rows)
        if not os.path.isdir(repo): print('missing repo', proj); continue
        bd = os.path.join(a.bip, 'projects', proj, 'bugs')
        for b in sorted((x for x in os.listdir(bd) if x.isdigit()), key=int):
            bi = open(os.path.join(bd, b, 'bug.info')).read(); rev = re.search(r'buggy_commit_id="(.+?)"', bi).group(1)
            rows += rows_for(repo, rev, patch_hunks(open(os.path.join(bd, b, 'bug_patch.txt'), errors='replace').read()), 'python', 'BIP-' + proj, f'{proj}-{b}', rng)
        print('BugsInPy', proj, len(rows) - n0, flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False); print('total', len(rows), 'skipped', skipped)


if __name__ == '__main__':
    main()
