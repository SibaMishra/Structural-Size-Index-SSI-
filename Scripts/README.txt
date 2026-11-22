Revision analysis for the SSI paper.

Files
  ssi_pipeline.py          Feature extractor copied from the SSI repository, plus frozen parameters and SSI scoring.
  extra_metrics.py         Token count, Halstead volume, Logical LOC, Maintainability Index, cognitive complexity,
                           named node features, exception handling counts.
  run_revision_analysis.py Runs every analysis in the revised paper and writes results.json and CSV files.

Run
  pip install tree-sitter tree-sitter-language-pack pandas numpy scikit-learn scipy
  python run_revision_analysis.py --ssi-repo PATH_TO_SSI_REPO --csb-repo PATH_TO_CODESIZEBERT --hx-dir PATH_TO_HUMANEVAL_X --out results

HumanEval-X files expected in --hx-dir: h_python.jsonl.gz, h_java.jsonl.gz, h_js.jsonl.gz, h_go.jsonl.gz
All random steps use fixed seeds, so a rerun gives the same numbers.

Tool comparison (added)
  pip install lizard radon complexipy
  python tool_comparison.py --ssi-repo PATH --csb-repo PATH --hx-dir PATH --results results
  Run after run_revision_analysis.py. Writes results/tool_results.json and results/tool_metrics_*.csv.
  make_tikz.py redraws the TikZ figures from results.json.

Extension studies (added)
  python run_revision_analysis.py ... --out results_fixed --fix-python     corrected Python mapping (default keeps repository mapping)
  python weighting_study.py --csb-repo PATH --results results              per language PCA, supervised ridge, node type histogram
  python mine_bugfix_functions.py --repos DIR_WITH_CLONES --out results/mined_functions.csv
        needs clones named requests, flask, click, commons-lang, gson, commons-io, express, axios, gin, chi (full history)
  python defect_analysis.py --results results                              AUC and cross validated models on the mined functions
  python extended_checks.py --ssi-repo PATH --hx-dir DIR --results results C++, cross corpus loadings, properties, runtime
        needs h_cpp.jsonl.gz next to the other HumanEval-X files
  python make_tikz.py results figures_tikz                                 regenerates all six TikZ figures

mine_benchmarks.py : builds the Defects4J (Java) and BugsInPy (Python) function level data set. Needs rjust/defects4j, soarsmu/BugsInPy,
                     and clones of the upstream project repositories (see the docstring). Writes results/benchmark_functions.csv.
defect_analysis.py --input benchmark_functions.csv --tag _bench : runs the defect test on that file, writes defect_results_bench.json.
defect_histogram.py : node type histogram model on the benchmark functions (needs the hist column written by mine_benchmarks.py). Writes results/defect_histogram_results.json.
