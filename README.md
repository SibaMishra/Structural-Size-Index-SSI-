# Structural Size Index (SSI)

This repository holds the extraction pipeline, datasets, notebooks, and analysis scripts for the Structural Size Index (SSI), a function level size metric built from Tree-sitter abstract syntax trees. SSI is the first principal component of 32 standardized AST features.

## Structure

```
.
├── Notebook/        Notebooks for the original study (feature extraction, SSI, RQ1 to RQ4)
├── Datasets/        CSV datasets, raw and derived
├── Results/         Result tables of the original study
├── Scripts/         Python scripts for the extended analyses of the journal version
├── requirements.txt Python dependencies
└── README.md
```

## Notebooks (original study)

Run them in this order.

1. `Notebook/mp-dataset-download-ast-features.ipynb` downloads CodeSearchNet, samples 500 functions per language (seed 42), and extracts 32 AST features per function.
2. `Notebook/loc-and-cyclomatic-complexity.ipynb` computes LOC and cyclomatic complexity on the same sample.
3. `Notebook/ssi-creation.ipynb` standardizes the 32 features and fits the PCA. It writes the frozen means, standard deviations, and loadings used by everything downstream.
4. `Notebook/comparativeanalysisofsoftwaremetrics.ipynb` merges SSI with the baselines and gives the RQ1 correlation table.
5. `Notebook/cross-language-validation.ipynb` runs the leave one language out Random Forest reconstruction.
6. `Notebook/humaneval-x-cross-language-validation.ipynb` downloads HumanEval-X and runs the matched task cross language comparison.
7. `Notebook/model-validationandexplainability.ipynb` and `Notebook/random-forest-on-ssi.ipynb` hold the Random Forest self prediction and permutation importance analysis. SSI is a deterministic function of the features it is predicted from, so these notebooks do not support any result in the paper. They are kept for transparency.

## Scripts (journal version)

All scripts use fixed seeds. Run them from inside `Scripts/`, because they import each other. `Scripts/README.txt` lists the options of each one.

| Script | Purpose |
|---|---|
| `ssi_pipeline.py` | Feature extraction, frozen parameters, SSI scoring. Optional corrected Python mapping and a C++ mapping. |
| `extra_metrics.py` | Token count, Halstead volume, logical LOC, Maintainability Index, cognitive complexity. |
| `run_revision_analysis.py` | Main analysis: baselines, HumanEval-X cross language tests, CodeSizeBERT checks. |
| `tool_comparison.py` | Comparison with Lizard, Radon, and Complexipy. |
| `weighting_study.py` | Per language PCA and supervised weights. |
| `compression_baseline.py` | Compressed size baseline. |
| `mine_bugfix_functions.py` | Function level bug fix data from commit history. |
| `mine_benchmarks.py` | Function level data from Defects4J and BugsInPy. |
| `defect_analysis.py` | Defect proneness tests. |
| `defect_histogram.py` | Node type histogram model for defects. |
| `extended_checks.py` | Held out C++, cross corpus stability, measurement properties, runtime. |

External inputs that are not in this repository: the CodeSizeBERT data (`smtnkc/CodeSizeBERT`), the HumanEval-X files (`THUDM/CodeGeeX`), the Defects4J framework (`rjust/defects4j`), BugsInPy (`soarsmu/BugsInPy`), and clones of the projects they point to. Defects4J project Chart is not used because its history is not in a public git repository.

## Datasets

`Datasets/` has raw extraction outputs and derived files. `Results/dataset_with_ssi.csv` is the 2,000 function CodeSearchNet reference corpus with SSI computed, and all frozen normalization parameters are fitted on it. The `humaneval_x_*.csv` files come from the public HumanEval-X benchmark (Java, Python, JavaScript, Go; 164 matched tasks) and use the same parameters.

## Setup

```
pip install -r requirements.txt
```

## License

To be added.
