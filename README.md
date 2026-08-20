# CNNLP_LAB

## Prerequisites

- Python 3.9+
- Jupyter Notebook / JupyterLab or Google Colab

Install the common dependencies:

```bash
pip install pandas numpy matplotlib nltk spacy scikit-learn
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

Some experiments read data from a mounted Google Drive folder (`/content/drive/MyDrive/CNNLP_LAB/...`) when run on Colab — update the `folder_path` variable to a local path if running elsewhere.

## Experiments

| # | Title | Topics Covered |
|---|-------|-----------------|
| 1 | Data Analysis | pandas groupby/aggregation, matplotlib (bar, pie, histogram, scatter) |
| 2 | Basic Text Preprocessing | text cleaning, tokenization (regex/NLTK/spaCy), stop-word removal |


Each notebook is self-contained: markdown cells state the task, followed by the code and its output/visualization.

## How to Run

1. Clone the repo and `cd` into it.
2. Place any required input files in `data/` (see each experiment's first cell for the expected filename).
3. Launch Jupyter: `jupyter notebook` and open the relevant `Experiment_N.ipynb`.
4. Run cells top to bottom — later cells in some notebooks depend on variables defined earlier (e.g. `df`, `text`).

## Conventions

- Each experiment starts with a markdown header naming the experiment and its sub-tasks (e.g. `Experiment 1.1`, `1.2`, ...).
- Charts are generated inline with matplotlib (`plt.show()`); no files are saved unless a task explicitly asks for export.
- Keep raw datasets and text inputs out of version control if they're large or sensitive — use `data/` with a `.gitignore` entry, or note the source instead.

## Notes

- Results and print-outs in each notebook are the source of truth; written explanations in this README summarize but don't replace them.
- Update the table above as new experiments are added.
