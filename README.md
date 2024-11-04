# Spellchecker Evaluation Project

This repository contains a comprehensive evaluation of some existing spellchecking tools, comparing their performance, accuracy, and processing speed.

## Overview

This project evaluates the following spellchecking libraries:
- [TextBlob](https://textblob.readthedocs.io/en/dev/)
- [SymSpell](https://github.com/mammothb/symspellpy)
- [PySpellChecker](https://pypi.org/project/pyspellchecker/)
- [Spello](https://pypi.org/project/spello/)
- [Autocorrect](https://github.com/filyp/autocorrect)

## Dataset

The evaluation uses [Peter Norvig's](https://www.kaggle.com/datasets/bittlingmayer/spelling) spelling correction dataset, preprocessed into a CSV format containing correct-error word pairs. This provides a standardized way to test spellchecker performance across common spelling mistakes.

## Key Metrics

- Precision: Accuracy of corrections made
- Recall: Proportion of errors successfully corrected
- F1 Score: Harmonic mean of precision and recall
- Processing Speed: Time taken to process the dataset
- Error Analysis: Detailed breakdown of correction types

## Project Structure
```
├── data/
│   ├── raw/                        # Original datasets
│   ├── processed/                  # Preprocessed CSV files for evaluation
│   └── pythonModels/               # used Spello model
├── src/
│   ├── data_preparation.py         # Data preprocessing scripts
│   ├── spell_checker_setup.py      # Spellchecker evaluation code
│   └── visualizations.py           # Visualization utilities
├── spellchecker_evaluation.ipynb   # Main analysis notebook
├── results/
│   ├── figures/                    # Generated visualizations
│   └── metrics/                    # Evaluation results
└── requirements.txt                # Project dependencies
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/MeisterSeSe/SpellCheckerEvaluation.git
cd spellchecker-evaluation
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Evaluation

The easiest way to reproduce the results is through the Jupyter notebook:
1. Start JupyterLab:
```bash
jupyter lab
```
2. Open `notebooks/spellchecker_evaluation.ipynb`
3. Run all cells to reproduce the analysis

Alternatively, run individual components:
```bash
python src/data_perparation.py #Preprocess data
python src/spell_checker_setup.py  # Run evaluation only
python src/visualizations.py  # Generate visualizations
```
Note: To use spello, you need a pretrained model (or train one yourself xD) and place it in `data/pythonModels`.
I used this [model](https://haptik-website-images.haptik.ai/spello_models/en.pkl.zip), unzip it and place it in `data/pythonModels`.
## Results Summary

![Evalution_results](/results/graphs/spellchecker_analysis.png)
![Evalution_results](/results/graphs/success_rate.png)
Key findings from our evaluation:
- SymSpell shows the best balance of speed and accuracy
- PySpellChecker achieves equal results but is orders of magnitude slower
- TextBlob provides additional NLP features but is significantly outperformed in both the classification metrics and the speed
- Spello provides promising results but may require more efficient trainings

We recommend that symspell should be selected as the spellchecker to be utilised, as it has been demonstrated to be both effective and highly efficient.
For detailed analysis and visualizations, please direct your attention to the Jupyter notebook.
