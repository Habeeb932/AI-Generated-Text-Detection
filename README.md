# Detecting LLM-Generated Text: A Comparative Study

Course project for **DA624: NLP with Large Language Models**. This repository compares classical machine-learning, CNN-based, and transformer-based approaches for distinguishing AI-generated from human-written text.

## Research question
How do classical n-gram features, a CNN text classifier, and a fine-tuned transformer compare on the selected MAGE dataset, and what might explain their different performance?

## Dataset
We use **MAGE (`yaful/MAGE`)**. The originally proposed Kaggle student-essay dataset was unavailable in our working environment, so the project proceeded with MAGE. The notebooks load the dataset through the Hugging Face `datasets` library. Please review the dataset card and its terms before reuse. The dataset itself is not included in this repository.

The prepared split sizes used in the experiments were:
- Training: 186,636 examples (93,318 per class)
- Validation: 56,740 examples
- Test: 59,981 examples

Labels: `0 = AI-generated`, `1 = human-written`.

## Models
1. **TF-IDF + Logistic Regression** (word n-grams)
2. **TF-IDF + Linear SVM** (word n-grams)
3. **Kim-style CNN** with trainable word embeddings, convolution and global max pooling
4. **DistilBERT** (`distilbert-base-uncased`) fine-tuned for binary sequence classification

## Test results
| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 73.70% | ~73.6%* | 82.17% |
| Linear SVM | 75.46% | 75.35% | 84.14% |
| Kim-style CNN | 67.46% | 66.63% | 77.42% |
| DistilBERT | 87.99% | 87.86% | 97.24% |

Exact recorded values are in [`results/test_metrics.csv`](results/test_metrics.csv). *The Logistic Regression notebook prints rounded class-wise F1 values, so its macro F1 here is an approximation.*

## Repository structure
```text
notebooks/
  01_classical_ml_svm_cnn.ipynb
  02_distilbert.ipynb
results/
  test_metrics.csv
models/
  README.md
```

## How to run
The notebooks are designed for Google Colab. Open each notebook in Colab, select a runtime appropriate to the model (GPU recommended for DistilBERT and CNN), and execute cells in order. Install any missing packages as indicated in the notebook. Dataset access requires an internet connection.

## Reproducibility notes
- The notebooks contain the data preparation, training, validation and test evaluation workflow.
- Test metrics are reported from the saved notebook outputs.
- Runtime, library versions, hardware, and random seeds can affect results.
- DistilBERT was run separately due to its computational cost; its notebook is therefore maintained separately.
- Do not compare these results with results from a different split or dataset without clearly noting the difference.

## Limitations
The results are specific to the selected MAGE splits and label distribution. They do not establish reliable detection of every future LLM or unseen generator. Dataset artifacts, source-specific writing patterns, and domain differences may influence performance. The CNN implementation is a compact course-project baseline and is not identical to every possible Kim (2014) configuration with externally pretrained GloVe/word2vec embeddings.

## Authors
Habeeb Nadendla — MTech Data Science

## Project Video
[Watch the project presentation on YouTube](https://www.youtube.com/watch?v=GBPyfzBFtMw)
