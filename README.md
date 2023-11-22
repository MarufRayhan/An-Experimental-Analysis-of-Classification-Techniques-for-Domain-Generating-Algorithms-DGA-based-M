#An Experimental Analysis of Classification Techniques for Domain Generating Algorithms (DGA) based Malicious Domains Detection

This repository contains work on the classification of malicious and non-malicious URLs using Machine Learning by tuning the n-gram range. The research paper can be found [here](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=JekLRZ4AAAAJ&citation_for_view=JekLRZ4AAAAJ:u5HHmVD_uO8C)

## Installation

To run the code snippets, follow these steps:

1. Install virtualenv:
   ```pip install virtualenv
      virtualenv mypython
      pip install requirements_latest.txt
      source mypython/bin/activate
   ```

## Repository Contents

- `dataPreprocessing.zip`: Contains code for dataset modification.
- `final__csv.csv`: Contains the labeled URLs, where malicious are labeled as 1 and non-malicious as 0.
- `res.csv`: Contains only benign URLs (80 classes).
- `Modified_final_updated_26_07_2020.ipynb`: Includes all necessary code for classification.
- `accuracy_check.ipynb`: Contains functions for accuracy checking, used in `Modified_final_updated_26_07_2020.ipynb`.
- `cnn_1d.ipynb`: Contains all necessary code for classification using a 1D Convolutional Neural Network.
- `tuning_logisticRegression.ipynb` and `tuning_svm.ipynb`: Include grid search code to find the best parameters.

## Tuning Instructions

To check and run different n-gram ranges and Machine Learning models, use `Modified_final_updated_26_07_2020.ipynb` and modify the variables as follows:

### Example 1:
- `n1 = 1` (n1 and n2 for N-gram range. E.g., (1,1) will be unigram)
- `n2 = 1`
- `model_name = "LinearSVC"`

### Example 2:
- `n1 = 2`
- `n2 = 2`
- `model_name = "MultinomialNB"`

## Running the CNN 1D Model

For the CNN 1D model, run the `cnn_1d.ipynb` notebook.

