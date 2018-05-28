# Naive Bayes Classifier
## Python Implementation

This repo contains a Python 3.6 implementation of the Multinomial Naive Bayes algorithm. It was designed as part of an assignment for [COMP30027 Machine Learning](https://handbook.unimelb.edu.au/2018/subjects/comp30027) at the [University of Melbourne](https://www.unimelb.edu.au/). It is not intended for use.

### Naive Bayes Overview
The [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) builds a model consisting of:  
• priors P(cj) (one per class)  
• posteriors P(xi |cj) (one per attribute value, per class)

Multinomial Naive Bayes evaluates a given test instance by taking the product of the prior probability of a given class, and the posterior probability of each attribute of the test instance given that class. The class with the highest probability given the test instance P(cj | xi) is the prediction of the classifier.

### Evaluation
The implementation uses a "testing on the training data" approach. This is not recommended in general as it can lead to model overfitting. Holdout or Cross Validation should be used in a real world setting.

### Data
Four datasets from the [UCI machine learning repository](https://archive.ics.uci.edu/ml/index.html) were used to train and evaluate the learner. They are included in this repo.
