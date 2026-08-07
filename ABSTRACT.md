# Detecting Fraudulent Credit Card Transactions Under Extreme Class Imbalance

**Abstract**

Credit card fraud detection is a canonical example of a machine learning
problem where predictive accuracy is worthless: in a dataset where 0.17% of
transactions are fraudulent, a model that never predicts fraud is 99.83%
accurate. This project treats the imbalance itself as the central technical
challenge rather than a preprocessing detail.

I will use the ULB Credit Card Fraud Detection dataset (284,807 transactions
over two days, of which 492 are fraudulent, with 28 anonymized PCA features
plus transaction time and amount). The work proceeds in four stages. First,
metric selection: I will evaluate models with precision-recall AUC and recall
at a fixed precision or at a fixed number of daily alerts. ROC-AUC will be
reported only as a secondary number, since with 578 negatives per positive it
obscures large changes in the false positive rate. Second, a validation design
that guards against leakage: exact duplicates are removed before splitting, all
resampling and scaling is done inside cross-validation folds, and a final
temporal holdout (train on day one, test on day two) estimates performance
after a distribution change. Third, a comparison across a ladder of models: a
trivial baseline, regularized logistic regression, tree ensembles, gradient
boosting tuned against PR-AUC, and an unsupervised anomaly detector trained on
legitimate transactions only crossed with three imbalance strategies (class
weighting, undersampling, SMOTE). Fourth, explicit threshold selection and
calibration, with a cost model that weights missed frauds by transaction amount
against a fixed review cost per false alarm.

Deliverables are a reproducible codebase, a precision-recall analysis with the
chosen operating point justified in cost terms, SHAP-based interpretation of
the winning model, and a discussion of limitations.
