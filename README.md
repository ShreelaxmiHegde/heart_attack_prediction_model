## Problem Statement
Predict the likelihood of a heart attack using patient health data.

## Models Evaluated
* Logistic Regression
* Gaussian Naive Bayes
* K-Nearest Neighbors (KNN)

## Model Selection

The following models were trained and evaluated using Accuracy, Precision, Recall, and F1-score:
Check: [Reports](https://github.com/ShreelaxmiHegde/heart_attack_prediction_model/reports)

| Model                              | Accuracy | Precision | Recall | F1-Score |
| ---------------------------------- | -------- | --------- | ------ | -------- |
| K-Nearest Neighbors (GridSearchCV) | 90.2%    | 93.3%     | 87.5%  | 90.3%    |
| Logistic Regression                | 88.5%    | 87.9%     | 90.6%  | 89.2%    |
| Gaussian Naive Bayes               | 86.9%    | 90.0%     | 84.4%  | 87.1%    |

### Selected Model: K-Nearest Neighbors (KNN)

Initially, Logistic Regression achieved the highest recall (90.6%), making it the strongest candidate for minimizing false negatives. However, after hyperparameter tuning using GridSearchCV, KNN achieved the highest accuracy (90.2%), precision (93.3%), and F1-score (90.3%) while maintaining a competitive recall of 87.5%.

Although KNN's recall was 3.1 percentage points lower than Logistic Regression, the improvements in overall predictive performance were considered significant. Therefore, KNN was selected as the final model for deployment.

---

Dataset Source: [Heart.csv](https://github.com/ShreelaxmiHegde/heart_attack_prediction_model/data/heart.csv)
