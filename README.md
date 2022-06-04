# Credit_Risk_Analysis
## Analysis Overview
This machine learning project used scikit-learn and imbalanced-learn to train and evaluate models to determine credit card risk using a credit card dataset from LendingClub, a peer-to-peer lending services company. Six different techniques were used to train and evaluate models with unbalanced classes to determine credit risk.

## Results
### RandomOverSampler model
![naive_random_oversampling](https://user-images.githubusercontent.com/97328622/172017615-b3ab9ae0-ca27-4bd6-bb86-3fabc615b32a.png)

* the balanced accuracy is 65%
* the high_risk precision is about 1% only with 57% sensitivity which makes F1 of 2% only
* Due to the high number of the low_risk population, its precision is almost 100% with a sensitivity of 68%

### SMOTE model
![smote_oversampling](https://user-images.githubusercontent.com/97328622/172017744-ad3417c4-f986-46e2-8ebb-79c73fab44ff.png)

* balance accuracy score is 63%
* the high_risk precision is about 1% only with 62% sensitivity which makes F1 of 2% only
* Due to the high number of the low_risk population, its precision is almost 100% with a sensitivity of 63%

### ClusterCentroids model
![cluster_centroids_model](https://user-images.githubusercontent.com/97328622/172018142-6b4d4c2a-b384-4a23-81a1-a44e5c848a13.png)

* Balanced accuracy is down to 51%
* The high_risk precision is still 1% only with 59% sensitivity which makes a F1 of 1%.
* Due to the high number of false positives, the low_risk sensitivity is only 43%

### SMOTEENN model
![smoteenn_model](https://user-images.githubusercontent.com/97328622/172018198-5c95e24d-8c67-4e61-b996-e7c86827a0ba.png)

* balanced accuracy score is 65%
* The high_risk precision is still 1% only with 70% sensitivity which makes a F1 of only 2%.
* Due to the high number of false positives, the low_risk sensitivity is 61%.

### BalancedRandomForestClassifier model
![balanced_random_forest_classifier_model](https://user-images.githubusercontent.com/97328622/172018293-b1a02631-e6af-452c-a0a3-96cc2e838c01.png)

* balanced accuracy score is 79%
* The high_risk precision is 3% only with 70% sensitivity which makes a F1 of only 6%.
* Due to the lower number of false positives, the low_risk sensitivity is 87% with 100% precision.

### EasyEnsembleClassifier model
![easy_ensemble_classifier_model](https://user-images.githubusercontent.com/97328622/172018535-7a239652-c92c-4212-bdfb-ab1273001dec.png)

* balanced accuracy is 93%
* The high_risk precision is low at 9% with 92% precision which makes F1 only 16%
* Due to a lower number of false positives, the low_risk sensitivity is now 94% with 100% precision

## Summary
All of the machine learning models had low precision scores for the high-risk loans in accurately predicting positives. The balanced accuracy score for the models varied with the lowest score for the undersampling method and a high score with the AdaBoost classifier. Recall scores also varied between models with the lowest scores for undersampling and highest with the classifying methods. Of the models created, the Easy Ensemble AdaBoost Classifier would be the best model to use to predict credit risk due to the high recall scores for both high and low risk loans, as well as an accuracy score of 92.5%. The precision for this model is still very off, indicating that the positives are not necessarily accurate, and so this model could be much improved and training and testing more data before putting it into use.
