# Credit_Risk_Analysis
## Analysis Overview
This machine learning project used scikit-learn and imbalanced-learn to train and evaluate models to determine credit card risk using a credit card dataset from LendingClub, a peer-to-peer lending services company. Six different techniques were used to train and evaluate models with unbalanced classes to determine credit risk.

## Results
### RandomOverSampler model
![naive_random_oversampling](https://user-images.githubusercontent.com/97328622/172017615-b3ab9ae0-ca27-4bd6-bb86-3fabc615b32a.png)

* the balanced accuracy is 65%
* the high_risk precision is about 1% only with 57% sensitivity
* Due to the high number of the low_risk population, its precision is almost 100% with a sensitivity of 68%

### SMOTE model
![smote_oversampling](https://user-images.githubusercontent.com/97328622/172017744-ad3417c4-f986-46e2-8ebb-79c73fab44ff.png)

* results are similar to previous model
* balance accuracy score is 63%
