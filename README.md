# Home_Credit_Default_Risks
A credit risk modeling project in Data Preparation and Visualization

In the banking and financial industry, credit risk modeling is one of the most important tasks. Instead of using complex models, data scientists must utilize the basic model because of its explainability. Therefore, the preprocessing steps need to be carefully taken. You will have to explore the data and perform data preparation to improve the performance of the default Logistic Regression model with the given dataset.

*Submissions are evaluated on GINI. Gini can be calculated via the AUC in the formula: GINI = 2 x AUC - 1*


#### [Link to Data Set](https://drive.google.com/drive/folders/1Yt3KR8-Flx2b5KczLErbU_5fjDCpvsdB?usp=sharing)


Repository Organization
------------
    ├── README.md          <- The top-level README for developers using this project.
    ├── EDA
    │   ├── 01_application_train - EDA 
    │   ├── 02_previous_application - EDA
    │   ├── 03_bureau_EDA
    │   ├── 04_bureau_balance_EDA
    │   ├── 05_credit_card_balance - EDA
    │   ├── 06_installments_payments - EDA.
    │   ├── 07_POS_CASH_balance - EDA
    │   └── utils      <- Store Essential Functions for EDA.                
    ├── processing
    │   ├── features      <- Store tables after feature engineering used to merge
    │   ├── 00_train_test_validation_code_split (split dataset to different subsets and marked by feature tvt_code) 
    │   ├── 01_application_FE_full_fill
    │   ├── 02_previous_application - FE
    │   ├── 03_bureau - FE
    │   ├── 04_bureau_balance - FE
    │   ├── 05_credit_card_balance - FE
    │   ├── 06_installments_payments - FE.
    │   ├── 07_POS_CASH_balance - FE
    │   ├── 100_combined_all_tables_fullfill
    │   └── utils_feature_engineering      <- Store Essential Functions for Feature Engineering.                
    └── model
         ├── submission   <- Store prediction results.
         └──  final_model
