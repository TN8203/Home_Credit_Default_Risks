# Home_Credit_Default_Risks
A credit risk modeling project in Data Preparation and Visualization

In the banking and financial industry, credit risk modeling is one of the most important tasks. Instead of using complex models, data scientists must utilize the basic model because of its explainability. Therefore, the preprocessing steps need to be carefully taken. You will have to explore the data and perform data preparation to improve the performance of the default Logistic Regression model with the given dataset.

### Evaluation
Submissions are evaluated on GINI. Gini can be calculated via the AUC in the formula: GINI = 2 x AUC - 1

### Submission File
For each SK_ID_CURR in the test set, you must predict a probability for the TARGET variable. The file should contain a header and have the following format:

SK_ID_CURR,TARGET
5, 0.5
13,0.5
16,0.5
etc.

### This is the notebook for the final Credit Default Risks project by Group 11, DSEB K63, NEU
### Team member
   * Tran Hai Nam
   * Vu Mai Dung
   * Nguyen Phuong Thao

#### [Link to Presentation Slide] (https://drive.google.com/file/d/1Fd0XV_igjwwUWq57-9pmoh8Xbqon4fEG/view?usp=sharing)


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

