EVO-Boost: Evolutionary Cascaded Boosting for Financial Risk Prediction
This is the official Python implementation of EVO-Boost, a lightweight, intrinsically interpretable ensemble model designed for credit risk and bankruptcy prediction.
Model Features
Sparse shallow decision chains with evolutionary feature selection
Risk-aware sample weighting for imbalanced financial data
Intrinsic interpretability with human-readable IF–THEN rules
Fixed cascade structure (6 chains) for regulatory compliance
Constant-time inference for real-time industrial deployment
File Structure
EVO_BOOST 2.0.py: Main model and training code
give me some credit.csv: Dataset (required)
README.md: This document
Requirements
Python 3.8+
NumPy, Pandas
Scikit-learn 0.24+
How to Run
Place the dataset give me some credit.csv in the same folder
Run the main script:
plaintext
python EVO_BOOST 2.0.py
Output Includes
Average AUC, Recall, Precision, F1 over 10×5-fold cross-validation
Full list of evolved decision chains
Interpretable IF–THEN rules from the best-performing chain
Output Explanation
Chain Feature Selection Summary: Lists all 6 chains and their in-sample F1 score
EVO-Boost Rules: Extracted rules with risk level, default probability, and sample size
Quantitative Results: Mean ± std of performance metrics
Dataset
The default dataset is the public Give Me Some Credit dataset from Kaggle, used for credit default prediction.
