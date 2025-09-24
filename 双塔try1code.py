# =========================
# 1. Import Libraries & Setup
# =========================
import pandas as pd
import numpy as np
import gc
import os
import json
import warnings
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import pyarrow.feather as feather
import toad

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, confusion_matrix,
    mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
)

warnings.filterwarnings("ignore")
pd.set_option('max_colwidth', 500)

# =========================
# 2. Utility Functions
# =========================
def save_feature_rank(model, filepath):
    """Save feature importance ranking from XGBoost model."""
    score_weight = model.get_score(importance_type='weight')
    score_gain = model.get_score(importance_type='gain')

    feature_rank = pd.DataFrame({
        'feature': list(score_weight.keys()),
        'weight': list(score_weight.values())
    }).set_index('feature')

    feature_rank['gain'] = pd.Series(score_gain)
    feature_rank['total_gain'] = feature_rank['weight'] * feature_rank['gain']
    feature_rank.sort_values('total_gain', ascending=False, inplace=True)
    feature_rank.to_csv(filepath)

def save_model(model, filepath, save_feature=True):
    """Save trained model and feature list."""
    model.dump_model(filepath + '_tree.txt')
    model.save_model(filepath + '_model.model')
    if save_feature:
        with open(filepath + '_feature.txt', 'w') as f:
            f.writelines([x + '\n' for x in model.feature_names])

def calculate_psi(base_df, test_df, features):
    """Calculate PSI between base and test datasets."""
    psi_df = pd.DataFrame(toad.metrics.PSI(
        base_df[features],
        test_df[features]
    )).sort_values(by=0, ascending=True)
    psi_df.rename(columns={0: 'psi'}, inplace=True)
    return psi_df

# =========================
# 3. Load & Preprocess Data
# =========================
# Withdrawal intention train & test
delqdata = pd.read_feather('/data/public_data/.../提现意愿数据_还款续借_23030101to250101.fther').set_index('first_trace_id')
order_data = pd.read_feather('/data/public_data/.../提现意愿数据_还款续借_250101to250301.fther').set_index('first_trace_id')

for df in [delqdata, order_data]:
    df['month'] = df['first_done_date'].dt.to_period('M').astype(str)

delqdata_sorted = delqdata.query("month >= '2024-01' and month <= '2025-01'") \
    .sort_values(by=['loan_account_id', 'month', 'first_done_date'])
order_data_sorted = order_data.query("month >= '2024-01' and month <= '2025-01'") \
    .sort_values(by=['loan_account_id', 'month', 'first_done_date'])

# Credit usage labels
trainlst = pd.read_feather('/data/public_data/.../trace_30daysgap_240301250101_捆绑额度以及30天内放款总金额.fther').set_index('first_trace_id')
trainlst['benbi_p30sum_afcredits_label'] = (trainlst["benbi_p30sum_afcredits"] > 0.5).astype(int)
trainlst['month'] = pd.to_datetime(trainlst['first_done_ts'] // 1000, unit='s').dt.to_period('M')
trainlst['cv'] = trainlst.index % 4

testlabel = pd.read_feather('/data/public_data/.../trace_30daysgap_testing_捆绑额度以及30天内放款总金额_modify.fther').set_index('first_trace_id')
testlabel['month'] = pd.to_datetime(testlabel['first_done_ts'] // 1000, unit='s').dt.to_period('M')
testlabel['benbi_p30sum_afcredits_label'] = (testlabel["benbi_p30sum_afcredits"] > 0.4).astype(int)
testlabel['wdr_t7'] = order_data['wdr_t7']

# Feature tables
tb1 = pd.read_feather('/data/public_data/.../循环贷场景240301250101_30daysgaptrace_selectedfea_creditusage.fther').set_index('traceId')
testall = pd.read_feather('/data/public_data/.../还款续借25年1月2月_30daysgap_selectedfea_creditusage.fther').set_index('traceId')

common_index = tb1.index.intersection(trainlst.index)
tb1_filtered = tb1.loc[common_index]

# =========================
# 4. Feature Filtering & Dataset Preparation
# =========================
remove_features = []
for c in tb1_filtered.columns:
    if any(keyword in c.lower() for keyword in [
        'idn_history_multi_loan_invalid', 'calccredits', 'withdraw',
        'curfirsttimediffbeforetime', 'curapplytimedifflastbillingtime',
        'curfinishtimediffcurpayouttime', 'idn_withdraw_history',
        'borrow_to_pay_historical_1sttrace_time', 'creditsusage', 'idn_before_order_gap'
    ]):
        remove_features.append(c)

roundfea1 = list(set(tb1_filtered.columns) - set(remove_features))

# Train/Test indices filtering
train_index = trainlst.query('index.isin(@tb1_filtered.index) and benbi_p30sum_afcredits <= 0.8 and cv == 2').index.tolist()
test_index = testlabel.query('index.isin(@testall.index) and benbi_p30sum_afcredits_label.notnull()').index.tolist()

X_train = tb1_filtered.reindex(train_index)[roundfea1]
y_train = trainlst.reindex(train_index)[["wdr_t7", "benbi_p30sum_afcredits_label"]]

X_test = testall.reindex(test_index)[roundfea1]
y_test = testlabel.reindex(test_index)[["wdr_t7", "benbi_p30sum_afcredits_label"]]

# =========================
# 5. Target Groups for PSI & Filtering
# =========================
combined_index = tb1_filtered.index.union(testall.index)

# Build month-specific target sets
target_1 = delqdata_sorted.query('index.isin(@combined_index) and risk_type == "复贷还款续借" and month == "2024-09"').index
target_5 = order_data.query('index.isin(@combined_index) and risk_type == "复贷还款续借"').index
target_10 = delqdata_sorted.query('index.isin(@combined_index) and month == "2024-09"').index
target_11 = delqdata_sorted.query('index.isin(@combined_index) and month == "2025-01"').index

# =========================
# 6. PSI Calculation
# =========================
# Example: calculate PSI for Sept 2024 vs Jan 2025 (all customers)
rank1 = pd.read_csv('/data/automl/.../round1.txt')
rank1['rank'] = range(1, len(rank1) + 1)
rank1_features = rank1['feature'].tolist()

psi1 = calculate_psi(
    tb1_filtered.query('index.isin(@target_1)')[rank1_features],
    testall.query('index.isin(@target_5)')[rank1_features],
)

psi5 = calculate_psi(
    tb1_filtered.query('index.isin(@target_10)')[rank1_features],
    testall.query('index.isin(@target_11)')[rank1_features],
)

print("PSI1 (Sept2024 vs Jan2025):\n", psi1.head())
print("PSI5 (All customers Sept2024 vs Jan2025):\n", psi5.head())

# =========================
# 7. XGBoost Training with Early Stopping
# =========================
from sklearn.metrics import roc_auc_score

# Prepare DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train.values)
dtest = xgb.DMatrix(X_test, label=y_test.values)

# Define training parameters (tuned for dual-output approximation)
params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',  # train two targets separately
    'tree_method': 'hist',
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.75,
    'eta': 0.05,
    'seed': 300,
    'alpha': 1,
    'gamma': 0.6,
    'min_child_weight': 600,
    'max_bin': 200
}

# Train model
print("Training XGBoost model...")
evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50,
    evals_result=evals_result,
    verbose_eval=100
)

# Save model and feature ranking
save_model(model, '/data/automl/duotower/dual_tower_model')
save_feature_rank(model, '/data/automl/duotower/dual_tower_feature_rank.csv')

# =========================
# 8. Evaluate Model Performance
# =========================
# Predict on test set
test_preds = model.predict(dtest)

# Split predictions for two targets
preds_wdr = test_preds  # if training one target at a time, adjust here
true_wdr = y_test["wdr_t7"].values

preds_clu = test_preds  # (same model here; you may train a separate model for CLU)
true_clu = y_test["benbi_p30sum_afcredits_label"].values

# Compute AUC for both tasks
auc_wdr = roc_auc_score(true_wdr, preds_wdr)
auc_clu = roc_auc_score(true_clu, preds_clu)

print(f"Test AUC - Withdrawal Intention (wdr_t7): {auc_wdr:.4f}")
print(f"Test AUC - Credit Usage (benbi_label): {auc_clu:.4f}")

# Plot feature importance (top 30)
xgb.plot_importance(model, importance_type='gain', max_num_features=30)
plt.title("Top 30 Feature Importances (by Gain)")
plt.show()
