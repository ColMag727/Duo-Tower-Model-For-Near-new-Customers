# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# =========================
# 2. Helper Functions
# =========================
def model_pred(train_set, oot_set, features, target):
    """
    Train a LightGBM binary classifier and return:
    - oot_score: predicted probabilities on OOT set
    - diff: difference between mean(oot_score) and mean(train_actual)
    """
    lgb_train = lgb.Dataset(train_set[features], train_set[target])
    lgb_oot = lgb.Dataset(oot_set[features], oot_set[target], reference=lgb_train)

    params = {
        'n_estimators': 200,
        'learning_rate': 0.1079,
        'num_leaves': 32,
        'max_depth': 3,
        'min_child_samples': 190,
        'min_child_weight': 7.02,
        'min_split_gain': 0.1975,
        'colsample_bytree': 0.77,
        'subsample': 0.54,
        'bagging_seed': 27,
        'reg_alpha': 1.27,
        'reg_lambda': 2.98,
        'scale_pos_weight': 1,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': 10
    }

    model = lgb.LGBMClassifier(**params)
    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]

    # Fit model
    if len(features) == 1:
        model.fit(
            train_set[features].values.reshape(-1, 1), train_set[target],
            eval_set=[(train_set[features].values.reshape(-1, 1), train_set[target])],
            eval_metric='auc',
            callbacks=callbacks
        )
        oot_score = model.predict_proba(oot_set[features].values.reshape(-1, 1))[:, 1]
    else:
        model.fit(
            train_set[features], train_set[target],
            eval_set=[(train_set[features], train_set[target])],
            eval_metric='auc',
            callbacks=callbacks
        )
        oot_score = model.predict_proba(oot_set[features])[:, 1]

    train_mean = train_set[target].mean()
    return oot_score, np.mean(oot_score) - train_mean


def population_shift(train_set, oot_set, feature_list, target):
    """
    Iteratively add features one by one to maximize shift in predicted mean,
    returning:
    - shift_vars: ordered list of features contributing to population shift
    - pop_shift: incremental shift contributions
    - oot_preds: predicted OOT mean after adding each feature
    """
    shift_vars = []
    remain_vars = feature_list.copy()
    pop_shift = []
    oot_preds = []

    # Baseline oot prediction
    oot_pred_baseline = model_pred(train_set, oot_set, feature_list, target)[1] + train_set[target].mean()

    for i in range(len(feature_list)):
        remain_vars = [v for v in remain_vars if v not in shift_vars]
        diffs = []
        for var in remain_vars:
            model_features = shift_vars + [var]
            diffs.append(model_pred(train_set, oot_set, model_features, target)[1])

        abs_diffs = [abs(d) for d in diffs]
        best_var = remain_vars[np.argmax(abs_diffs)]
        shift_vars.append(best_var)
        print(f"Selected shift var {i+1}: {best_var}")

        if i == 0:
            pop_shift.append(diffs[np.argmax(abs_diffs)])
        else:
            pop_shift.append(diffs[np.argmax(abs_diffs)] - oot_preds[-1] + train_set[target].mean())

        oot_preds.append(model_pred(train_set, oot_set, shift_vars, target)[1] + train_set[target].mean())

    return shift_vars, pop_shift, oot_preds


# =========================
# 3. Load and Preprocess Data
# =========================
order_data = pd.read_parquet('/data/public_data/cpu2/aojiali/delqdata_up0416/结清还款续借一次风控通过_2502010401_update0606_addtype47.parquet')
order_data['month'] = order_data['first_done_date'].dt.to_period('M').astype(str)

# Keep earliest trace per user-month
order_sorted = order_data.sort_values(by=['loan_account_id', 'month', 'first_done_date'])
order_sorted = order_sorted.loc[order_sorted.groupby(['loan_account_id', 'month'])['first_done_date'].idxmin()]

print(f"After filtering earliest trace: {order_sorted.shape}, Original: {order_data.shape}")

ldin = pd.read_csv('/data/public_data/cpu1/jingminzhou/cust_loan_account_id.csv')
print(f"Loan account id range: {ldin.loan_account_id.min()} ~ {ldin.loan_account_id.max()}")

# Load feature tables for Feb & Mar
test02 = pd.read_parquet('/data/public_data/cpu3/aojiali/etldata_次新客M3_250603/次新客round1fea_2502过件.parquet')
test03 = pd.read_parquet('/data/public_data/cpu3/aojiali/etldata_次新客M3_250603/次新客round1fea_2503过件.parquet')

# Filter qualified near-new customers
query_cond = '((sd_gap<=90 and risk_type.isin(["结清复贷","复贷还款续借"])) or (risk_type=="首贷还款续借")) and bf_user_type1!="47"'
testindex02 = order_sorted.query(f"{query_cond} and index.isin(@test02.index) and loan_account_id.isin(@ldin.loan_account_id.tolist())").index.tolist()
testindex03 = order_sorted.query(f"{query_cond} and index.isin(@test03.index) and loan_account_id.isin(@ldin.loan_account_id.tolist())").index.tolist()

print(f"Qualified customers: Feb={len(testindex02)}, Mar={len(testindex03)}")
