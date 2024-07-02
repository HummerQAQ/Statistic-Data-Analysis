import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle

# 讀取資料
file_path = '/Users/baizonghan/Downloads/performance.csv'
data = pd.read_csv(file_path)

# 編碼
data_encoded = pd.get_dummies(data, columns=['gender', 'stress', 'environment'], drop_first=True)

# 分割資料
X = data_encoded.drop(columns=['performance'])
y = data_encoded['performance']

# 標準化數值變量
scaler = StandardScaler()
X[['high_math', 'high_eng', 'high_hist', 'study_hrs', 'sleep_hrs']] = scaler.fit_transform(X[['high_math', 'high_eng', 'high_hist', 'study_hrs', 'sleep_hrs']])

# 添加交互項
poly = PolynomialFeatures(interaction_only=True, include_bias=False)
X_interaction = poly.fit_transform(X)

# 獲取新特徵名稱
interaction_features = poly.get_feature_names_out(X.columns)
X_interaction = pd.DataFrame(X_interaction, columns=interaction_features)

# 標準化交互特徵
interaction_scaler = StandardScaler()
X_interaction = interaction_scaler.fit_transform(X_interaction)
X_interaction = pd.DataFrame(X_interaction, columns=interaction_features)

# 移除異常值函數
def remove_outliers(X, y):
    model = sm.OLS(y, sm.add_constant(X)).fit()
    influence = model.get_influence()
    standardized_residuals = influence.resid_studentized_internal
    abs_z_scores = np.abs(standardized_residuals)
    filter_mask = (abs_z_scores < 3)
    return X[filter_mask], y[filter_mask]

# 分割訓練和測試集
X_train, X_test, y_train, y_test = train_test_split(X_interaction, y, test_size=0.2, random_state=0)
X_train_filtered, y_train_filtered = remove_outliers(X_train, y_train)

# 後向選擇函數
def backward_selection(X, y):
    features = X.columns.tolist()
    while len(features) > 0:
        pvals = sm.OLS(y, sm.add_constant(X[features])).fit().pvalues[1:]  # all coefs except intercept
        max_pval = pvals.max()
        if max_pval >= 0.05:
            excluded_feature = pvals.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features

# 選擇特徵
selected_features = backward_selection(X_train_filtered, y_train_filtered)
X_train_filtered_selected = X_train_filtered[selected_features]
X_test_selected = X_test[selected_features]
print(f'Selected Features: {selected_features}')

# 檢查並剔除包含 NaN 的特徵
model = sm.OLS(y_train_filtered, sm.add_constant(X_train_filtered[selected_features])).fit()
nan_features = model.pvalues[model.pvalues.isna()].index.tolist()
selected_features = [feature for feature in selected_features if feature not in nan_features]

# 打印剔除後的特徵
print(f'Selected Features after removing NaN features: {selected_features}')

# 計算VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X_train_filtered_selected.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_filtered_selected.values, i) for i in range(X_train_filtered_selected.shape[1])]

print(vif_data)

# 重新構建模型
X_train_filtered_selected = X_train_filtered[selected_features]
X_test_selected = X_test[selected_features]

# 重新訓練模型
model_refined = sm.OLS(y_train_filtered, sm.add_constant(X_train_filtered_selected)).fit()
print(model_refined.summary())

# 保存模型
with open('保存的原模型.pkl', 'wb') as f:
    pickle.dump(model_refined, f)

# 交叉驗證
kf = KFold(n_splits=5, shuffle=True, random_state=0)
model = LinearRegression()
scoring = {'r2': 'r2', 
           'mae': make_scorer(mean_absolute_error), 
           'rmse': make_scorer(mean_squared_error, squared=False)}

cv_results_r2 = cross_val_score(model, X_train_filtered_selected, y_train_filtered, cv=kf, scoring='r2')
cv_results_mae = cross_val_score(model, X_train_filtered_selected, y_train_filtered, cv=kf, scoring=scoring['mae'])
cv_results_rmse = cross_val_score(model, X_train_filtered_selected, y_train_filtered, cv=kf, scoring=scoring['rmse'])

print('Cross-Validation Results:')
print(f'R-square: {cv_results_r2.mean()}')
print(f'Mean Absolute Error (MAE): {cv_results_mae.mean()}')
print(f'Root Mean Square Error (RMSE): {cv_results_rmse.mean()}')


# 模型評估
y_train_pred = model_refined.predict(sm.add_constant(X_train_filtered_selected))
y_test_pred = model_refined.predict(sm.add_constant(X_test_selected))

# 訓練集評估
train_r2 = model_refined.rsquared
train_mae = mean_absolute_error(y_train_filtered, y_train_pred)
train_rmse = mean_squared_error(y_train_filtered, y_train_pred, squared=False)

print('\nTraining Set Evaluation:')
print(f'R-square: {train_r2}')
print(f'Mean Absolute Error (MAE): {train_mae}')
print(f'Root Mean Square Error (RMSE): {train_rmse}')

# 測試集評估
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

print('\nTest Set Evaluation:')
print(f'R-square: {test_r2}')
print(f'Mean Absolute Error (MAE): {test_mae}')
print(f'Root Mean Square Error (RMSE): {test_rmse}')


# python 原模型.py
