import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle


# 讀取資料
file_path = '/Users/baizonghan/Downloads/performance_test.csv'
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

# 「使用原模型」
with open('保存的原模型.pkl', 'rb') as f:
    model_refined = pickle.load(f)


# 評估新測試數據的函數
def evaluate_model_performance(model, X_test, y_test):
    y_pred = model.predict(sm.add_constant(X_test))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return r2, mae, rmse

# 使用之前的選擇特徵
# 假設這些特徵是從之前的訓練數據中選擇的
selected_features = ['study_hrs', 'high_math', 'high_hist', 'high_eng', 'stress_L', 'study_hrs stress_L', 'high_hist high_eng', 'gender_M stress_L', 'gender_M environment_Urban', 'stress_M environment_Urban']

# 檢查並剔除包含 NaN 的特徵
model = sm.OLS(y, sm.add_constant(X_interaction[selected_features])).fit()
nan_features = model.pvalues[model.pvalues.isna()].index.tolist()
selected_features = [feature for feature in selected_features if feature not in nan_features]

# 打印剔除後的特徵
print(f'Selected Features after removing NaN features: {selected_features}')

# 選擇新測試數據的特徵
X_test_selected = X_interaction[selected_features]

#預測結果
y_pred = model_refined.predict(sm.add_constant(X_test_selected))

# 評估新測試數據
new_test_r2, new_test_mae, new_test_rmse = evaluate_model_performance(model, X_test_selected, y)

# 打印新測試數據的評估結果
print('\nNew Test Set Evaluation:')
print(f'R-square: {new_test_r2}')
print(f'Mean Absolute Error (MAE): {new_test_mae}')
print(f'Root Mean Square Error (RMSE): {new_test_rmse}')

# 打印預測結果
print('\nPredicted Performance:')
print(y_pred)

# 將預測結果添加到原始數據中
data['predicted_performance'] = y_pred

# 將結果保存到新的 CSV 文件
data.to_csv('/Users/baizonghan/Downloads/predicted_performance.csv', index=False)

# python 五月三十號測試資料.py