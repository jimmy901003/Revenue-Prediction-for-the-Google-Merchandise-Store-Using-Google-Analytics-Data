import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor

df = pd.read_csv(r"E:\project\GA\data\nonzero_income_data.csv", parse_dates=['visitStartTime']).set_index('visitStartTime')

df['transactionRevenue'] = np.log1p(df['transactionRevenue'])
                     
# 定義特徵和目標變數
selected_features = ['channelGrouping', 'browser', 'operatingSystem', 'deviceCategory', 'continent', 'subContinent', 'country', 'region', 'metro', 'city', 'networkDomain', 'campaign', 'source', 'medium', 'DayOfWeek']
numeric_features = ['visitNumber', 'hits', 'pageviews', 'bounces', 'newVisits', 'Hour']
categorical_features = selected_features

df_sorted = df.sort_index()

# 設定切分的時間點
split_time = int(len(df_sorted) * 0.8)

# 切分訓練集和測試集
train_df = df_sorted[:split_time]
test_df = df_sorted[split_time:]

X_train = train_df[selected_features + numeric_features]
y_train = train_df['transactionRevenue']

X_test = test_df[selected_features + numeric_features]
y_test = test_df['transactionRevenue']

# 數值特徵的預處理管線
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# 類別特徵的預處理管線
categorical_transformer = Pipeline(steps=[
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 整合數值和類別特徵的預處理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


models = {
    'MLP Regressor': MLPRegressor(
        hidden_layer_sizes=(100, 50),  # 調整隱藏層的神經元數量和層數
        activation='relu',  # 指定激活函數，如 'identity', 'logistic', 'tanh', 'relu' 等
        alpha=0.0001,  # L2 正則化項
        learning_rate='adaptive',  # 學習率調整方式，如 'constant', 'invscaling', 'adaptive'
        max_iter=1000,  # 最大迭代次數
        random_state=42
    ),
    'Random Forest Regressor': RandomForestRegressor(
        n_estimators=100,  # 決策樹的數量
        max_depth=None,  # 樹的最大深度
        min_samples_split=2,  # 構建一個節點所需的最小樣本數
        min_samples_leaf=1,  # 葉子節點所需的最小樣本數
        random_state=42
    ),
    'Gradient Boosting Regressor': GradientBoostingRegressor(
        n_estimators=100,  # 決策樹的數量
        learning_rate=0.1,  # 學習率
        max_depth=3,  # 樹的最大深度
        min_samples_split=2,  # 構建一個節點所需的最小樣本數
        min_samples_leaf=1,  # 葉子節點所需的最小樣本數
        random_state=42
    ),
    'LGBM Regressor':LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
    ),
}

# 儲存結果分數的 DataFrame
results_df = pd.DataFrame()
residuals_df = pd.DataFrame()
import os
import joblib
for model_name, model in models.items():
    regression_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # 訓練模型
    regression_pipeline.fit(X_train, y_train)
    
    # 指定保存模型的資料夾
    # save_folder = r'E:\project\GA\models'

    # os.makedirs(save_folder, exist_ok=True)

    # 保存模型
    # model_filename = os.path.join(save_folder, f'{model_name}_model.pkl')
    # joblib.dump(regression_pipeline, model_filename)
    
    # 預測測試集
    y_pred = regression_pipeline.predict(X_test).flatten()  # 將預測值轉換成一維
    # 計算 RMSE
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)

    # 計算 RMSLE
    rmsle = round(np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
    
    # 計算 MAE
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    
    # 將結果儲存到 DataFrame
    results_df = results_df.append({'Model': model_name, 'RMSE': rmse, 'RMSLE': rmsle, 'MAE': mae}, ignore_index=True)
    
    plt.figure(figsize=(40, 15))
    # plt.plot(X_train.index, y_train, label='train', marker='o', color='skyblue')
    plt.plot(X_test.index, y_test, label='True Values', marker='o', color='#E8705A', alpha=0.5)
    plt.plot(X_test.index, y_pred, label=f'{model_name} Predictions', marker='o', color='#1090D0', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title(f'Predictions of {model_name} Model for the Next Time Period', fontsize=16)
    plt.legend(loc='upper right', fontsize=20)
    plt.show()
    
    # 計算殘差
    residuals = y_test - y_pred
    residuals_df[model_name] = residuals
    residuals_df.to_csv(r'E:\project\GA\data\residuals.csv', index=True)
    # 創建子圖
    gridspec_kw = {'width_ratios': [3, 0.7]}
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), gridspec_kw=gridspec_kw)
    
    # 繪製散點圖
    axes[0].scatter(X_test.index, residuals, label=f'{model_name} Residuals', color='#1090D0', alpha=0.8)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=2)  # 水平參考線
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'{model_name} Residual Plot')
    axes[0].legend()
    
    # 繪製直方圖
    axes[1].hist(residuals, bins=20, color='#1090D0', alpha=0.7, orientation='horizontal')
    axes[1].set_xlabel('Frequency')
    # 反轉直方圖的y軸
    axes[1].yaxis.tick_right()
    plt.subplots_adjust(wspace=0.02)
    axes[1].ticklabel_format(axis='both', style='plain')
    plt.show()
    

#建立ARIMA模型
from pmdarima import auto_arima
import pmdarima as pm

# 將訓練數據進行預處理
X_train_preprocessed = preprocessor.fit_transform(X_train)

# 測試數據進行預處理
X_test_preprocessed = preprocessor.transform(X_test)

# 使用auto_arima建立模型
model = pm.auto_arima(y_train, X=X_train_preprocessed, trace=True, suppress_warnings=True, seasonal=True, stepwise=True)

# 模型預測
y_pred, conf_int = model.predict(n_periods=len(test_df), X=X_test_preprocessed, return_conf_int=True)

# 計算模型表現
rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
rmsle = round(np.sqrt(mean_squared_log_error(y_test, y_pred)), 4)
mae = round(mean_absolute_error(y_test, y_pred), 4)

# 將結果添加到DataFrame中
results_df = results_df.append({'Model': 'arima', 'RMSE': rmse, 'RMSLE': rmsle, 'MAE': mae}, ignore_index=True)

# 計算殘差
residuals = y_test.values - y_pred.values
residuals_df['ARIMA'] = residuals
residuals_df.to_csv(r'E:\project\GA\data\residuals.csv', index=True)

# 創建子圖
gridspec_kw = {'width_ratios': [3, 0.7]}
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), gridspec_kw=gridspec_kw)

# 繪製散點圖
axes[0].scatter(X_test.index, residuals, label=f'{model_name} Residuals', color='#1090D0', alpha=0.8)
axes[0].axhline(0, color='red', linestyle='--', linewidth=2)  # 水平參考線
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Residuals')
axes[0].set_title(f'{model_name} Residual Plot')
axes[0].legend()

# 繪製直方圖
axes[1].hist(residuals, bins=20, color='#1090D0', alpha=0.7, orientation='horizontal')
axes[1].set_xlabel('Frequency')
# 反轉直方圖的y軸
axes[1].yaxis.tick_right()
plt.subplots_adjust(wspace=0.02)
axes[1].ticklabel_format(axis='both', style='plain')
plt.show()

