import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 設定顯示浮點數的格式
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# 設定 seaborn 样式
sns.set(style="darkgrid")
sns.set_palette("husl")

# 讀取資料
df = pd.read_csv(r"E:\project\GA\data\df_complete.csv", parse_dates=['visitStartTime']).set_index('visitStartTime')

# 隨機抽樣
df_revenue_status_1_sampled = df[df['RevenueStatus'] == 1].sample(n=11515, random_state=42)
df_revenue_status_0_sampled = df[df['RevenueStatus'] == 0].sample(n=20000, random_state=42)
df_sampled = pd.concat([df_revenue_status_1_sampled, df_revenue_status_0_sampled])

# 定義特徵和目標變數
selected_features = ['channelGrouping', 'browser', 'operatingSystem', 'deviceCategory', 'continent', 'subContinent', 'country', 'region', 'metro', 'city', 'networkDomain', 'campaign', 'source', 'medium', 'DayOfWeek']
numeric_features = ['visitNumber', 'hits', 'pageviews', 'bounces', 'newVisits', 'Hour']
categorical_features = selected_features

X = df_sampled[selected_features + numeric_features]
y = df_sampled['RevenueStatus']

# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# 模型字典
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(random_state=42),
}

# 訓練並評估每個模型
for model_name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # 評估模型
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    # 保存模型
    model_filename = f"{model_name}_model.joblib"
    joblib.dump(pipeline, model_filename)

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)
    print(f"Model saved as: {model_filename}")
    print("="*50)


from sklearn.model_selection import learning_curve


# 繪製學習曲線
def plot_learning_curve(pipeline, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 7)):
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.show()

test_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', LogisticRegression(random_state=42))])

plot_learning_curve(test_pipeline, X_train, y_train, cv=5, n_jobs=-1)

# 網格搜索最佳參數
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', GradientBoostingClassifier(random_state=42))])

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 4, 5],
    # 'classifier__min_samples_split': [2, 3, 4],
    # 'classifier__min_samples_leaf': [1, 2, 3]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best Parameters:", best_params)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)

print("Best Model:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# 保存模型和參數
# save_folder = r'E:\project\GA\models'
# os.makedirs(save_folder, exist_ok=True)

# model_filename = os.path.join(save_folder, 'best_classification_model.pkl')
# joblib.dump(best_model, model_filename)

# params_filename = os.path.join(save_folder, 'best_classification_model_params.pkl')
# joblib.dump(best_params, params_filename)