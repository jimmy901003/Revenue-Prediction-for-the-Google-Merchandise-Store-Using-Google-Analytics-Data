import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 讀取CSV檔案
df = pd.read_csv(r'E:\project\GA\data\train.csv')

# 設定Seaborn風格和調色板
sns.set(style="darkgrid")
sns.set_palette("husl")

# 展開JSON欄位
def expand_json_columns(df, columns_list):
    for column_name in columns_list:
        df[column_name] = df[column_name].apply(json.loads)
        expanded_df = pd.json_normalize(df[column_name])
        df = pd.concat([df, expanded_df], axis=1)
        df.drop(columns=[column_name], inplace=True)
    return df

df = expand_json_columns(df, ['device', 'totals', 'geoNetwork', 'trafficSource'])

# 刪除不需要的欄位
columns_to_drop = ['browserSize', 'browserVersion', 'mobileDeviceBranding',
                   'mobileInputSelector', 'mobileDeviceInfo', 'mobileDeviceMarketingName', 
                   'flashVersion', 'language', 'screenColors', 'screenResolution', 
                   'latitude', 'longitude', 'networkLocation', 
                   'adwordsClickInfo.criteriaParameters', 'isTrueDirect', 'referralPath', 
                   'adwordsClickInfo.page', 'adwordsClickInfo.slot', 'adwordsClickInfo.gclId', 
                   'adwordsClickInfo.adNetworkType', 'adwordsClickInfo.isVideoAd', 'campaignCode',
                   'keyword', 'adContent', 'mobileDeviceModel', 'operatingSystemVersion'
                   ]
df.drop(columns=columns_to_drop, inplace=True)

# 繪製缺失值的長條圖
def plot_missing_values(df, title='Missing Values Percentage'):
    missing_data = df.isnull().mean() * 100
    missing_data = missing_data.sort_values(ascending=False)

    plt.figure(figsize=(22, 16))
    plot = sns.barplot(x=missing_data.values, y=missing_data.index, palette='Set2')

    plt.title(title, fontsize=16)
    plt.xlabel('Percentage of Missing Values')
    plt.ylabel('Columns')
    plt.xlim(0, 100)
    plt.grid(axis='x')

    plot.tick_params(axis='y', labelsize=16)

    for i, v in enumerate(missing_data.values):
        decimal_places = 2 if v > 1 else 4
        plot.text(v + 0.3, i, f'{v:.{decimal_places}f}%', color='black', va='center')

    plt.tight_layout()
    plt.show()

plot_missing_values(df)

# 計算缺失值比例
missing_values = df[['transactionRevenue', 'bounces', 'newVisits', 'pageviews']].isnull().sum()
non_missing_values = df[['transactionRevenue', 'bounces', 'newVisits', 'pageviews']].count()
data = pd.DataFrame({'Missing Values': missing_values, 'Non-Missing Values': non_missing_values})
data['Missing Ratio'] = data['Missing Values'] / len(df)
data['Non-Missing Ratio'] = data['Non-Missing Values'] / len(df)

# 繪製缺失值比例的堆疊長條圖
ax = data[['Missing Ratio', 'Non-Missing Ratio']].plot(kind='bar', stacked=True, figsize=(12, 8))
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.3%}', (x + width / 2, y + height / 2), ha='center', va='center')

plt.title('Missing Values and Non-Missing Values Ratio')
plt.xlabel('Columns')
plt.ylabel('Ratio')
plt.show()

# 處理數值型欄位
numeric_columns = ['visits', 'hits', 'pageviews', 'bounces', 'newVisits', 'transactionRevenue']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

df['RevenueStatus'] = df['transactionRevenue'].apply(lambda x: 1 if x > 0 else 0)

# 繪製唯一值的長條圖
columns_to_check = ['bounces', 'newVisits', 'RevenueStatus']
for column in columns_to_check:
    plt.figure(figsize=(8, 6))
    plt.title(f'Unique Values for {column}')
    df[column].value_counts(dropna=False).sort_index().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

# 填充缺失值
df[['bounces', 'newVisits', 'transactionRevenue']] = df[['bounces', 'newVisits', 'transactionRevenue']].fillna(0)

# 繪製Top 20 Pageviews的長條圖
pageviews_counts = df['pageviews'].value_counts(dropna=False)
top_20_pageviews = pageviews_counts.head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_20_pageviews.index, y=top_20_pageviews.values, color='skyblue')
plt.title('Top 20 Pageviews Counts')
plt.xlabel('Pageviews')
plt.ylabel('Count')
plt.show()

# 繪製NaN Pageviews的盒狀圖
nan_values_df = df[df['pageviews'].isna()]
plt.figure(figsize=(12, 8))
plt.boxplot(x=nan_values_df['hits'], vert=False)
plt.ylabel('Hits', fontsize=16)
plt.title('Distribution of Hits for NaN Pageviews', fontsize=16)
plt.show()

# 繪製核密度估計圖
plt.figure(figsize=(16, 10))
clip_value = 15
sns.kdeplot(x=df['hits'].clip(0, clip_value), label='hits', fill=True, color='blue')
sns.kdeplot(x=df['pageviews'].clip(0, clip_value), label='pageviews', fill=True, color='orange')
plt.xlabel('Counts')
plt.ylabel('Density')
plt.title('Kernel Density Estimation - Hits vs Pageviews')
plt.legend(fontsize=20)
plt.xlim(0, 15)
plt.show()

# 填充缺失的Pageviews
if df['pageviews'].isnull().any():
    df['pageviews'] = df['pageviews'].fillna(df['hits'])



# def read_csv_and_expand_json(file_path, columns_to_expand):
#     # 讀取 CSV 檔案
#     df = pd.read_csv(file_path, parse_dates=['date'])

#     # 定義函數展開 JSON 欄位
#     def expand_json_columns(df, columns_list):
#         for column_name in columns_list:
#             # 將 JSON 字串解析成 Python 物件
#             df[column_name] = df[column_name].apply(json.loads)

#             # 將欄位展開成多個新欄位
#             expanded_df = pd.json_normalize(df[column_name])

#             # 將展開後的資料合併回原始 DataFrame
#             df = pd.concat([df, expanded_df], axis=1)

#             # 刪除原始的 JSON 欄位
#             df.drop(columns=[column_name], inplace=True)
            
#         return df

#     # 展開指定的 JSON 欄位
#     df = expand_json_columns(df, columns_to_expand)

#     # 指定數值型欄位
#     numeric_columns = ['visits', 'hits', 'pageviews', 'bounces', 'newVisits', 'transactionRevenue']

#     # 使用 pd.to_numeric 將指定欄位轉換為數值型，並將錯誤的值轉換為 NaN
#     df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

#     # 將 ['bounces', 'newVisits', 'transactionRevenue', 'pageviews'] 欄位的缺失值填充為 0
#     df[['bounces', 'newVisits', 'transactionRevenue']] = df[['bounces', 'newVisits', 'transactionRevenue']].fillna(0)
    
#     if df['pageviews'].isnull().any():

#         df['pageviews'] = df['pageviews'].fillna(df['hits'])
    
#     df['RevenueStatus'] = df['transactionRevenue'].apply(lambda x: 1 if x > 0 else 0)
    
#     df['isMobile'] = df['isMobile'].map({False: 0, True: 1})
    
    
#     columns_to_drop = ['browserSize', 'browserVersion', 'mobileDeviceBranding',
#                         'mobileInputSelector', 'mobileDeviceInfo', 'mobileDeviceMarketingName', 
#                         'flashVersion', 'language', 'screenColors', 'screenResolution', 
#                         'latitude', 'longitude', 'networkLocation', 
#                         'adwordsClickInfo.criteriaParameters', 'isTrueDirect', 'referralPath', 
#                         'adwordsClickInfo.page', 'adwordsClickInfo.slot', 'adwordsClickInfo.gclId', 
#                         'adwordsClickInfo.adNetworkType', 'adwordsClickInfo.isVideoAd', 'campaignCode',
#                         'keyword', 'adContent', 'mobileDeviceModel', 'operatingSystemVersion'
#                         ]

    
#     df.drop(columns=columns_to_drop, inplace=True)
#     df.set_index('date', inplace=True)
#     df['DayOfWeek'] = df.index.day_name()
#     return df