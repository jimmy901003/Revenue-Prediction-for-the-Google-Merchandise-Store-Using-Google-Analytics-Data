import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller

# 設定 pandas 顯示浮點數的格式
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# 設定 seaborn 样式
sns.set(style="darkgrid")
sns.set_palette("husl")

def read_csv_and_expand_json(file_path, columns_to_expand):
    # 讀取 CSV 檔案
    df = pd.read_csv(file_path, parse_dates=['visitStartTime'], date_parser=lambda x: pd.to_datetime(x, unit='s'))

    # 定義函數展開 JSON 欄位
    def expand_json_columns(df, columns_list):
        for column_name in columns_list:
            df[column_name] = df[column_name].apply(json.loads)
            expanded_df = pd.json_normalize(df[column_name])
            df = pd.concat([df, expanded_df], axis=1)
            df.drop(columns=[column_name], inplace=True)
        return df

    # 展開指定的 JSON 欄位
    df = expand_json_columns(df, columns_to_expand)

    # 指定數值型欄位
    numeric_columns = ['visits', 'hits', 'pageviews', 'bounces', 'newVisits', 'transactionRevenue']

    # 將指定欄位轉換為數值型，並將錯誤的值轉換為 NaN
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # 填充缺失值
    df[['bounces', 'newVisits', 'transactionRevenue']] = df[['bounces', 'newVisits', 'transactionRevenue']].fillna(0)
    if df['pageviews'].isnull().any():
        df['pageviews'] = df['pageviews'].fillna(df['hits'])

    # 創建 'RevenueStatus' 欄位
    df['RevenueStatus'] = df['transactionRevenue'].apply(lambda x: 1 if x > 0 else 0)

    # 將布爾型 'isMobile' 欄位映射為整數型
    df['isMobile'] = df['isMobile'].map({False: 0, True: 1})

    # 將指定欄位轉換為整數型
    df[['bounces', 'newVisits', 'transactionRevenue']] = df[['bounces', 'newVisits', 'transactionRevenue']].astype(int)

    # 刪除指定欄位
    columns_to_drop = ['browserSize', 'browserVersion', 'mobileDeviceBranding',
                       'mobileInputSelector', 'mobileDeviceInfo', 'mobileDeviceMarketingName',
                       'flashVersion', 'language', 'screenColors', 'screenResolution',
                       'latitude', 'longitude', 'networkLocation', 'date',
                       'adwordsClickInfo.criteriaParameters', 'isTrueDirect', 'referralPath',
                       'adwordsClickInfo.page', 'adwordsClickInfo.slot', 'adwordsClickInfo.gclId',
                       'adwordsClickInfo.adNetworkType', 'adwordsClickInfo.isVideoAd', 'campaignCode',
                       'keyword', 'adContent', 'mobileDeviceModel', 'operatingSystemVersion'
                       ]

    df.drop(columns=columns_to_drop, inplace=True)

    # 設定索引為 'visitStartTime'，並添加 'DayOfWeek' 和 'Hour' 欄位
    df.set_index('visitStartTime', inplace=True)
    df['DayOfWeek'] = df.index.day_name()
    df['Hour'] = df.index.hour

    return df



columns_to_expand = ['device', 'totals', 'geoNetwork', 'trafficSource']

df = read_csv_and_expand_json(r"E:\project\GA\data\train.csv", columns_to_expand)
# df.to_csv('df_complete.csv', index=True, encoding='utf-8')

# 讀取資料
# df = pd.read_csv(r"E:\project\GA\data\df_complete.csv", parse_dates=['visitStartTime']).set_index('visitStartTime')

# 過濾出交易收入大於0的資料
df_filtered = df[df['transactionRevenue'] > 0]
df_filtered.to_csv('nonzero_income_data.csv', index=True)

def plot_countplot(dataframe, column, title=None, xlabel=None, ylabel=None, palette="Spectral"):
    order = dataframe[column].value_counts().index
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=dataframe, x=column, order=order, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    total = len(dataframe[column])
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.text(x, y, percentage, ha='center', va='bottom')
    plt.show()

# 產生收益比例
plot_countplot(df, 'RevenueStatus', title='Number of Samples with and without Revenue', palette=['skyblue', 'salmon'])

# 收益分布
plt.figure(figsize=(12, 6))
sns.histplot(np.log1p(df['transactionRevenue'][df['RevenueStatus'] == 1]), bins=50, kde=True, color='skyblue')
plt.title('Distribution of Transaction Revenue')
plt.xlabel('Transaction Revenue')
plt.ylabel('Frequency')
plt.show()

def plot_grouped_statistics(dataframe, group_column, value_column):
    channel_revenue = dataframe.groupby(group_column)[value_column].agg(['count', 'mean'])
    channel_revenue_count_sorted = channel_revenue.sort_values(by='count', ascending=False)
    channel_revenue_mean_sorted = channel_revenue.sort_values(by='mean', ascending=False)
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    sns.barplot(x='count', y=channel_revenue_count_sorted.index, data=channel_revenue_count_sorted.reset_index(), palette='viridis')
    plt.title(f'Count by {group_column} Grouping', fontsize=14)
    plt.xlabel('Count')
    plt.ylabel(group_column)
    plt.subplot(1, 2, 2)
    sns.barplot(x='mean', y=channel_revenue_mean_sorted.index, data=channel_revenue_mean_sorted.reset_index(), palette='viridis')
    plt.title(f'Mean by {group_column} Grouping', fontsize=14)
    plt.xlabel('Mean')
    plt.ylabel(group_column)
    plt.tight_layout()
    plt.ticklabel_format(style='plain', axis='x')
    plt.show()

# 搜尋渠道
plot_grouped_statistics(df_filtered, 'channelGrouping', 'transactionRevenue')
# 搜尋瀏覽器
plot_grouped_statistics(df_filtered, 'browser', 'transactionRevenue')
# 作業系統
plot_grouped_statistics(df_filtered, 'operatingSystem', 'transactionRevenue')
# 地理位置
plot_grouped_statistics(df_filtered, 'subContinent', 'transactionRevenue')
# 星期
plot_grouped_statistics(df_filtered, 'DayOfWeek', 'transactionRevenue')
# 是否使用行動裝置
plot_countplot(df, 'isMobile', title='Distribution of Mobile Usage', palette=['skyblue', 'salmon'])

def plot_heatmap(df, columns_to_drop=None, cmap="YlGnBu", method='pearson'):
    if columns_to_drop is not None:
        df = df.drop(columns=columns_to_drop, errors='ignore')
    plt.figure(figsize=(12, 8))
    df_corr = df.corr(numeric_only=True, method=method)
    sns.heatmap(df_corr, cmap=cmap, annot=True, fmt=".2f")
    plt.title(f'{method}')
    plt.show()

plot_heatmap(df, columns_to_drop=['visits', 'visitId'])

plot_grouped_statistics(df_filtered, 'deviceCategory', 'transactionRevenue')

daily_RevenueStatus = df['RevenueStatus'].resample('D').sum()
daily_transactionRevenue = df['transactionRevenue'].resample('D').sum()

plt.figure(figsize=(30, 8))
ax1 = plt.subplot(111)
sns.lineplot(data=daily_RevenueStatus, color='dodgerblue', label='Revenue Status', legend=False)
ax1.set_xlabel('Date')
ax1.set_ylabel('Revenue Status', color='dodgerblue', fontsize=16)
ax1.tick_params('y', colors='dodgerblue')

ax2 = ax1.twinx()
sns.lineplot(data=daily_transactionRevenue, color='orangered', label='Transaction Revenue')
ax2.set_ylabel('Transaction Revenue', color='orangered', fontsize=16)
ax2.tick_params('y', colors='orangered')

plt.title('Daily Revenue Status and Transaction Revenue' ,fontsize=18)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# 執行單位根檢定
result_status = adfuller(daily_RevenueStatus)
result_transaction = adfuller(daily_transactionRevenue)

# 提取檢定統計量和p值
test_statistic_status, p_value_status, _, _, _, _ = result_status
test_statistic_transaction, p_value_transaction, _, _, _, _ = result_transaction

data = {
    'Test Statistic': [test_statistic_status, test_statistic_transaction],
    'P-Value': [p_value_status, p_value_transaction]
}

# 將字典轉換為DataFrame
df_adf_results = pd.DataFrame(data, index=['Revenue Status', 'Transaction Revenue'])

# 產生交易時間點
def plot_time_heatmap(df, values_column='RevenueStatus', annot=True):
    heatmap_data = df.pivot_table(values=values_column, index='DayOfWeek', columns=df.index.hour, aggfunc='sum')
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(weekday_order)
    plt.figure(figsize=(18, 16))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=annot, fmt='d', cbar_kws={'label': 'Transaction Revenue'})
    plt.title(f'{values_column} Heatmap Over Time')
    plt.show()

plot_time_heatmap(df, values_column='RevenueStatus')


