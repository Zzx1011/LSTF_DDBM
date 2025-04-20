import pandas as pd
import matplotlib.pyplot as plt


# 读取 CSV 文件
file_path = '/home/zzx/projects/rrg-timsbc/zzx/LTSF_DDBM/data/ETT/ETTm2.csv'
df = pd.read_csv(file_path, sep=',',  header=0)

print(df.keys())
# 设置第一列作为日期时间索引
date_column = 'date'
date_format = '%Y-%m-%d %H:%M:%S'
df[date_column] = pd.to_datetime(df[date_column], format='mixed')
df.set_index(date_column, inplace=True)

start_date = '2016-07-01 00:00:00'
end_date = '2016-07-17 00:00:00'
# 对数据进行切片
df = df.loc[start_date:end_date]
# 绘制每列数据的折线图
plt.figure(figsize=(12, 8))
for column in df.columns:
    plt.plot(df.index, df[column], label=f'Column {column}')

# 设置图形标题和标签
plt.title('Data Visualization')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Value')
plt.legend()

# 保存图片
save_path = '/home/zzx/projects/rrg-timsbc/zzx/LTSF_DDBM/data/ETT/data_visualization.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
    