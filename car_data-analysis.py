#  基于简单规则
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_profiling as pp
import missingno as msno


#设置解析
file_name='used_car_20200421.csv'

batch=0
#dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H')
for df in pd.read_csv(open(file_name, 'r'), 
                    #   parse_dates=['time'], 
                    #   index_col = ['time'], 
                    #   date_parser = dateparse,
                      chunksize = 100000):

    print(df)
    print(df.shape)


    print(df.info())# 数据信息查看 .info()可以看到每列的type，以及NAN缺失信息
    print(df.isnull().any())
    print(df.isnull().sum()) #判断列的缺失值总数
    # 数据的统计信息概览 数据记录数，平均值，标准方差， 最小值，下四分位数， 中位数， 上四分位数，最大值
    print(df.describe())

# 可视化
    plt.figure(figsize=(10, 6))
    #sample=df.sample(n=200,random_state=123)
    sample = df.sample(100)
    msno.matrix(sample)
    plt.show()
    msno.bar(sample)
    plt.show()

    msno.heatmap(sample)#figsize是指图的大小
    plt.show()
# report = pp.ProfileReport('used_car_testB_20200421.csv')
# report.to_file('report.html')

