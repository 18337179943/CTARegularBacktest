from datetime import datetime, timedelta
import imp
import sys
import json
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from  matplotlib.widgets import MultiCursor
import shutil
import plotly.graph_objects as go	# 引入plotly底层绘图库
from plotly.subplots import make_subplots
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor


__Author__ = 'ZCXY'


class Logger():
    def __init__(self, filen='Default.log') -> None:
        self.terminal = sys.stdout
        self.log = open(filen, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def remove_file(pa):
    '''删除文件'''
    if os.path.exists(pa): os.remove(pa)
    return pa

def new_time_folder(folder):
    '''获取最新日期的文件目录'''
    lists = os.listdir(folder)         # 列出目录的下所有文件和文件夹保存到lists
    lists.sort(key=lambda fn: os.path.getmtime(folder + "/" + fn)) # 按时间排序
    file_new = f'{folder}/{lists[-1]}'      # 获取最新的文件保存到file_new
    return file_new

def get_df_date(start=datetime(2010, 1, 1), end=datetime(2020, 1, 1), freq='month'):
    '''获取df时间序列'''
    df_t = pd.DataFrame()
    df_t['date'] = pd.date_range(start=start, end=end)
    if freq == 'month':
        df_t['day'] = df_t['date'].apply(lambda x: x.day)
        df_t = df_t[df_t['day']==1]
        # df_t['date'] = df_t['date'].apply(lambda x: x-timedelta(days=1))
    df_t.reset_index(drop=True, inplace=True)
    del df_t['day']
    return df_t

def flatten_list(li):
    '''把list展平'''
    return [j for i in li for j in flatten_list(i)] if isinstance(li, list) else [li]

def m_multiprocess(func, li, max_workers=4):
    '''多进程'''
    print(li)
    if not isinstance(li, list): li = [li]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:  # max_workers=10
        res = executor.map(func, li)
    # print(func.__name__, 'done.')
    return res

def get_maxn_values(df: pd.DataFrame, col, col_res, n=3):
    '''获取dataframe最大值和次大值'''
    df_r = df.sort_values(col, ascending=False)
    return df_r[col_res].iloc[:n].to_list()

def get_digit(pa):
    return re.findall("\d+", pa)[0]

def save_json(file, pa):
    with open(pa, 'w') as f:
        json.dump(file, f, indent=4)

def read_json(pa):
    with open(pa, 'r') as f:
        fi = json.load(f)
    return fi

def get_sy(contract, sy=1):
    '''contract like rb2110'''
    re_func = re.compile(r'(\d+|\s+)')
    # symbol, num = re_func.split(contract)
    symbol, num, _ = re_func.split(contract)
    # print(re_func.split(contract))
    return symbol if sy else num

def change_date(dt, method=''):
    '''修改时间格式'''
    if method == '':
        # change xxxx-xx-xx to xxxxxxxx
        dt_res = ''.join(dt.split('-'))
    else:
        month = dt[4:6] if dt[4] != '0' else dt[5]
        day = dt[6:] if dt[6] != '0' else dt[7]
        dt_res = f'{dt[:4]}{method}{month}{method}{day}'
    return dt_res


def check_letters(x):
    my_re = re.compile(r'[A-Za-z]', re.S)
    res = ''.join(re.findall(my_re, x))
    return res

def makedir(pa):
    if not os.path.exists(pa):
        os.makedirs(pa)
    return pa

def plot_(x):
    fig, ax = plt.subplots(2,2, figsize=(15, 15)) # 手动创建一个figure和四个ax对象

    plt.subplot(2,2,1) # Add a subplot to the current figure

    plt.plot(x, x, color='blue', label='Linear') # 绘制第一个子图

    plt.xlabel('x-axis')

    plt.ylabel('y-axis')

    plt.legend() # 开启图例

    plt.subplot(2,2,2)

    plt.plot(x, x**2, color='red', label='Quadratic') # 绘制第二个子图

    plt.xlabel('x-axis')

    plt.ylabel('y-axis')

    plt.legend() # 开启图例

    plt.subplot(2,2,3)

    plt.plot(x, x**3, color='green', label='Cubic') # 绘制第三个子图

    plt.xlabel('x-axis')

    plt.ylabel('y-axis')

    plt.legend() # 开启图例

    plt.subplot(2,2,4)

    plt.plot(x, np.log(x), color='purple', label='log') # 绘制第四个子图

    plt.xlabel('x-axis')

    plt.ylabel('y-axis')

    plt.legend() # 开启图例

    fig.suptitle("Simple plot demo with one figure and one axes")

    plt.show() # 展示绘图

def m_sharp_ratio(x):
    '''计算夏普比率'''
    return np.mean(x) / np.std(x)

def get_maincon():
    df = pd.read_csv('{pa_prefix}/datas/maincon.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

def timestamp_to_datetime(tsp):
    s = "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(tsp.strftime(s), s)

def str_to_datetime(tsp):
    s = "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(tsp, s) if isinstance(tsp, str) else tsp

def datetime_to_str(tsp):
    s = "%Y-%m-%d %H:%M:%S"
    return tsp.strftime(s)
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def m_indicshow(data):
    '''画图: close signal pos pnl return_rate return_rate_cost'''
    data=data.dropna(axis=0)
    fig2 = plt.figure()
    ax2= fig2.add_axes([0.05, 0.87, 0.85, 0.1])
    ax3= fig2.add_axes([0.05, 0.72, 0.85, 0.1],sharex=ax2)
    ax4= fig2.add_axes([0.05, 0.56, 0.85, 0.1],sharex=ax2)
    ax5= fig2.add_axes([0.05, 0.72, 0.85, 0.1],sharex=ax2)
    ax6= fig2.add_axes([0.05, 0.72, 0.85, 0.1],sharex=ax2)

    ax2.cla()
    ax3.cla()
    ax4.cla()
    ax5.cla()
    ax6.cla()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax5.grid()
    ax6.grid()

    ax2.plot(range(len(data)), data['close']) # 价格
    # ax2.plot(range(len(data)), list(data['BidPrice1']),color='w')
    data = data.dropna(axis=0)
    for j in range(len(data)):
        if data['signal'].iloc[j]>0 :
            ax2.plot(j, data['close'].iloc[j], '*r')
        if data['signal'].iloc[j]<0 :
            ax2.plot(j, data['close'].iloc[j], '^g')
    ax3.set_title('pos')
    ax4.set_title('pnl')
    ax5.set_title('return_rate')
    ax6.set_title('return_rate_cost')
    ax3.plot(range(len(data)), list(data['pos']))
    ax4.plot(range(len(data)), list(data['pnl']))
    ax5.plot(range(len(data)), list(data['return_rate']))
    ax6.plot(range(len(data)), list(data['return_rate_cost']))

    MultiCursor(fig2.canvas, (ax2,ax3,ax4,ax5,ax6), color='r', lw=1)
    plt.show()

    return 0

def dic_to_dataframe(dic):
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dic.items()]))
    return df

def train_val_test_pnl_plot(data_li, save_pa):
    plt.close()
    fig, axes = plt.subplots(3, 1, figsize=(40, 20))
    plt.suptitle('s',fontsize=20)
    for ax, data in zip(axes.flatten(), data_li):
        ax.plot(data.iloc[:, 0],data.iloc[:, 1])
        ax.set(xlabel=data.columns[0], ylabel=data.columns[1])
        ax.set_xticks([0, len(data)/2, len(data)-1])
        # ax.set_xticks(x_label)
        plt.savefig(f'{save_pa}.png')
    # plt.show()

def go_plot(pa1='./df_profit_real_pnl.csv', pa2='./df_profit_real.csv'):
    df1 = pd.read_csv(pa1, index_col='datetime')
    df2 = pd.read_csv(pa2, index_col='datetime')
    df2.iloc[:, :-1] = df2.iloc[:, :-1] / 83000
    df2.iloc[:, -1] = df2.iloc[:, -1] / 1000000

    for i in range(df1.shape[1]):
        
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[f"Balance_{df1.columns[i]}", "Daily Pnl"],
            vertical_spacing=0.15
        )

        balance_line = go.Scatter(
            x=df1.index,
            y=df1.iloc[:, i],
            mode="lines",
            name=f"Balance_{df1.columns[i]}"
        )
        pnl_bar = go.Bar(y=df2.iloc[:, i], name="Daily Pnl")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(pnl_bar, row=2, col=1)

        fig.update_layout(height=1000, width=1000)
        fig.show()

def m_plot_hist(datas, title, save_pa):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    plt.suptitle(title,fontsize=20)
    for ax, data in zip(axes.flatten(), datas):
        ax.hist(data, 10, density=True)
        ax.set(xlabel=data.name, ylabel='')
        # ax.set_xticks(x_label)
        plt.savefig(f'{save_pa}{title}.png')
    plt.close()

def m_plot_two_hist(datas, title, save_pa):
    fig, axes = plt.subplots(1, 2, figsize=(46, 20))
    plt.suptitle(title,fontsize=20)
    for ax, data in zip(axes.flatten(), datas):
        ax.hist(data, 100, density=True)
        ax.set(xlabel=title, ylabel='')
        # ax.set_xticks(x_label)
        plt.savefig(f'{save_pa}{title}.png')
    plt.close()

def m_plot_one_hist(datas, title, save_pa=''):
    '''画单个分布图并保存'''
    plt.figure(figsize=(18, 12))
    plt.title(title)
    plt.hist(datas,bins=100,color='pink',edgecolor='b')
    if len(save_pa):
        plt.savefig(f'{save_pa}{title}.png')
    plt.close()

def m_plot(datas, title, save_pa='', m_type='line'):
    '''画图并保存'''
    # plt.figure(figsize=(32, 18))
    plt.title(title)
    pd.DataFrame(datas).plot(kind=m_type, figsize=(32, 18))
    plt.xticks(rotation=20)
    if len(save_pa):
        plt.savefig(f'{save_pa}/{title}.png')
    plt.close()

def m_plot_corr(datas, title, save_pa=''):
    '''画df折线图并保存'''
    plt.figure(figsize=(18, 12))
    plt.title(title)
    # plt.plot(datas.iloc[:, 0], datas.iloc[:, 1])
    plt.scatter(datas.iloc[:, 0], datas.iloc[:, 1])
    if len(save_pa):
        plt.savefig(f'{save_pa}{title}.png')
    plt.close()

def del_folder_file(pa):
    '''删除文件夹下的所有文件'''
    shutil.rmtree(pa)
    makedir(pa)

def filter_str(filter_str, li, is_list=0):
    '''从list里筛选出含有filtetr_str的字符串'''
    res_li = list(filter(lambda x: filter_str in x, li))
    ret = res_li if is_list else res_li[0]
    return ret

def get_pa_prefix(sys_name):
    if sys_name == 'windows':
        pa_sys = 'D:/策略开发/futures_ml/'
        pa_prefix = '.' 
    else:
        pa_sys = '/home/ZhongCheng/futures_ml_linux/'
        pa_prefix = '/home/ZhongCheng/futures_ml_linux'
    return pa_sys, pa_prefix
    
# def del_file():
#     pa1 = 

if __name__ == '__main__':
    # s = {'w': 2, 'e': 4, 'r': 5}
    # pa = 's.json'
    # save_json(s, pa)
    # print(read_json(pa))
    pa = './datas/data_set/'
    del_folder_file(pa)