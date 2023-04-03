import subprocess
from os.path import join
import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
import operator
from functools import reduce
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

import time
import json
import numpy as np
import datetime
# from mpl_finance import candlestick_ohlc
from  matplotlib.widgets import MultiCursor
from m_base import *
# from mpl_finance import candlestick_ochl

def getJSON(filename):
    fd = open(filename, 'r')
    text = fd.read()
    fd.close()
    returndata = json.loads(text)
    return returndata
def getCoefFromJson(config):
    coef = []
    tcList =[]
    for k,v in config.items():
        if 'Lead' not in k:
            for i,w in enumerate(v['weight']):
                coef.append([k+'_%d'%(i+1), w])
        if 'TickLoc' in k:
            tcList = v['para'][1:-1]
            tcList = dict((i+1, tcList[i]) for i in range(10))
    coef = pd.DataFrame(coef, columns = ['name', 'coef'])
    return coef, tcList

def indicshow(fig2,ax2,ax3,ax5,data, ax6=None, mod=0):
    # print(data1[['AskPrice1','BidPrice1','aggrvalue','qty','pos','floating_profit',"threshod",'indicator']])
    # data=data1[['AskPrice1','BidPrice1','aggrvalue','qty','pos','floating_profit',"threshod",'indicator']].astype(float)
    # data=data.dropna(axis=0)
    data = data.fillna(0)
    data['close'] = [float(x) for x in np.array(data['close'])]
    data['signal'] = [float(x) for x in np.array(data['signal'])]
    # data['pred_sig'] = [float(x) for x in np.array(data['pred_sig'])]
    data['pos'] = [float(x) for x in np.array(data['pos'])]
    # data['pnl'] = [float(x) for x in np.array(data['pnl'])]
    data['pnl'] = [float(x) for x in np.array(data['pnl'])]
    ax2.cla()
    ax3.cla()
    # ax4.cla()
    ax5.cla()
    # ax7.cla()
    ax2.grid()
    ax3.grid()
    # ax4.grid()
    ax5.grid()
    if ax6 is not None:
        ax6.cla()
        ax6.grid()
    # ax7.grid()
    ax2.plot(range(len(data)), list(data['close']))
    # ax2.plot(range(len(data)), list(data['BidPrice1']),color='w')
    data = data.dropna(axis=0)

    if mod == 0:
        for j in range(len(data)):
            if data['signal'].iloc[j]==1:
                ax2.plot(j, data['close'].iloc[j], '*r')
            elif data['signal'].iloc[j]==2:
                ax2.plot(j, data['close'].iloc[j], '^g')
            elif data['signal'].iloc[j]==-1:
                ax2.plot(j, data['close'].iloc[j], '^b')
            elif data['signal'].iloc[j]==-2:
                ax2.plot(j, data['close'].iloc[j], '^y')

        ax3.set_title('pos')
        # ax4.set_title('indic')
        ax5.set_title('pnl')
        # ax6.set_title('threshold')
        # ax7.set_title('indic')
        ax5.plot(range(len(data)), list(data['pnl']))
        ax3.plot(range(len(data)), list(data['pos']))
        return MultiCursor(fig2.canvas, (ax2,ax3,ax5), color='r', lw=1)

def plot_show(symbol, pa='./datas/backtest_res/boll/RB_df_res.csv', save_pa='./datas/backtest_res/boll/RB_res.png', mod=0):

    plt.close()
    fig2 = plt.figure(figsize=(18, 12))
    ax2= fig2.add_axes([0.05, 0.75, 0.85, 0.2])
    ax5= fig2.add_axes([0.05, 0.5, 0.85, 0.2],sharex=ax2)
    ax3= fig2.add_axes([0.05, 0.2, 0.85, 0.2],sharex=ax2)
    # ax6= fig2.add_axes([0.05, 0.41, 0.85, 0.1],sharex=ax2)
    # ax7= fig2.add_axes([0.05, 0.28, 0.85, 0.1],sharex=ax2)
    # ax4= fig2.add_axes([0.05, 0.04, 0.85, 0.2], sharex=ax2)  ####left,bottom,width,height

    plt.title(f'{symbol}')
    data = pd.read_csv(pa)

    zs = indicshow(fig2, ax2, ax3, ax5, data, mod=mod)
    # plt.show()
    plt.savefig(save_pa)
    plt.close()

def plot_show_index_res(symbol, pa=None, save_pa=None, mod=4):
    # symbol_li = ['AP', 'FG', 'HC', 'L', 'M', 'PP', 'RM', 'RU', 'JD', 'JM', 'OI', 'V', 'P', 'sn'] # fg
    # symbol = 'AP'

    plt.close()
    fig2 = plt.figure(figsize=(23, 14))
    ax2= fig2.add_axes([0.05, 0.75, 0.85, 0.2])
    ax5= fig2.add_axes([0.05, 0.525, 0.85, 0.2],sharex=ax2)
    ax3= fig2.add_axes([0.05, 0.3, 0.85, 0.2],sharex=ax2)
    ax6= fig2.add_axes([0.05, 0.05, 0.85, 0.2],sharex=ax2)
    plt.title(f'{symbol}')
    if pa is None:
        pa = f'{pa_prefix}/simulation/optuna_params/total_val_raw/df_test_{symbol}.csv'
    data = pd.read_csv(pa)

    print(len(data))
    zs = indicshow(fig2, ax2, ax3, ax5, data, ax6, mod=mod)
    if save_pa is not None:
        plt.savefig(save_pa)
        plt.close()
    else:
        plt.show()

def plot_show_index(symbol, data, save_pa=None, mod=0):
    # symbol_li = ['AP', 'FG', 'HC', 'L', 'M', 'PP', 'RM', 'RU', 'JD', 'JM', 'OI', 'V', 'P', 'sn'] # fg
    # symbol = 'AP'

    plt.close()
    fig2 = plt.figure(figsize=(18, 12))
    ax2= fig2.add_axes([0.05, 0.5, 0.9, 0.45])
    # ax5= fig2.add_axes([0.05, 0.5, 0.85, 0.2],sharex=ax2)
    ax3= fig2.add_axes([0.05, 0.1, 0.9, 0.3],sharex=ax2)

    plt.title(f'{symbol}')
    zs = indicshow_index(fig2, ax2, ax3, data, mod=mod)
    if save_pa is not None:
        plt.savefig(save_pa)
        plt.close()
    else:
        plt.show()

def plot_pnl_all(pa=f'{pa_prefix}/simulation/optuna_params/total/total_test_analyze.csv', save_pa=f'{pa_prefix}/simulation/optuna_params/total/plot_pnl_all.csv'):
    df = pd.read_csv(pa)
    df.set_index('datetime', inplace=True)
    ax = df.plot()
    fig = ax.get_figure()
    fig.savefig(save_pa)
    plt.close()

def plot_pnl_seperate(df_res_all, save_pa=None):
    if save_pa is None:
        save_pa = makedir(f'{pa_prefix}/simulation/optuna_params/total/')
    for col in df_res_all.columns:
        plt.close()
        ax = pd.DataFrame(df_res_all[col]).plot(figsize=(18, 12))
        fig = ax.get_figure()
        fig.savefig(f'{save_pa}{col}.png')
    df_res_all.to_csv(f'{save_pa}df_res_all.csv')

if __name__ == '__main__':
    # fig1, ax1 = plt.subplots()
    # plot_show()
    # plot_pnl_seperate()
    symbol_li = ['AP', 'FG', 'HC', 'L', 'M', 'PP', 'RM', 'RU', 'JD', 'JM', 'OI', 'V', 'P', 'sn'] # fg
    symbol = 'AP'
    # for symbol in symbol_li:
    # plot_show1(symbol)
    pa = f'{pa_prefix}/simulation/optuna_params/madifrsi/df_test_RB.csv'
    plot_show_index_res(symbol, pa=pa, save_pa=None, mod=4)
