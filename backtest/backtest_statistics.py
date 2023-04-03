import statistics
import pandas as pd

import numpy as np
import sys

# sys_name = 'windows'
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
# sys.path.insert(0, pa_sys)
sys_name = 'windows'
pa_sys = '.'
pa_prefix = '.'
sys.path.insert(0, pa_sys)
from functools import partial
from m_base import *
from datas_process.m_futures_factors import SymbolsInfo
from m_base import m_plot_hist, str_to_datetime


class BacktesterStatistics:
    '''
    统计模型结果：
    1、信号的胜率
    2、盈亏比
    3、交易次数
    4、最长持仓周期
    5、平均持仓周期
    6、涨跌胜率
    '''
    def __init__(self, interval=15) -> None:
        '''
        df.columns = [datetime, close, signal, pos, profit, cost, pnl, pnl_cost]
        '''
        self.symbolinfo = SymbolsInfo()
        self.df_symbols_all = self.symbolinfo.df_symbols_all
        self.is_sep = 0
        self.interval = interval
        # self.symbol_size = self.df_symbols_all[self.df_symbols_all['symbol']==symbol.upper()]['size'].iloc[0]

    def _caculate_each_pnl0(self, df, return_df=0, method=0):
        '''计算每次成交的盈亏'''  # 把res_rate换成了res
        res, holding_period, res_rate, dt_li = [], [], [], []
        pnl, dt, pos = df['pnl'], df['datetime'], df['pos'] 
        df_s = df[df['signal']!=0]
        
        if abs(df_s['signal'].sum()) == len(df_s):
            res.append(pnl.iloc[-1] - pnl.iloc[0])
            holding_period.append(len(df))
            dt_li.append(dt.iloc[0])
        else:
            ind_signal = df_s.index.values
            
            for i in range(len(ind_signal)-2):
                ind_signal_i = ind_signal[i]
                if method == 0: # 非连续,有空仓
                    if pos.iloc[ind_signal[i+2]] != 0:
                        res.append(pnl.iloc[ind_signal[i+1]] - pnl.iloc[ind_signal_i])
                        res_rate.append((pnl.iloc[ind_signal[i+1]] - pnl.iloc[ind_signal_i])/pnl.iloc[ind_signal_i])
                        holding_period.append((dt.iloc[ind_signal[i+1]]-dt.iloc[ind_signal_i]).days)
                        dt_li.append(dt.iloc[ind_signal[i+1]])
                elif method == 1:   # 连续
                    res.append(pnl.iloc[ind_signal[i+1]] - pnl.iloc[ind_signal_i])
                    res_rate.append((pnl.iloc[ind_signal[i+1]] - pnl.iloc[ind_signal_i])/pnl.iloc[ind_signal_i])
                    holding_period.append((dt.iloc[ind_signal[i+1]]-dt.iloc[ind_signal_i]).days)
                    dt_li.append(dt.iloc[ind_signal[i+1]])

        if return_df:
            df_res = pd.DataFrame({'res': res, 'holding_period': holding_period, 'res_rate': res_rate, 'datetime': dt_li})
            return df_res
        # print(res_rate)
        # print('-----------------')
        return np.array(res), np.array(holding_period), np.array(res_rate), np.array(dt_li)

    def _caculate_each_pnl(self, df, return_df=0, method=0):
        '''计算每次成交的盈亏'''  # 把res_rate换成了res
        res, holding_period, res_rate, dt_li = [], [], [], []
        pnl, dt, pos = df['pnl'], df['datetime'], df['pos'] 
        df_s = df.copy()
        df_s['sign'] = df_s['pos'].apply(lambda x: np.sign(x))
        df_s['sign_change'] = np.where(df_s['sign'] != df_s['sign'].shift(1), 1, 0)
        df_s = df_s[(df_s['sign_change']==1) & (df_s['sign']!=0)]
        # df_s.to_csv('df_s.csv')
        
        # if abs(df_s['signal'].sum()) == len(df_s):
        #     res.append(pnl.iloc[-1] - pnl.iloc[0])
        #     holding_period.append(len(df))
        #     dt_li.append(dt.iloc[0])
        # else:
        if len(df_s) > 3:
            ind_signal = df_s.index.values
            
            for i in range(len(ind_signal)-2):
                ind_signal_i = ind_signal[i]
                if method == 0: # 非连续,有空仓
                    if pos.iloc[ind_signal[i+2]] != 0:
                        res.append(pnl.iloc[ind_signal[i+1]] - pnl.iloc[ind_signal_i])
                        res_rate.append((pnl.iloc[ind_signal[i+1]] - pnl.iloc[ind_signal_i])/pnl.iloc[ind_signal_i])
                        holding_period.append((dt.iloc[ind_signal[i+1]]-dt.iloc[ind_signal_i]).days)
                        dt_li.append(dt.iloc[ind_signal[i+1]])
                elif method == 1:   # 连续
                    res.append(pnl.iloc[ind_signal[i+1]] - pnl.iloc[ind_signal_i])
                    res_rate.append((pnl.iloc[ind_signal[i+1]] - pnl.iloc[ind_signal_i])/pnl.iloc[ind_signal_i])
                    holding_period.append((dt.iloc[ind_signal[i+1]]-dt.iloc[ind_signal_i]).days)
                    dt_li.append(dt.iloc[ind_signal[i+1]])

        if return_df:
            df_res = pd.DataFrame({'res': res, 'holding_period': holding_period, 'res_rate': res_rate, 'datetime': dt_li})
            return df_res
        # print(res_rate)
        # print('-----------------')
        return np.array(res), np.array(holding_period), np.array(res_rate), np.array(dt_li)

    def caculate_signal_win_rate(self, res):
        '''计算信号胜率'''
        signal_win_rate = round(np.sum(np.where(res>0, 1, 0)) / len(res), 3)
        return signal_win_rate

    def caculate_total_profit_loss_ratio(self, res):
        '''计算总盈亏比'''
        return round(np.sum(res[res>0]) / np.abs(np.sum(res[res<0])), 3)

    def caculate_average_profit_loss_ratio(self, res):
        '''计算平均盈亏比'''
        return round(np.mean(res[res>0]) / np.abs(np.mean(res[res<0])), 3)

    def caculate_trade_times(self, df, res):
        '''计算交易频率'''
        return round(len(res) / len(df), 5)

    def caculate_longest_holding_period(self, holding_period):
        '''最长持仓周期'''
        return holding_period.max()

    def caculate_average_holding_period(self, holding_period):
        '''平均持仓周期'''
        return round(holding_period.mean(), 2)

    def caculate_win_loss_rate(self, df: pd.DataFrame):
        '''持仓涨跌胜率'''
        df_up = df[df['pos']>0]['profit']
        df_down = df[df['pos']<0]['profit']
        up_rate = round(np.sum(np.where(df_up>0, 1, 0)) / len(df_up), 3)
        down_rate = round(np.sum(np.where(df_down>0, 1, 0)) / len(df_down), 3)
        return up_rate, down_rate

    def caculate_total_return(self, res_rate):
        '''计算总收益'''
        return round(np.sum(res_rate), 3)
    
    def max_ddpercent(self, df):
        '''百分比最大回撤'''
        df["highlevel"] = (
                df["pnl"].rolling(
                    min_periods=1, window=len(df), center=False).max()
            )
        df["drawdown"] = df["pnl"] - df["highlevel"]
        # df["ddpercent"] = df["drawdown"] / df["highlevel"]
        max_ddpercent = df["drawdown"].min()   
        return max_ddpercent
    
    def trade_days(self, df):
        '''交易日'''
        return (str_to_datetime(df['datetime'].iloc[-1]) - str_to_datetime(df['datetime'].iloc[0])).days

    def annual_return(self, df):
        '''年化收益率'''
        annual_return = (df['pnl'].iloc[-1]-1) / (self.trade_days(df) / 365)
        annual_return = round(annual_return, 4)
        return annual_return
    
    def sharp_ratio(self, df, dan=1):
        '''夏普比率'''
        pnl = df['pnl']
        if dan == 0:
            pnl_diff = df['pct'].dropna().values
        else:
            pnl_diff = pnl.values[1:] - pnl.values[:-1]
        interval_n = len(df) / self.trade_days(df)
        sharp_ratio = np.mean(pnl_diff) / np.std(pnl_diff) * np.sqrt(250*interval_n)  # 夏普比率
        return sharp_ratio

    def caculate_level(self, df: pd.DataFrame):
        '''计算平均杠杆率'''
        return df['level'].mean()

    def caculate_statistics(self, df, method=0, is_df=1, save_pa='./datas/backtest_res/boll/RB_df_statistic.csv'):
        '''计算统计结果:
        '''
        df_res = df.copy()
        df_res['datetime'] = pd.to_datetime(df_res['datetime'])
        res, holding_period, res_rate, dt = self._caculate_each_pnl(df_res, method=method)
        total_return = round(df_res['pnl'].iloc[-1]-1, 4)
        annual_return = self.annual_return(df)  
        
        if len(res_rate):
            win_rate = self.caculate_signal_win_rate(res)
            # total_profit_loss_ratio = self.caculate_total_profit_loss_ratio(res)
            average_profit_loss_ratio = self.caculate_average_profit_loss_ratio(res)
            trade_times = self.caculate_trade_times(df_res, res)
            longest_holding_period = self.caculate_longest_holding_period(holding_period)
            max_ddpercent = self.max_ddpercent(df_res)
            sharp_ratio = self.sharp_ratio(df)
            max_rate, min_rate, mean_rate = np.max(res_rate), np.min(res_rate), np.mean(res_rate)
            kama = -round(annual_return/max_ddpercent, 4)

        else: 
            win_rate, average_profit_loss_ratio, trade_times, longest_holding_period = None, None, None, None
            max_rate, min_rate, mean_rate, max_ddpercent, kama, sharp_ratio = None, None, None, None, None, None
        res_dic = {'总收益率': [total_return], '年化收益率': [annual_return], '收益回撤比': [kama], 
                   '百分比最大回撤': [max_ddpercent], '信号胜率': [win_rate], 
                   '平均盈亏比': [average_profit_loss_ratio], '换手率': [trade_times], '夏普比率': [sharp_ratio],
                   '最长持仓周期': [longest_holding_period], 
                   '每笔最大盈利': [max_rate], '每笔最大亏损': [min_rate], '每笔平均盈亏': [mean_rate]}
                   
        if is_df:
            df_statistic = pd.DataFrame(res_dic).T
            df_statistic.reset_index(inplace=True)
            df_statistic.columns = ['指标名称', '数值']
            if len(save_pa): df_statistic.to_csv(save_pa, index=False)
            return df_statistic

        return res_dic
    
    def get_index_value(self, df_statistic, name):
        '''获取统计结果的数值'''
        return df_statistic[df_statistic['指标名称']==name]['数值_total'].iloc[0]
    
    def caculate_statistics_total(self, df, df_statistic, save_pa='./datas/backtest_res/boll/RB_df_statistic.csv', is_df=1):
        '''计算统计结果:
        '''
        df_res = df.copy()
        df_res['datetime'] = pd.to_datetime(df_res['datetime'])
        total_return = round(df_res['pnl'].iloc[-1]-1, 4)
        annual_return = self.annual_return(df)  
        
        win_rate = self.get_index_value(df_statistic, '信号胜率')
        # total_profit_loss_ratio = self.caculate_total_profit_loss_ratio(res)
        average_profit_loss_ratio = self.get_index_value(df_statistic, '平均盈亏比')
        trade_times = self.get_index_value(df_statistic, '换手率')
        max_ddpercent = self.max_ddpercent(df_res)
        sharp_ratio = self.sharp_ratio(df)

        res_dic = {'总收益率': [total_return], '年化收益率': [annual_return], '收益回撤比': [-round(annual_return/max_ddpercent, 4)], 
                   '百分比最大回撤': [max_ddpercent], '信号胜率': [win_rate], '夏普比率': [sharp_ratio],
                   '平均盈亏比': [average_profit_loss_ratio], '换手率': [trade_times]}
                   
        if is_df:
            df_statistic = pd.DataFrame(res_dic).T
            df_statistic.reset_index(inplace=True)
            df_statistic.columns = ['指标名称', '数值']
            if len(save_pa): df_statistic.to_csv(save_pa, index=False)
            return df_statistic

        return res_dic

    def reset_df(self, df_0: pd.DataFrame):
        df = df_0.copy()
        if 'datetime' not in df.columns: df.reset_index(inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].apply(lambda x: x.date())
        df['date_change'] = np.where(df['date']!=df['date'].shift(-1), 1, 0)
        df = df[df['date_change']==1]
        df.reset_index(drop=True, inplace=True)
        del df['date_change']
        return df
    
    def train_test_pnl_sep(self, df_0: pd.DataFrame, sep_dt=datetime(2021, 1, 1), sp=''):
        '''训练集和测试集的pnl分离'''
        makedir(sp)
        df = self.reset_df(df_0)
        df_train = df[df['datetime']<sep_dt]
        df_test = df[df['datetime']>=sep_dt]
        init_balance = df_test['pnl'].iloc[0]
        df_test['pnl'] = (df_test['pnl'] - init_balance) / init_balance + 1
        df_train.to_csv(f'{sp}/df_train.csv', index=False)
        df_test.to_csv(f'{sp}/df_test.csv', index=False)
        return df_train, df_test

    def change_pnl_compound_interest_copy(self, df_0: pd.DataFrame, sp=''):
        '''pnl单利转换成复利'''
        df = self.reset_df(df_0)
        df['pct'] = df['pnl'] - df['pnl'].shift(1)
        df['pct'].iloc[0] = 0
        df['month'] = df['datetime'].apply(lambda x: x.month)
        df['change_month'] = np.where(df['month']!=df['month'].shift(1), 1, 0)
        # df['change_date'] = np.where(df['datetime']-df['datetime'].shift(1)>timedelta(days=1), 1, 0)
        df.reset_index(drop=True, inplace=True)
        df_change = df[df['change_month']==1]
        change_index = df_change.index.to_list()
        df['pnl_cp'] = df['pnl'].copy()
        for i in range(len(change_index)):
            i1 = change_index[i] 
            compound_v = df['pnl'].iloc[i1-1] if i1 != 0 else 1
            if i == len(change_index) - 1:
                df['pnl_cp'].iloc[i1:] = df['pnl_cp'].iloc[i1:] * compound_v
            else:
                i2 = change_index[i+1]
                df['pnl_cp'].iloc[i1:i2] = df['pnl_cp'].iloc[i1:i2] * compound_v
        df['diff'] = df['pnl'] - df['pnl_cp']
        if len(sp): df.to_csv(sp, index=False)
        
        return df
    
    def change_pnl_compound_interest(self, df_0: pd.DataFrame, sp=''):
        '''pnl单利转换成复利'''
        df = self.reset_df(df_0)
        df['pct'] = df['pnl'] - df['pnl'].shift(1)
        df['pct'].iloc[0] = 0
        # df['month'] = df['datetime'].apply(lambda x: x.month)
        # df['change_d'] = np.where(df['month']!=df['month'].shift(1), 1, 0)
        df['change_d'] = np.where(df['datetime']-df['datetime'].shift(1)>timedelta(days=1), 1, 0)
        df.reset_index(drop=True, inplace=True)
        df_change = df[df['change_d']==1]
        change_index = df_change.index.to_list()
        df['pct_adj'] = df['pct'].copy()
        for i in range(len(change_index)):
            i1 = change_index[i] 
            if i1 > 2:
                compound_v = df['pct_adj'].iloc[:i1-1].cumsum().iloc[-1]+1 if i1 != 0 else 1
                if i == len(change_index) - 1:
                    df['pct_adj'].iloc[i1:] = df['pct'].iloc[i1:] * compound_v
                else:
                    i2 = change_index[i+1]
                    df['pct_adj'].iloc[i1:i2] = compound_v * df['pct'].iloc[i1:i2]
        df['pnl_adj'] = df['pct_adj'].cumsum()+1
        df = df[['datetime', 'pnl_adj']]
        df.rename(columns={'pnl_adj': 'pnl'}, inplace=True)
        if len(sp): 
            makedir(sp)
            df.to_csv(f'{sp}/df_compound_pnl.csv', index=False)
        return df

    def get_yearly_statistic(self, df_0: pd.DataFrame, sp='', dan=1):
        '''获取年度的统计结果
        return  年份, 策略收益, 最大回撤, 夏普率, 波动率, Calmar, 月度胜率'''
        df = df_0.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].apply(lambda x: x.date())
        df['date_change'] = np.where(df['date']!=df['date'].shift(-1), 1, 0)
        df = df[df['date_change']==1]
        df['pct'] = df['pnl'] - df['pnl'].shift(1)
        df['year'] = df['datetime'].apply(lambda x: x.year)
        df['month'] = df['datetime'].apply(lambda x: x.month)
        df['month_change'] = np.where(df['month']!=df['month'].shift(-1), 1, 0)
        res_li = []
        for year, df_i in df.groupby('year'):
            if dan:
                df_i['pnl'] = df_i['pnl'] - df_i['pnl'].iloc[0] + 1   # 单利
            else:
                df_i['pnl'] = (df_i['pnl'] - df_i['pnl'].iloc[0]) / df_i['pnl'].iloc[0] + 1     # 复利
            total_return = df_i['pnl'].iloc[-1] - 1
            max_ddpecnt = self.max_ddpercent(df_i)
            sharp_ratio = self.sharp_ratio(df_i, dan=dan)
            vol = self.volatility(df_i, dan=dan)
            calmar = total_return / -max_ddpecnt
            df_i['month_change'].iloc[0] = 1
            df_i = df_i[df_i['month_change']==1]
            pct_m = np.sign((df_i['pnl'] - df_i['pnl'].shift(1)).iloc[1:])
            win_rate_m = len(np.where(pct_m>0)[0]) / len(pct_m)
            res_li.append([year, total_return, max_ddpecnt, sharp_ratio, vol, calmar, win_rate_m])
        df_res = pd.DataFrame(res_li)
        df_res.columns = ['年份', '策略收益', '最大回撤', '夏普率', '波动率', 'Calmar', '月度胜率']
        if len(df_res) > 2:
            annual_return = df_res['策略收益'].mean()
            annual_ddpercent = df_res['最大回撤'].min()
            annual_sharp = m_sharp_ratio(df_res['夏普率'])
            annual_vol = df_res['策略收益'].std()
            annual_calmar = annual_return / -annual_ddpercent
            annual_win_rate_m = df_res['月度胜率'].mean()
            df_res.loc[len(df_res)] = ['年化', annual_return, annual_ddpercent, annual_sharp, annual_vol, annual_calmar, annual_win_rate_m]

        if len(sp):
            df_res.to_csv(sp, index=False)
        
        return df_res

    def get_yearly_statistic1(self, df_0: pd.DataFrame, sp='', dan=1):
        '''获取年度的统计结果
        return  年份, 策略收益, 最大回撤, 夏普率, 波动率, Calmar, 月度胜率'''
        df = df_0.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].apply(lambda x: x.date())
        df['date_change'] = np.where(df['date']!=df['date'].shift(-1), 1, 0)
        df = df[df['date_change']==1]
        df['pct'] = df['pnl'] - df['pnl'].shift(1)
        df['year'] = df['datetime'].apply(lambda x: x.year)
        df['month'] = df['datetime'].apply(lambda x: x.month)
        df['month_change'] = np.where(df['month']!=df['month'].shift(-1), 1, 0)
        res_li = []
        for year, df_i in df.groupby('year'):
            if dan:
                df_i['pnl'] = df_i['pnl'] - df_i['pnl'].iloc[0] + 1   # 单利
            else:
                df_i['pnl'] = (df_i['pnl'] - df_i['pnl'].iloc[0]) / df_i['pnl'].iloc[0] + 1     # 复利
            total_return = df_i['pnl'].iloc[-1] - 1
            max_ddpecnt = self.max_ddpercent(df_i)
            sharp_ratio = self.sharp_ratio(df_i, dan=dan)
            vol = self.volatility(df_i, dan=dan)
            calmar = total_return / -max_ddpecnt
            df_i['month_change'].iloc[0] = 1
            df_i = df_i[df_i['month_change']==1]
            pct_m = np.sign((df_i['pnl'] - df_i['pnl'].shift(1)).iloc[1:])
            win_rate_m = len(np.where(pct_m>0)[0]) / len(pct_m)
            res_li.append([year, total_return, max_ddpecnt, sharp_ratio, vol, calmar, win_rate_m])
        df_res = pd.DataFrame(res_li)
        df_res.columns = ['年份', '策略收益', '最大回撤', '夏普率', '波动率', 'Calmar', '月度胜率']
        annual_return = df_res['策略收益'].mean()
        annual_ddpercent = df_res['最大回撤'].min()
        annual_sharp = m_sharp_ratio(df_res['夏普率'])
        annual_vol = df_res['策略收益'].std()
        annual_calmar = annual_return / -annual_ddpercent
        annual_win_rate_m = df_res['月度胜率'].mean()
        df_res.loc[len(df_res)] = ['年化', annual_return, annual_ddpercent, annual_sharp, annual_vol, annual_calmar, annual_win_rate_m]

        if len(sp):
            df_res.to_csv(sp, index=False)
        
        return df_res

    def get_daily_pnl(self, df0: pd.DataFrame, is_datetime=0):
        df = df0.copy()
        if 'datetime' not in df.columns: df.reset_index(inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].apply(lambda x: x.date())
        df['date_c'] = np.where(df['date']!=df['date'].shift(-1), 1,0)
        df = df[df['date_c']==1]
        df.reset_index(inplace=True)
        if is_datetime:
            del df['datetime']
            df.rename(columns={'date': 'datetime'}, inplace=True)
            return df[['datetime', 'pnl']]
        else:
            return df[['date', 'pnl']]

    def volatility(self, df0: pd.DataFrame, freq='1_year', dan=1):
        '''波动率'''
        df = df0.copy()
        if 'year' in freq: length = int(df.shape[0] * eval(freq.split('_')[0]))
        df = df.iloc[-length:]
        if 'datetime' not in df.columns:
            df.reset_index(inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['day'] = df['datetime'].apply(lambda x: x.day)
        df['day_change'] = np.where(df['day']!=df['day'].shift(1), 1, 0)
        df = df[df['day_change']==1]
        if dan:
            df['pct'] = df['pnl'] - df['pnl'].shift(1)
        else:
            df['pct'] = (df['pnl'] - df['pnl'].shift(1)) / df['pnl'].shift(1)
        return df['pct'].std()*np.sqrt(250)

    def get_mean_trade_count(self, pa, suffix):
        '''计算所有品种平均开仓次数'''
        pa_li = os.listdir(pa)
        df_pa_li = filter_str(suffix, pa_li, is_list=1)
        trade_rate_li = []
        for pa_i in df_pa_li:
            df = pd.read_csv(f'{pa}{pa_i}')
            trade_rate_li.append(round(len(df[df['signal']!=0]) / len(df), 5))
        return np.mean(trade_rate_li)

    def caculate_real_return(self, diff_days, total_return):
        '''按比例计算实际收益率'''
        return total_return / diff_days * 365

    def get_trade_result(self, pa=f'{pa_prefix}/datas/ml_result/model_1.0/symbol_result_10_index/raw12/total_test/'):
        ''''''
        li = os.listdir(pa)
        li = filter_str('df_', li, is_list=1)
        li.remove('df_res_all.csv')
        def set_dic(x, res_dic):
            res_dic['datetime'].append(x['datetime'])
            res_dic['trade_price'].append(np.mean(eval(x['trade_price'])))
            res_dic['direction'].append(np.sign(x['pos']))
            res_dic['hand'].append(np.abs(x['pos']))

        for pa_i in li:
            print(pa_i)
            res_dic = {'datetime': [], 'trade_price': [], 'direction': [], 'hand': []}
            x_func = partial(set_dic, res_dic=res_dic)
            df = pd.read_csv(f'{pa}{pa_i}')
            df = df[(df['trade_price']!= '[]') & (df['trade_price']!= '0')]
            df.apply(x_func, axis=1)
            df_res = pd.DataFrame(res_dic)
            save_pa = pa_i.split('.')[0] + '_trade_result.csv'
            df_res.to_csv(f'{pa}{save_pa}', index=False)
            print(pa_i, 'done')
            


def run_BacktesterStatistics():
    '''模型结果统计'''
    symbol = 'SN'
    save_pa = f'{pa_prefix}/datas/ml_result/total/[10, 1, 1, 1]_SN_60m_1.3_sample_10_1_return_rate_60m/y_pred_[10, 1, 1, 1]_SN_60m_1.3_sample_10_1_return_rate_60m_statistics'
    ms = BacktesterStatistics()
    ms.caculate_statistics_all(train_pa=train_pa, save_pa=save_pa, symbol=symbol)
    # ms.caculate_statistics_total()
    print('run_modelstatistics done.')


if __name__ == '__main__':
    run_BacktesterStatistics()

    
            
