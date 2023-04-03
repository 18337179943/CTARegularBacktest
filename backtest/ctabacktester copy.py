#%%
import sys, os
sys_name = 'windows'
pa_sys = '.'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from m_base import *
from vnpy.trader.optimize import OptimizationSetting
from vnpy_ctastrategy.backtesting import BacktestingEngine
from datetime import datetime, time, timedelta
from datas_process.m_futures_factors import SymbolsInfo, MainconInfo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest.backtest_statistics import BacktesterStatistics
from backtest.strategies.bollinger_signal_strategy import BollingerSignalStrategy
from backtest.strategies.bollinger_strategy import BollingerStrategy
from backtest.strategies.stockindex_signal_strategy import StockindexSignalStrategy
from backtest.strategies.stockindex_strategy import StockindexStrategy
from backtest.strategies.dow_strategy import DowStrategy
from backtest.strategies.dow_adj_strategy import DowAdjStrategy
from backtest.strategies.dow_adj2_strategy import DowAdj2Strategy
from backtest.strategies.index_strategy import IndexStrategy
# from backtest.model_statistics import ModelStatistics
from datas_process.m_futures_factors import SymbolsInfo
from backtest.data_analyze_show import plot_show
from functools import partial
from time import sleep
import multiprocessing
# import swifter
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决符号无法显示
# sys.stdout = Logger('./datas/backtest_res/log.txt')

__Author__ = 'ZCXY'

class BackTester():
    '''主连合约回测'''
    def __init__(self, startdate=datetime(2010, 1, 1), enddate=datetime(2023, 2, 20), strategy=BollingerSignalStrategy):  # datetime(2020, 12, 31)
        self.startdate = startdate
        self.enddate = enddate
        self.sig_meth = 0  # 0预测 1概率预测 2真实 3二分类
        self.res_pa = makedir(f'{pa_prefix}/datas/backtest_res')
        self.syinfo = SymbolsInfo()
        self.symbol_li = self.syinfo.symbol_li
        self.df_symbols_all = self.syinfo.df_symbols_all
        self.bs = BacktesterStatistics()
        self.capital = 1_000_000
        self.del_symbol = ['IC', 'IF', 'IH']
        # self.si = SymbolsInfo()
        self.strategy = strategy
    
    def set_strategy(self, strategy):
        ''''''
        self.strategy = strategy

    def backtesting(self, contract, startdate, enddate, plot=False, params={}):
        '''跑回测'''
        # print(contract)
        symbol = get_sy(contract)
        rate, pricetick, size, hand = self.syinfo.get_backtest_params(symbol)
        startdate = self.startdate if self.startdate > startdate else startdate
        enddate = self.enddate if self.enddate < enddate else enddate
        engine = BacktestingEngine()
        engine.set_parameters(
            vt_symbol=f"{contract}.LOCAL",
            interval="1m",
            start=startdate,
            end=enddate,
            rate=rate,
            slippage=pricetick,
            size=size,
            pricetick=pricetick,
            capital=self.capital,
        )
        params_adj = {'hand': hand, 'symbol_name': symbol, 'contract': contract, 'rate': rate, 'size': size, 'pricetick': pricetick,
                    'init_balance': self.capital}
        params_adj.update(params)
        engine.add_strategy(self.strategy, params_adj)
        # engine.add_strategy(AtrRsiStrategy, params)

        engine.load_data()
        engine.run_backtesting()
        df = engine.calculate_result()
        res = engine.calculate_statistics(output=True)
        
        if plot:
            engine.show_chart()

        res.update({'symbol': symbol})
        # print(contract)
        return engine, res, df

    def run_backtesting_all(self):
        '''单进程回测所有品种'''
        for symbol in self.symbol_li:
            print('begin: ', symbol)
            contract = symbol + '000'
            self.run_backtesting(contract)
            print('done: ', symbol)
    
    def run_backtesting(self, contract='RM000', pa='boll_signal'):
        '''单品种跑回测'''
        pa=f'{self.res_pa}/{pa}'
        engine, res, df = self.backtesting(contract, self.startdate, self.enddate, plot=False, params={'hand': 1})
        symbol = get_sy(contract)
        df_res = pd.DataFrame(engine.strategy.m_res.res_dic)
        sp = makedir(f'{pa}/{symbol}')
        df_res.to_csv(f'{sp}/{symbol}_df_res.csv', index=False)
        dic_to_dataframe(res).T.to_csv(f'{sp}/{symbol}_statistic.csv')
        df.to_csv(f'{sp}/{symbol}_daily_pnl.csv')
        plot_show(symbol, f'{sp}/{symbol}_df_res.csv', f'{sp}/{symbol}_res.png')
        print('done: ', symbol)
        return df_res


class MainconBackTester(BackTester):
    '''主力合约回测'''
    def __init__(self, startdate=datetime(2010, 1, 1), enddate=datetime(2022, 12, 31), strategy=BollingerStrategy):
        super().__init__(startdate, enddate, strategy)
        self.mainconinfo = MainconInfo()
    
    def symbol_backtesting(self, symbol, startdate=None, enddate=None, save_pa='boll', params={}, is_plot=1):
        '''单品种全合约回测'''
        if startdate==None:
            startdate, enddate = self.startdate, self.enddate
        df_contractinfo = self.mainconinfo.get_symbol_df_maincon(symbol, startdate, enddate, delay=30)  # 2 修改
        annual_return_li = []
        for i in range(df_contractinfo.shape[0]):
            q = df_contractinfo.iloc[i].to_list()
            params_adj = {'signal_pa': f'./datas/backtest_res/boll_signal/{symbol}/{symbol}_df_res.csv', 'is_oi': 0, 'stop_price_n': 8}
            params_adj.update(params)
            # try:
            engine, _, _ = self.backtesting(q[0], timestamp_to_datetime(q[1]), timestamp_to_datetime(q[2]), plot=False, params=params_adj)
            # except:
                # print(q, 'engine is 0')
                # engine = 0   # 修改
            if engine == 0:
                continue
            df_i = pd.DataFrame(engine.strategy.m_res.res_dic)
            # print('end: ', df_i['datetime'].iloc[-1])
            annual_return_li.append(df_i)
        df_res = pd.concat(annual_return_li, ignore_index=True)
        df_res = df_res.drop_duplicates(subset=['datetime'], keep='first')
        df_res = df_res[(df_res['trade_time']!='0') & (df_res['trade_time']!=0)]
        df_res['pnl'] = df_res['pct_change'].cumsum() + 1
        df_res.reset_index(drop=True, inplace=True)
        if len(df_res) > 50:
            save_folder = makedir(f'{save_pa}/{symbol}/') if self.res_pa in save_pa else makedir(f'{self.res_pa}/{save_pa}/{symbol}/')
            df_res_pa = f'{save_folder}/{symbol}_df_res.csv'
            df_res.to_csv(df_res_pa, index=False)
            self.bs.caculate_statistics(df_res, save_pa=f'{save_folder}/{symbol}_df_statistic.csv')
            if is_plot: plot_show(symbol, df_res_pa, f'{save_folder}/{symbol}_res.png')
        return df_res
    
    def symbol_backtesting1(self, startdate=None, enddate=None, save_pa='boll', params={}, symbol=None):
        '''改写回测传参顺序'''
        return self.symbol_backtesting(symbol, startdate=startdate, enddate=enddate, save_pa=save_pa, params=params)

    def symbol_backtesting_all(self, startdate=None, enddate=None, save_pa='boll', params={}, max_worker=4):
        '''全品种回测'''
        if startdate is None: startdate, enddate = self.startdate, self.enddate
        # sp = f'{self.res_pa}/{save_pa}'
        sp = save_pa
        symbol_li = list(filter(lambda x: x not in os.listdir(f'{sp}'), self.symbol_li))

        # while len(symbol_li):
            # symbol = symbol_li[0] if symbol_method else symbol_li[0] + '000' 
            # print(symbol)
            # n = min(max_worker, len(symbol_li))
            # run_symbol_li = symbol_li[-n:]
        func_bt = partial(self.symbol_backtesting1, startdate, enddate, save_pa, params)
        func_mp = partial(m_multiprocess, func_bt, symbol_li, max_worker)
        child_process = multiprocessing.Process(target=func_mp)
        child_process.start()
        while True:
            if not child_process.is_alive():
                child_process = None
                print('子进程关闭成功2')
                break
            else:
                sleep(2)
            # symbol_li = list(filter(lambda x: x not in os.listdir(f'{sp}'), symbol_li))
            # [symbol_li.remove(i) for i in run_symbol_li]

    def symbol_backtesting_total_copy(self, pa='boll', save_pa='total'):
        '''全品种总收益'''
        if self.res_pa not in pa: pa = f'{self.res_pa}/{pa}'
        if self.res_pa not in save_pa:
            sp = pa.split('/')[-1]
            sp = makedir(f'{self.res_pa}/{save_pa}/{sp}')
        else:
            sp = save_pa
        sy_li = os.listdir(pa)
        df_merge, df_pnl_merge = pd.DataFrame(), pd.DataFrame()
        df_statistic_merge = pd.DataFrame()
        for sy in sy_li:
            # print(sy)
            res_pa = f'{pa}/{sy}'
            df_res = pd.read_csv(f'{res_pa}/{sy}_df_res.csv')
            df_res['datetime'] = pd.to_datetime(df_res['datetime'])
            df_res.rename(columns={'pct_change': f'pct_change_{sy}', 'pnl': f'pnl_{sy}'}, inplace=True)
            df_pnl = df_res[['datetime', f'pnl_{sy}']]
            df_res = df_res[['datetime', f'pct_change_{sy}']]
            
            if not os.path.exists(f'{res_pa}/{sy}_df_statistic.csv'): 
                continue

            df_statistic = pd.read_csv(f'{res_pa}/{sy}_df_statistic.csv')
            df_statistic.rename(columns={'数值': f'数值_{sy}'}, inplace=True)

            if len(df_merge) == 0: 
                df_merge = df_res.copy()
                df_statistic_merge = df_statistic.copy()
                df_pnl_merge = df_pnl.copy()

            else: 
                df_merge = pd.merge(df_merge, df_res.copy(), how='outer', left_on='datetime', right_on='datetime')
                df_statistic_merge = pd.merge(df_statistic_merge, df_statistic.copy(), how='outer', left_on='指标名称', right_on='指标名称')
                df_pnl_merge = pd.merge(df_pnl_merge, df_pnl.copy(), how='outer', left_on='datetime', right_on='datetime')
        
        df_merge.sort_values('datetime', ascending=True, inplace=True)
        df_pnl_merge.sort_values('datetime', ascending=True, inplace=True)
        df_statistic_merge['数值_total'] = df_statistic.iloc[:, 1:].mean(axis=1)

        df_merge['pct_change'] = df_merge.iloc[:, 1:].mean(axis=1)
        df_merge['pnl'] = df_merge['pct_change'].cumsum() + 1

        df_pnl_merge.fillna(method='ffill', inplace=True)
        df_pnl_merge['pnl'] = df_pnl_merge.iloc[:, 1:].mean(axis=1)

        # df_merge.fillna(method='ffill', inplace=True)
        # df_merge['pnl'] = df_merge.mean(axis=1)
        df_merge.reset_index(drop=True, inplace=True)
        df_merge.to_csv(f'{sp}/total_df_pct.csv', index=False)
        df = df_merge[['datetime', 'pnl']]
        # df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df_statistic_merge.to_csv(f'{sp}/total_df_statistic_sy.csv', index=False)
        df_pnl_merge.to_csv(f'{sp}/total_df_pnl.csv', index=False)
        df_statistic = self.bs.caculate_statistics_total(df_pnl_merge, df_statistic_merge, save_pa=f'{sp}/total_df_statistic.csv')
        df_pnl_merge.set_index('datetime', inplace=True)
        # m_plot(df, 'total', sp)
        m_plot(df_pnl_merge['pnl'], 'total_pnl', sp)
        print('symbol_backtesting_total done.')
        return df_pnl_merge, df_statistic
    
    def symbol_backtesting_total(self, pa='boll', save_pa='total'):
        '''全品种总收益'''
        if self.res_pa not in pa: pa = f'{self.res_pa}/{pa}'
        if self.res_pa not in save_pa:
            sp = pa.split('/')[-1]
            sp = makedir(f'{self.res_pa}/{save_pa}/{sp}')
        else:
            sp = save_pa
        sy_li = os.listdir(pa)
        df_merge, df_pnl_merge = pd.DataFrame(), pd.DataFrame()
        df_statistic_merge = pd.DataFrame()
        for sy in sy_li:
            # print(sy)
            res_pa = f'{pa}/{sy}'
            df_res = pd.read_csv(f'{res_pa}/{sy}_df_res.csv')
            df_res['datetime'] = pd.to_datetime(df_res['datetime'])
            df_res.rename(columns={'pct_change': f'pct_change_{sy}', 'pnl': f'pnl_{sy}'}, inplace=True)
            df_pnl = df_res[['datetime', f'pnl_{sy}']]
            df_res = df_res[['datetime', f'pct_change_{sy}']]
            
            if not os.path.exists(f'{res_pa}/{sy}_df_statistic.csv'): 
                continue

            df_statistic = pd.read_csv(f'{res_pa}/{sy}_df_statistic.csv')
            df_statistic.rename(columns={'数值': f'数值_{sy}'}, inplace=True)

            if len(df_merge) == 0: 
                df_merge = df_res.copy()
                df_statistic_merge = df_statistic.copy()
                df_pnl_merge = df_pnl.copy()

            else: 
                df_merge = pd.merge(df_merge, df_res.copy(), how='outer', left_on='datetime', right_on='datetime')
                df_statistic_merge = pd.merge(df_statistic_merge, df_statistic.copy(), how='outer', left_on='指标名称', right_on='指标名称')
                df_pnl_merge = pd.merge(df_pnl_merge, df_pnl.copy(), how='outer', left_on='datetime', right_on='datetime')
        
        df_merge.sort_values('datetime', ascending=True, inplace=True)
        df_pnl_merge.sort_values('datetime', ascending=True, inplace=True)
        # df_pnl_merge.to_csv('df_pnl_merge.csv')

        df_statistic_merge['数值_total'] = df_statistic.iloc[:, 1:].mean(axis=1)

        df_merge['pct_change'] = df_merge.iloc[:, 1:].mean(axis=1)
        df_merge['pnl'] = df_merge['pct_change'].cumsum() + 1

        df_pnl_merge.fillna(method='ffill', inplace=True)
        df_pnl_merge['pnl'] = df_pnl_merge.iloc[:, 1:].mean(axis=1)

        # df_merge.fillna(method='ffill', inplace=True)
        # df_merge['pnl'] = df_merge.mean(axis=1)
        df_merge.reset_index(drop=True, inplace=True)
        df_merge.to_csv(f'{sp}/total_df_pct.csv', index=False)
        df = df_merge[['datetime', 'pnl']]
        # df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df_statistic_merge.to_csv(f'{sp}/total_df_statistic_sy.csv', index=False)
        df_pnl_merge.to_csv(f'{sp}/total_df_pnl.csv', index=False)
        df_statistic = self.bs.caculate_statistics_total(df_pnl_merge, df_statistic_merge, save_pa=f'{sp}/total_df_statistic.csv')
        df_pnl_merge.set_index('datetime', inplace=True)
        # m_plot(df, 'total', sp)
        m_plot(df_pnl_merge['pnl'], 'total_pnl', sp)
        print('symbol_backtesting_total done.')
        # exit()
        return df_pnl_merge, df_statistic
    

    def df_res_adj(self, pa='boll'):
        lp = f'{self.res_pa}/{pa}'
        sy_li = os.listdir(lp)
        for sy in sy_li:
            print(sy)
            df_res = pd.read_csv(f'{lp}/{sy}/{sy}_df_res.csv')
            df_res = df_res[df_res['trade_time']!='0']
            df_res['pnl'] = df_res['pct_change'].cumsum() + 1

            df_res_pa = f'{lp}/{sy}/{sy}_df_res.csv'
            df_res.to_csv(df_res_pa, index=False)
            self.bs.caculate_statistics(df_res, save_pa=f'{lp}/{sy}/{sy}_df_statistic.csv')
            plot_show(sy, df_res_pa, f'{lp}/{sy}/{sy}_res.png')
        

class DynamicBacktester(MainconBackTester):
    '''动态筛选品种回测'''
    def __init__(self, startdate=datetime(2010, 1, 1), enddate=datetime(2023, 1, 1), strategy=BollingerStrategy):
        super().__init__(startdate, enddate, strategy)

    def caculate_monthy_trade(self, lp='boll', sp='boll_dynamic', sy_li=[]):
        '''动态挑选品种回测，每个月末计算前一年的夏普，卡玛，开仓次数，
        剔除开仓次数少于5次或夏普，卡玛小于0的品种'''
        if self.res_pa not in lp: lp = f'{self.res_pa}/{lp}'
        if self.res_pa not in sp: sp = makedir(f'{self.res_pa}/total/{sp}')
        else: makedir(sp)

        if len(sy_li) == 0: sy_li = os.listdir(lp)
        df_total = pd.DataFrame()
        for sy in sy_li:
            # print(sy)
            res_li = []
            df_res = pd.read_csv(f'{lp}/{sy}/{sy}_df_res.csv')
            df_res['dt'] = pd.to_datetime(df_res['datetime'])
            df_res['year'] = df_res['dt'].apply(lambda x: x.year)
            df_res['month'] = df_res['dt'].apply(lambda x: x.month)
            df_res['month_change'] = np.where(df_res['month'] != df_res['month'].shift(1), 1, 0)
            res_ind_li = df_res[df_res['month_change']==1].index
            for i in res_ind_li:
                is_trade = 1
                year, month = df_res['year'].iloc[i], df_res['month'].iloc[i]
                try:
                    pre_ind = df_res[(df_res['year']==year-1) & (df_res['month']==month-1)].index[0]
                except:
                    res_li.append([f'{year}/{month}', is_trade])
                    continue
                df_i = df_res.iloc[pre_ind:i]
                df_i.reset_index(drop=True, inplace=True)
                df_i['pnl'] = df_i['pct_change'].cumsum() + 1
                # if len((df_i[df_i['signal']!=0])) < 5:
                #     is_trade = 0
                # else:
                try:
                    res_dic = self.bs.caculate_statistics(df_i, is_df=0)
                    if res_dic['夏普比率'][0] < 0 or res_dic['收益回撤比'][0] < 0: is_trade = 0 
                except:
                    is_trade = 0
                
                res_li.append([f'{year}-{month}', is_trade])
            df_sy = pd.DataFrame(res_li)
            df_sy.columns = ['ym', f'trade_{sy}']
            df_sy['ym'] = pd.to_datetime(df_sy['ym'])
            # df_sy.to_csv(f'{save_pa}/df_dynamic_{sy}.csv', index=False)
            if len(df_total): df_total = pd.merge(df_total, df_sy.copy(), left_on='ym', right_on='ym', how='outer')
            else: df_total = df_sy
        df_total.sort_values('ym', ascending=True, inplace=True)
        df_total.to_csv(f'{sp}/df_dynamic.csv', index=False)
        print('caculate_monthy_trade done.')
        return df_total
            
    def dynamicbacktesting(self, lp='boll', sp='boll_dynamic', is_monthy_trade=1, is_result=1):
        '''动态筛选品种和回测'''
        if self.res_pa not in lp: lp = f'{self.res_pa}/total/{lp}'
        if self.res_pa not in sp: sp = f'{self.res_pa}/total/{sp}'

        # df_dynamic = self.caculate_monthy_trade(lp, sp) if is_monthy_trade else \
        #                 pd.read_csv(f'{sp}/df_dynamic.csv')
        df_dynamic = pd.read_csv(f'{sp}/df_dynamic.csv')
        df_dynamic['ym'] = pd.to_datetime(df_dynamic['ym'])

        df_pnl_total = pd.read_csv(f'{lp}/total_df_pnl.csv')
        df_pnl_total.iloc[:, 1:] = df_pnl_total.iloc[:, 1:] - df_pnl_total.iloc[:, 1:].shift(1)
        df_pnl_total.to_csv(f'{lp}/total_df_pnl_pct.csv', index=False)
        del df_pnl_total['pnl']
        df_pnl_total['datetime'] = pd.to_datetime(df_pnl_total['datetime'])
        df_pnl_total['ym'] = df_pnl_total['datetime'].apply(lambda x: f'{x.year}-{x.month}')
        df_pnl_total['ym'] = pd.to_datetime(df_pnl_total['ym'])

        df_dynamic_map = pd.merge(df_pnl_total[['datetime', 'ym']], df_dynamic, how='outer', left_on='ym', right_on='ym')
        # df_dynamic_map.to_csv('df_dynamic_map.csv')
        del df_dynamic_map['ym']
        del df_pnl_total['ym']

        df_dynamic_map.set_index('datetime', inplace=True)
        df_pnl_total.set_index('datetime', inplace=True)

        for col in df_dynamic_map.columns: 
            sy = col.split('_')[1]
            df_dynamic_map.rename(columns={col: f'pnl_{sy}'}, inplace=True) 
        # df_dynamic_map.to_csv('df_dynamic_map.csv')
        # df_pnl_total.to_csv('df_pnl_total.csv')
        df_pnl_pct = df_dynamic_map * df_pnl_total
        # df_pnl_pct.to_csv('df_pnl_pct.csv')

        df_dynamic_map['symbol_n'] = df_dynamic_map.sum(axis=1)
        df_pnl_pct['pct'] = df_pnl_pct.sum(axis=1)
        df_pnl_pct['pct'] = df_pnl_pct['pct'] / df_dynamic_map['symbol_n']
        df_pnl_pct['pnl'] = df_pnl_pct['pct'].cumsum() + 1

        df_pnl_pct[['pnl']].to_csv(f'{sp}/df_pnl_pct.csv')
        m_plot(df_pnl_pct[['pnl']], 'dynamic_pnl', sp)

        if is_result: 
            # df_res = self.bs.get_yearly_statistic(df_pnl_pct, sp=sp)
            vol = self.bs.volatility(df_pnl_pct, '1_year')
            df_pnl_pct.reset_index(inplace=True)
            if vol == 0: vol = 0.1
            return vol, df_pnl_pct[['datetime', 'pct', 'pnl']]

        return df_pnl_pct

    def vol_adj_dynamicbacktesting_copy(self, startdate=None, enddate=None, sp='vol_boll_dynamic'):
        '''含有波动率调整的动态回测'''
        res_pa = makedir(f'{self.res_pa}/vol_params')
        vol_p = f'{res_pa}/df_vol.csv'
        # vol_p = remove_file(f'{res_pa}/df_vol.csv')
        if startdate is None: startdate, enddate = self.startdate, self.enddate
        df_t = get_df_date(startdate, enddate)
        # df_t['datedelta'] = df_t['date'] - df_t['date'].iloc[0]
        res_li = []
        for i in range(len(df_t)):
            pred_date = df_t['date'].iloc[i]
            enddate = pred_date - timedelta(days=1)
            startdate = enddate - timedelta(days=400)
            folder = f'{enddate.year}_{enddate.month}'
            # if os.path.exists(f'{self.res_pa}/total/{sp}/{folder}'):
            if df_t['date'].iloc[0] > startdate: 
                if os.path.exists(vol_p):
                    df_vol = pd.read_csv(f'{vol_p}')
                    df_vol.loc[len(df_vol)] = [pred_date, pred_date.year, pred_date.month, 1]
                else:
                    df_vol = pd.DataFrame({'date': [pred_date], 'year': [pred_date.year], 'month': [pred_date.month], 'vol_level': [1]})
                df_vol.to_csv(f'{vol_p}', index=False)
                continue
            
            params = {'vol_pa': vol_p}
            save_pa = makedir(f'{self.res_pa}/{sp}/{folder}')
            self.symbol_backtesting_all(startdate, enddate, save_pa, params, max_worker=4)

            save_pa_total = makedir(f'{self.res_pa}/total/{sp}/{folder}')
            self.symbol_backtesting_total(pa=save_pa, save_pa=save_pa_total)
            self.caculate_monthy_trade(lp=save_pa, sp=save_pa_total)
            vol, df_pnl_pct = self.dynamicbacktesting(lp=save_pa_total, sp=save_pa_total, is_result=1)

            df_vol = pd.read_csv(f'{vol_p}')
            df_vol.loc[len(df_vol)] = [pred_date, pred_date.year, pred_date.month, 0.1/vol]
            df_vol.to_csv(vol_p, index=False)
            res_li.append(df_pnl_pct.copy()) 
        
        df_res = pd.concat(res_li, ignore_index=True)
        df_res.drop_duplicates(subset=['datetime'], keep='first', inplace=True)
        df_res['pnl'] = df_res['pct'].cumsum() + 1
        df_res.to_csv(f'{res_pa}/df_res.csv', index=False)
        self.bs.get_yearly_statistic(df_res, f'{res_pa}/df_statistic.csv')
        df_res.set_index('datetime', inplace=True)
        m_plot(df_res['pnl'], sp, res_pa)
        return df_vol

    def vol_adj_dynamicbacktesting(self, startdate=None, enddate=None, sp='vol_boll_dynamic', vol_params=0.1, rate=0.0002):
        '''含有波动率调整的动态回测'''
        res_pa = makedir(f'{self.res_pa}/{sp}_params')
        vol_p = f'{res_pa}/df_vol.csv'
        # vol_p = remove_file(f'{res_pa}/df_vol.csv')
        if startdate is None: startdate, enddate = self.startdate, self.enddate
        df_t = get_df_date(startdate, enddate)
        # df_t['datedelta'] = df_t['date'] - df_t['date'].iloc[0]
        res_li = []
        symbol_n = 1
        for i in range(len(df_t)):
            pred_date = df_t['date'].iloc[i]
            enddate = pred_date - timedelta(days=1)
            startdate = enddate - timedelta(days=400)  # 400
            folder = f'{enddate.year}_{enddate.month}'
            # if os.path.exists(f'{self.res_pa}/total/{sp}/{folder}'):
            if df_t['date'].iloc[0] > startdate: 
                if os.path.exists(vol_p):
                    df_vol = pd.read_csv(f'{vol_p}')
                    df_vol.loc[len(df_vol)] = [pred_date, pred_date.year, pred_date.month, 1]
                else:
                    df_vol = pd.DataFrame({'date': [pred_date], 'year': [pred_date.year], 'month': [pred_date.month], 'vol_level': [1]})
                df_vol.to_csv(f'{vol_p}', index=False)
                continue
            
            params = {'vol_pa': vol_p, 'rate': rate}
            save_pa = makedir(f'{self.res_pa}/{sp}/{folder}')
            while symbol_n > len(os.listdir(save_pa)):
                self.symbol_backtesting_all(startdate, enddate, save_pa, params, max_worker=4)
                if symbol_n <= len(os.listdir(save_pa)): 
                    print(folder, 'end backtesting.')
                    symbol_n = len(os.listdir(save_pa))
                    break

            save_pa_total = makedir(f'{self.res_pa}/total/{sp}/{folder}')
            self.symbol_backtesting_total(pa=save_pa, save_pa=save_pa_total)
            self.caculate_monthy_trade(lp=save_pa, sp=save_pa_total)
            vol, df_pnl_pct = self.dynamicbacktesting(lp=save_pa_total, sp=save_pa_total, is_result=1)

            df_vol = pd.read_csv(f'{vol_p}')
            df_vol.loc[len(df_vol)] = [pred_date, pred_date.year, pred_date.month, vol_params/vol]
            df_vol.to_csv(vol_p, index=False)
            res_li.append(df_pnl_pct.copy()) 
            print(enddate, 'done.')
        
        df_res = pd.concat(res_li, ignore_index=True)
        df_res.drop_duplicates(subset=['datetime'], keep='first', inplace=True)
        df_res['pnl'] = df_res['pct'].cumsum() + 1
        df_res.to_csv(f'{res_pa}/df_res.csv', index=False)
        self.bs.get_yearly_statistic(df_res, f'{res_pa}/df_statistic.csv')
        df_res.set_index('datetime', inplace=True)
        m_plot(df_res['pnl'], sp, res_pa)
        return df_vol
    
    def get_monthly_info(self, pa='vol_boll_dynamic', sp='vol_params', df_pa='df_dynamic.csv', drop_duplicate='ym'):
        '''获取每周持仓信息和品种pct'''
        pa = f'{self.res_pa}/total/{pa}'
        df_t = get_df_date(self.startdate, self.enddate)
        res_li = []
        for date_i in df_t['date']:
            month_pa = f'{date_i.year}_{date_i.month}'
            pa_i = f'{pa}/{month_pa}/{df_pa}'
            if os.path.exists(pa_i):
                df_i = pd.read_csv(pa_i)
                res_li.append(df_i)
        df_res = pd.concat(res_li)
        df_res.drop_duplicates(subset=[drop_duplicate], keep='first', inplace=True)     
        df_res.to_csv(f'{self.res_pa}/{sp}/{df_pa}', index=False)
        return df_res
    
    def caculate_symbol_pnl(self, pa='vol_params', sp='vol_params'):
        '''计算每个品种资金净值'''
        df_dynamic = pd.read_csv(f'{self.res_pa}/{pa}/df_dynamic.csv')
        df_pnl_pct = pd.read_csv(f'{self.res_pa}/{pa}/total_df_pnl_pct.csv')
        del df_pnl_pct['pnl']
        df_pnl_pct['datetime'] = pd.to_datetime(df_pnl_pct['datetime'])
        df_pnl_pct['ym'] = df_pnl_pct['datetime'].apply(lambda x: f'{x.year}-{x.month}')
        df_dynamic['ym'] = pd.to_datetime(df_dynamic['ym'])
        df_pnl_pct['ym'] = pd.to_datetime(df_pnl_pct['ym'])
        df_dynamic = pd.merge(df_pnl_pct[['datetime', 'ym']], df_dynamic, how='outer', left_on='ym', right_on='ym')
        del df_dynamic['ym']
        del df_pnl_pct['ym']

        df_dynamic.set_index('datetime', inplace=True)
        df_pnl_pct.set_index('datetime', inplace=True)

        for col in df_dynamic.columns: 
            sy = col.split('_')[1]
            df_dynamic.rename(columns={col: f'pnl_{sy}'}, inplace=True)
        
        df_pnl_pct = df_dynamic * df_pnl_pct

        df_dynamic['symbol_n'] = df_dynamic.sum(axis=1)
        df_pnl_pct['pct'] = df_pnl_pct.sum(axis=1)
        df_pnl_pct['pct'] = df_pnl_pct['pct'] / df_dynamic['symbol_n']
        df_pnl_pct['pnl'] = df_pnl_pct['pct'].cumsum() + 1
        df_pnl_pct.iloc[:, :-2] = df_pnl_pct.iloc[:, :-2].cumsum() + 1
        # df_pnl_pct.to_csv(f'{self.res_pa}/{sp}/df_symbol_pnl.csv')
        # [m_plot(df_pnl_pct.iloc[:, i], f'pnl_{df_pnl_pct.columns[i]}', makedir(f'{self.res_pa}/{sp}/symbol_pnl_plot')) 
        #     for i in range(df_pnl_pct.shape[1]-2)]
        df_sy_res = pd.DataFrame(df_pnl_pct.iloc[-1, :-2]-1) 
        df_sy_res.reset_index(inplace=True)
        df_sy_res.columns = ['symbol', 'return']
        df_sy_res['symbol'] = df_sy_res['symbol'].apply(lambda x: x.split('_')[-1])
        df_sy_res.sort_values('return', inplace=True)
        df_sy_res.to_csv(f'{self.res_pa}/{sp}/df_symbol_res.csv', index=False)
        return df_pnl_pct

    def caculate_level(self, pa='vol_boll_dynamic_0'):
        '''计算平均杠杆率'''
        pa = f'{self.res_pa}/{pa}'
        res_dic = {}
        res_level_li = []
        df_t = get_df_date(self.startdate, self.enddate)
        for date_i in df_t['date']:
            month_pa = f'{date_i.year}_{date_i.month}'
            if not os.path.exists(f'{pa}/{month_pa}'): continue
            for symbol in os.listdir(f'{pa}/{month_pa}'):
                df_i = pd.read_csv(f'{pa}/{month_pa}/{symbol}/{symbol}_df_res.csv')
                df_i['datetime'] = pd.to_datetime(df_i['datetime'])
                if symbol not in res_dic.keys():
                    res_dic[symbol] = [df_i.copy()]
                else:
                    res_dic[symbol].append(df_i.copy())
            print(month_pa)
        
        for sy, li in res_dic.items():
            df_i = pd.concat(li, ignore_index=True)
            df_i.drop_duplicates(subset=['datetime'], keep='first', inplace=True)
            res_level_li.append(self.bs.caculate_level(df_i))
        
        mean_level = np.mean(res_level_li)
        print('level:', mean_level)
        return mean_level


class StockIndexBacktester(DynamicBacktester):
    '''股指期货回测'''
    def __init__(self, startdate=datetime(2010, 1, 1), enddate=datetime(2023, 1, 1), strategy=StockindexSignalStrategy):
        super().__init__(startdate, enddate, strategy)
        self.symbol_li = ['IH', 'IC', 'IF']
        self.capital = 10_000_000
    
    def run_backtesting_s(self, contract='IF000', pa='momentum_signal'):
        self.run_backtesting(contract=contract, pa=pa)
    
    def run_symbol_backtesting(self, symbol='IF', sp='momentum', params={'rate': 0.00013}, strategy=StockindexStrategy):
        ''''''
        self.set_strategy(strategy=strategy)
        df_res = self.symbol_backtesting(symbol=symbol, save_pa=sp, params=params)
        sp = makedir(f'{self.res_pa}/{sp}/{symbol}')
        self.bs.get_yearly_statistic(df_res, sp=f'{sp}/df_year.csv')
        df_res.set_index('datetime', inplace=True)
        m_plot(df_res[['pnl']], f'pnl_{symbol}', save_pa=sp)
    
    def vol_adj_symbol_dynamicbacktesting(self, symbol='IF', startdate=None, enddate=None, sp='vol_mon_dynamic'):
        '''含有波动率调整的动态回测'''
        self.set_strategy(strategy=StockindexStrategy)
        res_pa = makedir(f'{self.res_pa}/vol_params_{symbol}')
        vol_p = f'{res_pa}/df_vol.csv'
        # vol_p = remove_file(f'{res_pa}/df_vol.csv')
        if startdate is None: startdate, enddate = self.startdate, self.enddate
        df_t = get_df_date(startdate, enddate)
        # df_t['datedelta'] = df_t['date'] - df_t['date'].iloc[0]
        res_li = []
        for i in range(len(df_t)):
            pred_date = df_t['date'].iloc[i]
            enddate = pred_date - timedelta(days=1)
            startdate = enddate - timedelta(days=400)  # 400
            folder = f'{enddate.year}_{enddate.month}'
            # if os.path.exists(f'{self.res_pa}/total/{sp}/{folder}'):
            if df_t['date'].iloc[0] > startdate: 
                if os.path.exists(vol_p):
                    df_vol = pd.read_csv(f'{vol_p}')
                    df_vol.loc[len(df_vol)] = [pred_date, pred_date.year, pred_date.month, 1]
                else:
                    df_vol = pd.DataFrame({'date': [pred_date], 'year': [pred_date.year], 'month': [pred_date.month], 'vol_level': [1]})
                df_vol.to_csv(f'{vol_p}', index=False)
                continue
            
            params = {'vol_pa': vol_p, 'rate': 0.00013}
            save_pa = makedir(f'{self.res_pa}/{sp}_{symbol}/{folder}')
            df_i = self.symbol_backtesting(symbol, startdate=startdate, enddate=enddate, save_pa=save_pa, params=params, is_plot=0)
            vol = self.bs.volatility(df_i, '1_year')
            res_li.append(df_i)
            
            df_vol = pd.read_csv(f'{vol_p}')
            df_vol.loc[len(df_vol)] = [pred_date, pred_date.year, pred_date.month, 0.15/vol]
            df_vol.to_csv(vol_p, index=False)

        df_res = pd.concat(res_li, ignore_index=True)
        df_res.drop_duplicates(subset=['datetime'], keep='first', inplace=True)
        df_res['pnl'] = df_res['pct_change'].cumsum() + 1
        df_res.to_csv(f'{res_pa}/df_res.csv', index=False)
        self.bs.get_yearly_statistic(df_res, f'{res_pa}/df_statistic.csv')
        df_res.set_index('datetime', inplace=True)
        m_plot(df_res['pnl'], sp, res_pa)
        return df_vol
    
    def combo_stockindex_pnl(self, sp='vol_params_idx'):
        '''三个股指期货整合'''
        df = pd.read_csv(f'{self.res_pa}/vol_params_IF/df_res.csv')[['datetime', 'pnl']]
        # df1 = pd.read_csv(f'{self.res_pa}/vol_params_IH/df_res.csv')[['datetime', 'pnl']]
        df2 = pd.read_csv(f'{self.res_pa}/vol_params_IC/df_res.csv')[['datetime', 'pnl']]
        df['datetime'] = pd.to_datetime(df['datetime'])
        # df1['datetime'] = pd.to_datetime(df1['datetime'])
        df2['datetime'] = pd.to_datetime(df2['datetime'])
        # print(df.head(10))
        # print('----------')
        # print(df1.head(10))
        # print('----------')
        # print(df2.head(10))
        # input()
        df.rename(columns={'pnl': f'pnl_IF'}, inplace=True)
        # df1.rename(columns={'pnl': f'pnl_IH'}, inplace=True)
        df2.rename(columns={'pnl': f'pnl_IC'}, inplace=True)

        # df_merge = pd.merge(df, df1, how='outer', left_on='datetime', right_on='datetime')
        df_merge = pd.merge(df, df2, how='outer', left_on='datetime', right_on='datetime')
        df_merge.sort_values('datetime', ascending=True, inplace=True)
        df_merge.set_index('datetime', inplace=True)
        df_merge.fillna(method='ffill', inplace=True)
        df_merge = df_merge - df_merge.shift(1)
        df_merge['pct'] = df_merge.mean(axis=1)
        df_merge['pnl'] = df_merge['pct'].cumsum()+1
        df_res = self.bs.get_daily_pnl(df_merge)
        sp = makedir(f'{self.res_pa}/{sp}')
        df_res.to_csv(f'{sp}/df_pnl.csv')

        m_plot(df_merge[['pnl']], 'pnl_total', save_pa=f'{sp}')
        df_merge.reset_index(inplace=True)
        self.bs.get_yearly_statistic(df_merge, sp=f'{sp}/df_year_total.csv')


class DowBacktester(StockIndexBacktester):
    '''道氏理论回测'''
    def __init__(self, startdate=datetime(2010, 1, 1), enddate=datetime(2023, 1, 1), strategy=DowAdj2Strategy):
        super().__init__(startdate, enddate, strategy)
        self.capital = 1_000_000
        self.symbol_li = self.syinfo.symbol_li


class ComboPnl():
    def __init__(self) -> None:
        self.bs = BacktesterStatistics()
    
    def get_combo_pnl(self, pa_dic: dict, sp=''):
        '''合并pnl'''
        makedir(sp)
        df_merge = pd.DataFrame()
        for key, pa_i in pa_dic.items():
            df_i = pd.read_csv(pa_i)[['datetime', 'pnl']]
            df_i = self.bs.get_daily_pnl(df_i, is_datetime=1)
            df_i.rename(columns={'pnl': f'pnl_{key}'}, inplace=True)
            if len(df_merge) == 0: df_merge = df_i.copy()
            else: 
                df_merge = pd.merge(df_merge, df_i, left_on='datetime', right_on='datetime', how='outer')
        df_merge.sort_values('datetime', inplace=True, ascending=True)
        df_merge.fillna(method='ffill', inplace=True)
        df_merge['pnl_combo'] = df_merge.iloc[:, 1:].mean(axis=1)
        m_plot(df_merge.set_index('datetime')[['pnl_combo']], 'pnl_combo', sp)
        self.bs.get_yearly_statistic1(df_merge.rename(columns={'pnl_combo': 'pnl'}), f'{sp}/df_year.csv', dan=0)
        if len(sp): df_merge.to_csv(f'{sp}/df_combo_pnl.csv', index=False)
        return df_merge

    def get_backtest_statistic(self, pa, sp=''):
        ''''''
        key = sp.split('/')[-1]
        # pa = './datas/backtest_res/vol_dow_dynamic_params/df_res.csv'
        # sp = './datas/backtest_res/vol_dow_dynamic_params'
        df = pd.read_csv(pa)[['datetime', 'pnl']]
        df_compound = self.bs.change_pnl_compound_interest(df, sp)
        df_train, df_test = self.bs.train_test_pnl_sep(df_compound, sp=sp)
        m_plot(df_train.set_index('datetime')[['pnl']], f'pnl_{key}_train', sp)
        m_plot(df_test.set_index('datetime')[['pnl']], f'pnl_{key}_test', sp)
        self.bs.get_yearly_statistic1(df_train, f'{sp}/df_train_year.csv', dan=0)
        self.bs.get_yearly_statistic1(df_test, f'{sp}/df_test_year.csv', dan=0)
        print(sp, 'done.')
    
    def get_backtest_statistic1(self, pa, sp=''):
        ''''''
        key = sp.split('/')[-1]
        # pa = './datas/backtest_res/vol_dow_dynamic_params/df_res.csv'
        # sp = './datas/backtest_res/vol_dow_dynamic_params'
        df = pd.read_csv(pa)[['datetime', 'pnl']]
        df_train, df_test = self.bs.train_test_pnl_sep(df, sp=sp)
        df_compound_train = self.bs.change_pnl_compound_interest(df_train, sp)
        df_compound = self.bs.change_pnl_compound_interest(df_test, sp)
        m_plot(df_train.set_index('datetime')[['pnl']], f'pnl_{key}_train', sp)
        m_plot(df_test.set_index('datetime')[['pnl']], f'pnl_{key}_test', sp)
        self.bs.get_yearly_statistic1(df_train, f'{sp}/df_train_year.csv', dan=0)
        self.bs.get_yearly_statistic1(df_test, f'{sp}/df_test_year.csv', dan=0)
    
    def get_backtest_statistic_total(self):
        ''''''
        pa_dic = {
            'boll': ['./资料/整合策略/pnl/df_boll.csv',
                     './资料/整合策略/pnl/boll',],
            'mom': ['./资料/整合策略/pnl/df_mom.csv',
                     './资料/整合策略/pnl/mom',],
            'dow': ['./资料/整合策略/pnl/df_dow.csv',
                     './资料/整合策略/pnl/dow',]
        }

        pa_compound_train_dic = {
            'boll': './资料/整合策略/pnl/boll/df_train.csv',
            'mom': './资料/整合策略/pnl/mom/df_train.csv',
            'dow': './资料/整合策略/pnl/dow/df_train.csv'
        }

        pa_compound_test_dic = {
            'boll': './资料/整合策略/pnl/boll/df_test.csv',
            'mom': './资料/整合策略/pnl/mom/df_test.csv',
            'dow': './资料/整合策略/pnl/dow/df_test.csv'
        }

        for key, pa_li in pa_dic.items():
            print(key)
            self.get_backtest_statistic(pa_li[0], pa_li[1])
        
        self.get_combo_pnl(pa_compound_train_dic, './资料/整合策略/pnl/combo_train')
        self.get_combo_pnl(pa_compound_test_dic, './资料/整合策略/pnl/combo_test')

        print('get_backtest_statistic_total done.')
        return 



def run_BackTester(contract):
    bt = BackTester()
    # bt.run_backtesting_all()
    bt.run_backtesting(contract)
    print('子进程关闭成功 BackTester')

def run_MainconBackTester(symbol):
    mbt = MainconBackTester()
    # mbt.set_strategy(BollingerSignalStrategy)
    # mbt.symbol_backtesting(symbol, startdate=datetime(2013, 1, 1), enddate=datetime(2014, 2, 28), 
    #         save_pa=makedir('./datas/backtest_res/vol_boll_dynamic/2014_2'))
    mbt.symbol_backtesting(symbol, save_pa='boll')        # 1 step
    # mbt.run_backtesting(symbol)
    # mbt.symbol_backtesting_total(pa='boll_1slip')      # 2 step
    # mbt.df_res_adj()
    print('子进程关闭成功 MainconBackTester')

def run_DynamicBacktester():
    db = DynamicBacktester(startdate=datetime(2010, 1, 1), enddate=datetime(2023, 1, 1))
    # db.caculate_monthy_trade()      # 3 step
    # db.dynamicbacktesting()     # 4 step
    db.vol_adj_dynamicbacktesting()
    # db.symbol_backtesting
    # db.get_monthly_info(df_pa='total_df_pnl_pct.csv', drop_duplicate='datetime')
    # db.caculate_symbol_pnl()
    # db.caculate_level()

    print('子进程关闭成功 MainconBackTester')

def run_yearly_statistic():
    bt = BackTester()       # 5 step
    # pa = f'{bt.res_pa}/total/myindex/total_df_pnl.csv'
    # sp = f'{bt.res_pa}/total/myindex/total_df_year.csv'
    pa = f'{bt.res_pa}/df_res2.csv'
    sp = f'{bt.res_pa}/df_year2.csv'
    df = pd.read_csv(pa)[['datetime', 'pnl']]
    df['datetime'] = pd.to_datetime(df['datetime'])
    m_plot(df[['pnl']], 'df', f'{bt.res_pa}')
    # print(df['pnl'].iloc[-1])
    # df = bt.bs.change_pnl_compound_interest(df)
    # df.to_csv(f'{bt.res_pa}/df_res2.csv')
    # print(df['pnl'].iloc[-1])
    # sp = './资料/整合策略/pnl/df_year_combo.csv'
    # df = pd.read_csv('./资料/整合策略/pnl/df_combo1.csv')[['date', 'pnl']]
    # df.rename(columns={'date': 'datetime'}, inplace=True)

    # bt.bs.get_yearly_statistic(df, sp)

def child_func(func=run_BackTester, params=None):
    '''跑优化的子进程'''
    func(params)
    print('子进程关闭成功1')

def multi_p():
    mbt = MainconBackTester()
    # db = DowBacktester
    save_pa = makedir(f'./datas/backtest_res/boll')
    sy_li = mbt.symbol_li
    # sy_li = [i+'000' for i in mbt.symbol_li]
    symbol_n = 44
    
    while symbol_n > len(os.listdir(save_pa)):
        sy_li = list(filter(lambda x: x not in os.listdir(save_pa), sy_li))
        m_multiprocess(run_MainconBackTester, sy_li)
        if symbol_n == len(os.listdir(save_pa)): 
            break

def parant_func(child_func=child_func, symbol_method=0):
    '''跑优化的父进程'''
    syinfo = SymbolsInfo()
    symbol_li = syinfo.symbol_li
    # [symbol_li.remove(i) for i in BackTester().del_symbol]
    symbol_li = list(filter(lambda x: x not in os.listdir(f'./datas/backtest_res/boll_op'), symbol_li))
    print(symbol_li)
    # input()
    while len(symbol_li):
        symbol = symbol_li[0] if symbol_method else symbol_li[0] + '000' 
        print(symbol)
        child_process = multiprocessing.Process(target=partial(child_func, symbol))
        child_process.start()
        while True:
            if not child_process.is_alive():
                child_process = None
                print('子进程关闭成功2')
                break
            else:
                sleep(2)
        symbol_li = list(filter(lambda x: x not in os.listdir(f'./datas/backtest_res/boll_op'), symbol_li))

def run_StockIndexBacktester():
    sib = StockIndexBacktester(startdate=datetime(2010, 1, 1), enddate=datetime(2023, 1, 1))
    # sib.run_backtesting_s('IF000')
    # sib.symbol_li = ['IF']
    # sib.run_symbol_backtesting()
    # sib.vol_adj_symbol_dynamicbacktesting(symbol='IF')
    sib.combo_stockindex_pnl('vol_params_idx1')

def run_DowBacktester(symbol='I'):
    db = DowBacktester()
    params = {'rate': 0.00013, 'signal_pa': './datas/backtest_res/dow_signal/I/I_df_res.csv', 'atr_n': 21}
    # db.run_symbol_backtesting(symbol, sp='dow', params=params, strategy=DowAdj2Strategy)
    # db.symbol_backtesting_total(pa='dow') 
    db.vol_adj_dynamicbacktesting(sp='vol_dow_dynamic', vol_params=0.15, rate=0.00013)
    # db.caculate_level(pa='vol_dow_dynamic')
    print(symbol)
    # db.run_backtesting(contract='I000', pa='dow_signal')

def run_IndexBacktester(symbol='RB'):
    db = DowBacktester(startdate=datetime(2010, 1, 1), enddate=datetime(2023, 1, 1), strategy=IndexStrategy)
    params = {'rate': 0.00013, 'atr_n': 21, 'win_n': 1}
    db.run_symbol_backtesting(symbol, sp='myindex', params=params, strategy=IndexStrategy)
    # db.symbol_backtesting_total(pa='myindex') 
    db.vol_adj_dynamicbacktesting(sp='vol_myindex_dynamic', vol_params=0.15, rate=0.00013)

def run_backtest_statistic():
    '''计算复利'''
    cp = ComboPnl()
    cp.get_backtest_statistic_total()

if __name__ == "__main__":
    # multi_p() # 多进程

    # run_BackTester('I000')    # 单品种回测
    # run_MainconBackTester('PP')   # 单品种主力合约回测
    # run_DynamicBacktester()   # 动态筛选品种回测
    # run_yearly_statistic()    # 计算年化指标
    # parant_func(child_func=partial(child_func, run_BackTester), symbol_method=0)  # 单进程回测
    # parant_func(child_func=partial(child_func, run_MainconBackTester), symbol_method=1)   # 单进程回测
    # run_StockIndexBacktester()  # 股指期货回测
    # run_DowBacktester()   # 道氏理论回测
    # run_IndexBacktester()
    # run_backtest_statistic()
    pass



# %%
