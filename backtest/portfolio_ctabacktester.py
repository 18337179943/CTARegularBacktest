#%%
import sys, os
sys_name = 'windows'
pa_sys = '.'
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_sys)
from m_base import *
from vnpy.trader.optimize import OptimizationSetting
from vnpy.trader.constant import Interval
from vnpy_portfoliostrategy.backtesting import BacktestingEngine
from datetime import datetime, time, timedelta
from datas_process.m_futures_factors import SymbolsInfo, MainconInfo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest.backtest_statistics import BacktesterStatistics
from backtest.strategies.maatr_portfolio_strategy import MaatrPortfoliostrategy
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

class PortfolioBackTester():
    '''主连合约回测'''
    def __init__(self, startdate=datetime(2010, 1, 1), enddate=datetime(2023, 2, 20), strategy=MaatrPortfoliostrategy):  # datetime(2020, 12, 31)
        self.startdate = startdate
        self.enddate = enddate
        self.sig_meth = 0  # 0预测 1概率预测 2真实 3二分类
        self.res_pa = makedir(f'{pa_prefix}/datas/backtest_res')
        self.syinfo = SymbolsInfo()
        self.symbol_li = self.syinfo.symbol_li_p
        self.df_symbols_all = self.syinfo.df_symbols_all
        self.bs = BacktesterStatistics()
        self.capital = 10_000_000
        # self.si = SymbolsInfo()
        self.strategy = strategy
    
    def set_strategy(self, strategy):
        ''''''
        self.strategy = strategy

    def backtesting(self, sy_li=None, startdate=None, enddate=None, plot=False, params={}):
        '''跑回测'''
        if sy_li is None: sy_li = self.symbol_li
        if startdate is None: startdate, enddate = self.startdate, self.enddate
        rates, priceticks, sizes, self.vt_symbols, slippages = {}, {}, {}, [], {}
        for sy in sy_li:
            vt_symbol = f'{sy}000.LOCAL'
            self.vt_symbols.append(vt_symbol)
            slippages[vt_symbol] = 0
            # sizes[vt_symbol] = 1
            rates[vt_symbol], priceticks[vt_symbol], sizes[vt_symbol], _ = self.syinfo.get_backtest_params(sy)
        
        startdate = self.startdate if self.startdate > startdate else startdate
        enddate = self.enddate if self.enddate < enddate else enddate
        
        engine = BacktestingEngine()
        engine.set_parameters(
            vt_symbols=self.vt_symbols,
            rates=rates,
            slippages=slippages,
            sizes=sizes,
            priceticks=priceticks,
            capital=self.capital,
            interval=Interval.MINUTE,
            start=startdate,
            end=enddate
        )
        params_adj = {'pricetick_dic': priceticks, 'size_dic': sizes, 'init_balance': self.capital}
        params_adj.update(params)
        engine.add_strategy(self.strategy, params_adj)
        # engine.add_strategy(AtrRsiStrategy, params)

        engine.load_data()
        engine.run_backtesting()
        df = engine.calculate_result()
        res = engine.calculate_statistics(output=True)
        
        if plot:
            engine.show_chart()

        return engine, res, df

    def run_backtesting(self, sy_li=None, pa='portfolio'):
        '''单品种跑回测'''
        pa=f'{self.res_pa}/{pa}'
        if sy_li is None: sy_li = self.symbol_li
        engine, res, df = self.backtesting(sy_li, self.startdate, self.enddate, plot=False, params={'hand': 1})
        symbol = 'total'
        sp = makedir(f'{pa}/{symbol}')
        for vt_sy, sy in zip(self.vt_symbols, sy_li):
            try:
                df_res = pd.DataFrame(engine.strategy.m_res[vt_sy].res_dic)
            except:
                print(vt_sy)
                for k, v in engine.strategy.m_res[vt_sy].res_dic.items():
                    print(k, len(v))
                    input()
                
            df_res.to_csv(f'{sp}/{sy}_df_res.csv', index=False)
            df_res['datetime'] = pd.to_datetime(df_res['datetime'])
            m_plot(df_res.set_index('datetime')[['pnl']], sy, sp)
        dic_to_dataframe(res).T.to_csv(f'{sp}/{symbol}_statistic.csv')
        df['pnl'] = df['balance'] / df['balance'].iloc[0]
        df.to_csv(f'{sp}/{symbol}_daily_pnl.csv')
        m_plot(df[['pnl']], 'pnl', f'{sp}')
        # plot_show(symbol, f'{sp}/{symbol}_df_res.csv', f'{sp}/{symbol}_res.png')
        print('done: ', symbol)
        return df



def run_PortfolioBackTester():
    bt = PortfolioBackTester()
    # bt.run_backtesting_all()
    bt.run_backtesting(sy_li=None)
    print('子进程关闭成功 BackTester')


if __name__ == "__main__":
    # multi_p('maatr') # 多进程
    run_PortfolioBackTester()
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
