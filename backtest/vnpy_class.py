from vnpy_ctastrategy import (
    BarData,
    ArrayManager,
)
import numpy as np
from datetime import time
import pandas as pd


class ArrayManager_m(ArrayManager):
    def __init__(self, size: int = 100):
        super().__init__(size)
        """Constructor"""
        self.datetime_array = np.empty(size, dtype=object)

    def update_bar(self, bar: BarData) -> None:
        """
        Update new bar data into array manager.
        """
        super().update_bar(bar)
        self.datetime_array[:-1] = self.datetime_array[1:]
        self.datetime_array[-1] = bar.datetime

    @property
    def datetime(self) -> np.ndarray:
        """
        Get open price time series.
        """
        return self.datetime_array


class MMACD():
    def __init__(self, fast, slow, n, size=100) -> None:
        self.macd_arr = np.ones(size, dtype=float)
        self.mamacd_arr = np.ones(size, dtype=float)
        self.diff_arr = np.ones(size, dtype=float)
        self.fast, self.slow, self.n = fast, slow, n
    
    def append_arr(self, arr, val):
        ''''''
        arr[:-1] = arr[1:]
        arr[-1] = val
        return arr

    def macd(self, arr):
        ''''''
        if arr[-self.slow]:
            macd_v = np.mean(arr[-self.fast:]) - np.mean(arr[-self.slow:])
            self.macd_arr = self.append_arr(self.macd_arr, macd_v)
            if self.macd_arr[-self.n]:
                mamacd = np.mean(self.macd_arr[-self.n:])
                self.mamacd = self.append_arr(self.mamacd_arr, mamacd)
                diff = macd_v - mamacd
                self.diff_arr = self.append_arr(self.diff_arr, diff)

    def macd0(self, amn):
        if amn.close[-self.slow]:
            self.macd_arr, self.mamacd_arr, self.diff_arr = amn.macd(self.fast, self.slow, self.n, array=True)


class BacktestInfo():
    ''''''
    def __init__(self, rate, pricetick, size, init_balance, add_dic={}, is_slip=0) -> None:
        self.res_dic = self.reset_res_dic(add_dic)
        self.add_dic = add_dic
        self.rate = rate
        self.pricetick = pricetick
        self.size = size
        self.init_balance = init_balance
        self.trade_res = self.reset_trade_res()
        self.first = 1
        self.is_slip = is_slip
    
    def reset_res_dic(self, add_dic: dict):
        ''''''
        res_dic = {'datetime': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': [], 'trade_price': [], 'trade_time': [],
                'signal': [], 'pos': [], 'profit': [], 'cost': [], 'balance': [], 'pct_change':[], 'pnl': []}
        res_dic.update(add_dic)
        return res_dic
    
    def update_res_dic(self, dic):
        self.res_dic.update(dic)
        return self.res_dic
    
    def update_trade_res(self, trade, pos):
        self.trade_res['price'].append(trade.price)
        self.trade_res['pos'].append(pos)
        self.trade_res['datetime'].append(trade.datetime.strftime('%Y-%m-%d %H:%M:%S'))
        return self.trade_res
    
    def save_info(self, bar: BarData, params_dic={}, hand=1):
        ''''''
        bdt = bar.datetime.strftime('%Y-%m-%d %H:%M:%S')

        if len(self.add_dic):
            for key in self.add_dic: self.res_dic[key].append(params_dic[key])
            # if self.res_dic['temp_li'][-1][-1] == 0:
            #     print(bar.datetime, self.res_dic['temp_li'][-1], self.res_dic['temp_li'][-1][-1], '-------')

        signal = params_dic['signal']
        self.res_dic['datetime'].append(bdt)
        self.res_dic['open'].append(bar.open_price)
        self.res_dic['high'].append(bar.high_price)
        self.res_dic['low'].append(bar.low_price)
        self.res_dic['close'].append(bar.close_price)
        self.res_dic['volume'].append(bar.volume)
        self.res_dic['pos'].append(params_dic['pos'])
        # self.res_dic['market_value'].append(params_dic['pos']*self.size*bar.close_price)

        # if cost_type == 0:
        #     one_hand_cost = (self.rate*bar.close_price)*self.size
        # elif cost_type == 1:
        #     one_hand_cost = (self.rate*bar.close_price + 0.5*self.pricetick)*self.size

        if self.first:
            self.first = 0
            self.res_dic['profit'].append(0)
            self.res_dic['trade_price'].append([])
            self.res_dic['trade_time'].append(0)
            cost = (self.rate*bar.close_price)*self.size*hand if signal != 0 else 0
            balance = self.init_balance
            self.start_price = bar.close_price
            pct_change = 0
        else:
            profit_i, cost = 0, 0
            price_li = self.trade_res['price'].copy()
            price_li.insert(0, self.res_dic['close'][-2]), price_li.append(bar.close_price)
            pos_li = self.trade_res['pos'].copy()
            pos_li.insert(0, self.res_dic['pos'][-2])

            for i in range(len(price_li)-1):
                profit_i += (price_li[i+1] - price_li[i])*pos_li[i]*self.size
            
            self.res_dic['profit'].append(profit_i)
            self.res_dic['trade_price'].append(self.trade_res['price'])
            self.res_dic['trade_time'].append(self.trade_res['datetime'])
            
            trade_n = len(self.trade_res['price'])
            if trade_n:
                trade_pos = abs(self.res_dic['pos'][-1]-self.res_dic['pos'][-2])
                slip = self.pricetick*trade_pos*self.size if self.is_slip else 0
                cost = self.rate*self.size*np.mean(self.trade_res['price'])*trade_pos+slip    # hand*len(self.trade_res['price'])
            balance = self.res_dic['balance'][-1] + profit_i - cost
            pct_change = round((balance-self.res_dic['balance'][-1])/self.init_balance, 4)
            
            
        self.res_dic['signal'].append(signal)
        self.res_dic['cost'].append(cost)
        self.res_dic['balance'].append(balance)
        self.res_dic['pct_change'].append(pct_change)
        self.res_dic['pnl'].append(round(balance/self.init_balance, 4))
        self.trade_res = self.reset_trade_res()
        

    def reset_trade_res(self):
        '''每隔一小时重设交易记录'''
        trade_res = {'price': [], 'pos': [], 'datetime': []}
        return trade_res
    

class BacktestBalance():
    ''''''
    def __init__(self) -> None:
        self.balance_datetime = []
        self.balance = []

    def update_balance(self, dt, balance):
        dt = dt.strftime('%Y-%m-%d %H:%M:%S')
        if len(self.balance_datetime) != 0 and dt == self.balance_datetime[-1]:
            self.balance[-1] = balance
        else:
            self.balance_datetime.append(dt)
            self.balance.append(balance)

    def get_df(self):
        df = pd.DataFrame({'dateime': self.balance_datetime, 'balance': self.balance})
        return df
    

def filter_time(bar_time):
    '''过滤非交易时间段'''
    if (time(15, 0) <= bar_time < time(20, 59)) or \
        (time(8, 0) <= bar_time < time(8, 59)):
        return 1
    else: return 0
    
