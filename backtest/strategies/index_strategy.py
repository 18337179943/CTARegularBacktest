import sys, os
sys_name = 'windows'
pa_sys = 'D:/策略开发/futures_ml/'
pa_prefix = '.' 
sys.path.insert(0, pa_sys)
from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)
from backtest.vnpy_class import *
import pandas as pd
from datetime import datetime, timedelta
from datas_process.m_futures_factors import MyIndex
from vnpy.trader.constant import Interval


class IndexStrategy(CtaTemplate):
    """"""
    author = "ZCXY"

    hand = 1
    symbol = 'RB'
    contract = 'RB000'
    rate = 0.0001
    size = 10
    pricetick = 1
    is_slip = 0

    init_balance = 10_000_000
    win_n = 1

    level = 1
    hand_sep = 5
    stop_loss_step = 3

    index_n1 = 24
    index_ind = 1

    is_signal = 0

    atr_n = 21
    stop_n = 3

    signal_pa = './datas/backtest_res/boll/RB/RB_df_res.csv'
    vol_pa = ''

    signal = 0
    atr = 0
    max_hand = 1
    open_hand = 3
    left_hand = 0
    stop_hand = 0
    count_stop = 0
    stop_time = None
    vol_level = 1

    parameters = [
        "hand",
        "symbol",
        "contract",
        "rate",
        "size",
        "pricetick",
        "init_balance",
        "win_n",
        "stop_n",
        "signal_pa",
        "hand_sep",
        "stop_loss_step",
        "vol_pa",
        "atr_n",
        "index_n1",
        "index_ind",
        "is_signal",
        "is_slip"]

    variables = ["signal", 
            "atr", 
            "pre_pos",
            "max_hand",
            "open_hand",
            "left_hand",
            "stop_hand",
            "count_stop",
            "stop_time",
            "vol_level",
            "level"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager_m(100)

        if self.win_n > 1:
            self.bgn = BarGenerator(self.on_bar, self.win_n, self.on_nmin_bar)
        else:
            self.bgn = BarGenerator(self.on_bar, self.win_n, self.on_nmin_bar, interval=Interval.HOUR)
        self.amn = ArrayManager_m(100)

        self.is_cover, self.is_sell = 0, 0 
        self.is_buy, self.is_short = 0, 0 
        self.add_dic = {"index_v": [], "level": []}
        self.m_res = BacktestInfo(self.rate, self.pricetick, self.size, self.init_balance, self.add_dic, is_slip=self.is_slip)

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        if len(self.vol_pa): self.df_vol = pd.read_csv(self.vol_pa)
        # self.df_signal = pd.read_csv(self.signal_pa)
        self.load_bar(10)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.bg.update_tick(tick)
    
    def get_kline(self):
        df = pd.DataFrame({'datetime': self.amn.datetime, 'open': self.amn.open, 'high': self.amn.high, 'low': self.amn.low, 'close': self.amn.close, 
                           'volume': self.amn.volume, 'turnover': self.amn.turnover})
        # print(self.amn.turnover)
        # print('-------------')
        df.dropna(inplace=True)
        return df

    def generate_signal(self):
        '''gplearn _total_return_all_quantile index'''
        df = self.get_kline()
        
        if len(df) < 70:
            return 0, 0
        mi = MyIndex(df)
        index_s, index_v = getattr(mi, f'zcalpha{self.index_ind}')(just_index_v=0)
        signal=2 if index_s[-1] == -1 else index_s[-1]
        return signal, index_v[-1]
    
    def set_signal(self):
        '''根据信号进行下单'''
        price = self.am.close[-1]
        # if self.signal != 0:
        #     print(self.signal, self.max_hand, self.left_hand, self.open_hand)
        #     input()
        if self.pos == 0: 
            if self.signal == 1:
                self.buy(price, min(self.left_hand, self.max_hand))
            elif self.signal == 2:
                self.short(price, min(self.left_hand, self.max_hand))
        elif self.pos > 0:
            hand_open = min(self.left_hand, self.max_hand)
            hand_close = min(abs(self.pos), self.max_hand)
            if (self.signal == -1 or self.signal == 2) and hand_close > 0:
                self.sell(price, hand_close)
            elif self.signal == 1 and hand_open > 0:
                self.buy(price, hand_open)
        elif self.pos < 0:
            hand_open = min(self.left_hand, self.max_hand)
            hand_close = min(abs(self.pos), self.max_hand)
            if (self.signal == -2 or self.signal == 1) and hand_close > 0:
                self.cover(price, hand_close)
            elif self.signal == 2 and hand_open > 0:
                self.short(price, hand_open)

    def set_hand(self, bar: BarData):
        '''设置手数'''
        if self.pos == 0:
            if self.is_signal:
                self.open_hand, self.stop_hand, self.max_hand, self.left_hand = 1, 1, 1, 1
            else:
                self.open_hand = int(self.init_balance / (self.size * bar.close_price) * self.level)
                self.stop_hand = self.open_hand // self.stop_loss_step
                self.max_hand = max(int(self.open_hand / self.hand_sep), 1)
                self.left_hand = self.open_hand
        else:
            self.left_hand = self.open_hand - abs(self.pos)
    
    def adj_zero_price(self, bar: BarData):
        '''价格为0的bar'''
        if bar.close_price == 0: bar.close_price = self.am.close[-1]
        if bar.open_price == 0: bar.open_price = self.am.open[-1]
        if bar.high_price == 0: bar.high_price = bar.open_price
        if bar.low_price == 0: bar.low_price = bar.open_price
        return bar

    def set_level(self, bar: BarData):
        '''设置杠杆'''
        atr = self.amn.atr(self.atr_n)
        if len(self.vol_pa): 
            self.vol_level = self.df_vol[(self.df_vol['year']==bar.datetime.year) & \
                (self.df_vol['month']==bar.datetime.month)]['vol_level'].iloc[0]
        self.level = min(0.005 / atr * bar.close_price * self.vol_level, 4) 

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.cancel_all()

        bar = self.adj_zero_price(bar)
        if filter_time(bar.datetime.time()):
            return
        self.bgn.update_bar(bar)
        self.am.update_bar(bar)
        
        if not self.am.inited:
            return
        
        self.set_hand(bar)
        self.set_signal()

    def on_nmin_bar(self, bar: BarData):
        """"""
        self.cancel_all()
        self.amn.update_bar(bar)
        # if not self.amn.inited:
        #     return
        self.signal, index_v = self.generate_signal() # -1：平多，1：开多，0：不操作，2：开空，-2：平空

        if (self.pos == 0 and self.signal != 0) or (self.pos > 0 and self.signal == 2) or \
            (self.pos < 0 and self.signal == 1):
            self.set_level(bar)
        
        params_dic = {"signal": self.signal, "pos": self.pos, "level": self.level, "index_v": index_v}
        self.m_res.save_info(bar, params_dic)

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.m_res.update_trade_res(trade, self.pos)
        
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass
