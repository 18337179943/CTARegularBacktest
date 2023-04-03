import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import partial
import sys, os
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_prefix)
from m_base import *
from copy import deepcopy
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval, Exchange
import warnings
warnings.filterwarnings("ignore")

__Author__ = 'ZCXY'


class BackAdjustment():
    '''后复权数据处理'''
    def __init__(self) -> None:
        self.inactive_contract = ['LR']
        self.pa_maincon_info = './datas/maincon_info/maincon_info_adj.csv'
        self.pa_feather = './datas/Futures_Data'
        self.pa_adj_datas = makedir('./datas/adj_datas')
        self.df_maincon_adj = self.get_df_maincon_adj(self.pa_maincon_info)
    
    def get_daily_info(self, pa=f'./datas/Futures_Data/CZCE_Minute/20171016.feather'):
        '''
        获取每天信息
        return: df
        | tradedate | contract | close | volume | turnover | symbol | exchange | maincoin | sub_maincoin | 
        | tradedate | symbol | maincon0 | maincon1 | maincon2 | maincon0_close | maincon1_close | maincon2_close | 
        '''
        n = 3
        exchange = pa.split('/')[3].split('_')[0]
        res_li = []
        df = pd.read_feather(pa)
        for contract, df_i in df.groupby('TICKER'):
            res_li.append([df_i['date'].iloc[-1], contract, df_i['close'].iloc[-1], 
                           df_i['volume'].sum(), df_i['total_turnover'].sum(), get_sy(contract), exchange])
        df_res = pd.DataFrame(res_li)
        df_res.columns = ['tradedate', 'contract', 'close', 'volume', 'turnover', 'symbol', 'exchange']
        maincon_li = []
        for symbol, df_i in df_res.groupby('symbol'):
            contract_v_li = get_maxn_values(df_i, 'volume', 'contract', n)
            contract_t_li = get_maxn_values(df_i, 'turnover', 'contract', n)

            if len(contract_v_li) == 1:
                maincon0, maincon1, maincon2 = contract_v_li[0], contract_v_li[0], contract_v_li[0]
            elif len(contract_v_li) == 2:
                maincon0 = sorted([contract_v_li[0], contract_t_li[0]])[0]
                maincon1_li = sorted([contract_v_li[1], contract_t_li[1]])
                maincon1 = maincon1_li[1] if maincon1_li[0] == maincon0 else maincon1_li[0]
                maincon2 = maincon1
            else:
                maincon0 = sorted([contract_v_li[0], contract_t_li[0]])[0]
                maincon1_li = sorted([contract_v_li[1], contract_t_li[1]])
                maincon1 = maincon1_li[1] if maincon1_li[0] == maincon0 else maincon1_li[0]
                maincon2_li = sorted([contract_v_li[2], contract_t_li[2]])
                maincon2 = maincon2_li[1] if maincon2_li[0] == maincon1 else maincon2_li[0]

            maincon0_close, maincon1_close, maincon2_close = df_i[df_i['contract']==maincon0]['close'].iloc[0], \
                        df_i[df_i['contract']==maincon1]['close'].iloc[0], df_i[df_i['contract']==maincon2]['close'].iloc[0]
            
            maincon_li.append([df_i['tradedate'].iloc[0], exchange, symbol, maincon0, maincon1, maincon2, 
                                maincon0_close, maincon1_close, maincon2_close])

        df_maincon = pd.DataFrame(maincon_li)
        df_maincon.columns = ['tradedate', 'exchange', 'symbol', 'maincon0', 'maincon1', 'maincon2', 
                              'maincon0_close', 'maincon1_close', 'maincon2_close']
        # print(pa, 'done.')
        # df_maincon.to_csv('df_maincon1.csv')
        return df_maincon
    
    def get_datas_pa(self, pa=f'./datas/Futures_Data'):
        '''获取每日所有数据路径'''
        pa_li = os.listdir(pa)
        pa_li = [f'{pa}/{pa_i}' for pa_i in pa_li]
        pa_res_li = []
        for pa_i in pa_li:
            pa_i_li = os.listdir(pa_i)
            pa_i_li = [f'{pa_i}/{pa_j}' for pa_j in pa_i_li]
            pa_res_li.append(deepcopy(pa_i_li))
        pa_res_li = flatten_list(pa_res_li)
        return pa_res_li

    def get_maincon_info(self, pa=f'./datas/Futures_Data'):
        '''获取每日主力合约信息'''
        pa_res_li = self.get_datas_pa(pa)
        multi_res = multiprocess(self.get_daily_info, pa_res_li, 3)
        df_res_li = []
        for i in multi_res: df_res_li.append(i)
        df_res = pd.concat(df_res_li)
        df_res.sort_values(['tradedate', 'symbol'], ascending=True, inplace=True)
        save_pa = makedir('./datas/maincon_info')
        df_res.to_csv(f'{save_pa}/maincon_info.csv', index=False)
        print('get_maincon_info done.')
        return df_res
    
    def get_day_close(self, df: pd.DataFrame, contract0, i):
        '''set_maincon 使用到 获取某天的某个合约的收盘价'''
        exchange = df['exchange'].iloc[i]
        tradedate = change_date(df['tradedate'].iloc[i])
        pa = f'./datas/Futures_Data/{exchange}_Minute/{tradedate}.feather'
        df_read = pd.read_feather(pa)
        close = df_read[df_read['TICKER']==contract0]['close'].iloc[-1]
        return close

    def set_maincon(self, pa='./datas/maincon_info/maincon_info.csv'):
        '''计算每日主力合约'''
        df = pd.read_csv(pa)
        res_li = []
        for symbol, df_i in df.groupby('symbol'):
            # df_i['maincon0_pre'] = df_i['maincon0'].shift(1)
            print(symbol)
            if symbol in self.inactive_contract: continue
            df_i['adjfactor'] = None
            df_i['adjfactor'].iloc[0] = 1
            df_i['is_change'] = 0

            for i in range(1, df_i.shape[0]):
                dic_maincon = {df_i['maincon0'].iloc[i]:df_i['maincon0_close'].iloc[i],
                                df_i['maincon1'].iloc[i]:df_i['maincon1_close'].iloc[i],
                                df_i['maincon2'].iloc[i]:df_i['maincon2_close'].iloc[i]}
                dic_pre_maincon = {df_i['maincon0'].iloc[i-1]:df_i['maincon0_close'].iloc[i-1],
                                df_i['maincon1'].iloc[i-1]:df_i['maincon1_close'].iloc[i-1],
                                df_i['maincon2'].iloc[i-1]:df_i['maincon2_close'].iloc[i-1]}

                if df_i['maincon0'].iloc[i] < df_i['maincon0'].iloc[i-1]:
                    contract0 = df_i['maincon0'].iloc[i-1]
                    df_i['maincon0'].iloc[i] = contract0
                    try:
                        df_i['maincon0_close'].iloc[i] = dic_maincon[contract0]
                    except:
                        close = self.get_day_close(df_i, contract0, i)
                        df_i['maincon0_close'].iloc[i] = close

                if df_i['maincon1'].iloc[i] < df_i['maincon0'].iloc[i]:
                    contract1 = df_i['maincon2'].iloc[i]
                    df_i['maincon1'].iloc[i] = contract1
                    df_i['maincon1_close'].iloc[i] = dic_maincon[contract1]

                if df_i['maincon0'].iloc[i] != df_i['maincon0'].iloc[i-1]:
                    maincon0 = df_i['maincon0'].iloc[i]
                    if maincon0 in dic_pre_maincon.keys():
                        df_i['adjfactor'].iloc[i] = dic_pre_maincon[df_i['maincon0'].iloc[i-1]] / dic_pre_maincon[maincon0]
                        df_i['is_change'].iloc[i] = 1
                    else:
                        if df_i['maincon0'].iloc[i-1] != df_i['maincon0'].iloc[i+1]:
                            try: 
                                close = self.get_day_close(df_i, df_i['maincon0'].iloc[i], i-1)
                                df_i['adjfactor'].iloc[i] = dic_pre_maincon[df_i['maincon0'].iloc[i-1]] / close
                                df_i['is_change'].iloc[i] = 1
                            except:
                                df_i['maincon0'].iloc[i] = df_i['maincon0'].iloc[i-1]
                                if df_i['maincon0'].iloc[i] in dic_maincon.keys():
                                    df_i['maincon0_close'].iloc[i] = dic_maincon[df_i['maincon0'].iloc[i]]
                                else:
                                    try:
                                        close = self.get_day_close(df_i, df_i['maincon0'].iloc[i], i)
                                        df_i['maincon0_close'].iloc[i] = close
                                    except:
                                        print(df_i['maincon0'].iloc[i-1], df_i['maincon0'].iloc[i])
                                        print(df_i['tradedate'].iloc[i])
                            
                        else:
                            df_i['maincon0'].iloc[i] = df_i['maincon0'].iloc[i-1]
                            close = self.get_day_close(df_i, df_i['maincon0'].iloc[i], i-1)
                            df_i['maincon0_close'].iloc[i] = close

            # df_i.rename(columns={'tradedate': 'date'}, inplace=True)
            df_adj = df_i.dropna()
            df_adj['adjfactor'] = df_adj['adjfactor'].cumprod()

            del df_i['adjfactor']

            df_merge = pd.merge(df_i, df_adj[['adjfactor']], how='outer', left_index=True, right_index=True)
            # print(df_merge.head(20))
            df_merge.fillna(method='ffill', inplace=True)
            # print(df_merge.head(20))
            # input()
            res_li.append(df_merge.copy())

        df_concat = pd.concat(res_li)
        df_concat.to_csv(self.pa_maincon_info, index=False)
        print('set_maincon done.')
        return df_concat
    
    def get_df_maincon_adj(self, pa):
        '''获取主力合约信息表'''
        try:
            df_mainicon_adj = pd.read_csv(pa)
        except:
            self.get_maincon_info()
            df_mainicon_adj = self.set_maincon()
        return df_mainicon_adj

    def get_adjust_datas(self, pa='./datas/Futures_Data/CFFEX_Minute/20170421.feather'):
        '''计算后复权的k线数据'''
        df = pd.read_feather(pa)
        tradedate = get_sy(pa, 0)
        exchange = pa.split('/')[3].split('_')[0]
        df_maincon = self.df_maincon_adj[(self.df_maincon_adj['tradedate']==change_date(tradedate, '/')) & (self.df_maincon_adj['exchange']==exchange)]
        res_li = []
        for i in range(len(df_maincon)):
            maincon0 = df_maincon['maincon0'].iloc[i]
            adjfactor = df_maincon['adjfactor'].iloc[i]
            df_i = df[df['TICKER']==maincon0]
            df_i[['open', 'high', 'low', 'close']] = df_i[['open', 'high', 'low', 'close']] * adjfactor
            res_li.append(df_i.copy())
        
        df_res = pd.concat(res_li)
        sp = makedir(f'{self.pa_adj_datas}/{exchange}_Minute')
        df_res.to_csv(f'{sp}/{tradedate}.csv', index=False)
        print(f'{sp}/{tradedate}.csv done.')
        return df_res

    def multi_get_adjust_datas(self, pa=f'./datas/Futures_Data'):
        '''多进程计算后复权的k线数据'''
        pa_res_li = self.get_datas_pa(pa)
        multiprocess(self.get_adjust_datas, pa_res_li, 3)
        print('multi_get_adjust_datas done.')

    def save_datas_to_db(self, pa=f'./datas/adj_datas'):
        '''将csv数据保存到数据库里'''
        pa_exchange = os.listdir(pa)
        ct = ''
        for pa_e in pa_exchange: # [ind:]
            pa_date = os.listdir(f'{pa}/{pa_e}')
            for pa_d in pa_date:
                lpa = f'{pa}/{pa_e}/{pa_d}'
                bars = []
                data_df = pd.read_csv(lpa)
                data_df.dropna(inplace=True)
                data_df['datetime'] = pd.to_datetime(data_df['datetime'])
                data_df['datetime'] = data_df['datetime'].apply(lambda x: x-timedelta(hours=8, minutes=1))
                data_list = data_df.to_dict('records')
                for item in data_list:
                    dt = datetime.fromtimestamp(item['datetime'].timestamp())
                    bar = BarData(
                        symbol=get_sy(item['TICKER'])+'000',
                        exchange=Exchange.LOCAL,
                        datetime=dt,  # datetime.fromtimestamp(item['datetime'].timestamp()),
                        interval=Interval.MINUTE,
                        open_price=float(item['open']),
                        high_price=float(item['high']),
                        low_price=float(item['low']),
                        close_price=float(item['close']),
                        volume=float(item['volume']),
                        turnover=float(item['total_turnover']),
                        gateway_name="DB",
                    )
                    bars.append(bar)
                database_manager = get_database()
                database_manager.save_bar_data(bars)
                print(lpa, 'sql done.')
    

def run_BackAdjustment():
    ba = BackAdjustment()

    # ba.get_daily_info()   # 整合主力合约信息
    # ba.get_maincon_info()
    # ba.set_maincon()

    # ba.get_adjust_datas()     # 获取调整后的k线
    # ba.multi_get_adjust_datas()

    ba.save_datas_to_db()   # 把数据保存入库
    


if __name__ == '__main__':
    run_BackAdjustment()
    # pa = './datas/Futures_Data/CZCE_Minute/20140702.feather'
    # df = pd.read_feather(pa)
    # df.to_csv('df.csv')
