import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys, os
pa_prefix = '.' 
# pa_sys, pa_prefix = get_pa_prefix(sys_name)
sys.path.insert(0, pa_prefix)
import feather
import multiprocessing as mp
from datetime import timedelta
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval, Exchange
from m_base import *
import warnings
warnings.filterwarnings("ignore")

__Author__ = 'ZCXY'


class ExchangeDatas():
    def __init__(self) -> None:
        self.pa_1m = makedir('./datas/datas_1m')
        self.load_pa = './datas/Futures_Data'

    def change_date_to_contract(self, pa=f'./datas/Futures_Data/CFFEX_Minute'):
        '''合并相同合约的k线数据'''
        pa_li = os.listdir(pa)
        for pa_i in pa_li:
            df = feather.read_dataframe(f'{pa}/{pa_i}')
            # save_pa = pa_i.split('.')[0]
            for contract, df_i in df.groupby('TICKER'):
                symbol = get_sy(contract)
                pa_sy = makedir(f'{self.pa_1m}/{symbol}')
                try:
                    df_0 = pd.read_csv(f'{pa_sy}/{contract}.csv')
                    df_0['datetime'] = pd.to_datetime(df_0['datetime'])
                    df_0['trade_time'] = pd.to_datetime(df_0['trade_time'])
                    
                except:
                    df_0 = pd.DataFrame()
                # df_i = df_i[df_i['total_turnover']!=0]
                df_i['datetime'] = pd.to_datetime(df_i['datetime'])
                df_i['trade_time'] = pd.to_datetime(df_i['trade_time'])
                df_i['datetime'] = df_i['datetime'].apply(lambda x: x-timedelta(minutes=1))
                df_i['trade_time'] = df_i['trade_time'].apply(lambda x: x-timedelta(minutes=1))
                df_i = pd.concat([df_0, df_i], ignore_index=True)
                # df_i.sort_values('datetime', ascending=True, inplace=True)
                df_i.to_csv(f'{pa_sy}/{contract}.csv', index=False)
            print(f'{pa}/{pa_i} done.')

    def multi_change_date_to_contract(self, max_workers=4):
        '''多进程处理feather数据'''
        pa_li = os.listdir(self.load_pa)
        pa_li = [f'{self.load_pa}/{i}' for i in pa_li]
        multiprocess(self.change_date_to_contract, pa_li, max_workers)

    def save_datas_to_db(self, pa=f'./datas/Futures_Data', is_feather=1, is_maincon=0):
        '''将csv数据保存到数据库里'''
        pa_exchange = os.listdir(pa)

        for pa_e in pa_exchange: # [ind:]
            pa_date = os.listdir(f'{pa}/{pa_e}')
            for pa_d in pa_date:
                lpa = f'{pa}/{pa_e}/{pa_d}'
                bars = []
                if is_feather:
                    data_df = pd.read_feather(lpa)
                else:
                    data_df = pd.read_csv(lpa)
                data_df.dropna(inplace=True)
                data_df['datetime'] = pd.to_datetime(data_df['datetime'])
                data_df['datetime'] = data_df['datetime'].apply(lambda x: x-timedelta(hours=8, minutes=1))
                if is_maincon:
                    data_df['TICKER'] = data_df['TICKER'].apply(lambda x: get_sy(x)+'000')
                data_list = data_df.to_dict('records')
                for item in data_list:
                    dt = datetime.fromtimestamp(item['datetime'].timestamp())
                    bar = BarData(
                        symbol=item['TICKER'],
                        exchange=Exchange.LOCAL,
                        datetime=dt,  # datetime.fromtimestamp(item['datetime'].timestamp()),
                        interval=Interval.MINUTE,
                        open_price=float(item['open']),
                        high_price=float(item['high']),
                        low_price=float(item['low']),
                        close_price=float(item['close']),
                        volume=float(item['volume']),
                        turnover=float(item['total_turnover']),
                        open_interest=float(item['open_interest']),
                        gateway_name="DB",
                    )
                    bars.append(bar)
                database_manager = get_database()
                database_manager.save_bar_data(bars)
                print(lpa, 'sql done.')
    
    def save_datas_to_db1(self, pa=f'./datas/Futures_Data'):
        '''将csv数据保存到数据库里'''
        pa_exchange = os.listdir(pa)

        for pa_e in ['GFEX_Minute']: # [ind:]
            pa_date = os.listdir(f'{pa}/{pa_e}')
            for pa_d in pa_date:
                lpa = f'{pa}/{pa_e}/{pa_d}'
                bars = []
                data_df = pd.read_feather(lpa)
                data_df.dropna(inplace=True)
                data_df['datetime'] = pd.to_datetime(data_df['datetime'])
                data_df['datetime'] = data_df['datetime'].apply(lambda x: x-timedelta(hours=8, minutes=1))
                data_list = data_df.to_dict('records')
                for item in data_list:
                    dt = datetime.fromtimestamp(item['datetime'].timestamp())
                    bar = BarData(
                        symbol=item['TICKER'],
                        exchange=Exchange.LOCAL,
                        datetime=dt,  # datetime.fromtimestamp(item['datetime'].timestamp()),
                        interval=Interval.MINUTE,
                        open_price=float(item['open']),
                        high_price=float(item['high']),
                        low_price=float(item['low']),
                        close_price=float(item['close']),
                        volume=float(item['volume']),
                        turnover=float(item['total_turnover']),
                        open_interest=float(item['open_interest']),
                        gateway_name="DB",
                    )
                    bars.append(bar)
                database_manager = get_database()
                database_manager.save_bar_data(bars)
                print(lpa, 'sql done.')

    def change_date_to_rq(self, pa='./datas/datas_1m/A'):
        pa_li = os.listdir(pa)
        for pa_i in pa_li:
            sp = f'{pa}/{pa_i}'
            df = pd.read_csv(sp)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df['datetime'] = df['datetime'].apply(lambda x: x-timedelta(minutes=1))
            df['trade_time'] = df['trade_time'].apply(lambda x: x-timedelta(minutes=1))
            df.to_csv(sp)
            exit()

    

def run_ExchangeDatas():
    ed = ExchangeDatas()
    # ed.multi_change_date_to_contract(6)  # 多进程转换数据
    ed.save_datas_to_db(f'./datas/adj_datas', is_feather=0, is_maincon=1)
    # ed.change_date_to_contract(f'./datas/Futures_Data/CZCE_Minute')
    # ed.change_date_to_rq()



if __name__ == '__main__':
    run_ExchangeDatas()


