# flake8: noqa
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import MainWindow, create_qapp
from vnpy_ctastrategy import CtaStrategyApp
from vnpy_datamanager import DataManagerApp
from vnpy_riskmanager import RiskManagerApp

from vnpy_ctabacktester import CtaBacktesterApp
# from my_vnpy.my_gateway import CtpGateway
from vnpy_ctp import CtpGateway
from vnpy_uft import UftGateway
from vnpy_ust import UstGateway
# 写在顶部
from vnpy_portfoliostrategy import PortfolioStrategyApp


# from vnpy_ufx import UfxGateway
# from vnpy_spreadtrading import SpreadTradingApp

__Author__ = 'ZCXY'


def main():
    """"""
    qapp = create_qapp()

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)

    main_engine.add_gateway(CtpGateway)
    main_engine.add_app(CtaStrategyApp)
    main_engine.add_app(CtaBacktesterApp)
    main_engine.add_app(DataManagerApp)
    main_engine.add_app(RiskManagerApp)
    main_engine.add_app(PortfolioStrategyApp)
    main_engine.add_gateway(UftGateway)
    main_engine.add_gateway(UstGateway)
    # main_engine.add_app(SpreadTradingApp)

    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()

    qapp.exec()


if __name__ == "__main__":
    main()
