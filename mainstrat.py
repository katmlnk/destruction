# region imports
from AlgorithmImports import *
import numpy as np
# endregion

class DestroyingHQGAlgo(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(100000)
    
        self._tickers = [] ##universe tbd.....
        self._symbols = []
        
        for ticker in self._tickers:
            symbol = self.add_equity(ticker, Resolution.DAILY).symbol
            self._symbols.append(symbol)
        
        self._lookback_period = 90
        
        #this is scheduling
        self.schedule.on(self.date_rules.month_start(), 
                        self.time_rules.after_market_open("SPY", 30), 
                        self.Rebalance)
        
        # warm up historical data so it's ready
        self.set_warm_up(self._lookback_period, Resolution.DAILY)


    def Rebalance(self):
        # Get history for all symbols at once
        history = self.history(self._symbols, self._lookback_period + 1, Resolution.DAILY)
        
        if history.empty or len(history) < self._lookback_period + 1:
            return
        
        # Get close prices for all symbols
        close_prices = history['close'].unstack(level=0)
        
        # Calculate daily log returns
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        
        # Calculate mean returns (alpha) and covariance matrix
        alpha = log_returns.mean().values
        Sigma = log_returns.cov().values
        

        weights = self.Optimize(alpha, Sigma)

        if weights is None:
            return

        for symbol, weight in zip(self._symbols, weights):
            if weight > 0.001:
                self.set_holdings(symbol, float(weight))
            else:
                self.liquidate(symbol)


    def Optimize(self, alpha, Sigma):
        try:
            gamma = 10 #risk penalty severity
            N = len(alpha)
            inv_sigma = np.linalg.pinv(Sigma)

            raw_weights = (1/gamma) * inv_sigma.dot(alpha)

            # Long only
            raw_weights[raw_weights < 0] = 0

            # Cap weights
            weight_cap = 1
            raw_weights = np.minimum(raw_weights, weight_cap)

            if raw_weights.sum() == 0:
                return None

            weights = raw_weights / raw_weights.sum()
            return weights

        except:
            return None
