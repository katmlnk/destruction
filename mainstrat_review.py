# region imports
from AlgorithmImports import *
import numpy as np
# endregion


# NOTE: on terminology: In some of our examples, I realize that we use the term "alpha" instead "mu". both terms are correct, but we should always use "mu"
# "alpha" is also used for something else in the context of QF: excess risk-adjusted returns. "mu" removes that potential area that caused many ppl confusion last sem.
class DestroyingHQGAlgo(QCAlgorithm):
    # haha this class name is scary!! an algorithm to destroy HQG :O

    def initialize(self):
        # Set backtest period - 6 years of data
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(100000)
    
        # Define our investment universe
        self._tickers = [] ##universe tbd.....
        self._symbols = []
        
        for ticker in self._tickers:
            symbol = self.add_equity(ticker, Resolution.DAILY).symbol
            self._symbols.append(symbol)
        
        # 90 days = ~3 months of trading data
        self._lookback_period = 90
        
        #this is scheduling
        self.schedule.on(self.date_rules.month_start(), 
                        self.time_rules.after_market_open("SPY", 30), 
                        self.Rebalance)
        
        # warm up historical data so it's ready
        self.set_warm_up(self._lookback_period, Resolution.DAILY)


    def Rebalance(self):
        """
        1. Fetch historical price data
        2. Calculate expected returns (mu) and covariance matrix (Sigma)
        3. Solve optimization problem to get weights
        4. Execute trades to match target weights
        """
        
        # Get history for all symbols at once
        history = self.history(self._symbols, self._lookback_period + 1, Resolution.DAILY)
        
        if history.empty or len(history) < self._lookback_period + 1:
            # NOTE: yep, want to make sure we have enough data to make good decisions. if we only look at, say, one day of data, we will be mostly analyzing noise.
            return
        
        # Get close prices for all symbols
        close_prices = history['close'].unstack(level=0)
        
        # NOTE: the below method is correct, but I think it is helpful to abstract the calculateion of mu and Sigma into their own functions, as the calculations for each may grow a bit complex.
        
        # OLD CODE (kept for reference):
        # Calculate daily log returns: ln(P_t / P_{t-1})
        # Log returns approximately normal for small changes
        #log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        
        # Mean return vector (mu): expected return for each asset
        # Covariance matrix (Sigma): measures how assets move together
        #mu = log_returns.mean().values
        #Sigma = log_returns.cov().values

        # NOTE: added calculate_Mu() and calculate_Sigma()
        # This makes it easier to experiment with different methods, like:
        # - Simple historical mean vs. exponentially weighted mean
        # - Sample covariance vs. shrinkage estimators (e.g., Ledoit-Wolf)
        mu = self.calculate_Mu(close_prices)
        Sigma = self.calculate_Sigma(close_prices)
        
        # Solve the mean-variance optimization problem
        weights = self.Optimize(mu, Sigma)

        # If optimization failed, skip rebalancing this period
        if weights is None:
            return

        for symbol, weight in zip(self._symbols, weights):
            if weight > 0.001:
                self.set_holdings(symbol, float(weight))
            else:
                self.liquidate(symbol)


    def calculate_Mu(self, close_prices):
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        mu = log_returns.mean().values
        return mu

    def calculate_Sigma(self, close_prices):
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        Sigma = log_returns.cov().values
        return Sigma


    def Optimize(self, mu, Sigma):

        # NOTE:
        # technically, this code acheives the goal of finding a portfolio that balances risk and reward
        # however, it's structure leads us to having some biases + technical issues / edge cases

        try:
            gamma = 10 #risk penalty severity            
            N = len(mu)
            inv_sigma = np.linalg.pinv(Sigma)

            # w* = (1/gamma) * Sigma^{-1} * mu
            raw_weights = (1/gamma) * inv_sigma.dot(mu)     

            # Constraint 1: Long only (no short selling)
            # Set all negative weights to 0
            raw_weights[raw_weights < 0] = 0
            # NOTE: consider, what would happen if most output weights are negative? eg, [-.5, -.25, .25]
            # this will be normalized (below) to [0,0,1]. That's a lot of risk - all eggs in one basket.
            # PROBLEM: If mu has mostly negative values (bearish outlook), we end up concentrated in 1-2 assets
            # SOLUTION: Use a proper constrained optimizer that respects long-only from the start

            # Constraint 2: Cap individual weights
            # No single position can exceed 100% of portfolio
            weight_cap = 1
            raw_weights = np.minimum(raw_weights, weight_cap)
            # NOTE: consider, what would happen if there is only one nonzero positive weight and a weight cap of 0.25? eg, [0,0,0,1,0,0]
            # we would be left with [0,0,0,0.25,0,0] after cap
            # then will be normalized (below) back to [0,0,0,1,0,0]
            # PROBLEM: Capping then normalizing can undo the cap!
            # SOLUTION: Apply constraints simultaneously in optimization, not sequentially

            if raw_weights.sum() == 0:
                return None

            weights = raw_weights / raw_weights.sum()
            return weights

            # NOTE: cvxpy or gurobi may be better suited for this
            # example: https://github.com/Husky-Quantitative-Group/hqg-strategies/blob/main/projects/markowitz/basic/MeanVarBaseline.py

            # Why the current optimization function is a bit flawed:
            # This function solves the UNCONSTRAINED optimization problem first, THEN applies constraints as post-processing filters.
            # This is fundamentally incorrect because:
            # 
            # 1. Violates optimality: The unconstrained solution w* = (1/gamma) * Sigma^{-1} * mu is optimal for a DIFFERENT problem than what we actually want.
            #    Example: unconstrained optimal solution might be [0.00005, 0.3823, -0.492, 1.10965]
            #    After applying filters (set negatives to 0, cap at 1, normalize), we get [0, 0.23, 0, 0.77]
            #    These end up being completely different portfolios. The filtered version is not optimal for our constrained problem.
            # 
            # 2. Concentration risk: When we zero out negative weights then renormalize, we amplify the remaining weights.
            #    If only 1-2 assets had positive weights initially, we end up 100% concentrated in those assets.
            #    This is much riskier than the optimizer intended.
            # 
            # 3. Constraint interations: Applying constraints sequentially (clip negative -> cap -> normalize) means they interfere with each other.
            #    Example: [0, 0, 0, 1.5, 0] -> cap at 1 -> [0, 0, 0, 1, 0] -> normalize → [0, 0, 0, 1, 0]
            #    The normalization step can "undo" the capping step, so constraints aren't actually enforced.
            # 
            # With cvxpy...
            # We can add constraints DIRECTLY to the optimization problem, not as post-processing. The optimizer then searches within the feasible region defined by constraints.
            # 
            # Think of it like this:
            # Our goal is to find the lowest point on a path (constrained space) that runs through a valley (full solution space). (optimal point -- lowest risk minus return)
            # Current method: Find the lowest point in an entire valley, then walk uphill until you're back on the path. This can bring us to a part of the path that isn't the lowest.
            # CVXPY method: Find the lowest point that's ON the path from the start.
            # 
            # The constraints form a "grid" that limits the solution space. The optimizer finds the best point within that grid.
            # So the output is guaranteed to be (1) feasible (satisfies all constraints) and (2) optimal (best risk-return tradeoff given constraints).
            # 
            # FUTURE CONSIDERATION:
            # As we progress, we'll see that hard constraints (long-only, box constraints) are somewhat "unnatural" for mean-variance.
            # They reduce the feasible region and can force suboptimal solutions compared to the unconstrained case.
            # The math behind mean-variance assumes a smooth, convex optimization landscape.
            # Adding hard boundaries disrupts this, especially at the constraint edges.
            # 
            # Our advisor recommended that instead of constraining the optimization, we should focus on:
            # (1) Improving the inputs (mu and Sigma): Better estimators -> better unconstrained solution that naturally avoids extremes
            # (2) Modifying the objective function: maybe regularization terms (e.g., L1/L2 penalties) instead of hard constraints
            # This keeps the problem more mathematically clean while still controlling extreme positions.

        except:
            return None