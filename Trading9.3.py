import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import argrelextrema
from enum import Enum

class BotLogger:
    def __init__(self):
        self.log = []

    def record(self, event, msg):
        self.log.append((event, msg))
    
    def output_log(self, print_log=True, save_to_file="trade_log.txt"):
        output = '\n'.join([f"{event}: {msg}" for event, msg in self.log])
        if print_log:
            print(output)
        if save_to_file:
            with open(save_to_file, 'w') as f:
                f.write(output)

class Decision(Enum):
    BUY = 1
    SELL = 2
    HOLD = 3

class BacktestTradingBot:
    def __init__(self, tickers, start_date, end_date, lookback_period=14):
        self.lookback_period = lookback_period
        self.stock_data = {}  # Initialize the stock_data attribute as an empty dictionary
        self.historical_data = self._fetch_historical_data(tickers, start_date, end_date)
        self.portfolio = {'cash': 10000}
        self.initial_balance = 10000
        self.stop_loss_percentage = 0.03
        self.trailing_stop_loss_percentage = 0.08
        self.position_size_percentage = 0.02
        self.max_loss_percentage = 0.10
        self.trade_limit = 4 # Limit trades per week
        self.trade_count = 0  # Weekly trade counter
        self.cooldown_period = 0  # Days to wait after a trade before trading again
        self.last_trade_day = {}  # Dictionary to store the last trade day for each ticker
        self.logger = BotLogger()

    def _fetch_historical_data(self, tickers, start_date, end_date):
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        processed_data = {}
        for ticker in tickers:
            df = data[ticker][['Close', 'High', 'Low', 'Volume']]
            df['OBV'] = self._compute_obv(df['Close'], df['Volume'])
            processed_data[ticker] = df
        return processed_data

    def _compute_macd(self, prices, short_window=8, long_window=17, signal_window=9):
        short_ema = prices.ewm(span=short_window).mean()
        long_ema = prices.ewm(span=long_window).mean()
        macd = short_ema - long_ema
        macd_signal = macd.rolling(window=signal_window).mean()
        return macd, macd_signal

    def _compute_bollinger_bands(self, prices, window=10):
        sma = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = sma + (rolling_std * 1.5)
        lower_band = sma - (rolling_std * 1.5)
        return upper_band, lower_band
    
    def _compute_rsi(self, prices, window=8):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _compute_stochastic_oscillator(self, prices, window=6):
        low_min = prices.rolling(window=window).min()
        high_max = prices.rolling(window=window).max()
        k = 100 * ((prices - low_min) / (high_max - low_min))
        return k

    def _compute_obv(self, prices, volumes):
        obv = volumes.copy()
        obv[prices < prices.shift(1)] *= -1
        return obv.cumsum()

    def _compute_support_resistance(self, prices):
        # Determine local minima for support, and local maxima for resistance
        local_min = argrelextrema(prices.values, np.less_equal, order=5)[0]
        local_max = argrelextrema(prices.values, np.greater_equal, order=5)[0]

        support_levels = prices.iloc[local_min]
        resistance_levels = prices.iloc[local_max]
        
        return support_levels, resistance_levels
    
    def _compute_atr(self, data, window=14):
        """Compute the Average True Range (ATR) for given data."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean().fillna(tr)
        return atr

    def _analyze_stock(self, ticker, date_index):
        data = self.historical_data[ticker]

        avg_volume = data['Volume'][-10:].mean()
        current_volume = data['Volume'].iloc[date_index]
        current_close = data['Close'].iloc[date_index]

        macd, macd_signal = self._compute_macd(data['Close'])
        upper_band, lower_band = self._compute_bollinger_bands(data['Close'])
        rsi = self._compute_rsi(data['Close'])
        k = self._compute_stochastic_oscillator(data['Close'])
        obv = self._compute_obv(data['Close'], data['Volume'])
        atr = self._compute_atr(data)  # Calculate ATR for the stock

        recent_close = self.historical_data[ticker]["Close"].iloc[date_index - self.lookback_period:date_index]
        recent_obv = self.historical_data[ticker]["OBV"].iloc[date_index - self.lookback_period:date_index]
        recent_atr = atr.iloc[date_index - self.lookback_period:date_index]  # Get recent ATR for analysis

        if recent_close.empty or recent_obv.empty or recent_atr.empty:
            return Decision.HOLD  # If we don't have enough data for a full lookback, hold.

        obv_bullish_divergence = recent_close.min() == recent_close.iloc[-1] and recent_obv.min() != recent_obv.iloc[-1]
        obv_bearish_divergence = recent_close.max() == recent_close.iloc[-1] and recent_obv.max() != recent_obv.iloc[-1]

        atr_high_volatility = recent_atr.max() == recent_atr.iloc[-1]  # Check if the current ATR is at a high indicating high volatility

        buy_scores = {
            'price_above_lower_band': current_close > lower_band.iloc[date_index],
            'macd_above_signal': macd.iloc[date_index] > macd_signal.iloc[date_index],
            'volume_above_avg': current_volume > 1.5 * avg_volume,
            'rsi_below_25': rsi.iloc[date_index] < 25,
            'stochastic_below_20': k.iloc[date_index] < 20,
            'obv_bullish_divergence': obv_bullish_divergence,
            'atr_high_volatility': atr_high_volatility  # Add ATR-based score
        }

        sell_scores = {
            'price_below_upper_band': current_close < upper_band.iloc[date_index],
            'macd_below_signal': macd.iloc[date_index] < macd_signal.iloc[date_index],
            'rsi_above_75': rsi.iloc[date_index] > 75,
            'stochastic_above_80': k.iloc[date_index] > 80,
            'obv_bearish_divergence': obv_bearish_divergence,
            'atr_high_volatility': 2  # Assign a weight to the ATR score
        }

        score_weights = {
            'price_above_lower_band': 3,
            'macd_above_signal': 2,
            'volume_above_avg': 1,
            'rsi_below_25': 2,
            'stochastic_below_20': 1,
            'obv_bullish_divergence': 3,
            'price_below_upper_band': 3,
            'macd_below_signal': 2,
            'rsi_above_75': 2,
            'stochastic_above_80': 1,
            'obv_bearish_divergence': 3,
            'atr_high_volatility': 1  # The weight you assign to the ATR indicator
        }

        buy_score = sum(score_weights[key] for key, value in buy_scores.items() if value)
        sell_score = sum(score_weights[key] for key, value in sell_scores.items() if value)

        self.logger.record(f"{ticker}@{date_index}", f"BUY SCORE: {buy_score} | SELL SCORE: {sell_score}")

        buy_threshold = 3
        sell_threshold = 3

        if buy_score > buy_threshold:
            self.logger.record(f"{ticker}@{date_index}", "Decision: BUY")
            return Decision.BUY
        elif sell_score > sell_threshold:
            self.logger.record(f"{ticker}@{date_index}", "Decision: SELL")
            return Decision.SELL
        else:
            self.logger.record(f"{ticker}@{date_index}", "Decision: HOLD")
            return Decision.HOLD

    
    def reset_trade_count(self, date):
        # Check if it's the start of the week, and reset trade_count
        if date.weekday() == 0:  # Monday
            self.trade_count = 0      
    
    def _can_trade(self, ticker, date_index):
        # Check if a trade can be executed based on the cooldown period
        last_trade_day = self.last_trade_day.get(ticker, None)
        if last_trade_day is not None:
            if date_index - last_trade_day < self.cooldown_period:
                if self.last_trade_day.get(f"{ticker}_stop_loss_triggered", False):   
                        return True 
                return False
        return True
    
    def execute_trades(self):
        for date_index in range(self.lookback_period, len(self.historical_data[next(iter(self.historical_data))])):
            current_date = self.historical_data[next(iter(self.historical_data))].index[date_index].date() # Simplified date extraction
            self.reset_trade_count(current_date)
            
            for ticker in self.historical_data:
                decision = self._analyze_stock(ticker, date_index)
                current_price = self.historical_data[ticker]['Close'].iloc[date_index] # Extract current_price for the given ticker and date_index
                if decision == Decision.BUY:
                    quantity = self._calculate_position_size(current_price)  # Use the extracted current_price here
                    self._place_order(ticker, "buy", quantity, date_index)
                elif decision == Decision.SELL:
                    quantity = self.portfolio.get(ticker, 0)  # Determine how much to sell (all shares held)
                    self._place_order(ticker, "sell", quantity, date_index)
                self._check_trailing_stop_loss(ticker, date_index)


    def _check_trailing_stop_loss(self, ticker, date_index):
        current_price = self.historical_data[ticker]['Close'].iloc[date_index]
    
    # Update the highest observed price
        highest_price_key = f"{ticker}_highest_price"
        if highest_price_key not in self.portfolio:
            self.portfolio[highest_price_key] = current_price
        else:
            self.portfolio[highest_price_key] = max(self.portfolio[highest_price_key], current_price)
        
        trailing_key = f"{ticker}_trailing_stop"
        if trailing_key in self.portfolio:
        # If current price drops below the trailing stop, sell the asset
            if current_price < self.portfolio[trailing_key]:
                self._place_order(ticker, "sell", self.portfolio.get(ticker, 0), date_index)
                self.last_trade_day[f"{ticker}_stop_loss_triggered"] = date_index  # Mark that stop loss was triggered for cooldown
        # Update the trailing stop based on the highest observed price
            else:
                updated_trailing_stop = self.portfolio[highest_price_key] - (self.trailing_stop_loss_percentage * self.portfolio[highest_price_key])
                self.portfolio[trailing_key] = max(self.portfolio[trailing_key], updated_trailing_stop)

    
    def _calculate_position_size(self, price):
    #Calculate the number of shares based on the position size percentage.
        amount_to_invest = self.portfolio['cash'] * self.position_size_percentage
        number_of_shares = amount_to_invest // price  # Use // for integer division
        return int(number_of_shares)

    def _place_order(self, ticker, action, quantity, date_index):
        if self.trade_count >= self.trade_limit or not self._can_trade(ticker, date_index):
            return

        price = self.historical_data[ticker]['Close'].iloc[date_index]

        if action == "buy":
            atr = self._compute_atr(self.historical_data[ticker]).iloc[date_index]
            stop_loss_multiplier = 2

            order_quantity = self._calculate_position_size(price)
            if order_quantity == 0:  # If there's not enough cash to buy even one share
                return
            
            self.portfolio[ticker] = self.portfolio.get(ticker, 0) + order_quantity
            self.portfolio['cash'] -= price * order_quantity
            self.portfolio[f"{ticker}_stop_loss"] = price - atr * stop_loss_multiplier
            self.portfolio[f"{ticker}_trailing_stop"] = price * (1 - self.trailing_stop_loss_percentage)

            self.trade_count += 1
            self.last_trade_day[ticker] = date_index

        elif action == "sell":
            current_quantity = self.portfolio.get(ticker, 0)
            sell_quantity = min(current_quantity, quantity)

            self.portfolio[ticker] = current_quantity - sell_quantity
            self.portfolio['cash'] += price * sell_quantity
            if f"{ticker}_stop_loss" in self.portfolio:
                del self.portfolio[f"{ticker}_stop_loss"]
            if f"{ticker}_trailing_stop" in self.portfolio:
                del self.portfolio[f"{ticker}_trailing_stop"]
            
            self.trade_count += 1
            self.last_trade_day[ticker] = date_index
            
    def _check_max_loss(self, initial_balance, date_index):
        if self.portfolio['cash'] < initial_balance * (1 - self.max_loss_percentage):
            for ticker in self.historical_data:
                if ticker in self.portfolio and self.portfolio[ticker] > 0:
                    self._place_order(ticker, "sell", self.portfolio.get(ticker, 0), date_index)
    
    def _check_stop_loss(self, ticker, date_index):
        current_price = self.historical_data[ticker]['Close'].iloc[date_index]
        
        # Check for traditional stop-loss
        stop_loss_key = f"{ticker}_stop_loss"
        if stop_loss_key in self.portfolio:
            if current_price < self.portfolio[stop_loss_key]:
                self._place_order(ticker, "sell", self.portfolio.get(ticker, 0), date_index)
                self.last_trade_day[ticker] = date_index  # Updating the last trade day to start the cooldown
                return  # Exit the function after selling
        
        # Check for trailing stop-loss
        trailing_key = f"{ticker}_trailing_stop"
        if trailing_key in self.portfolio:
            # If current price drops below the trailing stop, sell the asset
            if current_price < self.portfolio[trailing_key]:
                self._place_order(ticker, "sell", self.portfolio.get(ticker, 0), date_index)
                self.last_trade_day[ticker] = date_index  # Updating the last trade day to start the cooldown
            # Update the trailing stop if the current price is higher than the previous trailing stop + a certain percentage
            elif current_price > self.portfolio[trailing_key] * (1 + self.trailing_stop_loss_percentage):
                self.portfolio[trailing_key] = current_price * (1 - self.trailing_stop_loss_percentage)

        
    def run(self):
        for date_index in range(self.lookback_period, len(self.historical_data[next(iter(self.historical_data))])):
            
            # Reset trade count if it's the start of a new week
            date = self.historical_data[next(iter(self.historical_data))].index[date_index]
            self.reset_trade_count(date)

            # Check if portfolio has exceeded max loss
            self._check_max_loss(self.initial_balance, date_index)

            # Track decisions for trading pairs to handle simultaneous signals
            decisions = {}
            
            for ticker in self.historical_data:
                decision = self._analyze_stock(ticker, date_index)
                decisions[ticker] = decision
                
                # Check stop-loss and take-profit conditions for the current ticker
                self._check_stop_loss(ticker, date_index)

            # Handle simultaneous signals for trading pairs
            if ('TQQQ' in decisions and 'SQQQ' in decisions) and (decisions['TQQQ'] == decisions['SQQQ']):
                # Log that we're skipping trades due to simultaneous signals
                self.logger.record(f"Date:{date}", f"Simultaneous signal detected for TQQQ and SQQQ. Skipping trades.")
                continue

            # If no simultaneous signals, execute trades as per decisions
            for ticker, decision in decisions.items():
                if decision == Decision.BUY:
                    self._place_order(ticker, "buy", 1, date_index)
                elif decision == Decision.SELL:
                    self._place_order(ticker, "sell", 1, date_index)
            
            last_day_index = len(self.historical_data[next(iter(self.historical_data))]) - 1
            closing_prices = {ticker: self.historical_data[ticker]['Close'].iloc[last_day_index] for ticker in self.historical_data}

    # Calculate total value of shares for each ticker
            total_shares_value = sum(self.portfolio.get(ticker, 0) * closing_prices[ticker] for ticker in self.historical_data)

    # Calculate total portfolio value (cash + shares)
            total_portfolio_value = self.portfolio['cash'] + total_shares_value

        print(f"Closing Prices: {closing_prices}")
        print(f"Total Portfolio Value: {total_portfolio_value}")
        print(self.portfolio)
        

# Tickers and date range
tickers = ["TQQQ", "SQQQ"]
start_date = "2020-01-01"
end_date = "2021-01-01"

bot = BacktestTradingBot(tickers, start_date, end_date)
bot.execute_trades()
bot.logger.output_log(print_log=True, save_to_file='trading_bot_log.txt')
bot.run()