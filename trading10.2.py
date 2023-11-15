import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from enum import Enum
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
from websocket import create_connection, WebSocketApp
import json
import threading
from datetime import datetime, timedelta


class BotLogger:
    def __init__(self):
        self.log = []

    def record_trade(self, action, ticker, quantity, price, timestamp):
        event = "TRADE"
        msg = f"Action: {action}, Ticker: {ticker}, Quantity: {quantity}, Price: {price}, Timestamp: {timestamp}"
        self.log.append((event, msg))

    def record_error(self, error, timestamp):
        event = "ERROR"
        msg = f"Error: {error}, Timestamp: {timestamp}"
        self.log.append((event, msg))

    def record_decision(self, decision, ticker, score, timestamp):
        event = "DECISION"
        msg = f"Decision: {decision.name}, Ticker: {ticker}, Score: {score}, Timestamp: {timestamp}"
        self.log.append((event, msg))

    def record(self, event, message):
        timestamp = datetime.now().isoformat()
        msg = f"{message}, Timestamp: {timestamp}"
        self.log.append((event, msg))

    def output_log(self, print_log=True, save_to_file="trade_logs.txt"):
        output = '\n'.join([f"{event}: {msg}" for event, msg in self.log])
        if print_log:
            print(output)
        if save_to_file:
            with open(save_to_file, 'a') as f:
                f.write(output+"\n")
    

class Decision(Enum):
        BUY = 1
        SELL = 2
        HOLD = 3

class TradingBot:
    def __init__(self, tickers, alpaca_api_key, alpaca_api_secret, base_url='https://paper-api.alpaca.markets'):
        self.api = tradeapi.REST(alpaca_api_key, alpaca_api_secret, base_url, api_version='v2')
        print(self.api)
        self.api_key = 'PKJACFWB5U4H52OAS0SY'
        self.api_secret = 'cYQf2wmUAImmF39niytdj0Wfvr3vkBhXmbrwImO4'
        self.stock_data = {}  # Initialize the stock_data attribute as an empty dictionary
        self.tickers = tickers  # Initialize the tickers list
        self.portfolio = {'cash': 100000}
        self.initial_balance = 10000
        self.stop_loss_percentage = 0.03
        self.trailing_stop_loss_percentage = 0.08
        self.position_size_percentage = 0.02
        self.max_loss_percentage = 0.10
        self.trade_limit = 10000  # Limit trades per week
        self.trade_count = 0  # Weekly trade counter
        self.cooldown_period = 0  # Days to wait after a trade before trading again
        self.last_trade_day = {}  # Dictionary to store the last trade day for each ticker
        self.avg_volume = {}  # Add an attribute to store average volume for each ticker
        self.atr_multiplier_for_buy = 1.5  # This means buy if the ATR is less than 1.5 times the ATR value
        self.atr_multiplier_for_sell = 2 
        self.logger = BotLogger()

    def _setup_websocket(self):
        ws_url = 'wss://data.alpaca.markets/stream'

        def on_open(ws):
            try:
                self.logger.record("WS_OPEN", "WebSocket opened.")
                auth_data = {
                    "action": "authenticate",
                    "data": {"key_id": self.api_key, "secret_key": self.api_secret}
            }
                ws.send(json.dumps(auth_data))
                listen_message = {
                    "action": "listen",
                    "data": {"streams": [f"AM.{ticker}" for ticker in self.tickers]}
                }
                ws.send(json.dumps(listen_message))
            except Exception as e:
                self.logger.record_error(f"Error on opening WebSocket: {e}")
                self.logger.output_log(save_to_file='trade_log.txt')

        def on_message(ws, message):
            try:
                self._handle_realtime_data(json.loads(message))
            except Exception as e:
                self.logger.record_error(f"Error handling message: {e}")
                self.logger.output_log(save_to_file='trade_log.txt')

        def on_close(ws,close_status_code, close_msg):
            self.logger.record("WS_CLOSE", "WebSocket closed.")
            self.logger.output_log(save_to_file='trade_log.txt')

        def on_error(ws, error):
            self.logger.record("WS_ERROR", str(error))
            self.logger.output_log(save_to_file='trade_log.txt')

        # Here we use WebSocketApp which provides the functionality we need
        self.ws = WebSocketApp(ws_url,
                               on_open=on_open,
                               on_message=on_message,
                               on_close=on_close,
                               on_error=on_error)

    def _handle_realtime_data(self, data):
        # Process the incoming WebSocket messages
        try:
            if 'stream' in data:
                if 'data' in data:
                # Extract the relevant data
                    symbol = data['data']['S']
                    close = data['data']['c']
                    high = data['data']['h']
                    low = data['data']['l']
                    volume = data['data']['v']
                # Store the real-time data in a suitable format
                    self.stock_data[symbol] = {
                        'Close': close,
                        'High': high,
                        'Low': low,
                        'Volume': volume,
                        'OBV': self._compute_obv(pd.Series([close]), pd.Series([volume]))
                }
        except Exception as e:
            self.logger.record_error(f"Error handling real-time data: {e}")
            self.logger.output_log(save_to_file='trade_log.txt')

    def start(self):
        # Step 5: Start the WebSocket
        self._setup_websocket()
        # Run the websocket in a separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.start()
        # self.logger.output_log(save_to_file='trade_log.txt')


    def _compute_macd(self, prices, short_window=8, long_window=17, signal_window=9):
        try:
            short_ema = prices.ewm(span=short_window).mean()
            long_ema = prices.ewm(span=long_window).mean()
            macd = short_ema - long_ema
            macd_signal = macd.rolling(window=signal_window).mean()
            return macd, macd_signal
        except Exception as e:
            self.logger.record_error(f"Error computing MACD: {e}")
            self.logger.output_log(save_to_file='trade_log.txt')
            return None, None  # Or handle the error accordingly

    def _compute_bollinger_bands(self, prices, window=10):
        try:
            sma = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            upper_band = sma + (rolling_std * 1.5)
            lower_band = sma - (rolling_std * 1.5)
            return upper_band, lower_band
        except Exception as e:
            self.logger.record_error(f"Error computing BB: {e}")
            self.logger.output_log(save_to_file='trade_log.txt')
        

    def _compute_rsi(self, prices, window=8):
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            self.logger.record_error(f"Error computing RSI: {e}")
            self.logger.output_log(save_to_file='trade_log.txt')
            return None, None 
        
    def _compute_stochastic_oscillator(self, prices, window=6):
        try:
            low_min = prices.rolling(window=window).min()
            high_max = prices.rolling(window=window).max()
            k = 100 * ((prices - low_min) / (high_max - low_min))
            return k
        except Exception as e:
            self.logger.record_error(f"Error computing SS: {e}")
            self.logger.output_log(save_to_file='trade_log.txt')
            return None, None  
        
    def _compute_obv(self, prices, volumes):
        try:
            obv = volumes.copy()
            obv[prices < prices.shift(1)] *= -1
            return obv.cumsum()
        except Exception as e:
            self.logger.record_error(f"Error computing MACD: {e}")
            self.logger.output_log(save_to_file='trade_log.txt')
            return None, None  

    def _compute_support_resistance(self, prices):
        try:
        # Determine local minima for support, and local maxima for resistance
            local_min = argrelextrema(prices.values, np.less_equal, order=5)[0]
            local_max = argrelextrema(prices.values, np.greater_equal, order=5)[0]
            support_levels = prices.iloc[local_min]
            resistance_levels = prices.iloc[local_max]
            return support_levels, resistance_levels
        except Exception as e:
            self.logger.record_error(f"Error computing SR: {e}")
            self.logger.output_log(save_to_file='trade_log.txt')
            return None, None  
    
    def _compute_atr(self, high, low, close, window=14):
        try:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(window=window).mean()
            return atr
        except Exception as e:
            self.logger.record_error(f"Error computing ATR: {e}")
            self.logger.output_log(save_to_file='trade_log.txt')
            return None, None  
    
    def _fetch_historical_volume(self, ticker, days=30):
        try:
        
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

        # Convert datetime objects to strings in the format 'YYYY-MM-DD'
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

        # Fetch historical data from Alpaca
            historical_data = self.api.get_bars(
                ticker, 
                tradeapi.TimeFrame.Day, 
                start=start_date_str, 
                end=end_date_str
            ).df
        # Calculate and return the average volume
            avg_volume = historical_data['volume'].mean() if not historical_data.empty else 0
            self.avg_volume[ticker] = avg_volume  # Store the average volume
            return avg_volume
        except Exception as e:
            self.logger.record_error(f"Error fetching historical volume for {ticker}: {e}")
            self.logger.output_log(save_to_file='trade_log.txt')
            return 0  # Or handle the error accordingly
    
    def _set_atr_thresholds(self, atr):
        """Sets the buy and sell thresholds based on the ATR value."""
        self.atr_buy_threshold = atr * self.atr_multiplier_for_buy
        self.atr_sell_threshold = atr * self.atr_multiplier_for_sell

    def _analyze_stock(self, ticker):
    # Fetch historical data for the indicators
        historical_data = self._fetch_historical_data(ticker)
        high_prices = historical_data['High']
        low_prices = historical_data['Low']
        close_prices = historical_data['Close']
        volumes = historical_data['Volume']

    # Compute indicators with historical data
        macd, macd_signal = self._compute_macd(close_prices)
        upper_band, lower_band = self._compute_bollinger_bands(close_prices)
        rsi = self._compute_rsi(close_prices)
        k = self._compute_stochastic_oscillator(close_prices)
        obv = self._compute_obv(close_prices, volumes)
        atr = self._compute_atr(high_prices, low_prices, close_prices)  # Compute ATR here

    # Fetch real-time data
        real_time_data = self._fetch_realtime_data(ticker)
        current_high = real_time_data['High']
        current_low = real_time_data['Low']
        current_close = real_time_data['Close']
        current_volume = real_time_data['Volume']

    # Recompute the indicators now including the real-time data
        atr = self._compute_atr(high_prices.append(pd.Series(current_high)), 
                            low_prices.append(pd.Series(current_low)), 
                            close_prices.append(pd.Series(current_close)))
        
        atr = self._compute_atr(high_prices, low_prices, close_prices)
        self._set_atr_thresholds(atr.iloc[-1])

    # Determine signals with ATR adjustment
        buy_signals = {
            'price_above_lower_band': current_close > lower_band.iloc[-1],
            'macd_above_signal': macd.iloc[-1] > macd_signal.iloc[-1],
            'rsi_below_30': rsi.iloc[-1] < 30,
            'k_below_20': k.iloc[-1] < 20,
            'volume_increase': current_volume > 1.5 * volumes.mean(),
            'atr_above_threshold': atr.iloc[-1] > self.atr_buy_threshold  # Example ATR threshold for buy
        }

        sell_signals = {
            'price_below_upper_band': current_close < upper_band.iloc[-1],
            'macd_below_signal': macd.iloc[-1] < macd_signal.iloc[-1],
            'rsi_above_70': rsi.iloc[-1] > 70,
            'k_above_80': k.iloc[-1] > 80,
            'volume_decrease': current_volume < 0.5 * volumes.mean(),
            'atr_above_threshold': atr.iloc[-1] > self.atr_sell_threshold  # Example ATR threshold for sell
        }

    # Compute the score for buy and sell signals
        buy_score = sum(buy_signals.values())
        sell_score = sum(sell_signals.values())

    # Determine the decision
        if buy_score >= 4:  # Increase threshold if necessary
            self.logger.record(f"{ticker}: BUY SIGNAL - {buy_score}")
            return Decision.BUY
        elif sell_score >= 4:  # Increase threshold if necessary
            self.logger.record(f"{ticker}: SELL SIGNAL - {sell_score}")
            return Decision.SELL
        else:
            self.logger.record(f"{ticker}: HOLD SIGNAL - No clear signal")
            return Decision.HOLD
    
    def reset_trade_count(self, date):
        if date.weekday() == 0:  
            self.trade_count = 0
            
    def _can_trade(self, ticker):
    # Alpaca's API doesn't require date index to check for trade capability in this context
        last_trade_day = self.last_trade_day.get(ticker)
        if last_trade_day is not None:
            cooldown_period_seconds = self.cooldown_period * 60
            cooldown_time = datetime.now() - last_trade_day
            if cooldown_time.total_seconds() < self.cooldown_period:
                if self.last_trade_day.get(f"{ticker}_stop_loss_triggered", False):
                    return True 
                return False
        return True

    def _check_trailing_stop_loss(self, ticker, date_index):
        current_price = self.historical_data[ticker]['Close'].iloc[date_index]
    
   
        highest_price_key = f"{ticker}_highest_price"
        if highest_price_key not in self.portfolio:
            self.portfolio[highest_price_key] = current_price
        else:
            self.portfolio[highest_price_key] = max(self.portfolio[highest_price_key], current_price)
        
        trailing_key = f"{ticker}_trailing_stop"
        if trailing_key in self.portfolio:
        
            if current_price < self.portfolio[trailing_key]:
                self._place_order(ticker, "sell", self.portfolio.get(ticker, 0), date_index)
                self.last_trade_day[f"{ticker}_stop_loss_triggered"] = date_index 
            else:
                updated_trailing_stop = self.portfolio[highest_price_key] - (self.trailing_stop_loss_percentage * self.portfolio[highest_price_key])
                self.portfolio[trailing_key] = max(self.portfolio[trailing_key], updated_trailing_stop)

    
    def _calculate_position_size(self, price):
        balance = self.portfolio['cash']
        position_size = balance * self.position_size_percentage
        quantity = position_size // price 
        return int(quantity)  


    def execute_trades(self):
        try:
            # Inverse pairs mapping
            inverse_pairs = {'TQQQ': 'SQQQ', 'SQQQ': 'TQQQ'}

            # Loop through the tickers
            for ticker in self.tickers:
                # Check if trading is allowed
                if not self._can_trade(ticker):
                    self.logger.record(f"Cannot trade {ticker} right now due to cooldown.")
                    continue

                # Analyze the stock and get a decision
                decision = self._analyze_stock(ticker)
                current_price = self._fetch_realtime_data(ticker)["Close"]
                timestamp = datetime.now()

                # Execute the trade decision
                if decision in (Decision.BUY, Decision.SELL):
                    inverse_ticker = inverse_pairs.get(ticker)
                    inverse_decision = Decision.SELL if decision == Decision.BUY else Decision.BUY
                    inverse_price = self._fetch_realtime_data(inverse_ticker)["Close"]

                    # Calculate the position sizes
                    quantity = self._calculate_position_size(current_price)
                    inverse_quantity = self._calculate_position_size(inverse_price)

                    # Place orders for both the ticker and its inverse
                    if quantity > 0:
                        self._place_order(ticker, decision.name.lower(), quantity, current_price, timestamp)

                    if inverse_quantity > 0:
                        self._place_order(inverse_ticker, inverse_decision.name.lower(), inverse_quantity, inverse_price, timestamp)

            # Log that execution is complete
            self.logger.record("EXECUTE_TRADES_COMPLETE", "Finished executing trades.")

        except Exception as e:
            # Log any exceptions
            self.logger.record_error(f"An error occurred during trade execution: {str(e)}")
            self.logger.output_log(save_to_file='trade_log.txt')

    def _place_order(self, ticker, action, quantity, price, timestamp):
        try:
            if action == "buy":
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
            elif action == "sell":
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
            if order.status == 'filled' or order.status == 'new':
            # Log the trade only if the order was successful
                self.logger.record_trade(action, ticker, quantity, price, timestamp)
            else:
            # Log the order submission attempt with its returned status if not successful
                self.logger.record_error(f"Order for {ticker} was not successful. Status: {order.status}", timestamp)           
        except Exception as e:
            self.logger.record_error(str(e), timestamp)
            
            self.logger.output_log(save_to_file='trade_log.txt')
            
        

if __name__ == "__main__":
    # Initialize your TradingBot with relevant parameters
    
    trading_bot = TradingBot(tickers=["TQQQ", "SQQQ"], alpaca_api_key="PKJACFWB5U4H52OAS0SY", alpaca_api_secret="cYQf2wmUAImmF39niytdj0Wfvr3vkBhXmbrwImO4")
    trading_bot.start()  # This starts the WebSocket and your bot
    
   