"""
Cache Manager
High-performance in-memory storage for real-time trading data
Simplified for Yahoo Finance only data with optimized RSI calculation support
"""

import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import logging

class CircularBuffer:
    """Memory-efficient circular buffer for time series data"""
    
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.full(size, np.nan, dtype=np.float64)
        self.index = 0
        self.count = 0
        self.lock = threading.Lock()
    
    def add(self, value: float):
        """Add a value to the buffer"""
        with self.lock:
            self.buffer[self.index] = float(value)
            self.index = (self.index + 1) % self.size
            self.count = min(self.count + 1, self.size)
    
    def get_array(self) -> np.ndarray:
        """Get the buffer as a properly ordered numpy array"""
        with self.lock:
            if self.count == 0:
                return np.array([])
            elif self.count < self.size:
                # Buffer not full yet
                return self.buffer[:self.count].copy()
            else:
                # Buffer is full, need to reorder
                return np.concatenate([
                    self.buffer[self.index:],
                    self.buffer[:self.index]
                ])
    
    def get_latest(self, n: int = 1) -> np.ndarray:
        """Get the n most recent values"""
        with self.lock:
            if n >= self.count:
                return self.get_array()
            
            if self.count < self.size:
                start_idx = max(0, self.count - n)
                return self.buffer[start_idx:self.count].copy()
            else:
                # Buffer is full
                start_idx = (self.index - n) % self.size
                if start_idx < self.index:
                    return self.buffer[start_idx:self.index].copy()
                else:
                    return np.concatenate([
                        self.buffer[start_idx:],
                        self.buffer[:self.index]
                    ])
    
    def get_last_value(self) -> Optional[float]:
        """Get the most recent value"""
        with self.lock:
            if self.count == 0:
                return None
            
            if self.count < self.size:
                return float(self.buffer[self.count - 1])
            else:
                last_idx = (self.index - 1) % self.size
                return float(self.buffer[last_idx])
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.count >= self.size
    
    def has_min_data(self, min_count: int) -> bool:
        """Check if buffer has minimum required data points"""
        return self.count >= min_count
    
    def __len__(self) -> int:
        return self.count

class StockDataCache:
    """Optimized cache for individual stock data with RSI components"""
    
    def __init__(self, symbol: str, max_prices: int = 200, rsi_period: int = 14):
        self.symbol = symbol
        self.rsi_period = rsi_period
        
        # Price and volume data
        self.prices = CircularBuffer(max_prices)
        self.volumes = CircularBuffer(max_prices // 4)  # Store less volume history
        self.timestamps = deque(maxlen=max_prices)
        
        # RSI calculation components
        self.gains = CircularBuffer(rsi_period * 2)  # Extra buffer for stability
        self.losses = CircularBuffer(rsi_period * 2)
        
        # Current RSI state
        self.current_rsi = None
        self.previous_rsi = None
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        self.last_price = None
        
        # Moving averages (for additional analysis)
        self.ema_12 = None
        self.ema_26 = None
        self.sma_20 = None
        self.sma_50 = None
        
        # Market state tracking
        self.current_market_state = 'UNKNOWN'
        self.last_update = None
        
        # Statistics
        self.update_count = 0
        self.rsi_calculation_count = 0
        
        self.lock = threading.RLock()
    
    def add_price_data(self, price: float, volume: int, timestamp: datetime, 
                      market_state: str = 'UNKNOWN'):
        """Add new price and volume data with RSI calculation"""
        with self.lock:
            # Store previous RSI
            self.previous_rsi = self.current_rsi
            
            # Add to buffers
            self.prices.add(price)
            self.volumes.add(volume)
            self.timestamps.append(timestamp)
            self.current_market_state = market_state
            self.last_update = timestamp
            self.update_count += 1
            
            # Calculate price change for RSI
            if self.last_price is not None:
                change = price - self.last_price
                
                if change > 0:
                    self.gains.add(change)
                    self.losses.add(0.0)
                else:
                    self.gains.add(0.0)
                    self.losses.add(abs(change))
                
                # Update RSI if we have enough data
                self._update_rsi()
            
            self.last_price = price
            
            # Update moving averages
            self._update_moving_averages(price)
    
    def _update_rsi(self):
        """Update RSI calculation using Wilder's smoothing method"""
        if not self.gains.has_min_data(self.rsi_period):
            return
        
        gains_array = self.gains.get_array()
        losses_array = self.losses.get_array()
        
        # Use Wilder's smoothing method
        if self.avg_gain == 0.0 and self.avg_loss == 0.0:
            # First calculation - use simple average of last rsi_period values
            recent_gains = gains_array[-self.rsi_period:]
            recent_losses = losses_array[-self.rsi_period:]
            self.avg_gain = np.mean(recent_gains)
            self.avg_loss = np.mean(recent_losses)
        else:
            # Incremental update using Wilder's smoothing
            if len(gains_array) > 0 and len(losses_array) > 0:
                latest_gain = gains_array[-1]
                latest_loss = losses_array[-1]
                
                # Wilder's smoothing: new_avg = ((n-1) * old_avg + new_value) / n
                alpha = 1.0 / self.rsi_period
                self.avg_gain = (1 - alpha) * self.avg_gain + alpha * latest_gain
                self.avg_loss = (1 - alpha) * self.avg_loss + alpha * latest_loss
        
        # Calculate RSI
        if self.avg_loss == 0:
            self.current_rsi = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            self.current_rsi = 100 - (100 / (1 + rs))
        
        self.rsi_calculation_count += 1
    
    def _update_moving_averages(self, current_price: float):
        """Update exponential and simple moving averages"""
        # EMA calculations
        if self.ema_12 is None:
            self.ema_12 = current_price
        else:
            multiplier_12 = 2 / (12 + 1)
            self.ema_12 = (current_price * multiplier_12) + (self.ema_12 * (1 - multiplier_12))
        
        if self.ema_26 is None:
            self.ema_26 = current_price
        else:
            multiplier_26 = 2 / (26 + 1)
            self.ema_26 = (current_price * multiplier_26) + (self.ema_26 * (1 - multiplier_26))
        
        # SMA calculations (only if we have enough data)
        prices_array = self.prices.get_array()
        
        if len(prices_array) >= 20:
            self.sma_20 = np.mean(prices_array[-20:])
        
        if len(prices_array) >= 50:
            self.sma_50 = np.mean(prices_array[-50:])
    
    def get_rsi(self) -> Optional[float]:
        """Get current RSI value"""
        return self.current_rsi
    
    def get_rsi_change(self) -> Optional[float]:
        """Get RSI change from previous calculation"""
        if self.current_rsi is not None and self.previous_rsi is not None:
            return self.current_rsi - self.previous_rsi
        return None
    
    def has_rsi_data(self) -> bool:
        """Check if RSI can be calculated"""
        return self.current_rsi is not None
    
    def get_latest_price(self) -> Optional[float]:
        """Get the most recent price"""
        return self.prices.get_last_value()
    
    def get_latest_volume(self) -> Optional[int]:
        """Get the most recent volume"""
        volume = self.volumes.get_last_value()
        return int(volume) if volume is not None else None
    
    def get_price_history(self, periods: int = None) -> np.ndarray:
        """Get price history"""
        if periods is None:
            return self.prices.get_array()
        return self.prices.get_latest(periods)
    
    def get_volume_history(self, periods: int = None) -> np.ndarray:
        """Get volume history"""
        if periods is None:
            return self.volumes.get_array()
        return self.volumes.get_latest(periods)
    
    def get_price_change(self) -> Optional[float]:
        """Get latest price change"""
        if self.last_price is not None:
            prices = self.prices.get_latest(2)
            if len(prices) >= 2:
                return prices[-1] - prices[-2]
        return None
    
    def get_price_change_percent(self) -> Optional[float]:
        """Get latest price change as percentage"""
        change = self.get_price_change()
        if change is not None:
            prices = self.prices.get_latest(2)
            if len(prices) >= 2 and prices[-2] != 0:
                return (change / prices[-2]) * 100
        return None
    
    def get_indicators(self) -> Dict:
        """Get all current indicator values"""
        return {
            'symbol': self.symbol,
            'price': self.get_latest_price(),
            'volume': self.get_latest_volume(),
            'price_change': self.get_price_change(),
            'price_change_percent': self.get_price_change_percent(),
            'rsi': self.current_rsi,
            'rsi_change': self.get_rsi_change(),
            'ema_12': self.ema_12,
            'ema_26': self.ema_26,
            'macd': (self.ema_12 - self.ema_26) if (self.ema_12 and self.ema_26) else None,
            'sma_20': self.sma_20,
            'sma_50': self.sma_50,
            'market_state': self.current_market_state,
            'last_update': self.last_update,
            'data_points': len(self.prices),
            'has_rsi': self.has_rsi_data()
        }
    
    def get_statistics(self) -> Dict:
        """Get cache statistics for this symbol"""
        return {
            'symbol': self.symbol,
            'update_count': self.update_count,
            'rsi_calculations': self.rsi_calculation_count,
            'price_data_points': len(self.prices),
            'volume_data_points': len(self.volumes),
            'rsi_data_points': len(self.gains),
            'has_rsi': self.has_rsi_data(),
            'buffer_usage_percent': (len(self.prices) / self.prices.size) * 100,
            'last_update': self.last_update,
            'market_state': self.current_market_state
        }

class CacheManager:
    """Main cache manager optimized for Yahoo Finance data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.symbols = config.get('symbols', [])
        self.rsi_period = config.get('rsi_period', 14)
        self.max_prices = config.get('max_prices', 200)
        
        # Initialize caches for all symbols
        self.stock_caches = {}
        for symbol in self.symbols:
            self.stock_caches[symbol] = StockDataCache(
                symbol=symbol,
                max_prices=self.max_prices,
                rsi_period=self.rsi_period
            )
        
        # System metrics
        self.update_count = 0
        self.start_time = datetime.now()
        self.last_update_time = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Cache manager initialized for {len(self.symbols)} symbols")
        self.logger.info(f"RSI period: {self.rsi_period}, Max prices: {self.max_prices}")
    
    async def update_price_data(self, symbol: str, data: Dict):
        """Update price data for a symbol (callback from market data feed)"""
        if symbol not in self.stock_caches:
            self.logger.info(f"Adding new symbol to cache: {symbol}")
            self.stock_caches[symbol] = StockDataCache(
                symbol=symbol,
                max_prices=self.max_prices,
                rsi_period=self.rsi_period
            )
        
        cache = self.stock_caches[symbol]
        
        try:
            cache.add_price_data(
                price=data['price'],
                volume=data['volume'],
                timestamp=data['timestamp'],
                market_state=data.get('market_state', 'UNKNOWN')
            )
            
            self.update_count += 1
            self.last_update_time = datetime.now()
            
            # Log RSI when first calculated
            if cache.has_rsi_data() and cache.rsi_calculation_count == 1:
                self.logger.info(f"RSI calculation started for {symbol}: {cache.get_rsi():.2f}")
            
            # Periodic logging
            if self.update_count % 50 == 0:
                ready_count = len(self.get_ready_symbols())
                self.logger.info(f"Processed {self.update_count} updates, {ready_count} symbols ready for RSI")
        
        except Exception as e:
            self.logger.error(f"Error updating cache for {symbol}: {e}")
    
    def get_stock_cache(self, symbol: str) -> Optional[StockDataCache]:
        """Get cache for a specific symbol"""
        return self.stock_caches.get(symbol)
    
    def get_all_rsi_values(self) -> Dict[str, float]:
        """Get current RSI values for all symbols that have RSI data"""
        rsi_values = {}
        for symbol, cache in self.stock_caches.items():
            rsi = cache.get_rsi()
            if rsi is not None:
                rsi_values[symbol] = rsi
        return rsi_values
    
    def get_all_indicators(self) -> Dict[str, Dict]:
        """Get all indicators for all symbols"""
        indicators = {}
        for symbol, cache in self.stock_caches.items():
            indicators[symbol] = cache.get_indicators()
        return indicators
    
    def get_ready_symbols(self) -> List[str]:
        """Get symbols that have enough data for RSI calculation"""
        ready = []
        for symbol, cache in self.stock_caches.items():
            if cache.has_rsi_data():
                ready.append(symbol)
        return ready
    
    def get_symbols_with_data(self) -> List[str]:
        """Get symbols that have any price data"""
        symbols_with_data = []
        for symbol, cache in self.stock_caches.items():
            if len(cache.prices) > 0:
                symbols_with_data.append(symbol)
        return symbols_with_data
    
    def get_market_summary(self) -> Dict:
        """Get summary of current market state"""
        market_states = {}
        open_symbols = []
        closed_symbols = []
        
        for symbol, cache in self.stock_caches.items():
            state = cache.current_market_state
            market_states[symbol] = state
            
            if state == 'REGULAR':
                open_symbols.append(symbol)
            else:
                closed_symbols.append(symbol)
        
        return {
            'total_symbols': len(self.stock_caches),
            'symbols_open': len(open_symbols),
            'symbols_closed': len(closed_symbols),
            'open_symbols': open_symbols,
            'closed_symbols': closed_symbols,
            'market_states': market_states
        }
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive cache system statistics"""
        uptime = datetime.now() - self.start_time
        ready_count = len(self.get_ready_symbols())
        data_count = len(self.get_symbols_with_data())
        
        # Memory usage estimate
        memory_per_cache = self.max_prices * 8 * 3  # 8 bytes per float, 3 buffers
        estimated_memory_mb = (len(self.stock_caches) * memory_per_cache) / (1024 * 1024)
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'total_updates': self.update_count,
            'symbols_configured': len(self.symbols),
            'symbols_tracked': len(self.stock_caches),
            'symbols_with_data': data_count,
            'symbols_ready_for_rsi': ready_count,
            'updates_per_minute': (self.update_count / uptime.total_seconds()) * 60 if uptime.total_seconds() > 0 else 0,
            'last_update': self.last_update_time,
            'estimated_memory_mb': estimated_memory_mb,
            'rsi_period': self.rsi_period,
            'max_prices_per_symbol': self.max_prices
        }
    
    def get_detailed_stats(self) -> Dict[str, Dict]:
        """Get detailed statistics for each symbol"""
        detailed_stats = {}
        for symbol, cache in self.stock_caches.items():
            detailed_stats[symbol] = cache.get_statistics()
        return detailed_stats
    
    def clear_cache(self, symbol: str = None):
        """Clear cache for specific symbol or all symbols"""
        if symbol:
            if symbol in self.stock_caches:
                del self.stock_caches[symbol]
                self.logger.info(f"Cleared cache for {symbol}")
        else:
            self.stock_caches.clear()
            self.update_count = 0
            self.start_time = datetime.now()
            self.last_update_time = None
            self.logger.info("Cleared all caches")
    
    def optimize_memory(self):
        """Optimize memory usage by cleaning old data"""
        for cache in self.stock_caches.values():
            # This could be extended to implement more sophisticated cleanup
            # For now, the circular buffers handle this automatically
            pass

# Example usage and testing
def test_cache_manager():
    """Test the simplified cache manager functionality"""
    import asyncio
    import random
    
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT'],
        'rsi_period': 14,
        'max_prices': 100
    }
    
    cache_manager = CacheManager(config)
    
    async def simulate_yahoo_data():
        base_prices = {'AAPL': 150.0, 'GOOGL': 2500.0, 'MSFT': 300.0}
        market_states = ['REGULAR', 'REGULAR', 'REGULAR', 'CLOSED']
        
        print("Simulating Yahoo Finance data updates...")
        
        for i in range(50):  # Simulate 50 updates
            for symbol in config['symbols']:
                # Generate realistic price movement
                base_price = base_prices[symbol]
                change_pct = random.uniform(-0.02, 0.02)  # ±2% change
                new_price = base_price * (1 + change_pct)
                base_prices[symbol] = new_price
                
                # Yahoo Finance format data
                data = {
                    'symbol': symbol,
                    'price': new_price,
                    'volume': random.randint(1000000, 10000000),
                    'timestamp': datetime.now(),
                    'market_state': random.choice(market_states),
                    'source': 'yahoo_finance'
                }
                
                await cache_manager.update_price_data(symbol, data)
            
            # Show progress every 10 updates
            if (i + 1) % 10 == 0:
                ready_symbols = cache_manager.get_ready_symbols()
                print(f"Update {i+1}: {len(ready_symbols)} symbols ready for RSI")
            
            await asyncio.sleep(0.05)  # Small delay
        
        # Print final results
        print("\n" + "="*60)
        print("CACHE MANAGER TEST RESULTS")
        print("="*60)
        
        # System stats
        stats = cache_manager.get_system_stats()
        print(f"System Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\nRSI Values:")
        rsi_values = cache_manager.get_all_rsi_values()
        for symbol, rsi in rsi_values.items():
            print(f"  {symbol}: {rsi:.2f}")
        
        print(f"\nMarket Summary:")
        market_summary = cache_manager.get_market_summary()
        print(f"  Open symbols: {market_summary['open_symbols']}")
        print(f"  Market states: {market_summary['symbols_open']}/{market_summary['total_symbols']} open")
    
    # Run the test
    asyncio.run(simulate_yahoo_data())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_cache_manager()