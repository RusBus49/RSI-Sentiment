"""
RSI Calculator
Optimized RSI calculation engine with support for multiple timeframes
and efficient batch processing for the Jetson Nano
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class RSISignal(Enum):
    """RSI signal types"""
    OVERSOLD_ENTRY = "oversold_entry"  # RSI crosses above 30
    OVERBOUGHT_ENTRY = "overbought_entry"  # RSI crosses below 70
    NEUTRAL_EXIT = "neutral_exit"  # RSI returns to neutral zone
    EXTREME_OVERSOLD = "extreme_oversold"  # RSI < 20
    EXTREME_OVERBOUGHT = "extreme_overbought"  # RSI > 80
    NO_SIGNAL = "no_signal"

@dataclass
class RSIState:
    """Current RSI state for a symbol"""
    symbol: str
    current_rsi: float
    previous_rsi: float
    period: int
    avg_gain: float
    avg_loss: float
    signal: RSISignal
    signal_strength: float  # 0-1 scale
    timestamp: datetime

class RSICalculator:
    """Optimized RSI calculator with multiple features"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.default_period = config.get('rsi_period', 14)
        self.oversold_threshold = config.get('oversold_threshold', 30)
        self.overbought_threshold = config.get('overbought_threshold', 70)
        self.extreme_oversold = config.get('extreme_oversold', 20)
        self.extreme_overbought = config.get('extreme_overbought', 80)
        
        # Signal detection settings
        self.signal_confirmation_periods = config.get('signal_confirmation_periods', 2)
        
        # Cache for RSI states
        self.rsi_states = {}
        self.previous_signals = {}
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RSI Calculator initialized with period={self.default_period}")
    
    def calculate_rsi_batch(self, price_data: Dict[str, np.ndarray], period: int = None) -> Dict[str, float]:
        """
        Calculate RSI for multiple symbols efficiently using vectorized operations
        
        Args:
            price_data: Dict of symbol -> price array
            period: RSI period (default from config)
        
        Returns:
            Dict of symbol -> RSI value
        """
        if period is None:
            period = self.default_period
        
        rsi_values = {}
        
        for symbol, prices in price_data.items():
            if len(prices) < period + 1:
                continue  # Not enough data
            
            rsi = self._calculate_rsi_array(prices, period)
            if rsi is not None:
                rsi_values[symbol] = rsi
        
        return rsi_values
    
    def _calculate_rsi_array(self, prices: np.ndarray, period: int) -> Optional[float]:
        """
        Calculate RSI from price array using Wilder's smoothing method
        Optimized for speed with NumPy operations
        """
        if len(prices) < period + 1:
            return None
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages (simple average for first period)
        if len(gains) < period:
            return None
        
        # Use Wilder's smoothing (exponential moving average with alpha = 1/period)
        alpha = 1.0 / period
        
        # Initialize with simple average
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Apply Wilder's smoothing to remaining data
        for i in range(period, len(gains)):
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        
        # Calculate RSI
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def update_rsi_incremental(self, symbol: str, new_price: float, previous_price: float) -> Optional[RSIState]:
        """
        Update RSI incrementally for a single symbol
        More efficient than recalculating from scratch
        """
        if symbol not in self.rsi_states:
            return None  # Need initial calculation first
        
        current_state = self.rsi_states[symbol]
        
        # Calculate price change
        change = new_price - previous_price
        gain = max(change, 0)
        loss = max(-change, 0)
        
        # Update running averages using Wilder's smoothing
        alpha = 1.0 / self.default_period
        new_avg_gain = alpha * gain + (1 - alpha) * current_state.avg_gain
        new_avg_loss = alpha * loss + (1 - alpha) * current_state.avg_loss
        
        # Calculate new RSI
        if new_avg_loss == 0:
            new_rsi = 100.0
        else:
            rs = new_avg_gain / new_avg_loss
            new_rsi = 100 - (100 / (1 + rs))
        
        # Detect signals
        signal, strength = self._detect_signal(
            current_rsi=new_rsi,
            previous_rsi=current_state.current_rsi,
            symbol=symbol
        )
        
        # Create new state
        new_state = RSIState(
            symbol=symbol,
            current_rsi=new_rsi,
            previous_rsi=current_state.current_rsi,
            period=self.default_period,
            avg_gain=new_avg_gain,
            avg_loss=new_avg_loss,
            signal=signal,
            signal_strength=strength,
            timestamp=datetime.now()
        )
        
        self.rsi_states[symbol] = new_state
        return new_state
    
    def initialize_rsi_state(self, symbol: str, price_history: np.ndarray) -> Optional[RSIState]:
        """
        Initialize RSI state for a symbol using historical price data
        """
        if len(price_history) < self.default_period + 1:
            return None
        
        rsi = self._calculate_rsi_array(price_history, self.default_period)
        if rsi is None:
            return None
        
        # Calculate current avg_gain and avg_loss for incremental updates
        deltas = np.diff(price_history[-self.default_period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Use Wilder's smoothing method
        alpha = 1.0 / self.default_period
        avg_gain = np.mean(gains[:self.default_period])
        avg_loss = np.mean(losses[:self.default_period])
        
        # Apply smoothing to get final averages
        for i in range(self.default_period, len(gains)):
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        
        # Create initial state
        state = RSIState(
            symbol=symbol,
            current_rsi=rsi,
            previous_rsi=rsi,  # No previous value initially
            period=self.default_period,
            avg_gain=avg_gain,
            avg_loss=avg_loss,
            signal=RSISignal.NO_SIGNAL,
            signal_strength=0.0,
            timestamp=datetime.now()
        )
        
        self.rsi_states[symbol] = state
        return state
    
    def _detect_signal(self, current_rsi: float, previous_rsi: float, symbol: str) -> Tuple[RSISignal, float]:
        """
        Detect RSI signals and calculate signal strength
        
        Returns:
            Tuple of (signal_type, signal_strength)
        """
        # Check for threshold crossings
        if previous_rsi <= self.oversold_threshold and current_rsi > self.oversold_threshold:
            # Potential long signal - oversold recovery
            strength = min(1.0, (current_rsi - self.oversold_threshold) / 10)
            return RSISignal.OVERSOLD_ENTRY, strength
        
        elif previous_rsi >= self.overbought_threshold and current_rsi < self.overbought_threshold:
            # Potential short signal - overbought decline
            strength = min(1.0, (self.overbought_threshold - current_rsi) / 10)
            return RSISignal.OVERBOUGHT_ENTRY, strength
        
        # Check for extreme levels
        elif current_rsi < self.extreme_oversold:
            strength = min(1.0, (self.extreme_oversold - current_rsi) / 10)
            return RSISignal.EXTREME_OVERSOLD, strength
        
        elif current_rsi > self.extreme_overbought:
            strength = min(1.0, (current_rsi - self.extreme_overbought) / 10)
            return RSISignal.EXTREME_OVERBOUGHT, strength
        
        # Check for neutral zone entry (exit signals)
        elif (previous_rsi < self.oversold_threshold or previous_rsi > self.overbought_threshold) and \
             (self.oversold_threshold <= current_rsi <= self.overbought_threshold):
            strength = 0.5  # Moderate strength for exit signals
            return RSISignal.NEUTRAL_EXIT, strength
        
        return RSISignal.NO_SIGNAL, 0.0
    
    def get_rsi_state(self, symbol: str) -> Optional[RSIState]:
        """Get current RSI state for a symbol"""
        return self.rsi_states.get(symbol)
    
    def get_all_rsi_states(self) -> Dict[str, RSIState]:
        """Get RSI states for all symbols"""
        return self.rsi_states.copy()
    
    def get_signals(self, min_strength: float = 0.3) -> Dict[str, RSIState]:
        """
        Get all symbols with active signals above minimum strength
        
        Args:
            min_strength: Minimum signal strength (0-1)
        
        Returns:
            Dict of symbol -> RSIState for symbols with signals
        """
        signals = {}
        for symbol, state in self.rsi_states.items():
            if state.signal != RSISignal.NO_SIGNAL and state.signal_strength >= min_strength:
                signals[symbol] = state
        return signals
    
    def get_oversold_signals(self, min_strength: float = 0.3) -> Dict[str, RSIState]:
        """Get symbols with oversold signals (potential long opportunities)"""
        signals = {}
        for symbol, state in self.rsi_states.items():
            if state.signal in [RSISignal.OVERSOLD_ENTRY, RSISignal.EXTREME_OVERSOLD] and \
               state.signal_strength >= min_strength:
                signals[symbol] = state
        return signals
    
    def get_overbought_signals(self, min_strength: float = 0.3) -> Dict[str, RSIState]:
        """Get symbols with overbought signals (potential short opportunities)"""
        signals = {}
        for symbol, state in self.rsi_states.items():
            if state.signal in [RSISignal.OVERBOUGHT_ENTRY, RSISignal.EXTREME_OVERBOUGHT] and \
               state.signal_strength >= min_strength:
                signals[symbol] = state
        return signals
    
    def calculate_divergence(self, symbol: str, price_history: np.ndarray, rsi_history: np.ndarray, 
                           lookback_periods: int = 20) -> Optional[str]:
        """
        Detect RSI divergence patterns
        
        Args:
            symbol: Stock symbol
            price_history: Recent price data
            rsi_history: Recent RSI data
            lookback_periods: Number of periods to analyze
        
        Returns:
            'bullish_divergence', 'bearish_divergence', or None
        """
        if len(price_history) < lookback_periods or len(rsi_history) < lookback_periods:
            return None
        
        # Get recent data
        recent_prices = price_history[-lookback_periods:]
        recent_rsi = rsi_history[-lookback_periods:]
        
        # Find local peaks and troughs
        price_peaks = self._find_peaks(recent_prices)
        price_troughs = self._find_troughs(recent_prices)
        rsi_peaks = self._find_peaks(recent_rsi)
        rsi_troughs = self._find_troughs(recent_rsi)
        
        # Bullish divergence: price makes lower lows, RSI makes higher lows
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            latest_price_trough = recent_prices[price_troughs[-1]]
            prev_price_trough = recent_prices[price_troughs[-2]]
            latest_rsi_trough = recent_rsi[rsi_troughs[-1]]
            prev_rsi_trough = recent_rsi[rsi_troughs[-2]]
            
            if latest_price_trough < prev_price_trough and latest_rsi_trough > prev_rsi_trough:
                return 'bullish_divergence'
        
        # Bearish divergence: price makes higher highs, RSI makes lower highs
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            latest_price_peak = recent_prices[price_peaks[-1]]
            prev_price_peak = recent_prices[price_peaks[-2]]
            latest_rsi_peak = recent_rsi[rsi_peaks[-1]]
            prev_rsi_peak = recent_rsi[rsi_peaks[-2]]
            
            if latest_price_peak > prev_price_peak and latest_rsi_peak < prev_rsi_peak:
                return 'bearish_divergence'
        
        return None
    
    def _find_peaks(self, data: np.ndarray, min_distance: int = 3) -> List[int]:
        """Find local peaks in data"""
        peaks = []
        for i in range(min_distance, len(data) - min_distance):
            if all(data[i] >= data[i-j] for j in range(1, min_distance + 1)) and \
               all(data[i] >= data[i+j] for j in range(1, min_distance + 1)):
                peaks.append(i)
        return peaks
    
    def _find_troughs(self, data: np.ndarray, min_distance: int = 3) -> List[int]:
        """Find local troughs in data"""
        troughs = []
        for i in range(min_distance, len(data) - min_distance):
            if all(data[i] <= data[i-j] for j in range(1, min_distance + 1)) and \
               all(data[i] <= data[i+j] for j in range(1, min_distance + 1)):
                troughs.append(i)
        return troughs
    
    def get_rsi_momentum(self, symbol: str, periods: int = 5) -> Optional[float]:
        """
        Calculate RSI momentum (rate of change)
        
        Args:
            symbol: Stock symbol
            periods: Number of periods to look back
        
        Returns:
            RSI momentum value (positive = increasing, negative = decreasing)
        """
        if symbol not in self.rsi_states:
            return None
        
        # This would need historical RSI values stored
        # For now, return a simple momentum based on current vs previous
        state = self.rsi_states[symbol]
        return state.current_rsi - state.previous_rsi
    
    def reset_state(self, symbol: str = None):
        """Reset RSI state for a symbol or all symbols"""
        if symbol:
            if symbol in self.rsi_states:
                del self.rsi_states[symbol]
                self.logger.info(f"Reset RSI state for {symbol}")
        else:
            self.rsi_states.clear()
            self.previous_signals.clear()
            self.logger.info("Reset all RSI states")
    
    def get_statistics(self) -> Dict:
        """Get RSI calculator statistics"""
        if not self.rsi_states:
            return {'symbols_tracked': 0}
        
        rsi_values = [state.current_rsi for state in self.rsi_states.values()]
        signals = [state for state in self.rsi_states.values() if state.signal != RSISignal.NO_SIGNAL]
        
        return {
            'symbols_tracked': len(self.rsi_states),
            'avg_rsi': np.mean(rsi_values),
            'min_rsi': np.min(rsi_values),
            'max_rsi': np.max(rsi_values),
            'active_signals': len(signals),
            'oversold_count': len([s for s in signals if s.signal in [RSISignal.OVERSOLD_ENTRY, RSISignal.EXTREME_OVERSOLD]]),
            'overbought_count': len([s for s in signals if s.signal in [RSISignal.OVERBOUGHT_ENTRY, RSISignal.EXTREME_OVERBOUGHT]])
        }

# Integration helper function
def create_rsi_calculator_from_cache(cache_manager, config: Dict = None) -> RSICalculator:
    """
    Create and initialize RSI calculator from cache manager data
    
    Args:
        cache_manager: CacheManager instance
        config: RSI configuration dict
    
    Returns:
        Initialized RSICalculator
    """
    if config is None:
        config = {
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'extreme_oversold': 20,
            'extreme_overbought': 80
        }
    
    calculator = RSICalculator(config)
    
    # Initialize RSI states for all symbols with sufficient data
    ready_symbols = cache_manager.get_ready_symbols()
    
    for symbol in ready_symbols:
        stock_cache = cache_manager.get_stock_cache(symbol)
        if stock_cache:
            price_history = stock_cache.get_price_history()
            if len(price_history) >= config['rsi_period'] + 1:
                calculator.initialize_rsi_state(symbol, price_history)
    
    return calculator

# Example usage and testing
def test_rsi_calculator():
    """Test RSI calculator functionality"""
    import asyncio
    from cache_manager import CacheManager
    
    # Configuration
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT'],
        'rsi_period': 14,
        'oversold_threshold': 30,
        'overbought_threshold': 70
    }
    
    # Create cache manager and RSI calculator
    cache_manager = CacheManager(config)
    rsi_calc = RSICalculator(config)
    
    async def simulate_trading_day():
        import random
        base_prices = {'AAPL': 150, 'GOOGL': 2500, 'MSFT': 300}
        
        # Generate initial price history
        for symbol in config['symbols']:
            prices = []
            price = base_prices[symbol]
            
            # Generate 30 periods of historical data
            for i in range(30):
                change = random.uniform(-0.03, 0.03)  # ±3% daily change
                price *= (1 + change)
                prices.append(price)
                
                # Add to cache
                data = {
                    'price': price,
                    'volume': random.randint(1000, 10000),
                    'timestamp': datetime.now()
                }
                await cache_manager.update_price_data(symbol, data)
            
            # Initialize RSI state
            price_array = np.array(prices)
            rsi_calc.initialize_rsi_state(symbol, price_array)
            base_prices[symbol] = price
        
        print("Initial RSI Values:")
        for symbol, state in rsi_calc.get_all_rsi_states().items():
            print(f"{symbol}: RSI = {state.current_rsi:.2f}")
        
        # Simulate real-time updates
        print("\nSimulating real-time updates...")
        for update in range(20):
            for symbol in config['symbols']:
                old_price = base_prices[symbol]
                change = random.uniform(-0.02, 0.02)  # ±2% change
                new_price = old_price * (1 + change)
                base_prices[symbol] = new_price
                
                # Update cache
                data = {
                    'price': new_price,
                    'volume': random.randint(1000, 10000),
                    'timestamp': datetime.now()
                }
                await cache_manager.update_price_data(symbol, data)
                
                # Update RSI incrementally
                rsi_calc.update_rsi_incremental(symbol, new_price, old_price)
            
            # Check for signals every few updates
            if update % 5 == 0:
                signals = rsi_calc.get_signals(min_strength=0.3)
                if signals:
                    print(f"\nUpdate {update} - Active Signals:")
                    for symbol, state in signals.items():
                        print(f"{symbol}: {state.signal.value} (strength: {state.signal_strength:.2f}, RSI: {state.current_rsi:.2f})")
        
        # Final statistics
        print(f"\nFinal Statistics:")
        stats = rsi_calc.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print(f"\nFinal RSI Values:")
        for symbol, state in rsi_calc.get_all_rsi_states().items():
            print(f"{symbol}: RSI = {state.current_rsi:.2f} (Signal: {state.signal.value})")
    
    # Run the simulation
    asyncio.run(simulate_trading_day())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_rsi_calculator()
        