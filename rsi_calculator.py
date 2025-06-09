"""
RSI Calculator
Simplified RSI calculation engine optimized for Yahoo Finance data
Focused on long-short RSI strategy with clean signal detection
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class RSISignal(Enum):
    """RSI signal types for long-short strategy"""
    LONG_ENTRY = "long_entry"          # RSI crosses above oversold threshold
    SHORT_ENTRY = "short_entry"        # RSI crosses below overbought threshold
    LONG_EXIT = "long_exit"            # RSI crosses back into neutral from oversold
    SHORT_EXIT = "short_exit"          # RSI crosses back into neutral from overbought
    EXTREME_OVERSOLD = "extreme_oversold"  # RSI < 20 (strong buy signal)
    EXTREME_OVERBOUGHT = "extreme_overbought"  # RSI > 80 (strong sell signal)
    NEUTRAL = "neutral"                # RSI in neutral zone
    NO_SIGNAL = "no_signal"            # No actionable signal

@dataclass
class RSIState:
    """Current RSI state for a symbol"""
    symbol: str
    current_rsi: float
    previous_rsi: float
    signal: RSISignal
    signal_strength: float  # 0-1 scale, higher = stronger signal
    timestamp: datetime
    price: float
    price_change_percent: float
    volume: int
    market_state: str
    
    def __str__(self) -> str:
        return (f"{self.symbol}: RSI={self.current_rsi:.1f} "
                f"Signal={self.signal.value} Strength={self.signal_strength:.2f}")

class RSICalculator:
    """Simplified RSI calculator optimized for Yahoo Finance integration"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # RSI parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.oversold_threshold = config.get('oversold_threshold', 30)
        self.overbought_threshold = config.get('overbought_threshold', 70)
        self.extreme_oversold = config.get('extreme_oversold', 20)
        self.extreme_overbought = config.get('extreme_overbought', 80)
        
        # Signal detection parameters
        self.min_signal_strength = config.get('min_signal_strength', 0.3)
        self.volume_confirmation = config.get('volume_confirmation', True)
        self.min_volume_threshold = config.get('min_volume_threshold', 100000)
        
        # State tracking
        self.rsi_states = {}
        self.signal_history = {}  # Track recent signals to avoid duplicates
        self.signal_count = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RSI Calculator initialized:")
        self.logger.info(f"  Period: {self.rsi_period}")
        self.logger.info(f"  Thresholds: {self.oversold_threshold}/{self.overbought_threshold}")
        self.logger.info(f"  Extreme levels: {self.extreme_oversold}/{self.extreme_overbought}")
    
    def update_from_cache(self, cache_manager) -> Dict[str, RSIState]:
        """Update RSI states from cache manager data"""
        updated_states = {}
        
        # Get all symbols with RSI data
        ready_symbols = cache_manager.get_ready_symbols()
        
        for symbol in ready_symbols:
            cache = cache_manager.get_stock_cache(symbol)
            if cache and cache.has_rsi_data():
                
                # Get all relevant data from cache
                indicators = cache.get_indicators()
                
                # Create RSI state
                state = self._create_rsi_state(
                    symbol=symbol,
                    current_rsi=indicators['rsi'],
                    previous_rsi=cache.previous_rsi,
                    price=indicators['price'],
                    price_change_percent=indicators.get('price_change_percent', 0),
                    volume=indicators.get('volume', 0),
                    market_state=indicators.get('market_state', 'UNKNOWN'),
                    timestamp=indicators.get('last_update', datetime.now())
                )
                
                if state:
                    self.rsi_states[symbol] = state
                    updated_states[symbol] = state
                    
                    # Log significant signals
                    if state.signal not in [RSISignal.NEUTRAL, RSISignal.NO_SIGNAL]:
                        self.logger.info(f"Signal: {state}")
        
        return updated_states
    
    def _create_rsi_state(self, symbol: str, current_rsi: float, previous_rsi: Optional[float],
                         price: float, price_change_percent: float, volume: int,
                         market_state: str, timestamp: datetime) -> Optional[RSIState]:
        """Create RSI state with signal detection"""
        
        if current_rsi is None:
            return None
        
        # Detect signal
        signal, strength = self._detect_signal(
            current_rsi=current_rsi,
            previous_rsi=previous_rsi,
            symbol=symbol,
            volume=volume,
            price_change_percent=price_change_percent
        )
        
        return RSIState(
            symbol=symbol,
            current_rsi=current_rsi,
            previous_rsi=previous_rsi or current_rsi,
            signal=signal,
            signal_strength=strength,
            timestamp=timestamp,
            price=price,
            price_change_percent=price_change_percent,
            volume=volume,
            market_state=market_state
        )
    
    def _detect_signal(self, current_rsi: float, previous_rsi: Optional[float], 
                      symbol: str, volume: int, price_change_percent: float) -> Tuple[RSISignal, float]:
        """Detect RSI signals with strength calculation"""
        
        if previous_rsi is None:
            return RSISignal.NO_SIGNAL, 0.0
        
        # Check for threshold crossings
        signal = RSISignal.NO_SIGNAL
        base_strength = 0.0
        
        # Long entry signals (oversold recovery)
        if previous_rsi <= self.oversold_threshold and current_rsi > self.oversold_threshold:
            signal = RSISignal.LONG_ENTRY
            # Strength based on how far RSI moved above threshold
            base_strength = min(1.0, (current_rsi - self.oversold_threshold) / 10)
        
        # Short entry signals (overbought decline)
        elif previous_rsi >= self.overbought_threshold and current_rsi < self.overbought_threshold:
            signal = RSISignal.SHORT_ENTRY
            # Strength based on how far RSI moved below threshold
            base_strength = min(1.0, (self.overbought_threshold - current_rsi) / 10)
        
        # Exit signals (return to neutral zone)
        elif (previous_rsi < self.oversold_threshold or previous_rsi > self.overbought_threshold) and \
             (self.oversold_threshold < current_rsi < self.overbought_threshold):
            
            if previous_rsi < self.oversold_threshold:
                signal = RSISignal.LONG_EXIT
            else:
                signal = RSISignal.SHORT_EXIT
            base_strength = 0.5  # Moderate strength for exit signals
        
        # Extreme level signals
        elif current_rsi <= self.extreme_oversold:
            signal = RSISignal.EXTREME_OVERSOLD
            base_strength = min(1.0, (self.extreme_oversold - current_rsi) / 10 + 0.7)
        
        elif current_rsi >= self.extreme_overbought:
            signal = RSISignal.EXTREME_OVERBOUGHT
            base_strength = min(1.0, (current_rsi - self.extreme_overbought) / 10 + 0.7)
        
        # Neutral zone
        elif self.oversold_threshold < current_rsi < self.overbought_threshold:
            signal = RSISignal.NEUTRAL
            base_strength = 0.0
        
        # Calculate final strength with confirmations
        final_strength = self._calculate_signal_strength(
            base_strength=base_strength,
            volume=volume,
            price_change_percent=price_change_percent,
            signal=signal
        )
        
        # Check for signal duplication (avoid repeated alerts)
        if self._is_duplicate_signal(symbol, signal):
            return RSISignal.NO_SIGNAL, 0.0
        
        # Record signal if significant
        if final_strength >= self.min_signal_strength:
            self._record_signal(symbol, signal)
            self.signal_count += 1
        
        return signal, final_strength
    
    def _calculate_signal_strength(self, base_strength: float, volume: int, 
                                 price_change_percent: float, signal: RSISignal) -> float:
        """Calculate final signal strength with confirmation factors"""
        
        if base_strength == 0.0:
            return 0.0
        
        strength = base_strength
        
        # Volume confirmation
        if self.volume_confirmation and volume > 0:
            if volume >= self.min_volume_threshold:
                strength *= 1.2  # Boost for high volume
            else:
                strength *= 0.8  # Reduce for low volume
        
        # Price movement confirmation
        if abs(price_change_percent) > 1.0:  # Significant price movement
            if signal in [RSISignal.LONG_ENTRY, RSISignal.EXTREME_OVERSOLD] and price_change_percent > 0:
                strength *= 1.1  # Price moving up confirms long signal
            elif signal in [RSISignal.SHORT_ENTRY, RSISignal.EXTREME_OVERBOUGHT] and price_change_percent < 0:
                strength *= 1.1  # Price moving down confirms short signal
        
        # Cap at 1.0
        return min(1.0, strength)
    
    def _is_duplicate_signal(self, symbol: str, signal: RSISignal) -> bool:
        """Check if this signal was recently generated"""
        if symbol not in self.signal_history:
            return False
        
        recent_signals = self.signal_history[symbol]
        
        # Check if same signal was generated in last 3 entries
        if len(recent_signals) >= 3:
            return signal in recent_signals[-3:]
        
        return False
    
    def _record_signal(self, symbol: str, signal: RSISignal):
        """Record signal to prevent duplicates"""
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        
        self.signal_history[symbol].append(signal)
        
        # Keep only last 10 signals per symbol
        if len(self.signal_history[symbol]) > 10:
            self.signal_history[symbol] = self.signal_history[symbol][-10:]
    
    def get_rsi_state(self, symbol: str) -> Optional[RSIState]:
        """Get current RSI state for a symbol"""
        return self.rsi_states.get(symbol)
    
    def get_all_rsi_states(self) -> Dict[str, RSIState]:
        """Get RSI states for all symbols"""
        return self.rsi_states.copy()
    
    def get_active_signals(self, min_strength: float = None) -> Dict[str, RSIState]:
        """Get all symbols with active signals above minimum strength"""
        if min_strength is None:
            min_strength = self.min_signal_strength
        
        active_signals = {}
        for symbol, state in self.rsi_states.items():
            if (state.signal not in [RSISignal.NEUTRAL, RSISignal.NO_SIGNAL] and 
                state.signal_strength >= min_strength):
                active_signals[symbol] = state
        
        return active_signals
    
    def get_long_signals(self, min_strength: float = None) -> Dict[str, RSIState]:
        """Get symbols with long signals (buy opportunities)"""
        if min_strength is None:
            min_strength = self.min_signal_strength
        
        long_signals = {}
        for symbol, state in self.rsi_states.items():
            if (state.signal in [RSISignal.LONG_ENTRY, RSISignal.EXTREME_OVERSOLD] and 
                state.signal_strength >= min_strength):
                long_signals[symbol] = state
        
        return long_signals
    
    def get_short_signals(self, min_strength: float = None) -> Dict[str, RSIState]:
        """Get symbols with short signals (sell opportunities)"""
        if min_strength is None:
            min_strength = self.min_signal_strength
        
        short_signals = {}
        for symbol, state in self.rsi_states.items():
            if (state.signal in [RSISignal.SHORT_ENTRY, RSISignal.EXTREME_OVERBOUGHT] and 
                state.signal_strength >= min_strength):
                short_signals[symbol] = state
        
        return short_signals
    
    def get_exit_signals(self, min_strength: float = None) -> Dict[str, RSIState]:
        """Get symbols with exit signals"""
        if min_strength is None:
            min_strength = self.min_signal_strength
        
        exit_signals = {}
        for symbol, state in self.rsi_states.items():
            if (state.signal in [RSISignal.LONG_EXIT, RSISignal.SHORT_EXIT] and 
                state.signal_strength >= min_strength):
                exit_signals[symbol] = state
        
        return exit_signals
    
    def get_market_overview(self) -> Dict:
        """Get overview of current market signals"""
        long_signals = self.get_long_signals()
        short_signals = self.get_short_signals()
        exit_signals = self.get_exit_signals()
        
        # Calculate average RSI by market state
        open_rsi = []
        closed_rsi = []
        
        for state in self.rsi_states.values():
            if state.market_state == 'REGULAR':
                open_rsi.append(state.current_rsi)
            else:
                closed_rsi.append(state.current_rsi)
        
        return {
            'total_symbols': len(self.rsi_states),
            'long_opportunities': len(long_signals),
            'short_opportunities': len(short_signals),
            'exit_signals': len(exit_signals),
            'symbols_open': len(open_rsi),
            'symbols_closed': len(closed_rsi),
            'avg_rsi_open': np.mean(open_rsi) if open_rsi else 0,
            'avg_rsi_closed': np.mean(closed_rsi) if closed_rsi else 0,
            'total_signals_generated': self.signal_count,
            'oversold_count': len([s for s in self.rsi_states.values() if s.current_rsi <= self.oversold_threshold]),
            'overbought_count': len([s for s in self.rsi_states.values() if s.current_rsi >= self.overbought_threshold]),
            'neutral_count': len([s for s in self.rsi_states.values() if self.oversold_threshold < s.current_rsi < self.overbought_threshold])
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive RSI calculator statistics"""
        if not self.rsi_states:
            return {'symbols_tracked': 0, 'signals_generated': self.signal_count}
        
        rsi_values = [state.current_rsi for state in self.rsi_states.values()]
        active_signals = self.get_active_signals()
        
        return {
            'symbols_tracked': len(self.rsi_states),
            'signals_generated': self.signal_count,
            'active_signals': len(active_signals),
            'avg_rsi': np.mean(rsi_values),
            'min_rsi': np.min(rsi_values),
            'max_rsi': np.max(rsi_values),
            'rsi_std': np.std(rsi_values),
            'oversold_threshold': self.oversold_threshold,
            'overbought_threshold': self.overbought_threshold,
            'min_signal_strength': self.min_signal_strength,
            'volume_confirmation': self.volume_confirmation
        }
    
    def reset_state(self, symbol: str = None):
        """Reset RSI state for a symbol or all symbols"""
        if symbol:
            if symbol in self.rsi_states:
                del self.rsi_states[symbol]
            if symbol in self.signal_history:
                del self.signal_history[symbol]
            self.logger.info(f"Reset RSI state for {symbol}")
        else:
            self.rsi_states.clear()
            self.signal_history.clear()
            self.signal_count = 0
            self.logger.info("Reset all RSI states")

# Helper function for integration testing
async def test_rsi_integration(cache_manager, rsi_calculator, num_cycles: int = 10):
    """Test RSI calculator integration with cache manager"""
    import random
    
    print("Testing RSI Calculator Integration")
    print("=" * 50)
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    base_prices = {'AAPL': 150.0, 'GOOGL': 2500.0, 'MSFT': 300.0}
    
    for cycle in range(num_cycles):
        print(f"\nCycle {cycle + 1}/{num_cycles}:")
        
        # Simulate price updates
        for symbol in symbols:
            # Generate realistic price movement
            change_pct = random.uniform(-0.03, 0.03)  # ±3% change
            new_price = base_prices[symbol] * (1 + change_pct)
            base_prices[symbol] = new_price
            
            # Create Yahoo Finance style data
            data = {
                'symbol': symbol,
                'price': new_price,
                'volume': random.randint(1000000, 10000000),
                'timestamp': datetime.now(),
                'market_state': 'REGULAR',
                'source': 'yahoo_finance'
            }
            
            await cache_manager.update_price_data(symbol, data)
        
        # Update RSI calculations
        updated_states = rsi_calculator.update_from_cache(cache_manager)
        
        # Display results
        for symbol, state in updated_states.items():
            print(f"  {symbol}: RSI={state.current_rsi:.1f} Price=${state.price:.2f} Signal={state.signal.value}")
        
        # Check for signals
        active_signals = rsi_calculator.get_active_signals()
        if active_signals:
            print(f"  🚨 Active signals: {len(active_signals)}")
            for symbol, state in active_signals.items():
                print(f"    {symbol}: {state.signal.value} (strength: {state.signal_strength:.2f})")
    
    # Final statistics
    print(f"\nFinal Results:")
    stats = rsi_calculator.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    market_overview = rsi_calculator.get_market_overview()
    print(f"\nMarket Overview:")
    for key, value in market_overview.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    # Simple test of RSI calculator
    import asyncio
    from cache_manager import CacheManager
    
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT'],
        'rsi_period': 14,
        'oversold_threshold': 30,
        'overbought_threshold': 70
    }
    
    async def main():
        cache_manager = CacheManager(config)
        rsi_calculator = RSICalculator(config)
        
        await test_rsi_integration(cache_manager, rsi_calculator, num_cycles=20)
    
    asyncio.run(main())