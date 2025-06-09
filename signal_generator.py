"""
Signal Generator - Phase 2 (Complete Version)
Advanced signal generation with multi-timeframe confirmation and risk management
Builds on Phase 1 RSI data to create actionable trading signals
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import asyncio

class SignalType(Enum):
    """Enhanced signal types for long-short strategy"""
    LONG_ENTRY = "long_entry"
    SHORT_ENTRY = "short_entry"
    LONG_EXIT = "long_exit"
    SHORT_EXIT = "short_exit"
    POSITION_INCREASE = "position_increase"  # Add to existing position
    POSITION_DECREASE = "position_decrease"  # Reduce existing position
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    NO_SIGNAL = "no_signal"

class SignalPriority(Enum):
    """Signal priority levels"""
    CRITICAL = 1    # Immediate action required (stop loss)
    HIGH = 2        # Strong signal, act quickly
    MEDIUM = 3      # Good signal, act when convenient
    LOW = 4         # Weak signal, monitor only

@dataclass
class TradingSignal:
    """Enhanced trading signal with comprehensive information"""
    symbol: str
    signal_type: SignalType
    priority: SignalPriority
    strength: float  # 0-1 scale
    confidence: float  # 0-1 scale based on confirmations
    
    # Price and timing
    current_price: float
    target_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Technical indicators
    rsi_current: float = 0.0
    rsi_previous: float = 0.0
    price_change_percent: float = 0.0
    volume: int = 0
    volume_vs_avg: float = 1.0  # Volume relative to average
    
    # Signal confirmations
    confirmations: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    # Position sizing
    suggested_position_size: float = 0.0  # As percentage of portfolio
    max_risk_per_trade: float = 0.02  # 2% default
    
    # Market context
    market_state: str = "UNKNOWN"
    market_trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    correlation_risk: float = 0.0  # Risk from correlated positions
    
    def __str__(self) -> str:
        priority_emoji = {
            SignalPriority.CRITICAL: "🚨",
            SignalPriority.HIGH: "🔴", 
            SignalPriority.MEDIUM: "🟡",
            SignalPriority.LOW: "🟢"
        }
        
        signal_emoji = {
            SignalType.LONG_ENTRY: "📈",
            SignalType.SHORT_ENTRY: "📉",
            SignalType.LONG_EXIT: "↗️",
            SignalType.SHORT_EXIT: "↘️",
            SignalType.STOP_LOSS: "⛔",
            SignalType.TAKE_PROFIT: "💰"
        }
        
        emoji = signal_emoji.get(self.signal_type, "📊")
        priority = priority_emoji.get(self.priority, "⚪")
        
        return (f"{priority} {emoji} {self.symbol}: {self.signal_type.value.upper()} "
                f"| Price: ${self.current_price:.2f} "
                f"| Strength: {self.strength:.2f} "
                f"| Confidence: {self.confidence:.2f}")

class SignalGenerator:
    """Advanced signal generator with multi-timeframe analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Signal generation parameters
        self.rsi_config = config.get('rsi', {})
        self.oversold_threshold = self.rsi_config.get('oversold_threshold', 30)
        self.overbought_threshold = self.rsi_config.get('overbought_threshold', 70)
        self.extreme_oversold = self.rsi_config.get('extreme_oversold', 20)
        self.extreme_overbought = self.rsi_config.get('extreme_overbought', 80)
        
        # Volume confirmation
        self.volume_confirmation = self.rsi_config.get('volume_confirmation', True)
        self.min_volume_threshold = self.rsi_config.get('min_volume_threshold', 100000)
        self.volume_spike_threshold = 1.5  # 50% above average
        
        # Multi-timeframe confirmation
        self.require_momentum_confirmation = config.get('require_momentum_confirmation', True)
        self.price_momentum_periods = config.get('price_momentum_periods', 3)
        
        # Risk management
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2%
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.06)  # 6%
        self.stop_loss_atr_multiplier = config.get('stop_loss_atr_multiplier', 2.0)
        
        # Signal filtering
        self.min_signal_strength = config.get('min_signal_strength', 0.4)
        self.min_confidence = config.get('min_confidence', 0.5)
        self.max_signals_per_cycle = config.get('max_signals_per_cycle', 3)
        
        # State tracking
        self.active_signals = {}
        self.signal_history = {}
        self.volume_averages = {}  # Track volume averages per symbol
        self.price_history = {}    # Short-term price history for momentum
        self.market_correlation = {}  # Track symbol correlations
        
        self.signal_count = 0
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Signal Generator initialized:")
        self.logger.info(f"  RSI thresholds: {self.oversold_threshold}/{self.overbought_threshold}")
        self.logger.info(f"  Risk per trade: {self.max_risk_per_trade*100:.1f}%")
        self.logger.info(f"  Min signal strength: {self.min_signal_strength}")
    
    async def generate_signals(self, cache_manager, rsi_calculator, 
                             portfolio_manager=None) -> List[TradingSignal]:
        """Generate trading signals from current market data"""
        
        # Get current RSI states
        rsi_states = rsi_calculator.get_all_rsi_states()
        if not rsi_states:
            return []
        
        # Update market context
        await self._update_market_context(cache_manager, rsi_states)
        
        # Generate signals for each symbol
        new_signals = []
        for symbol, rsi_state in rsi_states.items():
            try:
                signals = await self._analyze_symbol(
                    symbol, rsi_state, cache_manager, portfolio_manager
                )
                new_signals.extend(signals)
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
        
        # Filter and prioritize signals
        filtered_signals = self._filter_and_prioritize_signals(new_signals)
        
        # Update active signals
        self._update_active_signals(filtered_signals)
        
        # Log signal summary
        if filtered_signals:
            self.logger.info(f"Generated {len(filtered_signals)} signals")
            for signal in filtered_signals[:3]:  # Log top 3
                self.logger.info(f"  {signal}")
        
        return filtered_signals
    
    async def _update_market_context(self, cache_manager, rsi_states: Dict):
        """Update market-wide context for better signal generation"""
        
        # Calculate market-wide metrics
        all_rsi = [state.current_rsi for state in rsi_states.values()]
        market_rsi_avg = np.mean(all_rsi) if all_rsi else 50
        
        # Determine overall market trend
        oversold_count = len([rsi for rsi in all_rsi if rsi <= self.oversold_threshold])
        overbought_count = len([rsi for rsi in all_rsi if rsi >= self.overbought_threshold])
        
        if oversold_count > len(all_rsi) * 0.4:
            market_trend = "OVERSOLD"
        elif overbought_count > len(all_rsi) * 0.4:
            market_trend = "OVERBOUGHT"
        elif market_rsi_avg < 45:
            market_trend = "BEARISH"
        elif market_rsi_avg > 55:
            market_trend = "BULLISH"
        else:
            market_trend = "NEUTRAL"
        
        self.market_trend = market_trend
        self.market_rsi_avg = market_rsi_avg
        
        # Update volume averages
        await self._update_volume_averages(cache_manager)
        
        # Update price momentum tracking
        await self._update_price_momentum(cache_manager)
    
    async def _update_volume_averages(self, cache_manager):
        """Update rolling volume averages for each symbol"""
        symbols_with_data = cache_manager.get_symbols_with_data()
        
        for symbol in symbols_with_data:
            cache = cache_manager.get_stock_cache(symbol)
            if cache and len(cache.volumes) > 5:
                volume_history = cache.get_volume_history()
                if len(volume_history) > 0:
                    # Use last 20 periods for average
                    recent_volumes = volume_history[-20:] if len(volume_history) >= 20 else volume_history
                    avg_volume = np.mean(recent_volumes[recent_volumes > 0])  # Exclude zeros
                    self.volume_averages[symbol] = avg_volume
    
    async def _update_price_momentum(self, cache_manager):
        """Update short-term price momentum for confirmation"""
        symbols_with_data = cache_manager.get_symbols_with_data()
        
        for symbol in symbols_with_data:
            cache = cache_manager.get_stock_cache(symbol)
            if cache and len(cache.prices) >= self.price_momentum_periods:
                recent_prices = cache.get_price_history(self.price_momentum_periods + 1)
                
                if len(recent_prices) > self.price_momentum_periods:
                    # Calculate momentum as price change over last N periods
                    momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                    self.price_history[symbol] = {
                        'momentum': momentum,
                        'recent_prices': recent_prices,
                        'trend_strength': abs(momentum)
                    }
    
    async def _analyze_symbol(self, symbol: str, rsi_state, cache_manager, 
                            portfolio_manager) -> List[TradingSignal]:
        """Analyze individual symbol for trading signals"""
        
        signals = []
        
        # Get additional data
        cache = cache_manager.get_stock_cache(symbol)
        if not cache:
            return signals
        
        indicators = cache.get_indicators()
        
        # Calculate volume context
        current_volume = indicators.get('volume', 0)
        avg_volume = self.volume_averages.get(symbol, current_volume)
        volume_ratio = current_volume / max(avg_volume, 1)
        
        # Get price momentum
        momentum_data = self.price_history.get(symbol, {})
        price_momentum = momentum_data.get('momentum', 0)
        
        # Get current position (if portfolio manager available)
        current_position = 0
        if portfolio_manager:
            current_position = portfolio_manager.get_position_size(symbol)
        
        # Analyze for entry signals
        if current_position == 0:  # No current position
            entry_signal = self._check_entry_signals(
                symbol, rsi_state, indicators, volume_ratio, price_momentum
            )
            if entry_signal:
                signals.append(entry_signal)
        
        # Analyze for exit signals (if we have a position)
        elif current_position != 0:
            exit_signal = self._check_exit_signals(
                symbol, rsi_state, indicators, current_position, portfolio_manager
            )
            if exit_signal:
                signals.append(exit_signal)
        
        # Check for position sizing signals (regardless of current position)
        sizing_signal = self._check_position_sizing_signals(
            symbol, rsi_state, indicators, current_position
        )
        if sizing_signal:
            signals.append(sizing_signal)
        
        return signals
    
    def _check_entry_signals(self, symbol: str, rsi_state, indicators: Dict, 
                           volume_ratio: float, price_momentum: float) -> Optional[TradingSignal]:
        """Check for entry signals (long or short)"""
        
        current_rsi = rsi_state.current_rsi
        previous_rsi = rsi_state.previous_rsi
        current_price = indicators['price']
        
        # Long entry conditions
        if self._is_long_entry_condition(current_rsi, previous_rsi):
            return self._create_long_entry_signal(
                symbol, rsi_state, indicators, volume_ratio, price_momentum
            )
        
        # Short entry conditions  
        elif self._is_short_entry_condition(current_rsi, previous_rsi):
            return self._create_short_entry_signal(
                symbol, rsi_state, indicators, volume_ratio, price_momentum
            )
        
        return None
    
    def _is_long_entry_condition(self, current_rsi: float, previous_rsi: float) -> bool:
        """Check if conditions are met for long entry"""
        # RSI crossing above oversold threshold
        basic_condition = (previous_rsi <= self.oversold_threshold and 
                          current_rsi > self.oversold_threshold)
        
        # Extreme oversold condition (stronger signal)
        extreme_condition = current_rsi <= self.extreme_oversold
        
        # RSI momentum (RSI increasing)
        momentum_condition = current_rsi > previous_rsi
        
        return (basic_condition or extreme_condition) and momentum_condition
    
    def _is_short_entry_condition(self, current_rsi: float, previous_rsi: float) -> bool:
        """Check if conditions are met for short entry"""
        # RSI crossing below overbought threshold
        basic_condition = (previous_rsi >= self.overbought_threshold and 
                          current_rsi < self.overbought_threshold)
        
        # Extreme overbought condition (stronger signal)
        extreme_condition = current_rsi >= self.extreme_overbought
        
        # RSI momentum (RSI decreasing)
        momentum_condition = current_rsi < previous_rsi
        
        return (basic_condition or extreme_condition) and momentum_condition
    
    def _create_long_entry_signal(self, symbol: str, rsi_state, indicators: Dict,
                                volume_ratio: float, price_momentum: float) -> TradingSignal:
        """Create long entry signal with comprehensive analysis"""
        
        current_price = indicators['price']
        current_rsi = rsi_state.current_rsi
        
        # Calculate signal strength
        strength = self._calculate_long_strength(rsi_state, volume_ratio, price_momentum)
        
        # Calculate confidence based on confirmations
        confirmations = []
        confidence = 0.5  # Base confidence
        
        # RSI confirmation
        if current_rsi <= self.extreme_oversold:
            confirmations.append("Extreme oversold RSI")
            confidence += 0.2
        elif current_rsi <= self.oversold_threshold:
            confirmations.append("Oversold RSI")
            confidence += 0.1
        
        # Volume confirmation
        if volume_ratio >= self.volume_spike_threshold:
            confirmations.append("High volume")
            confidence += 0.15
        elif volume_ratio >= 1.2:
            confirmations.append("Above average volume")
            confidence += 0.1
        
        # Price momentum confirmation
        if price_momentum > 0.01:  # 1% positive momentum
            confirmations.append("Positive price momentum")
            confidence += 0.1
        elif price_momentum > 0:
            confirmations.append("Slight positive momentum")
            confidence += 0.05
        
        # Market trend confirmation
        if self.market_trend in ["BULLISH", "OVERSOLD"]:
            confirmations.append(f"Market trend: {self.market_trend}")
            confidence += 0.1
        
        confidence = min(1.0, confidence)
        
        # Calculate stop loss and take profit
        atr_estimate = current_price * 0.02  # Rough 2% ATR estimate
        stop_loss_price = current_price - (atr_estimate * self.stop_loss_atr_multiplier)
        take_profit_price = current_price + (atr_estimate * 3.0)  # 3:1 reward:risk
        
        # Calculate position size
        risk_amount = current_price - stop_loss_price
        position_size = min(
            self.max_risk_per_trade / (risk_amount / current_price),
            0.1  # Max 10% of portfolio in any single position
        )
        
        # Determine priority
        if current_rsi <= self.extreme_oversold and confidence >= 0.8:
            priority = SignalPriority.HIGH
        elif confidence >= 0.7:
            priority = SignalPriority.MEDIUM
        else:
            priority = SignalPriority.LOW
        
        return TradingSignal(
            symbol=symbol,
            signal_type=SignalType.LONG_ENTRY,
            priority=priority,
            strength=strength,
            confidence=confidence,
            current_price=current_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            rsi_current=current_rsi,
            rsi_previous=rsi_state.previous_rsi,
            price_change_percent=indicators.get('price_change_percent', 0),
            volume=indicators.get('volume', 0),
            volume_vs_avg=volume_ratio,
            confirmations=confirmations,
            suggested_position_size=position_size,
            max_risk_per_trade=self.max_risk_per_trade,
            market_state=indicators.get('market_state', 'UNKNOWN'),
            market_trend=self.market_trend
        )
    
    def _create_short_entry_signal(self, symbol: str, rsi_state, indicators: Dict,
                                 volume_ratio: float, price_momentum: float) -> TradingSignal:
        """Create short entry signal with comprehensive analysis"""
        
        current_price = indicators['price']
        current_rsi = rsi_state.current_rsi
        
        # Calculate signal strength
        strength = self._calculate_short_strength(rsi_state, volume_ratio, price_momentum)
        
        # Calculate confidence based on confirmations
        confirmations = []
        confidence = 0.5  # Base confidence
        
        # RSI confirmation
        if current_rsi >= self.extreme_overbought:
            confirmations.append("Extreme overbought RSI")
            confidence += 0.2
        elif current_rsi >= self.overbought_threshold:
            confirmations.append("Overbought RSI")
            confidence += 0.1
        
        # Volume confirmation
        if volume_ratio >= self.volume_spike_threshold:
            confirmations.append("High volume")
            confidence += 0.15
        elif volume_ratio >= 1.2:
            confirmations.append("Above average volume")
            confidence += 0.1
        
        # Price momentum confirmation (negative for short)
        if price_momentum < -0.01:  # 1% negative momentum
            confirmations.append("Negative price momentum")
            confidence += 0.1
        elif price_momentum < 0:
            confirmations.append("Slight negative momentum")
            confidence += 0.05
        
        # Market trend confirmation
        if self.market_trend in ["BEARISH", "OVERBOUGHT"]:
            confirmations.append(f"Market trend: {self.market_trend}")
            confidence += 0.1
        
        confidence = min(1.0, confidence)
        
        # Calculate stop loss and take profit
        atr_estimate = current_price * 0.02  # Rough 2% ATR estimate
        stop_loss_price = current_price + (atr_estimate * self.stop_loss_atr_multiplier)
        take_profit_price = current_price - (atr_estimate * 3.0)  # 3:1 reward:risk
        
        # Calculate position size (negative for short)
        risk_amount = stop_loss_price - current_price
        position_size = -min(  # Negative for short position
            self.max_risk_per_trade / (risk_amount / current_price),
            0.1  # Max 10% of portfolio in any single position
        )
        
        # Determine priority
        if current_rsi >= self.extreme_overbought and confidence >= 0.8:
            priority = SignalPriority.HIGH
        elif confidence >= 0.7:
            priority = SignalPriority.MEDIUM
        else:
            priority = SignalPriority.LOW
        
        return TradingSignal(
            symbol=symbol,
            signal_type=SignalType.SHORT_ENTRY,
            priority=priority,
            strength=strength,
            confidence=confidence,
            current_price=current_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            rsi_current=current_rsi,
            rsi_previous=rsi_state.previous_rsi,
            price_change_percent=indicators.get('price_change_percent', 0),
            volume=indicators.get('volume', 0),
            volume_vs_avg=volume_ratio,
            confirmations=confirmations,
            suggested_position_size=position_size,
            max_risk_per_trade=self.max_risk_per_trade,
            market_state=indicators.get('market_state', 'UNKNOWN'),
            market_trend=self.market_trend
        )
    
    def _calculate_long_strength(self, rsi_state, volume_ratio: float, 
                               price_momentum: float) -> float:
        """Calculate strength for long signals"""
        current_rsi = rsi_state.current_rsi
        
        # Base strength from RSI level
        if current_rsi <= self.extreme_oversold:
            base_strength = 0.9
        elif current_rsi <= self.oversold_threshold:
            base_strength = 0.7
        else:
            # RSI momentum based
            rsi_change = current_rsi - rsi_state.previous_rsi
            base_strength = min(0.6, 0.3 + (rsi_change / 10))
        
        # Volume multiplier
        volume_multiplier = min(1.3, 0.8 + (volume_ratio * 0.3))
        
        # Momentum multiplier
        momentum_multiplier = min(1.2, 1.0 + max(0, price_momentum * 10))
        
        # Market trend multiplier
        trend_multiplier = 1.1 if self.market_trend in ["BULLISH", "OVERSOLD"] else 1.0
        
        final_strength = base_strength * volume_multiplier * momentum_multiplier * trend_multiplier
        return min(1.0, final_strength)
    
    def _calculate_short_strength(self, rsi_state, volume_ratio: float, 
                                price_momentum: float) -> float:
        """Calculate strength for short signals"""
        current_rsi = rsi_state.current_rsi
        
        # Base strength from RSI level
        if current_rsi >= self.extreme_overbought:
            base_strength = 0.9
        elif current_rsi >= self.overbought_threshold:
            base_strength = 0.7
        else:
            # RSI momentum based
            rsi_change = rsi_state.previous_rsi - current_rsi  # Negative change for short
            base_strength = min(0.6, 0.3 + (rsi_change / 10))
        
        # Volume multiplier
        volume_multiplier = min(1.3, 0.8 + (volume_ratio * 0.3))
        
        # Momentum multiplier (negative momentum for short)
        momentum_multiplier = min(1.2, 1.0 + max(0, -price_momentum * 10))
        
        # Market trend multiplier
        trend_multiplier = 1.1 if self.market_trend in ["BEARISH", "OVERBOUGHT"] else 1.0
        
        final_strength = base_strength * volume_multiplier * momentum_multiplier * trend_multiplier
        return min(1.0, final_strength)
    
    def _check_exit_signals(self, symbol: str, rsi_state, indicators: Dict,
                          current_position: float, portfolio_manager) -> Optional[TradingSignal]:
        """Check for exit signals based on current position"""
        
        current_rsi = rsi_state.current_rsi
        current_price = indicators['price']
        
        # Long position exit conditions
        if current_position > 0:
            # RSI overbought exit
            if current_rsi >= self.overbought_threshold:
                return self._create_exit_signal(
                    symbol, SignalType.LONG_EXIT, rsi_state, indicators,
                    "RSI overbought - long exit"
                )
        
        # Short position exit conditions
        elif current_position < 0:
            # RSI oversold exit
            if current_rsi <= self.oversold_threshold:
                return self._create_exit_signal(
                    symbol, SignalType.SHORT_EXIT, rsi_state, indicators,
                    "RSI oversold - short exit"
                )
        
        return None
    
    def _create_exit_signal(self, symbol: str, signal_type: SignalType, 
                          rsi_state, indicators: Dict, reason: str) -> TradingSignal:
        """Create exit signal"""
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            priority=SignalPriority.MEDIUM,
            strength=0.7,  # Exit signals have moderate strength
            confidence=0.8,  # High confidence in RSI-based exits
            current_price=indicators['price'],
            rsi_current=rsi_state.current_rsi,
            rsi_previous=rsi_state.previous_rsi,
            price_change_percent=indicators.get('price_change_percent', 0),
            volume=indicators.get('volume', 0),
            confirmations=[reason],
            market_state=indicators.get('market_state', 'UNKNOWN'),
            market_trend=self.market_trend
        )
    
    def _check_position_sizing_signals(self, symbol: str, rsi_state, indicators: Dict,
                                     current_position: float) -> Optional[TradingSignal]:
        """Check for position sizing adjustment signals"""
        
        # This could be expanded for position scaling
        # For now, we'll keep it simple
        return None
    
    def _filter_and_prioritize_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter signals based on strength/confidence and prioritize"""
        
        # Filter by minimum thresholds
        filtered = [
            signal for signal in signals 
            if (signal.strength >= self.min_signal_strength and 
                signal.confidence >= self.min_confidence)
        ]
        
        # Sort by priority, then strength, then confidence
        filtered.sort(key=lambda s: (s.priority.value, -s.strength, -s.confidence))
        
        # Limit number of signals per cycle
        return filtered[:self.max_signals_per_cycle]
    
    def _update_active_signals(self, new_signals: List[TradingSignal]):
        """Update the active signals tracking"""
        
        # Add new signals
        for signal in new_signals:
            self.active_signals[signal.symbol] = signal
            self.signal_count += 1
        
        # Remove old signals (could add time-based expiry here)
        # For now, signals are replaced when new ones are generated
    
    def get_active_signals(self) -> Dict[str, TradingSignal]:
        """Get currently active signals"""
        return self.active_signals.copy()
    
    def get_signals_by_priority(self, priority: SignalPriority) -> List[TradingSignal]:
        """Get signals filtered by priority level"""
        return [
            signal for signal in self.active_signals.values()
            if signal.priority == priority
        ]
    
    def get_entry_signals(self) -> List[TradingSignal]:
        """Get all entry signals (long and short)"""
        return [
            signal for signal in self.active_signals.values()
            if signal.signal_type in [SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY]
        ]
    
    def get_exit_signals(self) -> List[TradingSignal]:
        """Get all exit signals"""
        return [
            signal for signal in self.active_signals.values()
            if signal.signal_type in [SignalType.LONG_EXIT, SignalType.SHORT_EXIT]
        ]
    
    def get_statistics(self) -> Dict:
        """Get signal generator statistics"""
        active_signals = list(self.active_signals.values())
        
        if not active_signals:
            return {
                'total_signals_generated': self.signal_count,
                'active_signals': 0,
                'entry_signals': 0,
                'exit_signals': 0,
                'avg_signal_strength': 0.0,
                'avg_confidence': 0.0,
                'priority_breakdown': {
                    'critical': 0,
                    'high': 0,
                    'medium': 0,
                    'low': 0
                },
                'market_trend': getattr(self, 'market_trend', 'UNKNOWN'),
                'market_rsi_avg': getattr(self, 'market_rsi_avg', 50),
                'symbols_with_signals': 0
            }
        
        return {
            'total_signals_generated': self.signal_count,
            'active_signals': len(active_signals),
            'entry_signals': len(self.get_entry_signals()),
            'exit_signals': len(self.get_exit_signals()),
            'avg_signal_strength': np.mean([s.strength for s in active_signals]),
            'avg_confidence': np.mean([s.confidence for s in active_signals]),
            'priority_breakdown': {
                'critical': len(self.get_signals_by_priority(SignalPriority.CRITICAL)),
                'high': len(self.get_signals_by_priority(SignalPriority.HIGH)),
                'medium': len(self.get_signals_by_priority(SignalPriority.MEDIUM)),
                'low': len(self.get_signals_by_priority(SignalPriority.LOW))
            },
            'market_trend': getattr(self, 'market_trend', 'UNKNOWN'),
            'market_rsi_avg': getattr(self, 'market_rsi_avg', 50),
            'symbols_with_signals': len(set(s.symbol for s in active_signals))
        }
    
    def clear_signals(self, symbol: str = None):
        """Clear signals for a specific symbol or all symbols"""
        if symbol:
            if symbol in self.active_signals:
                del self.active_signals[symbol]
                self.logger.info(f"Cleared signals for {symbol}")
        else:
            self.active_signals.clear()
            self.signal_count = 0
            self.logger.info("Cleared all signals")

# Integration and testing functions
async def test_signal_generator_integration():
    """Test signal generator with Phase 1 components"""
    import random
    from cache_manager import CacheManager
    from rsi_calculator import RSICalculator
    
    print("=== SIGNAL GENERATOR INTEGRATION TEST ===")
    print()
    
    # Configuration
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        'rsi_period': 14,
        'rsi': {
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'extreme_oversold': 20,
            'extreme_overbought': 80,
            'volume_confirmation': True,
            'min_volume_threshold': 100000
        },
        'min_signal_strength': 0.4,
        'min_confidence': 0.5,
        'max_risk_per_trade': 0.02,
        'max_signals_per_cycle': 3
    }
    
    # Initialize components
    cache_manager = CacheManager(config)
    rsi_calculator = RSICalculator(config)
    signal_generator = SignalGenerator(config)
    
    # Simulate market data with trending behavior
    print("Simulating market conditions to generate signals...")
    base_prices = {'AAPL': 150.0, 'GOOGL': 2500.0, 'MSFT': 300.0, 'TSLA': 200.0}
    
    # Phase 1: Create oversold conditions
    print("\n📉 Phase 1: Creating oversold conditions...")
    for cycle in range(15):
        for symbol in config['symbols']:
            # Trending down to create oversold
            change_pct = random.uniform(-0.04, -0.01)  # Consistent downward movement
            new_price = base_prices[symbol] * (1 + change_pct)
            base_prices[symbol] = new_price
            
            data = {
                'symbol': symbol,
                'price': new_price,
                'volume': random.randint(500000, 2000000),  # Higher volume
                'timestamp': datetime.now(),
                'market_state': 'REGULAR',
                'source': 'simulation'
            }
            
            await cache_manager.update_price_data(symbol, data)
        
        # Update RSI
        rsi_calculator.update_from_cache(cache_manager)
        
        # Show RSI progression
        if cycle % 5 == 4:
            rsi_values = rsi_calculator.get_all_rsi_values()
            print(f"  Cycle {cycle+1}: RSI values: {[(s, f'{v:.0f}') for s, v in rsi_values.items()]}")
    
    # Check for long signals
    print("\n📈 Checking for long entry signals...")
    signals = await signal_generator.generate_signals(cache_manager, rsi_calculator)
    
    if signals:
        print(f"✅ Generated {len(signals)} signals:")
        for signal in signals:
            print(f"  {signal}")
            print(f"    Confirmations: {', '.join(signal.confirmations)}")
            print(f"    Suggested position: {signal.suggested_position_size:.2%}")
            print(f"    Stop loss: ${signal.stop_loss_price:.2f}")
            print(f"    Take profit: ${signal.take_profit_price:.2f}")
            print()
    else:
        print("⚠️ No signals generated yet - may need more extreme conditions")
    
    # Phase 2: Create recovery (should trigger long signals)
    print("📈 Phase 2: Creating price recovery...")
    for cycle in range(8):
        for symbol in config['symbols']:
            # Price recovery with high volume
            change_pct = random.uniform(0.01, 0.03)  # Upward movement
            new_price = base_prices[symbol] * (1 + change_pct)
            base_prices[symbol] = new_price
            
            data = {
                'symbol': symbol,
                'price': new_price,
                'volume': random.randint(1000000, 3000000),  # Very high volume
                'timestamp': datetime.now(),
                'market_state': 'REGULAR',
                'source': 'simulation'
            }
            
            await cache_manager.update_price_data(symbol, data)
        
        # Update RSI and check for signals
        rsi_calculator.update_from_cache(cache_manager)
        signals = await signal_generator.generate_signals(cache_manager, rsi_calculator)
        
        if signals:
            print(f"\n🚨 New signals in recovery cycle {cycle+1}:")
            for signal in signals:
                print(f"  {signal}")
    
    # Phase 3: Create overbought conditions
    print("\n📈 Phase 3: Creating overbought conditions...")
    for cycle in range(15):
        for symbol in config['symbols']:
            # Trending up to create overbought
            change_pct = random.uniform(0.01, 0.04)  # Consistent upward movement
            new_price = base_prices[symbol] * (1 + change_pct)
            base_prices[symbol] = new_price
            
            data = {
                'symbol': symbol,
                'price': new_price,
                'volume': random.randint(800000, 2500000),
                'timestamp': datetime.now(),
                'market_state': 'REGULAR',
                'source': 'simulation'
            }
            
            await cache_manager.update_price_data(symbol, data)
        
        # Update RSI
        rsi_calculator.update_from_cache(cache_manager)
        
        # Check for short signals
        signals = await signal_generator.generate_signals(cache_manager, rsi_calculator)
        
        if cycle % 5 == 4:
            rsi_values = rsi_calculator.get_all_rsi_values()
            print(f"  Cycle {cycle+1}: RSI values: {[(s, f'{v:.0f}') for s, v in rsi_values.items()]}")
            
            if signals:
                print("  🚨 Short signals detected:")
                for signal in signals:
                    if signal.signal_type == SignalType.SHORT_ENTRY:
                        print(f"    {signal}")
    
    # Final results
    print("\n" + "="*60)
    print("SIGNAL GENERATOR TEST RESULTS")
    print("="*60)
    
    final_signals = await signal_generator.generate_signals(cache_manager, rsi_calculator)
    stats = signal_generator.get_statistics()
    
    print(f"Total signals generated: {stats['total_signals_generated']}")
    print(f"Currently active signals: {stats['active_signals']}")
    print(f"Entry signals: {stats.get('entry_signals', 0)}")
    print(f"Exit signals: {stats.get('exit_signals', 0)}")
    print(f"Average signal strength: {stats.get('avg_signal_strength', 0):.2f}")
    print(f"Average confidence: {stats.get('avg_confidence', 0):.2f}")
    print(f"Market trend: {stats['market_trend']}")
    
    if final_signals:
        print(f"\nFinal active signals:")
        for signal in final_signals:
            print(f"  {signal}")
    
    return len(final_signals) > 0

# Advanced testing with realistic market scenarios
async def test_realistic_market_scenarios():
    """Test signal generator with realistic market scenarios"""
    print("=== REALISTIC MARKET SCENARIOS TEST ===")
    print()
    from cache_manager import CacheManager
    from rsi_calculator import RSICalculator

    # Test different market conditions
    scenarios = [
        {
            'name': 'Bear Market Bounce',
            'description': 'Strong oversold bounce in bear market',
            'price_pattern': [
                (-0.03, -0.01),  # Strong decline
                (-0.02, 0.00),   # Continued weakness  
                (-0.01, 0.01),   # Stabilization
                (0.01, 0.03),    # Recovery bounce
                (0.00, 0.02)     # Consolidation
            ],
            'volume_pattern': [1.5, 1.8, 2.0, 2.5, 1.2]  # Volume spikes during bounce
        },
        {
            'name': 'Bull Market Top',
            'description': 'Overbought conditions in bull market',
            'price_pattern': [
                (0.01, 0.03),    # Strong uptrend
                (0.02, 0.04),    # Acceleration
                (0.00, 0.02),    # Slowing momentum
                (-0.01, 0.01),   # Divergence
                (-0.02, 0.00)    # Reversal signal
            ],
            'volume_pattern': [1.2, 1.4, 1.1, 0.9, 1.6]  # Volume divergence
        }
    ]
    
    config = {
        'symbols': ['AAPL'],  # Test with one symbol for clarity
        'rsi_period': 14,
        'rsi': {
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'extreme_oversold': 20,
            'extreme_overbought': 80,
            'volume_confirmation': True
        },
        'min_signal_strength': 0.3,  # Lower threshold for testing
        'min_confidence': 0.4
    }
    
    for scenario in scenarios:
        print(f"\n🎯 Testing: {scenario['name']}")
        print(f"   {scenario['description']}")
        print("-" * 50)
        
        # Initialize fresh components
        cache_manager = CacheManager(config)
        rsi_calculator = RSICalculator(config)
        signal_generator = SignalGenerator(config)
        
        base_price = 150.0
        base_volume = 1000000
        
        # Build RSI history first
        for i in range(20):
            price_change = random.uniform(-0.01, 0.01)
            base_price *= (1 + price_change)
            
            data = {
                'symbol': 'AAPL',
                'price': base_price,
                'volume': base_volume,
                'timestamp': datetime.now(),
                'market_state': 'REGULAR',
                'source': 'simulation'
            }
            
            await cache_manager.update_price_data('AAPL', data)
            rsi_calculator.update_from_cache(cache_manager)
        
        # Apply scenario pattern
        for phase_idx, ((min_change, max_change), volume_mult) in enumerate(zip(scenario['price_pattern'], scenario['volume_pattern'])):
            print(f"\n  Phase {phase_idx + 1}: Price change {min_change:.1%} to {max_change:.1%}, Volume {volume_mult}x")
            
            for cycle in range(5):  # 5 cycles per phase
                price_change = random.uniform(min_change, max_change)
                base_price *= (1 + price_change)
                
                data = {
                    'symbol': 'AAPL',
                    'price': base_price,
                    'volume': int(base_volume * volume_mult * random.uniform(0.8, 1.2)),
                    'timestamp': datetime.now(),
                    'market_state': 'REGULAR',
                    'source': 'simulation'
                }
                
                await cache_manager.update_price_data('AAPL', data)
                rsi_calculator.update_from_cache(cache_manager)
                
                # Check for signals
                signals = await signal_generator.generate_signals(cache_manager, rsi_calculator)
                
                if signals:
                    for signal in signals:
                        print(f"    🚨 {signal}")
                        print(f"       Confirmations: {', '.join(signal.confirmations)}")
            
            # Show RSI at end of phase
            rsi_values = rsi_calculator.get_all_rsi_values()
            if rsi_values:
                rsi = list(rsi_values.values())[0]
                print(f"    RSI: {rsi:.1f}")
        
        print(f"\n  ✅ {scenario['name']} scenario complete")

# Quick functionality test
async def quick_signal_test():
    """Quick test to verify signal generation works"""
    from cache_manager import CacheManager
    from rsi_calculator import RSICalculator
    
    config = {
        'symbols': ['AAPL', 'GOOGL'],
        'rsi_period': 14,
        'rsi': {
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'extreme_oversold': 20,
            'extreme_overbought': 80
        },
        'min_signal_strength': 0.3,
        'min_confidence': 0.4
    }
    
    print("🚀 Quick Signal Generator Test")
    print("=" * 40)
    
    # Initialize components
    cache_manager = CacheManager(config)
    rsi_calculator = RSICalculator(config)
    signal_generator = SignalGenerator(config)
    
    # Generate extreme oversold condition for AAPL
    base_price = 150.0
    for i in range(25):  # Build enough history
        # Create strong downtrend
        if i < 20:
            change = random.uniform(-0.03, -0.01)  # Strong decline
        else:
            change = random.uniform(0.01, 0.02)   # Recovery
        
        base_price *= (1 + change)
        
        data = {
            'symbol': 'AAPL',
            'price': base_price,
            'volume': random.randint(1000000, 3000000),
            'timestamp': datetime.now(),
            'market_state': 'REGULAR',
            'source': 'simulation'
        }
        
        await cache_manager.update_price_data('AAPL', data)
        rsi_calculator.update_from_cache(cache_manager)
        
        if i >= 20:  # Check for signals during recovery
            signals = await signal_generator.generate_signals(cache_manager, rsi_calculator)
            if signals:
                print(f"✅ Signal generated on cycle {i+1}:")
                for signal in signals:
                    print(f"  {signal}")
                return True
    
    print("⚠️ No signals generated")
    return False

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    print("Signal Generator Test Options:")
    print("1. Integration test (comprehensive)")
    print("2. Realistic scenarios test")
    print("3. Quick functionality test")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(test_signal_generator_integration())
    elif choice == "2":
        asyncio.run(test_realistic_market_scenarios())
    elif choice == "3":
        success = asyncio.run(quick_signal_test())
        print(f"\n{'✅ SUCCESS' if success else '⚠️ PARTIAL'}: Signal generator test completed")
    else:
        print("Invalid choice")