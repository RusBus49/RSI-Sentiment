"""
Risk Manager - Phase 2
Advanced risk management and position sizing for RSI long-short strategy
Validates and adjusts signals from signal_generator based on portfolio risk
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio

# Import our signal types
from signal_generator import TradingSignal, SignalType, SignalPriority

class RiskLevel(Enum):
    """Portfolio risk assessment levels"""
    LOW = "low"           # < 25% of max risk
    MODERATE = "moderate" # 25-50% of max risk
    HIGH = "high"         # 50-75% of max risk
    CRITICAL = "critical" # > 75% of max risk
    EXCESSIVE = "excessive" # > 100% of max risk

class RiskAction(Enum):
    """Risk management actions"""
    APPROVE = "approve"               # Signal approved as-is
    REDUCE_SIZE = "reduce_size"       # Reduce position size
    REJECT = "reject"                 # Reject signal entirely
    REQUIRE_EXIT = "require_exit"     # Force position exit
    HOLD_NEW_POSITIONS = "hold_new_positions"  # Stop new positions

@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    total_portfolio_value: float
    current_positions_value: float
    unrealized_pnl: float
    total_risk_exposure: float
    max_single_position_risk: float
    correlation_risk: float
    leverage: float
    cash_available: float
    
    # Risk ratios
    portfolio_utilization: float  # positions / total value
    risk_utilization: float       # current risk / max risk
    concentration_risk: float     # largest position / total
    
    # Daily metrics
    daily_pnl: float
    daily_var: float              # Value at Risk
    max_drawdown: float
    
    risk_level: RiskLevel = RiskLevel.LOW

@dataclass 
class PositionRisk:
    """Risk assessment for individual position"""
    symbol: str
    current_position_size: float  # Current position value
    current_risk: float          # Current position risk
    correlation_exposure: float  # Risk from correlated positions
    sector_exposure: float       # Sector concentration risk
    liquidity_risk: float        # Risk from low liquidity
    total_risk_contribution: float
    
    # Position metrics
    days_held: int
    unrealized_pnl: float
    unrealized_pnl_percent: float
    stop_loss_distance: float
    
    # Risk flags
    risk_flags: List[str] = field(default_factory=list)

@dataclass
class RiskAdjustedSignal:
    """Signal with risk management adjustments"""
    original_signal: TradingSignal
    risk_action: RiskAction
    adjusted_position_size: float
    risk_score: float  # 0-1, higher = riskier
    rejection_reason: Optional[str] = None
    size_reduction_reason: Optional[str] = None
    approved: bool = True
    
    def __str__(self) -> str:
        action_emoji = {
            RiskAction.APPROVE: "✅",
            RiskAction.REDUCE_SIZE: "⚠️", 
            RiskAction.REJECT: "❌",
            RiskAction.REQUIRE_EXIT: "🚨",
            RiskAction.HOLD_NEW_POSITIONS: "⏸️"
        }
        
        emoji = action_emoji.get(self.risk_action, "❓")
        size_change = ""
        
        if self.risk_action == RiskAction.REDUCE_SIZE:
            original_size = abs(self.original_signal.suggested_position_size)
            adjusted_size = abs(self.adjusted_position_size)
            reduction = (1 - adjusted_size / original_size) * 100 if original_size > 0 else 0
            size_change = f" (-{reduction:.0f}%)"
        
        return (f"{emoji} {self.original_signal.symbol}: {self.risk_action.value.upper()}{size_change} "
                f"| Risk Score: {self.risk_score:.2f} "
                f"| Adj Size: {self.adjusted_position_size:.2%}")

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Portfolio limits
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.06)  # 6% max total risk
        self.max_single_position = config.get('max_single_position', 0.10)  # 10% max per position
        self.max_sector_exposure = config.get('max_sector_exposure', 0.25)  # 25% max per sector
        self.max_correlation_exposure = config.get('max_correlation_exposure', 0.20)  # 20% correlated risk
        
        # Risk per trade limits
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2% max per trade
        self.min_risk_per_trade = config.get('min_risk_per_trade', 0.005)  # 0.5% min per trade
        
        # Portfolio management
        self.target_portfolio_utilization = config.get('target_portfolio_utilization', 0.80)  # 80% invested
        self.max_leverage = config.get('max_leverage', 1.0)  # No leverage by default
        
        # Risk thresholds
        self.risk_reduction_threshold = config.get('risk_reduction_threshold', 0.75)  # Reduce at 75%
        self.position_hold_threshold = config.get('position_hold_threshold', 0.85)   # Hold new at 85%
        self.force_exit_threshold = config.get('force_exit_threshold', 0.95)         # Force exit at 95%
        
        # Correlation and sector mapping
        self.sector_mapping = config.get('sector_mapping', {})
        self.correlation_groups = config.get('correlation_groups', {})
        
        # Daily limits
        self.max_daily_loss = config.get('max_daily_loss', 0.02)  # 2% max daily loss
        self.max_consecutive_losses = config.get('max_consecutive_losses', 3)
        
        # State tracking
        self.current_portfolio_value = 100000.0  # Default starting value
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.risk_override_active = False
        self.last_risk_assessment = None
        
        # Position tracking
        self.position_risks = {}
        self.sector_exposures = {}
        self.correlation_exposures = {}
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Risk Manager initialized:")
        self.logger.info(f"  Max portfolio risk: {self.max_portfolio_risk*100:.1f}%")
        self.logger.info(f"  Max single position: {self.max_single_position*100:.1f}%")
        self.logger.info(f"  Max risk per trade: {self.max_risk_per_trade*100:.1f}%")
    
    async def assess_portfolio_risk(self, portfolio_manager=None) -> RiskMetrics:
        """Comprehensive portfolio risk assessment"""
        
        # Get current portfolio state
        if portfolio_manager:
            portfolio_value = portfolio_manager.get_total_value()
            positions = portfolio_manager.get_all_positions()
            unrealized_pnl = portfolio_manager.get_unrealized_pnl()
            cash_available = portfolio_manager.get_cash_balance()
            daily_pnl = portfolio_manager.get_daily_pnl()
        else:
            # Default values for testing
            portfolio_value = self.current_portfolio_value
            positions = {}
            unrealized_pnl = 0.0
            cash_available = portfolio_value * 0.2  # Assume 20% cash
            daily_pnl = self.daily_pnl
        
        # Calculate position values and risks
        total_position_value = 0.0
        total_risk_exposure = 0.0
        max_single_risk = 0.0
        
        for symbol, position in positions.items():
            position_value = abs(position.get('value', 0))
            position_risk = abs(position.get('risk', 0))
            
            total_position_value += position_value
            total_risk_exposure += position_risk
            max_single_risk = max(max_single_risk, position_risk)
        
        # Calculate correlation risk
        correlation_risk = self._calculate_correlation_risk(positions)
        
        # Calculate leverage
        leverage = total_position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate ratios
        portfolio_utilization = total_position_value / portfolio_value if portfolio_value > 0 else 0
        risk_utilization = total_risk_exposure / (portfolio_value * self.max_portfolio_risk) if portfolio_value > 0 else 0
        concentration_risk = max_single_risk / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate VaR (simplified)
        daily_var = self._calculate_var(positions, portfolio_value)
        
        # Calculate max drawdown (simplified)
        max_drawdown = max(0, -unrealized_pnl / portfolio_value) if portfolio_value > 0 else 0
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_utilization, concentration_risk, leverage)
        
        risk_metrics = RiskMetrics(
            total_portfolio_value=portfolio_value,
            current_positions_value=total_position_value,
            unrealized_pnl=unrealized_pnl,
            total_risk_exposure=total_risk_exposure,
            max_single_position_risk=max_single_risk,
            correlation_risk=correlation_risk,
            leverage=leverage,
            cash_available=cash_available,
            portfolio_utilization=portfolio_utilization,
            risk_utilization=risk_utilization,
            concentration_risk=concentration_risk,
            daily_pnl=daily_pnl,
            daily_var=daily_var,
            max_drawdown=max_drawdown,
            risk_level=risk_level
        )
        
        self.last_risk_assessment = risk_metrics
        return risk_metrics
    
    async def evaluate_signals(self, signals: List[TradingSignal], 
                             portfolio_manager=None) -> List[RiskAdjustedSignal]:
        """Evaluate and adjust signals based on risk management"""
        
        if not signals:
            return []
        
        # Get current risk assessment
        risk_metrics = await self.assess_portfolio_risk(portfolio_manager)
        
        # Evaluate each signal
        risk_adjusted_signals = []
        
        for signal in signals:
            try:
                adjusted_signal = await self._evaluate_single_signal(
                    signal, risk_metrics, portfolio_manager
                )
                risk_adjusted_signals.append(adjusted_signal)
                
            except Exception as e:
                self.logger.error(f"Error evaluating signal for {signal.symbol}: {e}")
                # Create rejected signal
                risk_adjusted_signals.append(RiskAdjustedSignal(
                    original_signal=signal,
                    risk_action=RiskAction.REJECT,
                    adjusted_position_size=0.0,
                    risk_score=1.0,
                    rejection_reason=f"Evaluation error: {str(e)}",
                    approved=False
                ))
        
        # Apply portfolio-level constraints
        risk_adjusted_signals = self._apply_portfolio_constraints(
            risk_adjusted_signals, risk_metrics
        )
        
        # Log risk management summary
        self._log_risk_summary(risk_adjusted_signals, risk_metrics)
        
        return risk_adjusted_signals
    
    async def _evaluate_single_signal(self, signal: TradingSignal, 
                                    risk_metrics: RiskMetrics,
                                    portfolio_manager=None) -> RiskAdjustedSignal:
        """Evaluate individual signal for risk"""
        
        # Calculate signal risk score
        risk_score = self._calculate_signal_risk_score(signal, risk_metrics)
        
        # Check for immediate rejections
        rejection_reason = self._check_rejection_criteria(signal, risk_metrics)
        if rejection_reason:
            return RiskAdjustedSignal(
                original_signal=signal,
                risk_action=RiskAction.REJECT,
                adjusted_position_size=0.0,
                risk_score=risk_score,
                rejection_reason=rejection_reason,
                approved=False
            )
        
        # Calculate adjusted position size
        adjusted_size, size_reduction_reason = self._calculate_adjusted_position_size(
            signal, risk_metrics, portfolio_manager
        )
        
        # Determine risk action
        if adjusted_size == 0:
            risk_action = RiskAction.REJECT
            approved = False
        elif abs(adjusted_size) < abs(signal.suggested_position_size) * 0.95:  # 5% tolerance
            risk_action = RiskAction.REDUCE_SIZE
            approved = True
        else:
            risk_action = RiskAction.APPROVE
            approved = True
        
        return RiskAdjustedSignal(
            original_signal=signal,
            risk_action=risk_action,
            adjusted_position_size=adjusted_size,
            risk_score=risk_score,
            size_reduction_reason=size_reduction_reason,
            approved=approved
        )
    
    def _calculate_signal_risk_score(self, signal: TradingSignal, 
                                   risk_metrics: RiskMetrics) -> float:
        """Calculate comprehensive risk score for signal (0-1 scale)"""
        
        risk_factors = []
        
        # Position size risk
        position_size_risk = abs(signal.suggested_position_size) / self.max_single_position
        risk_factors.append(min(1.0, position_size_risk))
        
        # Portfolio utilization risk
        portfolio_risk = risk_metrics.risk_utilization
        risk_factors.append(portfolio_risk)
        
        # Concentration risk
        concentration_risk = risk_metrics.concentration_risk / self.max_single_position
        risk_factors.append(concentration_risk)
        
        # Signal strength risk (inverse - lower strength = higher risk)
        signal_strength_risk = 1.0 - signal.strength
        risk_factors.append(signal_strength_risk * 0.5)  # Weight it less
        
        # Confidence risk (inverse)
        confidence_risk = 1.0 - signal.confidence
        risk_factors.append(confidence_risk * 0.3)  # Weight it less
        
        # Sector concentration risk
        sector_risk = self._calculate_sector_risk(signal.symbol, signal.suggested_position_size)
        risk_factors.append(sector_risk)
        
        # Correlation risk
        correlation_risk = self._calculate_signal_correlation_risk(signal, risk_metrics)
        risk_factors.append(correlation_risk)
        
        # Market conditions risk
        if hasattr(signal, 'market_state') and signal.market_state != 'REGULAR':
            risk_factors.append(0.2)  # Add risk for non-regular market hours
        
        # Calculate weighted average
        weights = [0.25, 0.20, 0.15, 0.10, 0.10, 0.10, 0.10]  # Position size and portfolio risk get highest weight
        
        if len(risk_factors) != len(weights):
            # Fallback to simple average if lengths don't match
            return np.mean(risk_factors)
        
        weighted_risk = sum(r * w for r, w in zip(risk_factors, weights))
        return min(1.0, weighted_risk)
    
    def _check_rejection_criteria(self, signal: TradingSignal, 
                                risk_metrics: RiskMetrics) -> Optional[str]:
        """Check if signal should be immediately rejected"""
        
        # Portfolio risk too high
        if risk_metrics.risk_level == RiskLevel.EXCESSIVE:
            return "Portfolio risk excessive"
        
        # Daily loss limit exceeded
        if risk_metrics.daily_pnl < -self.max_daily_loss * risk_metrics.total_portfolio_value:
            return "Daily loss limit exceeded"
        
        # Cash insufficient for position
        required_cash = abs(signal.suggested_position_size) * risk_metrics.total_portfolio_value
        if required_cash > risk_metrics.cash_available:
            return "Insufficient cash available"
        
        # Position size too small
        if abs(signal.suggested_position_size) < self.min_risk_per_trade:
            return "Position size below minimum"
        
        # Signal strength/confidence too low for current risk level
        if risk_metrics.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            if signal.strength < 0.7 or signal.confidence < 0.7:
                return "Signal quality too low for current risk level"
        
        # Too many consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            if signal.signal_type in [SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY]:
                return "Too many consecutive losses - holding new positions"
        
        # Market hours check
        if hasattr(signal, 'market_state') and signal.market_state == 'CLOSED':
            return "Market is closed"
        
        return None
    
    def _calculate_adjusted_position_size(self, signal: TradingSignal, 
                                        risk_metrics: RiskMetrics,
                                        portfolio_manager=None) -> Tuple[float, Optional[str]]:
        """Calculate risk-adjusted position size"""
        
        original_size = signal.suggested_position_size
        adjusted_size = original_size
        size_reduction_reason = None
        
        # Risk-based position sizing
        max_risk_based_size = self._calculate_risk_based_max_size(signal, risk_metrics)
        if abs(adjusted_size) > abs(max_risk_based_size):
            adjusted_size = max_risk_based_size
            size_reduction_reason = "Risk limits"
        
        # Portfolio utilization limits
        if risk_metrics.portfolio_utilization > self.target_portfolio_utilization:
            utilization_factor = self.target_portfolio_utilization / risk_metrics.portfolio_utilization
            portfolio_adjusted_size = adjusted_size * utilization_factor
            if abs(portfolio_adjusted_size) < abs(adjusted_size):
                adjusted_size = portfolio_adjusted_size
                size_reduction_reason = "Portfolio utilization"
        
        # Concentration limits
        max_concentration_size = self.max_single_position
        if abs(adjusted_size) > max_concentration_size:
            adjusted_size = max_concentration_size if adjusted_size > 0 else -max_concentration_size
            size_reduction_reason = "Concentration limits"
        
        # Cash availability limits
        required_cash = abs(adjusted_size) * risk_metrics.total_portfolio_value
        if required_cash > risk_metrics.cash_available:
            cash_adjusted_size = (risk_metrics.cash_available / risk_metrics.total_portfolio_value)
            if adjusted_size < 0:
                cash_adjusted_size = -cash_adjusted_size
            adjusted_size = cash_adjusted_size
            size_reduction_reason = "Cash availability"
        
        # Sector exposure limits
        sector_adjusted_size = self._apply_sector_limits(signal.symbol, adjusted_size)
        if abs(sector_adjusted_size) < abs(adjusted_size):
            adjusted_size = sector_adjusted_size
            size_reduction_reason = "Sector exposure limits"
        
        # Correlation limits
        correlation_adjusted_size = self._apply_correlation_limits(signal, adjusted_size)
        if abs(correlation_adjusted_size) < abs(adjusted_size):
            adjusted_size = correlation_adjusted_size
            size_reduction_reason = "Correlation limits"
        
        # Risk level adjustments
        if risk_metrics.risk_level == RiskLevel.HIGH:
            adjusted_size *= 0.75  # Reduce by 25%
            size_reduction_reason = "High portfolio risk"
        elif risk_metrics.risk_level == RiskLevel.CRITICAL:
            adjusted_size *= 0.5   # Reduce by 50%
            size_reduction_reason = "Critical portfolio risk"
        
        # Minimum size check
        if abs(adjusted_size) < self.min_risk_per_trade:
            adjusted_size = 0.0
            size_reduction_reason = "Below minimum position size"
        
        return adjusted_size, size_reduction_reason
    
    def _calculate_risk_based_max_size(self, signal: TradingSignal, 
                                     risk_metrics: RiskMetrics) -> float:
        """Calculate maximum position size based on risk"""
        
        # Available risk budget
        available_risk = (self.max_portfolio_risk * risk_metrics.total_portfolio_value) - risk_metrics.total_risk_exposure
        
        # Calculate position risk
        if signal.stop_loss_price and signal.current_price:
            risk_per_dollar = abs(signal.current_price - signal.stop_loss_price) / signal.current_price
        else:
            risk_per_dollar = self.max_risk_per_trade  # Default risk assumption
        
        # Maximum position value based on available risk
        max_position_value = available_risk / risk_per_dollar if risk_per_dollar > 0 else 0
        
        # Convert to position size as percentage
        max_position_size = max_position_value / risk_metrics.total_portfolio_value if risk_metrics.total_portfolio_value > 0 else 0
        
        # Apply sign from original signal
        if signal.suggested_position_size < 0:
            max_position_size = -max_position_size
        
        return max_position_size
    
    def _apply_portfolio_constraints(self, signals: List[RiskAdjustedSignal], 
                                   risk_metrics: RiskMetrics) -> List[RiskAdjustedSignal]:
        """Apply portfolio-level constraints to approved signals"""
        
        # Sort by risk score (lowest risk first)
        approved_signals = [s for s in signals if s.approved]
        approved_signals.sort(key=lambda s: s.risk_score)
        
        # Apply portfolio-level limits
        total_new_risk = 0.0
        max_new_risk = (self.max_portfolio_risk * risk_metrics.total_portfolio_value) - risk_metrics.total_risk_exposure
        
        for signal in approved_signals:
            # Calculate risk for this signal
            if signal.original_signal.stop_loss_price and signal.original_signal.current_price:
                risk_per_dollar = abs(signal.original_signal.current_price - signal.original_signal.stop_loss_price) / signal.original_signal.current_price
                signal_risk = abs(signal.adjusted_position_size) * risk_metrics.total_portfolio_value * risk_per_dollar
            else:
                signal_risk = abs(signal.adjusted_position_size) * risk_metrics.total_portfolio_value * self.max_risk_per_trade
            
            # Check if adding this signal would exceed risk budget
            if total_new_risk + signal_risk > max_new_risk:
                # Reduce position size or reject
                available_risk = max_new_risk - total_new_risk
                if available_risk > 0 and risk_per_dollar > 0:
                    # Reduce position size
                    max_position_value = available_risk / risk_per_dollar
                    new_position_size = max_position_value / risk_metrics.total_portfolio_value
                    if signal.adjusted_position_size < 0:
                        new_position_size = -new_position_size
                    
                    signal.adjusted_position_size = new_position_size
                    signal.risk_action = RiskAction.REDUCE_SIZE
                    signal.size_reduction_reason = "Portfolio risk budget"
                    
                    total_new_risk += available_risk
                else:
                    # Reject signal
                    signal.risk_action = RiskAction.REJECT
                    signal.adjusted_position_size = 0.0
                    signal.approved = False
                    signal.rejection_reason = "Portfolio risk budget exceeded"
            else:
                total_new_risk += signal_risk
        
        return signals
    
    def _calculate_correlation_risk(self, positions: Dict) -> float:
        """Calculate portfolio correlation risk"""
        # Simplified correlation risk calculation
        # In practice, this would use actual correlation matrices
        
        if not positions:
            return 0.0
        
        # Group positions by correlation groups
        correlation_groups = {}
        for symbol, position in positions.items():
            group = self._get_correlation_group(symbol)
            if group not in correlation_groups:
                correlation_groups[group] = 0.0
            correlation_groups[group] += abs(position.get('value', 0))
        
        # Calculate correlation risk as max group exposure
        max_group_exposure = max(correlation_groups.values()) if correlation_groups else 0.0
        
        return max_group_exposure
    
    def _calculate_var(self, positions: Dict, portfolio_value: float) -> float:
        """Calculate Value at Risk (simplified)"""
        if not positions or portfolio_value <= 0:
            return 0.0
        
        # Simplified VaR calculation
        # Assume 2% daily volatility and 95% confidence level
        total_position_value = sum(abs(p.get('value', 0)) for p in positions.values())
        daily_volatility = 0.02  # 2% daily volatility assumption
        confidence_level = 1.96  # 95% confidence level (z-score)
        
        var = total_position_value * daily_volatility * confidence_level
        return var
    
    def _determine_risk_level(self, risk_utilization: float, 
                            concentration_risk: float, leverage: float) -> RiskLevel:
        """Determine overall portfolio risk level"""
        
        # Risk utilization thresholds
        if risk_utilization >= 1.0:
            return RiskLevel.EXCESSIVE
        elif risk_utilization >= 0.85:
            return RiskLevel.CRITICAL
        elif risk_utilization >= 0.60:
            return RiskLevel.HIGH
        elif risk_utilization >= 0.30:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _calculate_sector_risk(self, symbol: str, position_size: float) -> float:
        """Calculate sector concentration risk for a symbol"""
        sector = self._get_sector(symbol)
        current_sector_exposure = self.sector_exposures.get(sector, 0.0)
        new_exposure = current_sector_exposure + abs(position_size)
        
        # Risk increases exponentially as we approach limits
        risk_ratio = new_exposure / self.max_sector_exposure
        return min(1.0, risk_ratio ** 2)
    
    def _calculate_signal_correlation_risk(self, signal: TradingSignal, 
                                         risk_metrics: RiskMetrics) -> float:
        """Calculate correlation risk for a specific signal"""
        correlation_group = self._get_correlation_group(signal.symbol)
        current_exposure = self.correlation_exposures.get(correlation_group, 0.0)
        new_exposure = current_exposure + abs(signal.suggested_position_size)
        
        risk_ratio = new_exposure / self.max_correlation_exposure
        return min(1.0, risk_ratio)
    
    def _apply_sector_limits(self, symbol: str, position_size: float) -> float:
        """Apply sector exposure limits to position size"""
        sector = self._get_sector(symbol)
        current_exposure = self.sector_exposures.get(sector, 0.0)
        available_exposure = self.max_sector_exposure - current_exposure
        
        if abs(position_size) > available_exposure:
            return available_exposure if position_size > 0 else -available_exposure
        
        return position_size
    
    def _apply_correlation_limits(self, signal: TradingSignal, position_size: float) -> float:
        """Apply correlation limits to position size"""
        correlation_group = self._get_correlation_group(signal.symbol)
        current_exposure = self.correlation_exposures.get(correlation_group, 0.0)
        available_exposure = self.max_correlation_exposure - current_exposure
        
        if abs(position_size) > available_exposure:
            return available_exposure if position_size > 0 else -available_exposure
        
        return position_size
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol"""
        return self.sector_mapping.get(symbol, 'UNKNOWN')
    
    def _get_correlation_group(self, symbol: str) -> str:
        """Get correlation group for symbol"""
        for group, symbols in self.correlation_groups.items():
            if symbol in symbols:
                return group
        return symbol  # Default to symbol itself if no group found
    
    def _log_risk_summary(self, signals: List[RiskAdjustedSignal], 
                         risk_metrics: RiskMetrics):
        """Log risk management summary"""
        
        approved = len([s for s in signals if s.approved])
        rejected = len([s for s in signals if not s.approved])
        reduced = len([s for s in signals if s.risk_action == RiskAction.REDUCE_SIZE])
        
        self.logger.info(f"Risk Management Summary:")
        self.logger.info(f"  Portfolio Risk Level: {risk_metrics.risk_level.value.upper()}")
        self.logger.info(f"  Risk Utilization: {risk_metrics.risk_utilization:.1%}")
        self.logger.info(f"  Signals: {approved} approved, {reduced} reduced, {rejected} rejected")
        
        if signals:
            avg_risk_score = np.mean([s.risk_score for s in signals])
            self.logger.info(f"  Average Risk Score: {avg_risk_score:.2f}")
    
    def get_risk_statistics(self) -> Dict:
        """Get comprehensive risk management statistics"""
        
        if not self.last_risk_assessment:
            return {'status': 'No risk assessment available'}
        
        risk_metrics = self.last_risk_assessment
        
        return {
            'risk_level': risk_metrics.risk_level.value,
            'portfolio_value': risk_metrics.total_portfolio_value,
            'risk_utilization': risk_metrics.risk_utilization,
            'portfolio_utilization': risk_metrics.portfolio_utilization,
            'concentration_risk': risk_metrics.concentration_risk,
            'correlation_risk': risk_metrics.correlation_risk,
            'leverage': risk_metrics.leverage,
            'daily_pnl': risk_metrics.daily_pnl,
            'daily_var': risk_metrics.daily_var,
            'max_drawdown': risk_metrics.max_drawdown,
            'cash_available': risk_metrics.cash_available,
            'consecutive_losses': self.consecutive_losses,
            'risk_override_active': self.risk_override_active,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_single_position': self.max_single_position,
            'max_risk_per_trade': self.max_risk_per_trade
        }
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L for risk calculations"""
        self.daily_pnl = pnl
        
        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
    
    def emergency_risk_override(self, active: bool = True):
        """Emergency risk override (use with caution)"""
        self.risk_override_active = active
        if active:
            self.logger.warning("🚨 EMERGENCY RISK OVERRIDE ACTIVATED")
        else:
            self.logger.info("Risk override deactivated")
    
    def update_sector_exposures(self, portfolio_manager=None):
        """Update sector exposure tracking"""
        if not portfolio_manager:
            return
        
        positions = portfolio_manager.get_all_positions()
        sector_totals = {}
        
        for symbol, position in positions.items():
            sector = self._get_sector(symbol)
            position_value = abs(position.get('value', 0))
            
            if sector not in sector_totals:
                sector_totals[sector] = 0.0
            sector_totals[sector] += position_value
        
        # Convert to percentages
        total_portfolio = portfolio_manager.get_total_value()
        if total_portfolio > 0:
            self.sector_exposures = {
                sector: value / total_portfolio 
                for sector, value in sector_totals.items()
            }
    
    def update_correlation_exposures(self, portfolio_manager=None):
        """Update correlation exposure tracking"""
        if not portfolio_manager:
            return
        
        positions = portfolio_manager.get_all_positions()
        correlation_totals = {}
        
        for symbol, position in positions.items():
            group = self._get_correlation_group(symbol)
            position_value = abs(position.get('value', 0))
            
            if group not in correlation_totals:
                correlation_totals[group] = 0.0
            correlation_totals[group] += position_value
        
        # Convert to percentages
        total_portfolio = portfolio_manager.get_total_value()
        if total_portfolio > 0:
            self.correlation_exposures = {
                group: value / total_portfolio 
                for group, value in correlation_totals.items()
            }

# Testing and integration functions
async def test_risk_manager_integration():
    """Test risk manager with signal generator integration"""
    from signal_generator import SignalGenerator, TradingSignal, SignalType, SignalPriority
    from cache_manager import CacheManager
    from rsi_calculator import RSICalculator
    import random
    
    print("=== RISK MANAGER INTEGRATION TEST ===")
    print()
    
    # Configuration with risk management settings
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
        'rsi_period': 14,
        'rsi': {
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'extreme_oversold': 20,
            'extreme_overbought': 80,
            'volume_confirmation': True
        },
        'min_signal_strength': 0.4,
        'min_confidence': 0.5,
        'max_risk_per_trade': 0.02,
        'max_portfolio_risk': 0.06,
        'max_single_position': 0.10,
        'max_sector_exposure': 0.25,
        'sector_mapping': {
            'AAPL': 'TECHNOLOGY',
            'GOOGL': 'TECHNOLOGY', 
            'MSFT': 'TECHNOLOGY',
            'TSLA': 'AUTOMOTIVE',
            'NVDA': 'TECHNOLOGY'
        },
        'correlation_groups': {
            'BIG_TECH': ['AAPL', 'GOOGL', 'MSFT'],
            'AI_CHIPS': ['NVDA'],
            'EV_AUTO': ['TSLA']
        }
    }
    
    # Initialize components
    cache_manager = CacheManager(config)
    rsi_calculator = RSICalculator(config)
    signal_generator = SignalGenerator(config)
    risk_manager = RiskManager(config)
    
    print("Creating market conditions to generate multiple signals...")
    
    # Create multiple signals across different symbols
    base_prices = {'AAPL': 150.0, 'GOOGL': 2500.0, 'MSFT': 300.0, 'TSLA': 200.0, 'NVDA': 400.0}
    
    # Phase 1: Create oversold conditions across multiple symbols
    for cycle in range(20):
        for symbol in config['symbols']:
            # Create varying oversold conditions
            if symbol in ['AAPL', 'GOOGL', 'MSFT']:  # Tech stocks decline together
                change_pct = random.uniform(-0.04, -0.01)
            elif symbol == 'TSLA':  # TSLA declines more
                change_pct = random.uniform(-0.05, -0.02)
            else:  # NVDA declines moderately
                change_pct = random.uniform(-0.03, -0.01)
            
            new_price = base_prices[symbol] * (1 + change_pct)
            base_prices[symbol] = new_price
            
            data = {
                'symbol': symbol,
                'price': new_price,
                'volume': random.randint(1000000, 3000000),
                'timestamp': datetime.now(),
                'market_state': 'REGULAR',
                'source': 'simulation'
            }
            
            await cache_manager.update_price_data(symbol, data)
        
        rsi_calculator.update_from_cache(cache_manager)
    
    # Phase 2: Create recovery to trigger multiple signals
    print("\nCreating recovery to trigger multiple signals...")
    for cycle in range(10):
        for symbol in config['symbols']:
            change_pct = random.uniform(0.01, 0.03)  # Recovery
            new_price = base_prices[symbol] * (1 + change_pct)
            base_prices[symbol] = new_price
            
            data = {
                'symbol': symbol,
                'price': new_price,
                'volume': random.randint(1500000, 4000000),  # High volume
                'timestamp': datetime.now(),
                'market_state': 'REGULAR',
                'source': 'simulation'
            }
            
            await cache_manager.update_price_data(symbol, data)
        
        rsi_calculator.update_from_cache(cache_manager)
    
    # Generate signals
    print("\n📊 Generating signals...")
    signals = await signal_generator.generate_signals(cache_manager, rsi_calculator)
    
    if signals:
        print(f"Generated {len(signals)} raw signals:")
        for signal in signals:
            print(f"  {signal}")
            print(f"    Suggested size: {signal.suggested_position_size:.2%}")
    else:
        print("⚠️ No signals generated")
        return False
    
    # Test risk management
    print(f"\n🛡️ Applying risk management...")
    
    # Test different portfolio risk scenarios
    test_scenarios = [
        {"name": "Low Risk Portfolio", "portfolio_value": 100000, "existing_risk": 0.01},
        {"name": "High Risk Portfolio", "portfolio_value": 100000, "existing_risk": 0.05},
        {"name": "Small Portfolio", "portfolio_value": 25000, "existing_risk": 0.02}
    ]
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        # Simulate portfolio state
        risk_manager.current_portfolio_value = scenario['portfolio_value']
        
        # Create mock portfolio manager
        class MockPortfolioManager:
            def __init__(self, value, risk):
                self.value = value
                self.risk = risk
            
            def get_total_value(self):
                return self.value
            
            def get_all_positions(self):
                return {}  # Empty for simplicity
            
            def get_unrealized_pnl(self):
                return 0.0
            
            def get_cash_balance(self):
                return self.value * 0.3  # 30% cash
            
            def get_daily_pnl(self):
                return 0.0
        
        mock_portfolio = MockPortfolioManager(scenario['portfolio_value'], scenario['existing_risk'])
        
        # Evaluate signals with risk management
        risk_adjusted_signals = await risk_manager.evaluate_signals(signals, mock_portfolio)
        
        # Display results
        approved = [s for s in risk_adjusted_signals if s.approved]
        rejected = [s for s in risk_adjusted_signals if not s.approved]
        reduced = [s for s in risk_adjusted_signals if s.risk_action == RiskAction.REDUCE_SIZE]
        
        print(f"Results: {len(approved)} approved, {len(reduced)} reduced, {len(rejected)} rejected")
        
        for signal in risk_adjusted_signals:
            print(f"  {signal}")
            if signal.rejection_reason:
                print(f"    Reason: {signal.rejection_reason}")
            if signal.size_reduction_reason:
                print(f"    Reduction: {signal.size_reduction_reason}")
    
    # Test extreme risk scenarios
    print(f"\n🚨 Testing extreme risk scenarios...")
    
    extreme_scenarios = [
        {"name": "Portfolio Risk Exceeded", "existing_positions": 0.08},  # 8% existing risk
        {"name": "Daily Loss Limit", "daily_pnl": -2500},  # $2,500 loss on $100k portfolio
        {"name": "Low Cash", "cash_ratio": 0.05}  # Only 5% cash
    ]
    
    for scenario in extreme_scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        # Reset risk manager
        risk_manager.current_portfolio_value = 100000
        
        if 'daily_pnl' in scenario:
            risk_manager.update_daily_pnl(scenario['daily_pnl'])
        
        class ExtremeMockPortfolioManager:
            def get_total_value(self):
                return 100000
            
            def get_all_positions(self):
                if 'existing_positions' in scenario:
                    return {
                        'EXISTING': {'value': 50000, 'risk': scenario['existing_positions'] * 100000}
                    }
                return {}
            
            def get_unrealized_pnl(self):
                return 0.0
            
            def get_cash_balance(self):
                if 'cash_ratio' in scenario:
                    return 100000 * scenario['cash_ratio']
                return 20000
            
            def get_daily_pnl(self):
                return scenario.get('daily_pnl', 0.0)
        
        extreme_portfolio = ExtremeMockPortfolioManager()
        risk_adjusted_signals = await risk_manager.evaluate_signals(signals, extreme_portfolio)
        
        approved = len([s for s in risk_adjusted_signals if s.approved])
        rejected = len([s for s in risk_adjusted_signals if not s.approved])
        
        print(f"Extreme scenario results: {approved} approved, {rejected} rejected")
        
        # Show rejection reasons
        for signal in risk_adjusted_signals:
            if not signal.approved and signal.rejection_reason:
                print(f"  {signal.original_signal.symbol}: {signal.rejection_reason}")
    
    # Final statistics
    print(f"\n" + "="*60)
    print("RISK MANAGER TEST RESULTS")
    print("="*60)
    
    risk_stats = risk_manager.get_risk_statistics()
    print("Risk Manager Statistics:")
    for key, value in risk_stats.items():
        if isinstance(value, float):
            if key.endswith('_ratio') or key.endswith('utilization') or key.endswith('_risk'):
                print(f"  {key}: {value:.1%}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    return True

# Quick risk management test
async def quick_risk_test():
    """Quick test of risk management functionality"""
    from signal_generator import TradingSignal, SignalType, SignalPriority
    
    print("🛡️ Quick Risk Manager Test")
    print("=" * 40)
    
    config = {
        'max_portfolio_risk': 0.06,
        'max_single_position': 0.10,
        'max_risk_per_trade': 0.02,
        'sector_mapping': {'AAPL': 'TECHNOLOGY'},
        'correlation_groups': {}
    }
    
    risk_manager = RiskManager(config)
    
    # Create test signal
    test_signal = TradingSignal(
        symbol='AAPL',
        signal_type=SignalType.LONG_ENTRY,
        priority=SignalPriority.HIGH,
        strength=0.8,
        confidence=0.7,
        current_price=150.0,
        suggested_position_size=0.05,  # 5% position
        stop_loss_price=145.0,
        take_profit_price=160.0
    )
    
    print(f"Test signal: {test_signal}")
    print(f"Original size: {test_signal.suggested_position_size:.1%}")
    
    # Test with different risk levels
    test_cases = [
        {"name": "Normal Risk", "portfolio": 100000, "existing_risk": 0.02},
        {"name": "High Risk", "portfolio": 100000, "existing_risk": 0.055},
        {"name": "Small Portfolio", "portfolio": 10000, "existing_risk": 0.01}
    ]
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        
        class SimplePortfolio:
            def get_total_value(self):
                return case['portfolio']
            
            def get_all_positions(self):
                return {} if case['existing_risk'] == 0 else {
                    'EXISTING': {'value': case['portfolio'] * 0.5, 'risk': case['existing_risk'] * case['portfolio']}
                }
            
            def get_unrealized_pnl(self):
                return 0.0
            
            def get_cash_balance(self):
                return case['portfolio'] * 0.3
            
            def get_daily_pnl(self):
                return 0.0
        
        portfolio = SimplePortfolio()
        risk_adjusted = await risk_manager.evaluate_signals([test_signal], portfolio)
        
        if risk_adjusted:
            result = risk_adjusted[0]
            print(f"  Action: {result.risk_action.value}")
            print(f"  Adjusted size: {result.adjusted_position_size:.1%}")
            print(f"  Risk score: {result.risk_score:.2f}")
            print(f"  Approved: {result.approved}")
            
            if result.rejection_reason:
                print(f"  Rejection: {result.rejection_reason}")
            if result.size_reduction_reason:
                print(f"  Reduction: {result.size_reduction_reason}")
        else:
            print("  No result returned")
    
    print(f"\n✅ Risk management test completed")
    return True

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    print("Risk Manager Test Options:")
    print("1. Integration test (comprehensive)")
    print("2. Quick functionality test")
    print("3. Extreme scenarios test")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(test_risk_manager_integration())
    elif choice == "2":
        asyncio.run(quick_risk_test())
    elif choice == "3":
        # Run just the extreme scenarios from integration test
        asyncio.run(test_risk_manager_integration())
    else:
        print("Invalid choice")