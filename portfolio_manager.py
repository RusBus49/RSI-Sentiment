"""
Portfolio Manager - Phase 2
Comprehensive portfolio tracking and position management for RSI long-short strategy
Executes approved signals and maintains real-time portfolio state
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import random

# Import our components
from signal_generator import TradingSignal, SignalType
from risk_manager import RiskAdjustedSignal, RiskAction

class PositionStatus(Enum):
    """Position status types"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"      # Partially closed
    PENDING = "pending"      # Order placed but not filled
    FAILED = "failed"        # Order failed

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

@dataclass
class Position:
    """Individual position tracking"""
    symbol: str
    position_size: float        # Number of shares (negative for short)
    entry_price: float
    current_price: float
    entry_time: datetime
    last_update: datetime
    
    # Position metadata
    position_type: str = "" # "long" or "short"
    status: PositionStatus = PositionStatus.OPEN
    
    # Risk management
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    max_risk_amount: float = 0.0
    
    # P&L tracking
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Trading costs
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    total_fees: float = 0.0
    
    # Performance metrics
    max_favorable_excursion: float = 0.0  # Best unrealized profit
    max_adverse_excursion: float = 0.0    # Worst unrealized loss
    days_held: int = 0
    
    # Signal that created this position
    entry_signal_id: Optional[str] = None
    entry_signal_strength: float = 0.0
    entry_signal_confidence: float = 0.0
    
    def __post_init__(self):
        self.position_type = "long" if self.position_size > 0 else "short"
        self.market_value = abs(self.position_size) * self.current_price
        self.update_pnl()
    
    def update_price(self, new_price: float, timestamp: datetime = None):
        """Update position with new market price"""
        self.current_price = new_price
        self.last_update = timestamp or datetime.now()
        self.market_value = abs(self.position_size) * new_price
        self.update_pnl()
        self.update_excursions()
        self.days_held = (self.last_update - self.entry_time).days
    
    def update_pnl(self):
        """Calculate and update P&L metrics"""
        if self.position_size == 0:
            self.unrealized_pnl = 0.0
            self.unrealized_pnl_percent = 0.0
        else:
            # Calculate unrealized P&L
            if self.position_size > 0:  # Long position
                self.unrealized_pnl = (self.current_price - self.entry_price) * self.position_size
            else:  # Short position
                self.unrealized_pnl = (self.entry_price - self.current_price) * abs(self.position_size)
            
            # Calculate percentage return
            if self.entry_price > 0:
                if self.position_size > 0:  # Long
                    self.unrealized_pnl_percent = (self.current_price - self.entry_price) / self.entry_price
                else:  # Short
                    self.unrealized_pnl_percent = (self.entry_price - self.current_price) / self.entry_price
            
            # Total P&L including realized
            self.total_pnl = self.unrealized_pnl + self.realized_pnl - self.total_fees
    
    def update_excursions(self):
        """Update maximum favorable and adverse excursions"""
        current_pnl = self.unrealized_pnl
        
        # Update max favorable excursion (best profit)
        if current_pnl > self.max_favorable_excursion:
            self.max_favorable_excursion = current_pnl
        
        # Update max adverse excursion (worst loss)
        if current_pnl < self.max_adverse_excursion:
            self.max_adverse_excursion = current_pnl
    
    def close_position(self, exit_price: float, exit_time: datetime = None, 
                      commission: float = 0.0) -> float:
        """Close the position and return realized P&L"""
        if self.status == PositionStatus.CLOSED:
            return 0.0
        
        exit_time = exit_time or datetime.now()
        
        # Calculate final P&L
        if self.position_size > 0:  # Long position
            realized_pnl = (exit_price - self.entry_price) * self.position_size
        else:  # Short position
            realized_pnl = (self.entry_price - exit_price) * abs(self.position_size)
        
        # Update position state
        self.realized_pnl += realized_pnl
        self.exit_commission = commission
        self.total_fees = self.entry_commission + self.exit_commission
        self.total_pnl = self.realized_pnl - self.total_fees
        
        self.status = PositionStatus.CLOSED
        self.position_size = 0
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_percent = 0.0
        self.last_update = exit_time
        
        return realized_pnl
    
    def partial_close(self, close_size: float, exit_price: float, 
                     exit_time: datetime = None, commission: float = 0.0) -> float:
        """Partially close the position"""
        if self.status == PositionStatus.CLOSED or close_size == 0:
            return 0.0
        
        exit_time = exit_time or datetime.now()
        
        # Ensure we don't close more than we have
        max_close = abs(self.position_size)
        actual_close = min(abs(close_size), max_close)
        
        if self.position_size < 0:  # Short position
            actual_close = -actual_close
        
        # Calculate realized P&L for closed portion
        if self.position_size > 0:  # Long position
            realized_pnl = (exit_price - self.entry_price) * actual_close
        else:  # Short position
            realized_pnl = (self.entry_price - exit_price) * abs(actual_close)
        
        # Update position
        self.position_size -= actual_close
        self.realized_pnl += realized_pnl
        self.total_fees += commission
        self.last_update = exit_time
        
        # Update status
        if abs(self.position_size) < 0.01:  # Essentially closed
            self.status = PositionStatus.CLOSED
            self.position_size = 0
        else:
            self.status = PositionStatus.PARTIAL
        
        # Recalculate unrealized P&L for remaining position
        self.update_pnl()
        
        return realized_pnl
    
    def is_stop_loss_triggered(self) -> bool:
        """Check if stop loss should be triggered"""
        if not self.stop_loss_price or self.status != PositionStatus.OPEN:
            return False
        
        if self.position_size > 0:  # Long position
            return self.current_price <= self.stop_loss_price
        else:  # Short position
            return self.current_price >= self.stop_loss_price
    
    def is_take_profit_triggered(self) -> bool:
        """Check if take profit should be triggered"""
        if not self.take_profit_price or self.status != PositionStatus.OPEN:
            return False
        
        if self.position_size > 0:  # Long position
            return self.current_price >= self.take_profit_price
        else:  # Short position
            return self.current_price <= self.take_profit_price
    
    def get_risk_amount(self) -> float:
        """Get current risk amount (potential loss to stop loss)"""
        if not self.stop_loss_price or self.position_size == 0:
            return 0.0
        
        if self.position_size > 0:  # Long position
            risk_per_share = max(0, self.entry_price - self.stop_loss_price)
        else:  # Short position
            risk_per_share = max(0, self.stop_loss_price - self.entry_price)
        
        return risk_per_share * abs(self.position_size)
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'entry_time': self.entry_time.isoformat(),
            'last_update': self.last_update.isoformat(),
            'position_type': self.position_type,
            'status': self.status.value,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'max_favorable_excursion': self.max_favorable_excursion,
            'max_adverse_excursion': self.max_adverse_excursion,
            'days_held': self.days_held,
            'entry_signal_strength': self.entry_signal_strength,
            'entry_signal_confidence': self.entry_signal_confidence
        }

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: float
    cash_balance: float
    total_invested: float
    total_positions: int
    
    # P&L metrics
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float
    daily_pnl: float
    
    # Performance ratios
    total_return_percent: float
    daily_return_percent: float
    portfolio_utilization: float
    
    # Risk metrics
    total_risk_exposure: float
    max_single_position_risk: float
    portfolio_beta: float = 1.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Position breakdown
    long_positions: int = 0
    short_positions: int = 0
    long_value: float = 0.0
    short_value: float = 0.0
    
    # Sector/correlation exposure
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    correlation_exposure: Dict[str, float] = field(default_factory=dict)

class PortfolioManager:
    """Comprehensive portfolio management system"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Portfolio initialization
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.cash_balance = self.initial_capital
        self.total_value = self.initial_capital
        
        # Position tracking
        self.positions = {}  # symbol -> Position
        self.closed_positions = []  # List of closed positions for analysis
        
        # Performance tracking
        self.daily_values = []  # Track daily portfolio values
        self.daily_returns = []  # Track daily returns
        self.max_portfolio_value = self.initial_capital
        self.max_drawdown = 0.0
        
        # Trading costs
        self.commission_per_trade = config.get('commission_per_trade', 1.0)
        self.commission_percent = config.get('commission_percent', 0.0)
        
        # Position sizing
        self.min_position_value = config.get('min_position_value', 1000.0)
        self.max_positions = config.get('max_positions', 20)
        
        # Performance metrics
        self.start_date = datetime.now()
        self.last_update = datetime.now()
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commissions = 0.0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Portfolio Manager initialized:")
        self.logger.info(f"  Initial capital: ${self.initial_capital:,.2f}")
        self.logger.info(f"  Commission per trade: ${self.commission_per_trade:.2f}")
        self.logger.info(f"  Max positions: {self.max_positions}")
    
    async def execute_signal(self, risk_adjusted_signal: RiskAdjustedSignal, 
                           current_price: float) -> bool:
        """Execute an approved risk-adjusted signal"""
        
        if not risk_adjusted_signal.approved:
            self.logger.warning(f"Attempted to execute unapproved signal for {risk_adjusted_signal.original_signal.symbol}")
            return False
        
        signal = risk_adjusted_signal.original_signal
        adjusted_size = risk_adjusted_signal.adjusted_position_size
        
        # Calculate position details
        position_value = abs(adjusted_size) * self.total_value
        required_cash = position_value
        
        # Check cash availability
        if required_cash > self.cash_balance:
            self.logger.warning(f"Insufficient cash for {signal.symbol}: need ${required_cash:.2f}, have ${self.cash_balance:.2f}")
            return False
        
        # Calculate commission
        commission = self.commission_per_trade + (position_value * self.commission_percent)
        
        # Calculate position size in shares
        shares = position_value / current_price
        if adjusted_size < 0:  # Short position
            shares = -shares
        
        try:
            if signal.signal_type in [SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY]:
                return await self._open_position(signal, shares, current_price, commission, risk_adjusted_signal)
            
            elif signal.signal_type in [SignalType.LONG_EXIT, SignalType.SHORT_EXIT]:
                return await self._close_position(signal.symbol, current_price, commission)
            
            else:
                self.logger.warning(f"Unsupported signal type: {signal.signal_type}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error executing signal for {signal.symbol}: {e}")
            return False
    
    async def _open_position(self, signal: TradingSignal, shares: float, price: float, 
                           commission: float, risk_adjusted_signal: RiskAdjustedSignal) -> bool:
        """Open a new position"""
        
        symbol = signal.symbol
        
        # Check if position already exists
        if symbol in self.positions and self.positions[symbol].status == PositionStatus.OPEN:
            self.logger.warning(f"Position already exists for {symbol}")
            return False
        
        # Create new position
        position = Position(
            symbol=symbol,
            position_size=shares,
            entry_price=price,
            current_price=price,
            entry_time=datetime.now(),
            last_update=datetime.now(),
            stop_loss_price=signal.stop_loss_price,
            take_profit_price=signal.take_profit_price,
            entry_commission=commission,
            entry_signal_strength=signal.strength,
            entry_signal_confidence=signal.confidence,
            entry_signal_id=f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Calculate risk amount
        position.max_risk_amount = position.get_risk_amount()
        
        # Update portfolio
        position_value = abs(shares) * price
        self.cash_balance -= (position_value + commission)
        self.total_commissions += commission
        self.positions[symbol] = position
        self.total_trades += 1
        
        # Update portfolio value
        await self._update_portfolio_value()
        
        self.logger.info(f"📈 Opened {position.position_type} position: {symbol}")
        self.logger.info(f"   Size: {shares:.2f} shares @ ${price:.2f}")
        self.logger.info(f"   Value: ${position_value:.2f}")
        self.logger.info(f"   Stop Loss: ${signal.stop_loss_price:.2f}")
        self.logger.info(f"   Take Profit: ${signal.take_profit_price:.2f}")
        
        return True
    
    async def _close_position(self, symbol: str, exit_price: float, commission: float) -> bool:
        """Close an existing position"""
        
        if symbol not in self.positions:
            self.logger.warning(f"No position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        
        if position.status != PositionStatus.OPEN:
            self.logger.warning(f"Position for {symbol} is not open (status: {position.status})")
            return False
        
        # Close the position
        realized_pnl = position.close_position(exit_price, datetime.now(), commission)
        
        # Update portfolio
        position_value = abs(position.position_size) * exit_price
        self.cash_balance += position_value - commission
        self.total_commissions += commission
        
        # Track performance
        if realized_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        # Update portfolio value
        await self._update_portfolio_value()
        
        self.logger.info(f"📉 Closed {position.position_type} position: {symbol}")
        self.logger.info(f"   Exit: ${exit_price:.2f}")
        self.logger.info(f"   P&L: ${realized_pnl:.2f}")
        self.logger.info(f"   Days held: {position.days_held}")
        
        return True
    
    async def update_positions(self, market_data: Dict):
        """Update all positions with current market data"""
        
        updated_count = 0
        
        for symbol, position in self.positions.items():
            if symbol in market_data:
                data = market_data[symbol]
                new_price = data.get('price')
                timestamp = data.get('timestamp', datetime.now())
                
                if new_price and new_price > 0:
                    position.update_price(new_price, timestamp)
                    updated_count += 1
        
        if updated_count > 0:
            await self._update_portfolio_value()
            await self._check_stop_losses_and_take_profits(market_data)
        
        return updated_count
    
    async def _check_stop_losses_and_take_profits(self, market_data: Dict):
        """Check and execute stop losses and take profits"""
        
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if position.status != PositionStatus.OPEN:
                continue
            
            current_price = position.current_price
            
            # Check stop loss
            if position.is_stop_loss_triggered():
                self.logger.warning(f"🛑 Stop loss triggered for {symbol} at ${current_price:.2f}")
                positions_to_close.append((symbol, current_price, "stop_loss"))
            
            # Check take profit
            elif position.is_take_profit_triggered():
                self.logger.info(f"💰 Take profit triggered for {symbol} at ${current_price:.2f}")
                positions_to_close.append((symbol, current_price, "take_profit"))
        
        # Execute closes
        for symbol, price, reason in positions_to_close:
            commission = self.commission_per_trade + (abs(self.positions[symbol].position_size) * price * self.commission_percent)
            await self._close_position(symbol, price, commission)
    
    async def _update_portfolio_value(self):
        """Update total portfolio value and metrics"""
        
        # Calculate total position values
        total_position_value = 0.0
        total_unrealized_pnl = 0.0
        
        for position in self.positions.values():
            if position.status == PositionStatus.OPEN:
                total_position_value += position.market_value
                total_unrealized_pnl += position.unrealized_pnl
        
        # Update portfolio value
        old_value = self.total_value
        self.total_value = self.cash_balance + total_position_value
        
        # Track maximum value and drawdown
        if self.total_value > self.max_portfolio_value:
            self.max_portfolio_value = self.total_value
        
        current_drawdown = (self.max_portfolio_value - self.total_value) / self.max_portfolio_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Calculate daily return
        if old_value > 0:
            daily_return = (self.total_value - old_value) / old_value
            self.daily_returns.append(daily_return)
        
        # Record daily value
        self.daily_values.append({
            'date': datetime.now(),
            'total_value': self.total_value,
            'cash': self.cash_balance,
            'positions_value': total_position_value,
            'unrealized_pnl': total_unrealized_pnl
        })
        
        # Keep only last 252 days (1 trading year)
        if len(self.daily_values) > 252:
            self.daily_values = self.daily_values[-252:]
            self.daily_returns = self.daily_returns[-252:]
        
        self.last_update = datetime.now()
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get comprehensive portfolio metrics"""
        
        # Calculate position statistics
        total_positions = len([p for p in self.positions.values() if p.status == PositionStatus.OPEN])
        long_positions = len([p for p in self.positions.values() if p.status == PositionStatus.OPEN and p.position_size > 0])
        short_positions = total_positions - long_positions
        
        # Calculate values
        total_position_value = sum(p.market_value for p in self.positions.values() if p.status == PositionStatus.OPEN)
        long_value = sum(p.market_value for p in self.positions.values() if p.status == PositionStatus.OPEN and p.position_size > 0)
        short_value = total_position_value - long_value
        
        # Calculate P&L
        total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values() if p.status == PositionStatus.OPEN)
        total_realized_pnl = sum(p.realized_pnl for p in self.closed_positions)
        total_pnl = total_unrealized_pnl + total_realized_pnl - self.total_commissions
        
        # Calculate returns
        total_return = (self.total_value - self.initial_capital) / self.initial_capital
        daily_return = self.daily_returns[-1] if self.daily_returns else 0.0
        
        # Calculate risk metrics
        total_risk_exposure = sum(p.get_risk_amount() for p in self.positions.values() if p.status == PositionStatus.OPEN)
        max_single_risk = max([p.get_risk_amount() for p in self.positions.values() if p.status == PositionStatus.OPEN], default=0.0)
        
        # Portfolio utilization
        portfolio_utilization = total_position_value / self.total_value if self.total_value > 0 else 0.0
        
        # Calculate Sharpe ratio (simplified)
        if len(self.daily_returns) > 10:
            avg_return = np.mean(self.daily_returns)
            return_std = np.std(self.daily_returns)
            sharpe_ratio = (avg_return / return_std) * np.sqrt(252) if return_std > 0 else 0.0  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Daily P&L
        daily_pnl = (self.total_value - self.daily_values[-2]['total_value']) if len(self.daily_values) > 1 else 0.0
        
        return PortfolioMetrics(
            total_value=self.total_value,
            cash_balance=self.cash_balance,
            total_invested=total_position_value,
            total_positions=total_positions,
            total_unrealized_pnl=total_unrealized_pnl,
            total_realized_pnl=total_realized_pnl,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            total_return_percent=total_return,
            daily_return_percent=daily_return,
            portfolio_utilization=portfolio_utilization,
            total_risk_exposure=total_risk_exposure,
            max_single_position_risk=max_single_risk,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.max_drawdown,
            long_positions=long_positions,
            short_positions=short_positions,
            long_value=long_value,
            short_value=short_value
        )
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all positions in format expected by risk manager"""
        positions = {}
        for symbol, position in self.positions.items():
            if position.status == PositionStatus.OPEN:
                positions[symbol] = {
                    'value': position.market_value,
                    'risk': position.get_risk_amount(),
                    'pnl': position.unrealized_pnl,
                    'size': position.position_size,
                    'type': position.position_type
                }
        return positions
    
    def get_position_size(self, symbol: str) -> float:
        """Get position size for symbol (for risk manager)"""
        if symbol in self.positions and self.positions[symbol].status == PositionStatus.OPEN:
            return self.positions[symbol].position_size
        return 0.0
    
    def get_total_value(self) -> float:
        """Get total portfolio value"""
        return self.total_value
    
    def get_cash_balance(self) -> float:
        """Get available cash balance"""
        return self.cash_balance
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L"""
        return sum(p.unrealized_pnl for p in self.positions.values() if p.status == PositionStatus.OPEN)
    
    def get_daily_pnl(self) -> float:
        """Get daily P&L"""
        if len(self.daily_values) > 1:
            return self.daily_values[-1]['total_value'] - self.daily_values[-2]['total_value']
        return 0.0
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        metrics = self.get_portfolio_metrics()
        
        # Calculate additional statistics
        win_rate = (self.winning_trades / max(1, self.winning_trades + self.losing_trades)) * 100
        avg_trade_pnl = metrics.total_realized_pnl / max(1, self.total_trades)
        
        # Calculate position statistics
        open_positions = [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
        avg_holding_period = np.mean([p.days_held for p in open_positions]) if open_positions else 0
        
        return {
            'portfolio_value': self.total_value,
            'initial_capital': self.initial_capital,
            'total_return_percent': metrics.total_return_percent * 100,
            'total_pnl': metrics.total_pnl,
            'unrealized_pnl': metrics.total_unrealized_pnl,
            'realized_pnl': metrics.total_realized_pnl,
            'cash_balance': self.cash_balance,
            'positions_count': metrics.total_positions,
            'long_positions': metrics.long_positions,
            'short_positions': metrics.short_positions,
            'portfolio_utilization': metrics.portfolio_utilization * 100,
            'max_drawdown': self.max_drawdown * 100,
            'sharpe_ratio': metrics.sharpe_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_commissions': self.total_commissions,
            'max_portfolio_value': self.max_portfolio_value,
            'max_drawdown': self.max_drawdown,
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            'closed_positions': [pos.to_dict() for pos in self.closed_positions],
            'daily_values': self.daily_values[-30:] if len(self.daily_values) > 30 else self.daily_values,  # Last 30 days
            'performance_summary': self.get_performance_summary()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            self.logger.info(f"Portfolio state saved to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving portfolio state: {e}")
            return False
    
    def load_portfolio_state(self, filename: str):
        """Load portfolio state from file"""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            # Restore basic properties
            self.initial_capital = state.get('initial_capital', 100000.0)
            self.cash_balance = state.get('cash_balance', self.initial_capital)
            self.total_value = state.get('total_value', self.initial_capital)
            self.total_trades = state.get('total_trades', 0)
            self.winning_trades = state.get('winning_trades', 0)
            self.losing_trades = state.get('losing_trades', 0)
            self.total_commissions = state.get('total_commissions', 0.0)
            self.max_portfolio_value = state.get('max_portfolio_value', self.initial_capital)
            self.max_drawdown = state.get('max_drawdown', 0.0)
            
            # Restore dates
            if 'start_date' in state:
                self.start_date = datetime.fromisoformat(state['start_date'])
            if 'last_update' in state:
                self.last_update = datetime.fromisoformat(state['last_update'])
            
            # Restore daily values
            self.daily_values = state.get('daily_values', [])
            
            self.logger.info(f"Portfolio state loaded from {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading portfolio state: {e}")
            return False

# Integration and testing functions
async def test_portfolio_manager_integration():
    """Test portfolio manager with signal generator and risk manager"""
    from signal_generator import SignalGenerator, TradingSignal, SignalType, SignalPriority
    from risk_manager import RiskManager
    from cache_manager import CacheManager
    from rsi_calculator import RSICalculator
    import random
    
    print("=== PORTFOLIO MANAGER INTEGRATION TEST ===")
    print()
    
    # Configuration
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        'initial_capital': 100000.0,
        'commission_per_trade': 1.0,
        'commission_percent': 0.001,  # 0.1%
        'min_position_value': 1000.0,
        'max_positions': 10,
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
        'sector_mapping': {
            'AAPL': 'TECHNOLOGY',
            'GOOGL': 'TECHNOLOGY', 
            'MSFT': 'TECHNOLOGY',
            'TSLA': 'AUTOMOTIVE'
        }
    }
    
    # Initialize all components
    cache_manager = CacheManager(config)
    rsi_calculator = RSICalculator(config)
    signal_generator = SignalGenerator(config)
    risk_manager = RiskManager(config)
    portfolio_manager = PortfolioManager(config)
    
    print(f"📊 Initial Portfolio:")
    print(f"   Capital: ${portfolio_manager.initial_capital:,.2f}")
    print(f"   Cash: ${portfolio_manager.cash_balance:,.2f}")
    print(f"   Total Value: ${portfolio_manager.total_value:,.2f}")
    print()
    
    # Simulate market data and generate trading signals
    base_prices = {'AAPL': 150.0, 'GOOGL': 2500.0, 'MSFT': 300.0, 'TSLA': 200.0}
    
    # Phase 1: Create oversold conditions and recovery
    print("📉 Creating market conditions for trading signals...")
    
    # Build RSI history first
    for cycle in range(20):
        for symbol in config['symbols']:
            # Create declining market
            change_pct = random.uniform(-0.04, -0.01)
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
    
    # Phase 2: Recovery to generate long signals
    print("📈 Creating recovery to generate long signals...")
    for cycle in range(10):
        for symbol in config['symbols']:
            change_pct = random.uniform(0.01, 0.03)  # Recovery
            new_price = base_prices[symbol] * (1 + change_pct)
            base_prices[symbol] = new_price
            
            data = {
                'symbol': symbol,
                'price': new_price,
                'volume': random.randint(1500000, 4000000),
                'timestamp': datetime.now(),
                'market_state': 'REGULAR',
                'source': 'simulation'
            }
            
            await cache_manager.update_price_data(symbol, data)
        
        rsi_calculator.update_from_cache(cache_manager)
        
        # Generate and execute signals
        signals = await signal_generator.generate_signals(cache_manager, rsi_calculator)
        
        if signals:
            print(f"\n🚨 Cycle {cycle + 1}: Generated {len(signals)} signals")
            
            # Apply risk management
            risk_adjusted_signals = await risk_manager.evaluate_signals(signals, portfolio_manager)
            
            # Execute approved signals
            for risk_signal in risk_adjusted_signals:
                if risk_signal.approved:
                    symbol = risk_signal.original_signal.symbol
                    current_price = base_prices[symbol]
                    
                    success = await portfolio_manager.execute_signal(risk_signal, current_price)
                    
                    if success:
                        print(f"   ✅ Executed: {risk_signal}")
                    else:
                        print(f"   ❌ Failed: {risk_signal.original_signal.symbol}")
                else:
                    print(f"   🚫 Rejected: {risk_signal}")
            
            # Update positions with current prices
            market_data = {symbol: {'price': price, 'timestamp': datetime.now()} 
                          for symbol, price in base_prices.items()}
            await portfolio_manager.update_positions(market_data)
            
            # Display portfolio status
            metrics = portfolio_manager.get_portfolio_metrics()
            print(f"   💰 Portfolio: ${metrics.total_value:,.2f} | P&L: ${metrics.total_pnl:,.2f} | Positions: {metrics.total_positions}")
    
    # Phase 3: Continue market simulation to test position management
    print(f"\n📊 Continuing simulation to test position management...")
    
    for cycle in range(20):
        # Random price movements
        for symbol in config['symbols']:
            change_pct = random.uniform(-0.02, 0.02)  # Random walk
            new_price = base_prices[symbol] * (1 + change_pct)
            base_prices[symbol] = new_price
            
            data = {
                'symbol': symbol,
                'price': new_price,
                'volume': random.randint(1000000, 2000000),
                'timestamp': datetime.now(),
                'market_state': 'REGULAR',
                'source': 'simulation'
            }
            
            await cache_manager.update_price_data(symbol, data)
        
        # Update positions
        market_data = {symbol: {'price': price, 'timestamp': datetime.now()} 
                      for symbol, price in base_prices.items()}
        updated = await portfolio_manager.update_positions(market_data)
        
        # Check for signals periodically
        if cycle % 5 == 0:
            rsi_calculator.update_from_cache(cache_manager)
            signals = await signal_generator.generate_signals(cache_manager, rsi_calculator, portfolio_manager)
            
            if signals:
                risk_adjusted_signals = await risk_manager.evaluate_signals(signals, portfolio_manager)
                
                for risk_signal in risk_adjusted_signals:
                    if risk_signal.approved:
                        symbol = risk_signal.original_signal.symbol
                        current_price = base_prices[symbol]
                        await portfolio_manager.execute_signal(risk_signal, current_price)
        
        # Show periodic updates
        if cycle % 10 == 9:
            metrics = portfolio_manager.get_portfolio_metrics()
            print(f"   Update {cycle + 1}: ${metrics.total_value:,.2f} | P&L: ${metrics.total_pnl:,.2f} | Positions: {metrics.total_positions}")
    
    # Final results
    print(f"\n" + "="*60)
    print("PORTFOLIO MANAGER TEST RESULTS")
    print("="*60)
    
    performance = portfolio_manager.get_performance_summary()
    
    print(f"📈 Portfolio Performance:")
    print(f"   Initial Capital: ${performance['initial_capital']:,.2f}")
    print(f"   Final Value: ${performance['portfolio_value']:,.2f}")
    print(f"   Total Return: {performance['total_return_percent']:+.2f}%")
    print(f"   Total P&L: ${performance['total_pnl']:+,.2f}")
    print(f"   Unrealized P&L: ${performance['unrealized_pnl']:+,.2f}")
    print(f"   Realized P&L: ${performance['realized_pnl']:+,.2f}")
    
    print(f"\n📊 Trading Statistics:")
    print(f"   Total Trades: {performance['total_trades']}")
    print(f"   Winning Trades: {performance['winning_trades']}")
    print(f"   Losing Trades: {performance['losing_trades']}")
    print(f"   Win Rate: {performance['win_rate']:.1f}%")
    print(f"   Avg Trade P&L: ${performance['avg_trade_pnl']:+.2f}")
    print(f"   Total Commissions: ${performance['total_commissions']:.2f}")
    
    print(f"\n🎯 Risk Metrics:")
    print(f"   Portfolio Utilization: {performance['portfolio_utilization']:.1f}%")
    print(f"   Max Drawdown: {performance['max_drawdown']:.2f}%")
    print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"   Risk Exposure: ${performance['risk_exposure']:,.2f}")
    
    print(f"\n📋 Current Positions:")
    for symbol, position in portfolio_manager.positions.items():
        if position.status == PositionStatus.OPEN:
            print(f"   {symbol}: {position.position_type} | Size: {position.position_size:.0f} shares")
            print(f"      Value: ${position.market_value:,.2f} | P&L: ${position.unrealized_pnl:+,.2f} ({position.unrealized_pnl_percent:+.1%})")
            print(f"      Entry: ${position.entry_price:.2f} | Current: ${position.current_price:.2f}")
            print(f"      Stop: ${position.stop_loss_price:.2f} | Target: ${position.take_profit_price:.2f}")
    
    # Save portfolio state
    portfolio_manager.save_portfolio_state("test_portfolio_state.json")
    
    return performance['total_return_percent'] != 0  # Success if we had some trading activity

# Quick portfolio test
async def quick_portfolio_test():
    """Quick test of portfolio functionality"""
    print("💼 Quick Portfolio Manager Test")
    print("=" * 40)
    
    config = {
        'initial_capital': 50000.0,
        'commission_per_trade': 1.0,
        'commission_percent': 0.0,
        'max_positions': 5
    }
    
    portfolio = PortfolioManager(config)
    
    # Create a simple test signal
    from signal_generator import TradingSignal, SignalType, SignalPriority
    from risk_manager import RiskAdjustedSignal, RiskAction
    
    test_signal = TradingSignal(
        symbol='AAPL',
        signal_type=SignalType.LONG_ENTRY,
        priority=SignalPriority.HIGH,
        strength=0.8,
        confidence=0.7,
        current_price=150.0,
        suggested_position_size=0.05,  # 5%
        stop_loss_price=145.0,
        take_profit_price=160.0
    )
    
    risk_adjusted = RiskAdjustedSignal(
        original_signal=test_signal,
        risk_action=RiskAction.APPROVE,
        adjusted_position_size=0.05,
        risk_score=0.3,
        approved=True
    )
    
    print(f"Initial portfolio: ${portfolio.total_value:,.2f}")
    
    # Execute signal
    success = await portfolio.execute_signal(risk_adjusted, 150.0)
    print(f"Signal execution: {'✅ Success' if success else '❌ Failed'}")
    
    if success:
        print(f"Portfolio after trade: ${portfolio.total_value:,.2f}")
        print(f"Cash remaining: ${portfolio.cash_balance:,.2f}")
        print(f"Positions: {len(portfolio.positions)}")
        
        # Simulate price movement
        await portfolio.update_positions({'AAPL': {'price': 155.0, 'timestamp': datetime.now()}})
        
        metrics = portfolio.get_portfolio_metrics()
        print(f"After price update: ${metrics.total_value:,.2f}")
        print(f"Unrealized P&L: ${metrics.total_unrealized_pnl:+,.2f}")
        
        # Test stop loss
        await portfolio.update_positions({'AAPL': {'price': 144.0, 'timestamp': datetime.now()}})
        
        final_metrics = portfolio.get_portfolio_metrics()
        print(f"After stop loss: ${final_metrics.total_value:,.2f}")
        print(f"Positions remaining: {final_metrics.total_positions}")
    
    return success

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    print("Portfolio Manager Test Options:")
    print("1. Integration test (comprehensive)")
    print("2. Quick functionality test")
    print("3. Position management test")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(test_portfolio_manager_integration())
    elif choice == "2":
        asyncio.run(quick_portfolio_test())
    elif choice == "3":
        # Run a focused test on position management
        async def position_test():
            await test_portfolio_manager_integration()
        asyncio.run(position_test())
    else:
        print("Invalid choice")
    

def save_portfolio_state(self, filename: str = None):
    """Save portfolio state to file"""
    if filename is None:
        filename = f"portfolio_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    state = {
        'initial_capital': self.initial_capital,
        'cash_balance': self.cash_balance,
        'total_value': self.total_value,
        'start_date': self.start_date.isoformat(),
        'last_update': self.last_update.isoformat(),
        'total_trades': self.total_trades,
        'winning_trades': self.winning_trades,
        'losing_trades': self.losing_trades,
        'total_commissions': self.total_commissions,
        'max_portfolio_value': self.max_portfolio_value,
        'max_drawdown': self.max_drawdown,
        'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
        'closed_positions': [pos.to_dict() for pos in self.closed_positions],
        'daily_values': self.daily_values[-30:] if len(self.daily_values) > 30 else self.daily_values,  # Last 30 days
        'performance_summary': self.get_performance_summary()
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        self.logger.info(f"Portfolio state saved to {filename}")
        return True
    except Exception as e:
        self.logger.error(f"Error saving portfolio state: {e}")
        return False

def load_portfolio_state(self, filename: str):
    """Load portfolio state from file"""
    try:
        with open(filename, 'r') as f:
            state = json.load(f)
        
        # Restore basic properties
        self.initial_capital = state.get('initial_capital', 100000.0)
        self.cash_balance = state.get('cash_balance', self.initial_capital)
        self.total_value = state.get('total_value', self.initial_capital)
        self.total_trades = state.get('total_trades', 0)
        self.winning_trades = state.get('winning_trades', 0)
        self.losing_trades = state.get('losing_trades', 0)
        self.total_commissions = state.get('total_commissions', 0.0)
        self.max_portfolio_value = state.get('max_portfolio_value', self.initial_capital)
        self.max_drawdown = state.get('max_drawdown', 0.0)
        
        # Restore dates
        if 'start_date' in state:
            self.start_date = datetime.fromisoformat(state['start_date'])
        if 'last_update' in state:
            self.last_update = datetime.fromisoformat(state['last_update'])
        
        # Restore daily values
        self.daily_values = state.get('daily_values', [])
        
        self.logger.info(f"Portfolio state loaded from {filename}")
        return True
        
    except Exception as e:
        self.logger.error(f"Error loading portfolio state: {e}")
        return False

# Integration and testing functions
async def test_portfolio_manager_integration():
    """Test portfolio manager with signal generator and risk manager"""
    from signal_generator import SignalGenerator, TradingSignal, SignalType, SignalPriority
    from risk_manager import RiskManager
    from cache_manager import CacheManager
    from rsi_calculator import RSICalculator
    
    print("=== PORTFOLIO MANAGER INTEGRATION TEST ===")
    print()
    
    # Configuration
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        'initial_capital': 100000.0,
        'commission_per_trade': 1.0,
        'commission_percent': 0.001,  # 0.1%
        'min_position_value': 1000.0,
        'max_positions': 10,
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
        'sector_mapping': {
            'AAPL': 'TECHNOLOGY',
            'GOOGL': 'TECHNOLOGY', 
            'MSFT': 'TECHNOLOGY',
            'TSLA': 'AUTOMOTIVE'
        }
    }
    
    # Initialize all components
    cache_manager = CacheManager(config)
    rsi_calculator = RSICalculator(config)
    signal_generator = SignalGenerator(config)
    risk_manager = RiskManager(config)
    
    # Import PortfolioManager from the main file
    from portfolio_manager import PortfolioManager
    portfolio_manager = PortfolioManager(config)
    
    print(f"📊 Initial Portfolio:")
    print(f"   Capital: ${portfolio_manager.initial_capital:,.2f}")
    print(f"   Cash: ${portfolio_manager.cash_balance:,.2f}")
    print(f"   Total Value: ${portfolio_manager.total_value:,.2f}")
    print()
    
    # Simulate market data and generate trading signals
    base_prices = {'AAPL': 150.0, 'GOOGL': 2500.0, 'MSFT': 300.0, 'TSLA': 200.0}
    
    # Phase 1: Create oversold conditions and recovery
    print("📉 Creating market conditions for trading signals...")
    
    # Build RSI history first
    for cycle in range(20):
        for symbol in config['symbols']:
            # Create declining market
            change_pct = random.uniform(-0.04, -0.01)
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
    
    # Phase 2: Recovery to generate long signals
    print("📈 Creating recovery to generate long signals...")
    for cycle in range(10):
        for symbol in config['symbols']:
            change_pct = random.uniform(0.01, 0.03)  # Recovery
            new_price = base_prices[symbol] * (1 + change_pct)
            base_prices[symbol] = new_price
            
            data = {
                'symbol': symbol,
                'price': new_price,
                'volume': random.randint(1500000, 4000000),
                'timestamp': datetime.now(),
                'market_state': 'REGULAR',
                'source': 'simulation'
            }
            
            await cache_manager.update_price_data(symbol, data)
        
        rsi_calculator.update_from_cache(cache_manager)
        
        # Generate and execute signals
        signals = await signal_generator.generate_signals(cache_manager, rsi_calculator)
        
        if signals:
            print(f"\n🚨 Cycle {cycle + 1}: Generated {len(signals)} signals")
            
            # Apply risk management
            risk_adjusted_signals = await risk_manager.evaluate_signals(signals, portfolio_manager)
            
            # Execute approved signals
            for risk_signal in risk_adjusted_signals:
                if risk_signal.approved:
                    symbol = risk_signal.original_signal.symbol
                    current_price = base_prices[symbol]
                    
                    success = await portfolio_manager.execute_signal(risk_signal, current_price)
                    
                    if success:
                        print(f"   ✅ Executed: {risk_signal}")
                    else:
                        print(f"   ❌ Failed: {risk_signal.original_signal.symbol}")
                else:
                    print(f"   🚫 Rejected: {risk_signal}")
            
            # Update positions with current prices
            market_data = {symbol: {'price': price, 'timestamp': datetime.now()} 
                          for symbol, price in base_prices.items()}
            await portfolio_manager.update_positions(market_data)
            
            # Display portfolio status
            metrics = portfolio_manager.get_portfolio_metrics()
            print(f"   💰 Portfolio: ${metrics.total_value:,.2f} | P&L: ${metrics.total_pnl:,.2f} | Positions: {metrics.total_positions}")
    
    # Phase 3: Continue market simulation to test position management
    print(f"\n📊 Continuing simulation to test position management...")
    
    for cycle in range(20):
        # Random price movements
        for symbol in config['symbols']:
            change_pct = random.uniform(-0.02, 0.02)  # Random walk
            new_price = base_prices[symbol] * (1 + change_pct)
            base_prices[symbol] = new_price
            
            data = {
                'symbol': symbol,
                'price': new_price,
                'volume': random.randint(1000000, 2000000),
                'timestamp': datetime.now(),
                'market_state': 'REGULAR',
                'source': 'simulation'
            }
            
            await cache_manager.update_price_data(symbol, data)
        
        # Update positions
        market_data = {symbol: {'price': price, 'timestamp': datetime.now()} 
                      for symbol, price in base_prices.items()}
        updated = await portfolio_manager.update_positions(market_data)
        
        # Check for signals periodically
        if cycle % 5 == 0:
            rsi_calculator.update_from_cache(cache_manager)
            signals = await signal_generator.generate_signals(cache_manager, rsi_calculator, portfolio_manager)
            
            if signals:
                risk_adjusted_signals = await risk_manager.evaluate_signals(signals, portfolio_manager)
                
                for risk_signal in risk_adjusted_signals:
                    if risk_signal.approved:
                        symbol = risk_signal.original_signal.symbol
                        current_price = base_prices[symbol]
                        await portfolio_manager.execute_signal(risk_signal, current_price)
        
        # Show periodic updates
        if cycle % 10 == 9:
            metrics = portfolio_manager.get_portfolio_metrics()
            print(f"   Update {cycle + 1}: ${metrics.total_value:,.2f} | P&L: ${metrics.total_pnl:,.2f} | Positions: {metrics.total_positions}")
    
    # Final results
    print(f"\n" + "="*60)
    print("PORTFOLIO MANAGER TEST RESULTS")
    print("="*60)
    
    performance = portfolio_manager.get_performance_summary()
    
    print(f"📈 Portfolio Performance:")
    print(f"   Initial Capital: ${performance['initial_capital']:,.2f}")
    print(f"   Final Value: ${performance['portfolio_value']:,.2f}")
    print(f"   Total Return: {performance['total_return_percent']:+.2f}%")
    print(f"   Total P&L: ${performance['total_pnl']:+,.2f}")
    print(f"   Unrealized P&L: ${performance['unrealized_pnl']:+,.2f}")
    print(f"   Realized P&L: ${performance['realized_pnl']:+,.2f}")
    
    print(f"\n📊 Trading Statistics:")
    print(f"   Total Trades: {performance['total_trades']}")
    print(f"   Winning Trades: {performance['winning_trades']}")
    print(f"   Losing Trades: {performance['losing_trades']}")
    print(f"   Win Rate: {performance['win_rate']:.1f}%")
    print(f"   Avg Trade P&L: ${performance['avg_trade_pnl']:+.2f}")
    print(f"   Total Commissions: ${performance['total_commissions']:.2f}")
    
    print(f"\n🎯 Risk Metrics:")
    print(f"   Portfolio Utilization: {performance['portfolio_utilization']:.1f}%")
    print(f"   Max Drawdown: {performance['max_drawdown']:.2f}%")
    print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"   Risk Exposure: ${performance['risk_exposure']:,.2f}")
    
    print(f"\n📋 Current Positions:")
    for symbol, position in portfolio_manager.positions.items():
        if position.status.value == 'open':  # Use .value to access enum value
            print(f"   {symbol}: {position.position_type} | Size: {position.position_size:.0f} shares")
            print(f"      Value: ${position.market_value:,.2f} | P&L: ${position.unrealized_pnl:+,.2f} ({position.unrealized_pnl_percent:+.1%})")
            print(f"      Entry: ${position.entry_price:.2f} | Current: ${position.current_price:.2f}")
            print(f"      Stop: ${position.stop_loss_price:.2f} | Target: ${position.take_profit_price:.2f}")
    
    # Save portfolio state
    portfolio_manager.save_portfolio_state("test_portfolio_state.json")
    
    return performance['total_return_percent'] != 0  # Success if we had some trading activity

# Quick portfolio test
async def quick_portfolio_test():
    """Quick test of portfolio functionality"""
    print("💼 Quick Portfolio Manager Test")
    print("=" * 40)
    
    config = {
        'initial_capital': 50000.0,
        'commission_per_trade': 1.0,
        'commission_percent': 0.0,
        'max_positions': 5
    }
    
    # Import from main file
    from portfolio_manager import PortfolioManager
    portfolio = PortfolioManager(config)
    
    # Create a simple test signal
    from signal_generator import TradingSignal, SignalType, SignalPriority
    from risk_manager import RiskAdjustedSignal, RiskAction
    
    test_signal = TradingSignal(
        symbol='AAPL',
        signal_type=SignalType.LONG_ENTRY,
        priority=SignalPriority.HIGH,
        strength=0.8,
        confidence=0.7,
        current_price=150.0,
        suggested_position_size=0.05,  # 5%
        stop_loss_price=145.0,
        take_profit_price=160.0
    )
    
    risk_adjusted = RiskAdjustedSignal(
        original_signal=test_signal,
        risk_action=RiskAction.APPROVE,
        adjusted_position_size=0.05,
        risk_score=0.3,
        approved=True
    )
    
    print(f"Initial portfolio: ${portfolio.total_value:,.2f}")
    
    # Execute signal
    success = await portfolio.execute_signal(risk_adjusted, 150.0)
    print(f"Signal execution: {'✅ Success' if success else '❌ Failed'}")
    
    if success:
        print(f"Portfolio after trade: ${portfolio.total_value:,.2f}")
        print(f"Cash remaining: ${portfolio.cash_balance:,.2f}")
        print(f"Positions: {len(portfolio.positions)}")
        
        # Simulate price movement
        await portfolio.update_positions({'AAPL': {'price': 155.0, 'timestamp': datetime.now()}})
        
        metrics = portfolio.get_portfolio_metrics()
        print(f"After price update: ${metrics.total_value:,.2f}")
        print(f"Unrealized P&L: ${metrics.total_unrealized_pnl:+,.2f}")
        
        # Test stop loss
        await portfolio.update_positions({'AAPL': {'price': 144.0, 'timestamp': datetime.now()}})
        
        final_metrics = portfolio.get_portfolio_metrics()
        print(f"After stop loss: ${final_metrics.total_value:,.2f}")
        print(f"Positions remaining: {final_metrics.total_positions}")
    
    return success

# Position management focused test
async def position_management_test():
    """Test focused on position lifecycle management"""
    print("🎯 Position Management Test")
    print("=" * 40)
    
    config = {
        'initial_capital': 100000.0,
        'commission_per_trade': 0.0,  # No commissions for clean testing
        'commission_percent': 0.0,
        'max_positions': 10
    }
    
    from portfolio_manager import PortfolioManager
    portfolio = PortfolioManager(config)
    
    # Test 1: Open position
    from signal_generator import TradingSignal, SignalType, SignalPriority
    from risk_manager import RiskAdjustedSignal, RiskAction
    
    signal = TradingSignal(
        symbol='TSLA',
        signal_type=SignalType.LONG_ENTRY,
        priority=SignalPriority.HIGH,
        strength=0.9,
        confidence=0.8,
        current_price=200.0,
        suggested_position_size=0.10,  # 10%
        stop_loss_price=190.0,
        take_profit_price=220.0
    )
    
    risk_adjusted = RiskAdjustedSignal(
        original_signal=signal,
        risk_action=RiskAction.APPROVE,
        adjusted_position_size=0.10,
        risk_score=0.2,
        approved=True
    )
    
    print("Test 1: Opening position")
    success = await portfolio.execute_signal(risk_adjusted, 200.0)
    print(f"   Position opened: {'✅' if success else '❌'}")
    
    if success:
        position = portfolio.get_position('TSLA')
        print(f"   Shares: {position.position_size:.0f}")
        print(f"   Value: ${position.market_value:,.2f}")
        print(f"   Stop Loss: ${position.stop_loss_price:.2f}")
        print(f"   Take Profit: ${position.take_profit_price:.2f}")
    
    print(f"\nTest 2: Price movements and P&L tracking")
    test_prices = [205.0, 210.0, 195.0, 215.0, 185.0]  # Last one should trigger stop loss
    
    for i, price in enumerate(test_prices):
        await portfolio.update_positions({'TSLA': {'price': price, 'timestamp': datetime.now()}})
        
        if 'TSLA' in portfolio.positions:
            position = portfolio.positions['TSLA']
            print(f"   Price ${price:.2f}: P&L ${position.unrealized_pnl:+,.2f} ({position.unrealized_pnl_percent:+.1%})")
            
            if position.is_stop_loss_triggered():
                print(f"   🛑 Stop loss would trigger at ${price:.2f}")
            elif position.is_take_profit_triggered():
                print(f"   💰 Take profit would trigger at ${price:.2f}")
        else:
            print(f"   Price ${price:.2f}: Position closed (stop loss triggered)")
            break
    
    print(f"\nTest 3: Final portfolio state")
    metrics = portfolio.get_portfolio_metrics()
    performance = portfolio.get_performance_summary()
    
    print(f"   Portfolio Value: ${metrics.total_value:,.2f}")
    print(f"   Total P&L: ${metrics.total_pnl:+,.2f}")
    print(f"   Total Trades: {performance['total_trades']}")
    print(f"   Positions: {metrics.total_positions}")
    
    if performance['total_trades'] > 0:
        print(f"   Win Rate: {performance['win_rate']:.1f}%")
    
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Portfolio Manager Completion Test Options:")
    print("1. Integration test (comprehensive)")
    print("2. Quick functionality test")
    print("3. Position management test")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(test_portfolio_manager_integration())
    elif choice == "2":
        asyncio.run(quick_portfolio_test())
    elif choice == "3":
        asyncio.run(position_management_test())
    else:
        print("Invalid choice")

# Additional utility functions for merging with main file
def add_missing_methods_to_portfolio_manager():
    """
    Instructions for merging this file with the main portfolio_manager.py:
    
    1. Copy the save_portfolio_state method (lines 15-40)
    2. Copy the load_portfolio_state method (lines 42-70)
    3. Add the test functions at the end of the main file
    4. Update any enum references to use .value when needed
    """
    pass