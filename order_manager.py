"""
Order Manager - Phase 3
Advanced order management system for RSI long-short strategy
Handles order execution, order states, and broker integration simulation
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Import our Phase 2 components
from signal_generator import TradingSignal, SignalType
from risk_manager import RiskAdjustedSignal, RiskAction
from portfolio_manager import PortfolioManager

class OrderType(Enum):
    """Order types supported by the system"""
    MARKET = "market"                # Execute at market price immediately
    LIMIT = "limit"                  # Execute only at specified price or better
    STOP_MARKET = "stop_market"      # Market order triggered by stop price
    STOP_LIMIT = "stop_limit"        # Limit order triggered by stop price
    TRAILING_STOP = "trailing_stop"  # Stop that trails the market price
    BRACKET = "bracket"              # Entry with automatic stop/target orders

class OrderStatus(Enum):
    """Order status states"""
    PENDING = "pending"              # Order created, not yet submitted
    SUBMITTED = "submitted"          # Order sent to broker/exchange
    ACCEPTED = "accepted"            # Order accepted by broker/exchange  
    PARTIALLY_FILLED = "partially_filled"  # Order partially executed
    FILLED = "filled"                # Order completely executed
    CANCELLED = "cancelled"          # Order cancelled before execution
    REJECTED = "rejected"            # Order rejected by broker/exchange
    EXPIRED = "expired"              # Order expired (time-based)
    FAILED = "failed"                # Order failed due to system error

class OrderSide(Enum):
    """Order side (buy/sell)"""
    BUY = "buy"
    SELL = "sell"

class TimeInForce(Enum):
    """Time in force options"""
    DAY = "day"                      # Good for trading day
    GTC = "gtc"                      # Good till cancelled  
    IOC = "ioc"                      # Immediate or cancel
    FOK = "fok"                      # Fill or kill
    GTD = "gtd"                      # Good till date

@dataclass
class OrderFill:
    """Individual order fill record"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    
    @property
    def value(self) -> float:
        """Get total value of fill"""
        return abs(self.quantity) * self.price

@dataclass 
class Order:
    """Comprehensive order representation"""
    # Basic order info
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float                  # Shares to buy/sell
    
    # Pricing
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None
    trail_percent: Optional[float] = None
    
    # Order management
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    
    # Execution tracking
    filled_quantity: float = 0.0
    remaining_quantity: float = field(init=False)
    avg_fill_price: float = 0.0
    total_commission: float = 0.0
    
    # Timestamps
    created_time: datetime = field(default_factory=datetime.now)
    submitted_time: Optional[datetime] = None
    last_update_time: datetime = field(default_factory=datetime.now)
    expiry_time: Optional[datetime] = None
    
    # Related orders (for bracket orders)
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = field(default_factory=list)
    
    # Signal context
    signal_id: Optional[str] = None
    signal_strength: float = 0.0
    signal_confidence: float = 0.0
    
    # Fills
    fills: List[OrderFill] = field(default_factory=list)
    
    # Error tracking
    rejection_reason: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        self.remaining_quantity = self.quantity
        if self.time_in_force == TimeInForce.DAY and not self.expiry_time:
            # Set expiry to end of trading day (4 PM ET)
            today = datetime.now().date()
            self.expiry_time = datetime.combine(today, datetime.min.time().replace(hour=16))
            if datetime.now() > self.expiry_time:
                # If after 4 PM, set to next trading day
                self.expiry_time += timedelta(days=1)
    
    def add_fill(self, fill: OrderFill):
        """Add a fill to this order"""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.remaining_quantity = max(0, self.quantity - abs(self.filled_quantity))
        self.total_commission += fill.commission
        self.last_update_time = fill.timestamp
        
        # Update average fill price
        if self.fills:
            total_value = sum(abs(f.quantity) * f.price for f in self.fills)
            total_quantity = sum(abs(f.quantity) for f in self.fills)
            self.avg_fill_price = total_value / total_quantity if total_quantity > 0 else 0.0
        
        # Update status
        if abs(self.filled_quantity) >= abs(self.quantity):
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def cancel(self, reason: str = "User requested"):
        """Cancel the order"""
        if self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED]:
            self.status = OrderStatus.CANCELLED
            self.rejection_reason = reason
            self.last_update_time = datetime.now()
            return True
        return False
    
    def reject(self, reason: str):
        """Reject the order"""
        self.status = OrderStatus.REJECTED
        self.rejection_reason = reason
        self.last_update_time = datetime.now()
    
    def expire(self):
        """Mark order as expired"""
        if self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED]:
            self.status = OrderStatus.EXPIRED
            self.last_update_time = datetime.now()
    
    def is_active(self) -> bool:
        """Check if order is still active (can be filled)"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, 
                              OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED]
    
    def is_complete(self) -> bool:
        """Check if order is completely done (filled, cancelled, rejected, expired)"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                              OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.FAILED]
    
    def get_fill_summary(self) -> Dict:
        """Get summary of order fills"""
        if not self.fills:
            return {
                'fill_count': 0,
                'total_quantity': 0.0,
                'avg_price': 0.0,
                'total_value': 0.0,
                'total_commission': 0.0
            }
        
        total_quantity = sum(abs(f.quantity) for f in self.fills)
        total_value = sum(f.value for f in self.fills)
        
        return {
            'fill_count': len(self.fills),
            'total_quantity': total_quantity,
            'avg_price': self.avg_fill_price,
            'total_value': total_value,
            'total_commission': self.total_commission,
            'first_fill': self.fills[0].timestamp,
            'last_fill': self.fills[-1].timestamp
        }
    
    def to_dict(self) -> Dict:
        """Convert order to dictionary for serialization"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'avg_fill_price': self.avg_fill_price,
            'total_commission': self.total_commission,
            'created_time': self.created_time.isoformat(),
            'submitted_time': self.submitted_time.isoformat() if self.submitted_time else None,
            'last_update_time': self.last_update_time.isoformat(),
            'fills': [fill.__dict__ for fill in self.fills],
            'signal_strength': self.signal_strength,
            'signal_confidence': self.signal_confidence
        }

class BrokerSimulator:
    """Simulates broker/exchange behavior for testing"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Execution simulation parameters
        self.fill_probability = config.get('fill_probability', 0.95)  # 95% chance of fill
        self.partial_fill_probability = config.get('partial_fill_probability', 0.1)  # 10% chance of partial fill
        self.slippage_range = config.get('slippage_range', (0.0, 0.002))  # 0-0.2% slippage
        self.execution_delay_range = config.get('execution_delay_range', (0.1, 2.0))  # 0.1-2 second delay
        
        # Commission structure
        self.commission_per_share = config.get('commission_per_share', 0.0)
        self.commission_per_trade = config.get('commission_per_trade', 1.0)
        self.commission_percent = config.get('commission_percent', 0.001)  # 0.1%
        self.min_commission = config.get('min_commission', 1.0)
        
        # Market simulation
        self.market_volatility = config.get('market_volatility', 0.02)  # 2% daily volatility
        self.bid_ask_spread = config.get('bid_ask_spread', 0.001)  # 0.1% spread
        
        # Rejection scenarios
        self.rejection_probability = config.get('rejection_probability', 0.02)  # 2% rejection rate
        self.rejection_reasons = [
            "Insufficient buying power",
            "Market closed",
            "Symbol not tradeable", 
            "Price too far from market",
            "Position size exceeds limit"
        ]
        
        self.logger = logging.getLogger(__name__)
    
    async def execute_order(self, order: Order, current_price: float) -> bool:
        """Simulate order execution"""
        
        # Simulate execution delay
        delay = np.random.uniform(*self.execution_delay_range)
        await asyncio.sleep(delay)
        
        # Check for rejection
        if np.random.random() < self.rejection_probability:
            reason = np.random.choice(self.rejection_reasons)
            order.reject(reason)
            self.logger.warning(f"Order {order.order_id} rejected: {reason}")
            return False
        
        # Check if order should fill based on type and price
        should_fill, fill_price = self._should_order_fill(order, current_price)
        
        if not should_fill:
            order.status = OrderStatus.ACCEPTED  # Order accepted but not filled yet
            return True
        
        # Determine fill quantity (partial vs full)
        fill_quantity = order.remaining_quantity
        if np.random.random() < self.partial_fill_probability:
            # Partial fill - fill 30-80% of remaining quantity
            fill_percentage = np.random.uniform(0.3, 0.8)
            fill_quantity = order.remaining_quantity * fill_percentage
            fill_quantity = max(1, int(fill_quantity))  # At least 1 share
        
        # Calculate commission
        commission = self._calculate_commission(fill_quantity, fill_price)
        
        # Create fill
        fill = OrderFill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity if order.side == OrderSide.BUY else -fill_quantity,
            price=fill_price,
            timestamp=datetime.now(),
            commission=commission
        )
        
        # Add fill to order
        order.add_fill(fill)
        
        self.logger.info(f"Order {order.order_id} filled: {abs(fill_quantity)} shares @ ${fill_price:.2f}")
        
        return True
    
    def _should_order_fill(self, order, current_price):
        """Determine if order should fill and at what price"""
        
        # Market orders always fill (with slippage)
        if order.order_type == OrderType.MARKET:
            slippage = np.random.uniform(*self.slippage_range)
            if order.side == OrderSide.BUY:
                fill_price = current_price * (1 + slippage)
            else:
                fill_price = current_price * (1 - slippage)
            return True, fill_price
        
        # Limit orders
        elif order.order_type == OrderType.LIMIT and order.limit_price:
            if order.side == OrderSide.BUY and current_price <= order.limit_price:
                # Buy limit order fills at limit price or better
                fill_price = min(order.limit_price, current_price)
                return True, fill_price
            elif order.side == OrderSide.SELL and current_price >= order.limit_price:
                # Sell limit order fills at limit price or better
                fill_price = max(order.limit_price, current_price)
                return True, fill_price
        
        # Stop market orders
        elif order.order_type == OrderType.STOP_MARKET and order.stop_price:
            if order.side == OrderSide.BUY and current_price >= order.stop_price:
                # Buy stop triggered - execute as market order
                slippage = np.random.uniform(*self.slippage_range)
                fill_price = current_price * (1 + slippage)
                return True, fill_price
            elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                # Sell stop triggered - execute as market order
                slippage = np.random.uniform(*self.slippage_range)
                fill_price = current_price * (1 - slippage)
                return True, fill_price
        
        # Stop limit orders
        elif order.order_type == OrderType.STOP_LIMIT and order.stop_price and order.limit_price:
            triggered = False
            if order.side == OrderSide.BUY and current_price >= order.stop_price:
                triggered = True
            elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                triggered = True
            
            if triggered:
                # Now check if limit price would fill
                temp_order = Order(
                    order_id="temp", 
                    symbol=order.symbol, 
                    side=order.side, 
                    order_type=OrderType.LIMIT, 
                    quantity=order.quantity, 
                    limit_price=order.limit_price
                )
                return self._should_order_fill(temp_order, current_price)
        
        return False, 0.0
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a fill"""
        commission = 0.0
        
        # Per share commission
        commission += abs(quantity) * self.commission_per_share
        
        # Per trade commission
        commission += self.commission_per_trade
        
        # Percentage commission
        trade_value = abs(quantity) * price
        commission += trade_value * self.commission_percent
        
        # Apply minimum commission
        commission = max(commission, self.min_commission)
        
        return commission
    
    def get_order_history(self, symbol: str = None, limit: int = 100) -> List[Order]:
        """Get order history, optionally filtered by symbol"""
        history = self.order_history
        
        if symbol:
            history = [order for order in history if order.symbol == symbol]
        
        return history[-limit:] if limit else history
    
    def get_statistics(self) -> Dict:
        """Get order management statistics"""
        active_count = len([order for order in self.orders.values() if order.is_active()])
        
        # Calculate fill rate
        filled_orders = len([order for order in self.order_history if order.status == OrderStatus.FILLED])
        fill_rate = (filled_orders / max(1, self.total_orders)) * 100
        
        # Calculate average execution time for filled orders
        execution_times = []
        for order in self.order_history:
            if order.status == OrderStatus.FILLED and order.submitted_time and order.fills:
                first_fill_time = min(fill.timestamp for fill in order.fills)
                execution_time = (first_fill_time - order.submitted_time).total_seconds()
                execution_times.append(execution_time)
        
        avg_execution_time = np.mean(execution_times) if execution_times else 0.0
        
        return {
            'total_orders': self.total_orders,
            'successful_orders': self.successful_orders,
            'rejected_orders': self.rejected_orders,
            'cancelled_orders': self.cancelled_orders,
            'active_orders': active_count,
            'fill_rate_percent': fill_rate,
            'avg_execution_time_seconds': avg_execution_time,
            'orders_in_history': len(self.order_history),
            'symbols_with_active_orders': len(self.active_orders)
        }
    
    def get_symbol_statistics(self, symbol: str) -> Dict:
        """Get statistics for a specific symbol"""
        symbol_orders = [order for order in self.order_history if order.symbol == symbol]
        active_symbol_orders = self.get_active_orders(symbol)
        
        if not symbol_orders and not active_symbol_orders:
            return {'symbol': symbol, 'no_data': True}
        
        # Calculate symbol-specific metrics
        filled_orders = [order for order in symbol_orders if order.status == OrderStatus.FILLED]
        total_volume = sum(order.filled_quantity for order in filled_orders)
        total_value = sum(fill.value for order in filled_orders for fill in order.fills)
        
        return {
            'symbol': symbol,
            'total_orders': len(symbol_orders),
            'active_orders': len(active_symbol_orders),
            'filled_orders': len(filled_orders),
            'total_volume': total_volume,
            'total_value': total_value,
            'avg_order_size': total_volume / max(1, len(filled_orders))
        }


class OrderManager:
    """Advanced order management system"""
    
    def __init__(self, config: Dict, portfolio_manager: PortfolioManager):
        self.config = config
        self.portfolio_manager = portfolio_manager
        
        # Initialize broker simulator
        self.broker = BrokerSimulator(config.get('broker_simulation', {}))
        
        # Order tracking
        self.orders = {}  # order_id -> Order
        self.active_orders = {}  # symbol -> list of active orders
        self.order_history = []  # Completed orders
        
        # Order management settings
        self.max_orders_per_symbol = config.get('max_orders_per_symbol', 5)
        self.order_timeout_minutes = config.get('order_timeout_minutes', 60)
        self.retry_failed_orders = config.get('retry_failed_orders', True)
        
        # Callbacks
        self.order_update_callbacks = []
        self.fill_callbacks = []
        
        # Statistics
        self.total_orders = 0
        self.successful_orders = 0
        self.rejected_orders = 0
        self.cancelled_orders = 0
        
        # Background tasks
        self.monitor_task = None
        self.is_running = False
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Order Manager initialized")
        self.logger.info(f"  Max orders per symbol: {self.max_orders_per_symbol}")
        self.logger.info(f"  Order timeout: {self.order_timeout_minutes} minutes")
    
    async def start(self):
        """Start the order manager"""
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_orders())
        self.logger.info("Order Manager started")
    
    async def stop(self):
        """Stop the order manager"""
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Order Manager stopped")
    
    async def submit_order_from_signal(self, risk_adjusted_signal: RiskAdjustedSignal, 
                                     current_price: float) -> Optional[str]:
        """Create and submit order from risk-adjusted signal"""
        
        if not risk_adjusted_signal.approved:
            self.logger.warning(f"Cannot submit order for unapproved signal: {risk_adjusted_signal.original_signal.symbol}")
            return None
        
        signal = risk_adjusted_signal.original_signal
        
        # Determine order side
        if signal.signal_type in [SignalType.LONG_ENTRY]:
            side = OrderSide.BUY
        elif signal.signal_type in [SignalType.SHORT_ENTRY, SignalType.LONG_EXIT, SignalType.SHORT_EXIT]:
            side = OrderSide.SELL
        else:
            self.logger.warning(f"Unsupported signal type for order: {signal.signal_type}")
            return None
        
        # Calculate quantity in shares
        position_value = abs(risk_adjusted_signal.adjusted_position_size) * self.portfolio_manager.get_total_value()
        quantity = int(position_value / current_price)
        
        if quantity == 0:
            self.logger.warning(f"Order quantity is 0 for {signal.symbol}")
            return None
        
        # Create order
        order = Order(
            order_id=self._generate_order_id(),
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,  # Start with market orders for simplicity
            quantity=quantity,
            signal_id=f"{signal.symbol}_{signal.signal_type.value}",
            signal_strength=signal.strength,
            signal_confidence=signal.confidence
        )
        
        return await self.submit_order(order)
    
    async def submit_order(self, order: Order) -> Optional[str]:
        """Submit an order for execution"""
        
        # Validate order
        validation_error = self._validate_order(order)
        if validation_error:
            order.reject(validation_error)
            self.logger.error(f"Order validation failed: {validation_error}")
            return None
        
        # Check order limits per symbol
        if self._get_active_order_count(order.symbol) >= self.max_orders_per_symbol:
            order.reject("Too many active orders for symbol")
            self.logger.warning(f"Too many active orders for {order.symbol}")
            return None
        
        # Add to tracking
        self.orders[order.order_id] = order
        if order.symbol not in self.active_orders:
            self.active_orders[order.symbol] = []
        self.active_orders[order.symbol].append(order.order_id)
        
        # Update status and submit
        order.status = OrderStatus.SUBMITTED
        order.submitted_time = datetime.now()
        self.total_orders += 1
        
        self.logger.info(f"📝 Order submitted: {order.order_id}")
        self.logger.info(f"   {order.symbol} | {order.side.value.upper()} {order.quantity} @ {order.order_type.value}")
        
        # Notify callbacks
        await self._notify_order_update_callbacks(order)
        
        return order.order_id
    
    async def cancel_order(self, order_id: str, reason: str = "User requested") -> bool:
        """Cancel an active order"""
        
        if order_id not in self.orders:
            self.logger.warning(f"Cannot cancel unknown order: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if order.cancel(reason):
            self.cancelled_orders += 1
            await self._move_to_history(order)
            self.logger.info(f"❌ Order cancelled: {order_id} - {reason}")
            await self._notify_order_update_callbacks(order)
            return True
        
        return False
    
    async def cancel_symbol_orders(self, symbol: str, reason: str = "Symbol cancellation") -> int:
        """Cancel all active orders for a symbol"""
        
        cancelled_count = 0
        
        if symbol in self.active_orders:
            order_ids = self.active_orders[symbol].copy()
            for order_id in order_ids:
                if await self.cancel_order(order_id, reason):
                    cancelled_count += 1
        
        return cancelled_count
    
    async def update_market_data(self, market_data: Dict):
        """Update with current market data to trigger order executions"""
        
        for symbol, data in market_data.items():
            current_price = data.get('price')
            if not current_price or current_price <= 0:
                continue
            
            # Check orders for this symbol
            if symbol in self.active_orders:
                order_ids = self.active_orders[symbol].copy()
                
                for order_id in order_ids:
                    if order_id in self.orders:
                        order = self.orders[order_id]
                        
                        if order.is_active():
                            try:
                                # Attempt to execute order
                                success = await self.broker.execute_order(order, current_price)
                                
                                if success:
                                    await self._notify_order_update_callbacks(order)
                                    
                                    # If order is complete, process fills
                                    if order.is_complete():
                                        await self._process_completed_order(order)
                                
                            except Exception as e:
                                self.logger.error(f"Error executing order {order_id}: {e}")
                                order.status = OrderStatus.FAILED
                                order.error_message = str(e)
                                await self._move_to_history(order)
    
    async def _process_completed_order(self, order: Order):
        """Process a completed order"""
        
        if order.status == OrderStatus.FILLED and order.fills:
            # Update portfolio with fills
            for fill in order.fills:
                # Notify portfolio manager (this would integrate with your portfolio_manager)
                await self._notify_fill_callbacks(fill)
            
            self.successful_orders += 1
            self.logger.info(f"✅ Order completed: {order.order_id}")
            
            fill_summary = order.get_fill_summary()
            self.logger.info(f"   Filled {fill_summary['total_quantity']} shares @ avg ${fill_summary['avg_price']:.2f}")
            self.logger.info(f"   Total value: ${fill_summary['total_value']:.2f}, Commission: ${fill_summary['total_commission']:.2f}")
        
        elif order.status == OrderStatus.REJECTED:
            self.rejected_orders += 1
            self.logger.warning(f"🚫 Order rejected: {order.order_id} - {order.rejection_reason}")
        
        # Move to history
        await self._move_to_history(order)
    
    async def _move_to_history(self, order: Order):
        """Move completed order to history"""
        
        if order.symbol in self.active_orders:
            if order.order_id in self.active_orders[order.symbol]:
                self.active_orders[order.symbol].remove(order.order_id)
            
            # Clean up empty lists
            if not self.active_orders[order.symbol]:
                del self.active_orders[order.symbol]
        
        self.order_history.append(order)
        
        # Keep only last 1000 orders in history
        if len(self.order_history) > 1000:
            self.order_history = self.order_history[-1000:]
    
    async def _monitor_orders(self):
        """Background task to monitor order timeouts and expiry"""
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check for expired orders
                expired_orders = []
                
                for order_id, order in self.orders.items():
                    if not order.is_active():
                        continue
                    
                    # Check expiry time
                    if order.expiry_time and current_time > order.expiry_time:
                        order.expire()
                        expired_orders.append(order)
                        continue
                    
                    # Check timeout
                    if order.submitted_time:
                        minutes_since_submit = (current_time - order.submitted_time).total_seconds() / 60
                        if minutes_since_submit > self.order_timeout_minutes:
                            order.cancel("Order timeout")
                            expired_orders.append(order)
                
                # Process expired orders
                for order in expired_orders:
                    await self._process_completed_order(order)
                
                # Sleep for 1 minute before next check
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in order monitor: {e}")
                await asyncio.sleep(60)
    
    def _validate_order(self, order: Order) -> Optional[str]:
        """Validate order before submission"""
        
        if not order.symbol:
            return "Symbol is required"
        
        if order.quantity <= 0:
            return "Quantity must be positive"
        
        if order.order_type == OrderType.LIMIT and not order.limit_price:
            return "Limit price required for limit orders"
        
        if order.order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT] and not order.stop_price:
            return "Stop price required for stop orders"
        
        if order.order_type == OrderType.STOP_LIMIT and not order.limit_price:
            return "Limit price required for stop limit orders"
        
        # Check portfolio constraints
        portfolio_value = self.portfolio_manager.get_total_value()
        cash_available = self.portfolio_manager.get_cash_balance()
        
        if order.side == OrderSide.BUY:
            # Rough estimate of required cash (will be refined with actual fill price)
            estimated_cost = order.quantity * (order.limit_price or 100)  # Use limit price or estimate
            if estimated_cost > cash_available * 1.1:  # 10% buffer for slippage
                return "Insufficient cash for order"
        
        return None
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _get_active_order_count(self, symbol: str) -> int:
        """Get count of active orders for symbol"""
        if symbol not in self.active_orders:
            return 0
        return len(self.active_orders[symbol])
    
    def add_order_update_callback(self, callback: Callable):
        """Add callback for order status updates"""
        self.order_update_callbacks.append(callback)
    
    def add_fill_callback(self, callback: Callable):
        """Add callback for order fills"""
        self.fill_callbacks.append(callback)
    
    async def _notify_order_update_callbacks(self, order: Order):
        """Notify callbacks of order updates"""
        for callback in self.order_update_callbacks:
            try:
                await callback(order)
            except Exception as e:
                self.logger.error(f"Error in order update callback: {e}")
    
    async def _notify_fill_callbacks(self, fill: OrderFill):
        """Notify callbacks of order fills"""
        for callback in self.fill_callbacks:
            try:
                await callback(fill)
            except Exception as e:
                self.logger.error(f"Error in fill callback: {e}")
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: str = None) -> List[Order]:
        """Get active orders, optionally filtered by symbol"""
        active = []
        
        if symbol:
            if symbol in self.active_orders:
                for order_id in self.active_orders[symbol]:
                    if order_id in self.orders:
                        order = self.orders[order_id]
                        if order.is_active():
                            active.append(order)
        else:
            for order_id, order in self.orders.items():
                if order.is_active():
                    active.append(order)
    
        return active

    def get_order_history(self, symbol: str = None, limit: int = 100) -> List:
        """Get order history, optionally filtered by symbol"""
        history = self.order_history
        
        if symbol:
            history = [order for order in history if order.symbol == symbol]
        
        return history[-limit:] if limit else history

    def get_statistics(self) -> Dict:
        """Get order management statistics"""
        active_count = len([order for order in self.orders.values() if order.is_active()])
        
        # Calculate fill rate
        filled_orders = len([order for order in self.order_history if order.status.value == 'filled'])
        fill_rate = (filled_orders / max(1, self.total_orders)) * 100
        
        # Calculate average execution time for filled orders
        execution_times = []
        for order in self.order_history:
            if order.status.value == 'filled' and order.submitted_time and order.fills:
                first_fill_time = min(fill.timestamp for fill in order.fills)
                execution_time = (first_fill_time - order.submitted_time).total_seconds()
                execution_times.append(execution_time)
        
        avg_execution_time = np.mean(execution_times) if execution_times else 0.0
        
        return {
            'total_orders': self.total_orders,
            'successful_orders': self.successful_orders,
            'rejected_orders': self.rejected_orders,
            'cancelled_orders': self.cancelled_orders,
            'active_orders': active_count,
            'fill_rate_percent': fill_rate,
            'avg_execution_time_seconds': avg_execution_time,
            'orders_in_history': len(self.order_history),
            'symbols_with_active_orders': len(self.active_orders)
        }

    def get_symbol_statistics(self, symbol: str) -> Dict:
        """Get statistics for a specific symbol"""
        symbol_orders = [order for order in self.order_history if order.symbol == symbol]
        active_symbol_orders = self.get_active_orders(symbol)
        
        if not symbol_orders and not active_symbol_orders:
            return {'symbol': symbol, 'no_data': True}
        
        # Calculate symbol-specific metrics
        filled_orders = [order for order in symbol_orders if order.status.value == 'filled']
        total_volume = sum(order.filled_quantity for order in filled_orders)
        total_value = sum(fill.value for order in filled_orders for fill in order.fills)
        
        return {
            'symbol': symbol,
            'total_orders': len(symbol_orders),
            'active_orders': len(active_symbol_orders),
            'filled_orders': len(filled_orders),
            'total_volume': total_volume,
            'total_value': total_value,
            'avg_order_size': total_volume / max(1, len(filled_orders))
        }

# Integration and testing functions
async def test_order_manager_integration():
    """Test order manager with full Phase 2 integration"""
    from signal_generator import SignalGenerator, TradingSignal, SignalType, SignalPriority
    from risk_manager import RiskManager, RiskAdjustedSignal, RiskAction
    from portfolio_manager import PortfolioManager
    from cache_manager import CacheManager
    from rsi_calculator import RSICalculator
    from order_manager import OrderManager, Order, OrderStatus, OrderSide
    import random

    print("=== ORDER MANAGER INTEGRATION TEST ===")
    print()
    
    # Configuration
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT'],
        'initial_capital': 100000.0,
        'commission_per_trade': 1.0,
        'commission_percent': 0.001,
        'rsi_period': 14,
        'rsi': {
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'extreme_oversold': 20,
            'extreme_overbought': 80
        },
        'max_risk_per_trade': 0.02,
        'max_portfolio_risk': 0.06,
        'max_single_position': 0.10,
        'broker_simulation': {
            'fill_probability': 0.95,
            'partial_fill_probability': 0.15,
            'slippage_range': (0.0, 0.003),
            'execution_delay_range': (0.1, 1.0),
            'commission_per_trade': 1.0,
            'rejection_probability': 0.05
        },
        'max_orders_per_symbol': 3,
        'order_timeout_minutes': 30
    }
    
    # Initialize all components
    cache_manager = CacheManager(config)
    rsi_calculator = RSICalculator(config)
    signal_generator = SignalGenerator(config)
    risk_manager = RiskManager(config)
    portfolio_manager = PortfolioManager(config)
    order_manager = OrderManager(config, portfolio_manager)
    
    print(f"🏗️ Initialized all components:")
    print(f"   Portfolio: ${portfolio_manager.total_value:,.2f}")
    print(f"   Max orders per symbol: {config['max_orders_per_symbol']}")
    print(f"   Fill probability: {config['broker_simulation']['fill_probability']*100:.0f}%")
    print()
    
    # Start order manager
    await order_manager.start()
    
    # Set up order callbacks
    async def order_update_callback(order):
        status_emoji = {
            'submitted': "📝",
            'accepted': "✅",
            'filled': "💰",
            'partially_filled': "🔄",
            'rejected': "❌",
            'cancelled': "🚫"
        }
        emoji = status_emoji.get(order.status.value, "📋")
        print(f"   {emoji} Order {order.order_id[:8]}: {order.status.value.upper()}")
    
    async def fill_callback(fill):
        side_emoji = "📈" if fill.side.value == 'buy' else "📉"
        print(f"   {side_emoji} Fill: {fill.symbol} {abs(fill.quantity):.0f} shares @ ${fill.price:.2f}")
    
    order_manager.add_order_update_callback(order_update_callback)
    order_manager.add_fill_callback(fill_callback)
    
    # Simulate market data and generate orders
    base_prices = {'AAPL': 150.0, 'GOOGL': 2500.0, 'MSFT': 300.0}
    
    # Phase 1: Build RSI data
    print("📊 Building RSI data...")
    for cycle in range(20):
        for symbol in config['symbols']:
            change_pct = random.uniform(-0.03, -0.01)  # Declining market
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
    
    # Phase 2: Generate signals and create orders
    print("🚨 Generating trading signals and orders...")
    order_count = 0
    
    for cycle in range(15):
        # Create price recovery
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
        
        # Generate signals every few cycles
        if cycle % 3 == 0:
            signals = await signal_generator.generate_signals(cache_manager, rsi_calculator)
            
            if signals:
                print(f"\nCycle {cycle + 1}: Generated {len(signals)} signals")
                
                # Apply risk management
                risk_adjusted_signals = await risk_manager.evaluate_signals(signals, portfolio_manager)
                
                # Create orders from approved signals
                for risk_signal in risk_adjusted_signals:
                    if risk_signal.approved and order_count < 10:  # Limit for testing
                        symbol = risk_signal.original_signal.symbol
                        current_price = base_prices[symbol]
                        
                        print(f"  🎯 Creating order for {symbol} @ ${current_price:.2f}")
                        
                        order_id = await order_manager.submit_order_from_signal(risk_signal, current_price)
                        
                        if order_id:
                            order_count += 1
                            print(f"     Order ID: {order_id}")
                        else:
                            print(f"     ❌ Order creation failed")
        
        # Update market data for order execution
        market_data = {symbol: {'price': price, 'timestamp': datetime.now()} 
                      for symbol, price in base_prices.items()}
        await order_manager.update_market_data(market_data)
        
        # Small delay between cycles
        await asyncio.sleep(0.5)
    
    # Phase 3: Continue market simulation for order fills
    print(f"\n📈 Continuing market simulation for order execution...")
    
    for cycle in range(20):
        # Random price movements
        for symbol in config['symbols']:
            change_pct = random.uniform(-0.02, 0.02)
            new_price = base_prices[symbol] * (1 + change_pct)
            base_prices[symbol] = new_price
        
        # Update market data
        market_data = {symbol: {'price': price, 'timestamp': datetime.now()} 
                      for symbol, price in base_prices.items()}
        await order_manager.update_market_data(market_data)
        
        # Show progress every 5 cycles
        if cycle % 5 == 4:
            stats = order_manager.get_statistics()
            active_orders = len(order_manager.get_active_orders())
            print(f"   Cycle {cycle + 1}: {active_orders} active orders, {stats['successful_orders']} filled")
        
        await asyncio.sleep(0.2)
    
    # Final results
    print(f"\n" + "="*60)
    print("ORDER MANAGER TEST RESULTS")
    print("="*60)
    
    stats = order_manager.get_statistics()
    print(f"📊 Order Statistics:")
    print(f"   Total Orders: {stats['total_orders']}")
    print(f"   Successful: {stats['successful_orders']}")
    print(f"   Rejected: {stats['rejected_orders']}")
    print(f"   Cancelled: {stats['cancelled_orders']}")
    print(f"   Fill Rate: {stats['fill_rate_percent']:.1f}%")
    print(f"   Avg Execution Time: {stats['avg_execution_time_seconds']:.1f}s")
    
    print(f"\n📋 Active Orders:")
    active_orders = order_manager.get_active_orders()
    if active_orders:
        for order in active_orders:
            print(f"   {order.order_id}: {order.symbol} {order.side.value} {order.quantity} - {order.status.value}")
    else:
        print("   No active orders")
    
    print(f"\n📈 Recent Order History:")
    recent_orders = order_manager.get_order_history(limit=5)
    for order in recent_orders[-5:]:
        status_emoji = "✅" if order.status.value == 'filled' else "❌"
        print(f"   {status_emoji} {order.symbol}: {order.side.value} {order.quantity} - {order.status.value}")
        if order.fills:
            fill_summary = order.get_fill_summary()
            print(f"      Filled: {fill_summary['total_quantity']} @ ${fill_summary['avg_price']:.2f}")
    
    # Symbol-specific statistics
    print(f"\n📊 Symbol Statistics:")
    for symbol in config['symbols']:
        symbol_stats = order_manager.get_symbol_statistics(symbol)
        if not symbol_stats.get('no_data'):
            print(f"   {symbol}: {symbol_stats['total_orders']} orders, "
                  f"{symbol_stats['filled_orders']} filled, "
                  f"${symbol_stats['total_value']:.0f} volume")
    
    # Stop order manager
    await order_manager.stop()
    
    return stats['total_orders'] > 0

# Quick order test
async def quick_order_test():
    """Quick test of order functionality"""
    from order_manager import OrderManager, Order, OrderType, OrderSide
    from portfolio_manager import PortfolioManager
    
    print("⚡ Quick Order Manager Test")
    print("=" * 40)
    
    config = {
        'initial_capital': 50000.0,
        'broker_simulation': {
            'fill_probability': 1.0,  # 100% fill for testing
            'execution_delay_range': (0.1, 0.3),
            'commission_per_trade': 1.0
        }
    }
    
    portfolio_manager = PortfolioManager(config)
    order_manager = OrderManager(config, portfolio_manager)
    
    await order_manager.start()
    
    # Create test order
    order = Order(
        order_id="TEST_001",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100
    )
    
    print(f"Creating test order: {order.symbol} {order.side.value} {order.quantity}")
    
    # Submit order
    order_id = await order_manager.submit_order(order)
    print(f"Order submitted: {order_id}")
    
    # Simulate market data
    market_data = {
        'AAPL': {'price': 150.0, 'timestamp': datetime.now()}
    }
    
    print("Updating market data for execution...")
    await order_manager.update_market_data(market_data)
    
    # Check order status
    await asyncio.sleep(1)  # Allow execution
    
    final_order = order_manager.get_order(order_id)
    if final_order:
        print(f"Final status: {final_order.status.value}")
        if final_order.fills:
            print(f"Fills: {len(final_order.fills)}")
            for fill in final_order.fills:
                print(f"  {abs(fill.quantity)} shares @ ${fill.price:.2f}")
    
    stats = order_manager.get_statistics()
    print(f"Stats: {stats['total_orders']} orders, {stats['successful_orders']} successful")
    
    await order_manager.stop()
    
    return True

# Order lifecycle test
async def order_lifecycle_test():
    """Test focused on order lifecycle and status changes"""
    from order_manager import OrderManager, Order, OrderType, OrderSide, OrderStatus
    from portfolio_manager import PortfolioManager
    
    print("🔄 Order Lifecycle Test")
    print("=" * 40)
    
    config = {
        'initial_capital': 100000.0,
        'broker_simulation': {
            'fill_probability': 0.8,  # 80% fill probability
            'partial_fill_probability': 0.3,  # 30% partial fill chance
            'execution_delay_range': (0.1, 0.5),
            'commission_per_trade': 1.0,
            'rejection_probability': 0.1  # 10% rejection rate
        },
        'order_timeout_minutes': 5  # Short timeout for testing
    }
    
    portfolio_manager = PortfolioManager(config)
    order_manager = OrderManager(config, portfolio_manager)
    
    await order_manager.start()
    
    # Test different order types and scenarios
    test_cases = [
        {"name": "Market Order - Should Fill", "type": OrderType.MARKET, "price": None},
        {"name": "Limit Order - Above Market", "type": OrderType.LIMIT, "price": 155.0},
        {"name": "Limit Order - Below Market", "type": OrderType.LIMIT, "price": 145.0},
        {"name": "Stop Order", "type": OrderType.STOP_MARKET, "price": 148.0}
    ]
    
    market_price = 150.0
    order_results = []
    
    print(f"Market price: ${market_price:.2f}")
    print()
    
    for i, test_case in enumerate(test_cases):
        print(f"Test {i+1}: {test_case['name']}")
        
        # Create order
        order = Order(
            order_id=f"TEST_{i+1:03d}",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=test_case['type'],
            quantity=100,
            limit_price=test_case['price'] if test_case['type'] == OrderType.LIMIT else None,
            stop_price=test_case['price'] if test_case['type'] == OrderType.STOP_MARKET else None
        )
        
        # Submit order
        order_id = await order_manager.submit_order(order)
        print(f"  Submitted: {order_id}")
        
        # Update with market data
        market_data = {'AAPL': {'price': market_price, 'timestamp': datetime.now()}}
        await order_manager.update_market_data(market_data)
        
        # Wait a bit for execution
        await asyncio.sleep(1)
        
        # Check final status
        final_order = order_manager.get_order(order_id)
        if final_order:
            print(f"  Status: {final_order.status.value}")
            if final_order.fills:
                print(f"  Fills: {len(final_order.fills)} fills")
                for fill in final_order.fills:
                    print(f"    {abs(fill.quantity)} @ ${fill.price:.2f}")
            else:
                print(f"  No fills")
            
            order_results.append({
                'test': test_case['name'],
                'status': final_order.status.value,
                'fills': len(final_order.fills)
            })
        
        print()
    
    # Test order cancellation
    print("Test: Order Cancellation")
    cancel_order = Order(
        order_id="TEST_CANCEL",
        symbol="AAPL", 
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=50,
        limit_price=140.0  # Well below market - won't fill
    )
    
    cancel_order_id = await order_manager.submit_order(cancel_order)
    print(f"  Submitted limit order: {cancel_order_id}")
    
    # Try to cancel
    await asyncio.sleep(0.5)
    cancelled = await order_manager.cancel_order(cancel_order_id, "Test cancellation")
    print(f"  Cancelled: {cancelled}")
    
    final_cancel_order = order_manager.get_order(cancel_order_id)
    if final_cancel_order:
        print(f"  Final status: {final_cancel_order.status.value}")
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    for result in order_results:
        status_emoji = "✅" if result['status'] == 'filled' else "⏸️" if result['status'] == 'accepted' else "❌"
        print(f"  {status_emoji} {result['test']}: {result['status']} ({result['fills']} fills)")
    
    stats = order_manager.get_statistics()
    print(f"\nFinal stats: {stats['total_orders']} orders, {stats['fill_rate_percent']:.1f}% fill rate")
    
    await order_manager.stop()
    
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Order Manager Completion Test Options:")
    print("1. Integration test (comprehensive)")
    print("2. Quick functionality test")
    print("3. Order lifecycle test")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(test_order_manager_integration())
    elif choice == "2":
        asyncio.run(quick_order_test())
    elif choice == "3":
        asyncio.run(order_lifecycle_test())
    else:
        print("Invalid choice")