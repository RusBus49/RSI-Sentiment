"""
Main Controller for RSI-Sentiment Trading System
Optimized for Jetson Nano with integrated web dashboard support
Manages all system components and provides real-time data streams
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path
import time

# Handle optional imports with fallbacks
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    print("Warning: aiofiles not available - using synchronous file operations")

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("Warning: websockets not available - real-time updates disabled")

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("Warning: aiohttp not available - web API disabled")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available - system monitoring limited")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("Warning: GPUtil not available - GPU monitoring disabled")

# Import all trading system components
from market_data_feed import MarketDataFeed
from cache_manager import CacheManager
from rsi_calculator import RSICalculator, RSISignal
from signal_generator import SignalGenerator, TradingSignal, SignalType
from risk_manager import RiskManager, RiskAdjustedSignal
from portfolio_manager import PortfolioManager
from order_manager import OrderManager
from market_hours_helper import MarketHoursChecker

class SystemState(Enum):
    """System operational states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MARKET_CLOSED = "market_closed"

@dataclass
class SystemMetrics:
    """Real-time system metrics for dashboard"""
    timestamp: datetime
    state: SystemState
    uptime_seconds: float
    
    # Performance metrics
    cpu_usage: float
    memory_usage_mb: float
    gpu_usage: float  # Jetson GPU usage
    gpu_memory_mb: float
    
    # Trading metrics
    total_signals: int
    active_positions: int
    portfolio_value: float
    daily_pnl: float
    total_pnl: float
    win_rate: float
    
    # Component health
    market_feed_health: bool
    cache_health: bool
    rsi_health: bool
    
    # Market status
    market_open: bool
    next_market_event: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'state': self.state.value,
            'uptime_seconds': self.uptime_seconds,
            'cpu_usage': self.cpu_usage,
            'memory_usage_mb': self.memory_usage_mb,
            'gpu_usage': self.gpu_usage,
            'gpu_memory_mb': self.gpu_memory_mb,
            'total_signals': self.total_signals,
            'active_positions': self.active_positions,
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'win_rate': self.win_rate,
            'market_feed_health': self.market_feed_health,
            'cache_health': self.cache_health,
            'rsi_health': self.rsi_health,
            'market_open': self.market_open,
            'next_market_event': self.next_market_event
        }

class MainController:
    """Main controller that orchestrates the entire trading system"""
    
    def __init__(self, config_file: str = "trading_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        
        # System state
        self.state = SystemState.INITIALIZING
        self.start_time = datetime.now()
        self.is_running = False
        
        # Components (will be initialized in setup)
        self.market_hours_checker = None
        self.cache_manager = None
        self.rsi_calculator = None
        self.signal_generator = None
        self.risk_manager = None
        self.portfolio_manager = None
        self.order_manager = None
        self.market_feed = None
        
        # WebSocket connections for real-time updates
        self.websocket_clients = set()
        
        # Event callbacks for dashboard
        self.event_callbacks = {
            'signal': [],
            'order': [],
            'position': [],
            'alert': [],
            'metrics': []
        }
        
        # Background tasks
        self.tasks = []
        
        # Performance monitoring
        self.metrics_history = []
        self.max_metrics_history = 1000
        
        # Setup logging
        self._setup_logging()
        self.logger.info("=" * 70)
        self.logger.info("RSI-SENTIMENT TRADING SYSTEM - MAIN CONTROLLER")
        self.logger.info("Optimized for NVIDIA Jetson Nano")
        self.logger.info("=" * 70)
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Add portfolio management config if not present
            if 'portfolio' not in config:
                config['portfolio'] = {
                    'initial_capital': 100000.0,
                    'commission_per_trade': 1.0,
                    'commission_percent': 0.001,
                    'min_position_value': 1000.0,
                    'max_positions': 10
                }
            
            # Add risk management config if not present
            if 'risk' not in config:
                config['risk'] = {
                    'max_portfolio_risk': 0.06,
                    'max_single_position': 0.10,
                    'max_risk_per_trade': 0.02,
                    'max_sector_exposure': 0.25
                }
            
            # Add signal generation config
            if 'signals' not in config:
                config['signals'] = {
                    'min_signal_strength': 0.4,
                    'min_confidence': 0.5,
                    'max_signals_per_cycle': 3,
                    'require_momentum_confirmation': True
                }
            
            # Merge configs for components
            config.update(config.get('portfolio', {}))
            config.update(config.get('risk', {}))
            config.update(config.get('signals', {}))
            
            return config
            
        except FileNotFoundError:
            self.logger.error(f"Config file {self.config_file} not found")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file: {e}")
            raise
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # File handler
        log_file = log_config.get('file', 'trading_system.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Setup root logger
        self.logger = logging.getLogger('MainController')
        self.logger.setLevel(log_level)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing trading system components...")
            
            # Initialize market hours checker
            self.market_hours_checker = MarketHoursChecker(self.config.get('system', {}))
            self.market_hours_checker.log_market_status()
            
            # Check if we should run
            should_run, reason = self.market_hours_checker.should_system_run()
            if not should_run and self.config.get('system', {}).get('market_hours_only', True):
                self.logger.warning(f"System not starting: {reason}")
                self.state = SystemState.MARKET_CLOSED
                return False
            
            # Initialize components in dependency order
            self.logger.info("Initializing cache manager...")
            self.cache_manager = CacheManager(self.config)
            
            self.logger.info("Initializing RSI calculator...")
            self.rsi_calculator = RSICalculator(self.config)
            
            self.logger.info("Initializing portfolio manager...")
            self.portfolio_manager = PortfolioManager(self.config)
            
            self.logger.info("Initializing risk manager...")
            self.risk_manager = RiskManager(self.config)
            
            self.logger.info("Initializing signal generator...")
            self.signal_generator = SignalGenerator(self.config)
            
            self.logger.info("Initializing order manager...")
            self.order_manager = OrderManager(self.config, self.portfolio_manager)
            
            # Setup order callbacks
            self.order_manager.add_order_update_callback(self._on_order_update)
            self.order_manager.add_fill_callback(self._on_order_fill)
            
            self.logger.info("Initializing market data feed...")
            self.market_feed = MarketDataFeed(self.config)
            self.market_feed.add_callback(self._on_market_data)
            
            self.state = SystemState.RUNNING
            self.logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            self.state = SystemState.ERROR
            raise
    
    async def start(self):
        """Start the trading system"""
        if not await self.initialize():
            return
        
        self.is_running = True
        self.logger.info("Starting trading system...")
        
        try:
            # Start components
            await self.order_manager.start()
            
            # Create background tasks
            self.tasks = [
                asyncio.create_task(self.market_feed.start()),
                asyncio.create_task(self._signal_generation_loop()),
                asyncio.create_task(self._position_monitoring_loop()),
                asyncio.create_task(self._metrics_collection_loop()),
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._market_hours_monitor())
            ]
            
            self.logger.info("🚀 Trading system started successfully")
            self._broadcast_event('system', {'event': 'started', 'timestamp': datetime.now().isoformat()})
            
            # Wait for tasks (they should run indefinitely)
            await asyncio.gather(*self.tasks)
            
        except asyncio.CancelledError:
            self.logger.info("System tasks cancelled")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.state = SystemState.ERROR
            raise
    
    async def stop(self):
        """Gracefully stop the trading system"""
        self.logger.info("Stopping trading system...")
        self.is_running = False
        self.state = SystemState.STOPPING
        
        # Cancel all background tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Stop components
        if self.market_feed:
            await self.market_feed.stop()
        
        if self.order_manager:
            await self.order_manager.stop()
        
        # Save state
        await self._save_system_state()
        
        self.state = SystemState.STOPPED
        self.logger.info("✅ Trading system stopped successfully")
        self._broadcast_event('system', {'event': 'stopped', 'timestamp': datetime.now().isoformat()})
    
    async def _on_market_data(self, symbol: str, data: Dict):
        """Handle incoming market data"""
        try:
            # Update cache
            await self.cache_manager.update_price_data(symbol, data)
            
            # Update positions with new prices
            market_data = {symbol: data}
            await self.portfolio_manager.update_positions(market_data)
            
            # Update order manager
            await self.order_manager.update_market_data(market_data)
            
            # Broadcast price update to dashboard
            self._broadcast_event('price', {
                'symbol': symbol,
                'price': data['price'],
                'volume': data['volume'],
                'timestamp': data['timestamp'].isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error processing market data for {symbol}: {e}")
    
    async def _signal_generation_loop(self):
        """Main signal generation loop"""
        signal_interval = self.config.get('signals', {}).get('generation_interval', 30)  # 30 seconds default
        
        while self.is_running:
            try:
                # Update RSI calculations
                self.rsi_calculator.update_from_cache(self.cache_manager)
                
                # Generate signals
                signals = await self.signal_generator.generate_signals(
                    self.cache_manager, 
                    self.rsi_calculator,
                    self.portfolio_manager
                )
                
                if signals:
                    self.logger.info(f"Generated {len(signals)} trading signals")
                    
                    # Apply risk management
                    risk_adjusted_signals = await self.risk_manager.evaluate_signals(
                        signals, 
                        self.portfolio_manager
                    )
                    
                    # Process approved signals
                    for risk_signal in risk_adjusted_signals:
                        if risk_signal.approved:
                            await self._execute_signal(risk_signal)
                        
                        # Broadcast signal to dashboard
                        self._broadcast_signal(risk_signal)
                
                await asyncio.sleep(signal_interval)
                
            except Exception as e:
                self.logger.error(f"Error in signal generation loop: {e}")
                await asyncio.sleep(signal_interval)
    
    async def _execute_signal(self, risk_signal: RiskAdjustedSignal):
        """Execute an approved trading signal"""
        try:
            symbol = risk_signal.original_signal.symbol
            current_price = self.cache_manager.get_stock_cache(symbol).get_latest_price()
            
            if not current_price:
                self.logger.warning(f"No price available for {symbol}")
                return
            
            # Create and submit order
            order_id = await self.order_manager.submit_order_from_signal(
                risk_signal, 
                current_price
            )
            
            if order_id:
                self.logger.info(f"✅ Order submitted: {order_id} for {symbol}")
                self._broadcast_event('order_created', {
                    'order_id': order_id,
                    'symbol': symbol,
                    'signal_type': risk_signal.original_signal.signal_type.value,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                self.logger.warning(f"Failed to submit order for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
    
    async def _position_monitoring_loop(self):
        """Monitor positions and update portfolio"""
        monitor_interval = 10  # 10 seconds
        
        while self.is_running:
            try:
                # Get current market data
                market_data = {}
                for symbol in self.config['symbols']:
                    cache = self.cache_manager.get_stock_cache(symbol)
                    if cache:
                        price = cache.get_latest_price()
                        if price:
                            market_data[symbol] = {
                                'price': price,
                                'timestamp': datetime.now()
                            }
                
                # Update positions
                if market_data:
                    await self.portfolio_manager.update_positions(market_data)
                
                # Update risk exposures
                self.risk_manager.update_sector_exposures(self.portfolio_manager)
                self.risk_manager.update_correlation_exposures(self.portfolio_manager)
                
                # Broadcast portfolio update
                self._broadcast_portfolio_update()
                
                await asyncio.sleep(monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(monitor_interval)
    
    async def _metrics_collection_loop(self):
        """Collect system and trading metrics"""
        metrics_interval = 5  # 5 seconds
        
        while self.is_running:
            try:
                metrics = await self._collect_metrics()
                
                # Store metrics history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_metrics_history:
                    self.metrics_history = self.metrics_history[-self.max_metrics_history:]
                
                # Broadcast metrics
                self._broadcast_event('metrics', metrics.to_dict())
                
                await asyncio.sleep(metrics_interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(metrics_interval)
    
    async def _collect_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()