"""
Phase 1 Integration Test
Combines market data feed, cache manager, and RSI calculator
Tests the complete data flow for the RSI trading system
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List

# Import our Phase 1 components
from market_data_feed import MarketDataFeed
from cache_manager import CacheManager
from rsi_calculator import RSICalculator, create_rsi_calculator_from_cache, RSISignal

class Phase1TradingSystem:
    """Integration class for Phase 1 components"""
    
    def __init__(self, config_file: str = None):
        # Load configuration
        if config_file:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
                "market_data": {
                    "update_interval": 10,
                    "api_key": "FGPMEXHAEYSHJGQI"
                },
                "rsi": {
                    "rsi_period": 14,
                    "oversold_threshold": 30,
                    "overbought_threshold": 70,
                    "extreme_oversold": 20,
                    "extreme_overbought": 80,
                    "signal_confirmation_periods": 2
                },
                "cache": {
                    "max_prices": 200,
                    "max_volumes": 50
                }
            }
        
        # Initialize components
        self.market_feed = None
        self.cache_manager = None
        self.rsi_calculator = None
        
        # System state
        self.is_running = False
        self.start_time = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize all system components"""
        self.logger.info("Initializing Phase 1 Trading System...")
        
        # Initialize cache manager
        cache_config = {
            'symbols': self.config['symbols'],
            **self.config.get('cache', {})
        }
        self.cache_manager = CacheManager(cache_config)
        
        # Initialize market data feed
        feed_config = {
            'symbols': self.config['symbols'],
            **self.config.get('market_data', {})
        }
        self.market_feed = MarketDataFeed(feed_config)
        
        # Connect market feed to cache manager
        self.market_feed.add_callback(self.cache_manager.update_price_data)
        
        # Initialize RSI calculator (will be populated once we have data)
        self.rsi_calculator = RSICalculator(self.config.get('rsi', {}))
        
        self.logger.info("System initialization complete")
    
    async def start(self):
        """Start the trading system"""
        if not self.market_feed or not self.cache_manager or not self.rsi_calculator:
            await self.initialize()
        
        self.is_running = True
        self.start_time = datetime.now()
        
        self.logger.info("Starting Phase 1 Trading System...")
        
        # Start market data feed
        feed_task = asyncio.create_task(self.market_feed.start())
        
        # Start monitoring loop
        monitor_task = asyncio.create_task(self._monitoring_loop())
        
        # Start RSI initialization check
        rsi_init_task = asyncio.create_task(self._rsi_initialization_loop())
        
        try:
            # Run all tasks concurrently
            await asyncio.gather(feed_task, monitor_task, rsi_init_task)
        except asyncio.CancelledError:
            self.logger.info("System shutdown requested")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the trading system"""
        self.is_running = False
        
        if self.market_feed:
            await self.market_feed.stop()
        
        self.logger.info("Phase 1 Trading System stopped")
    
    async def _rsi_initialization_loop(self):
        """Check for symbols ready for RSI calculation and initialize them"""
        while self.is_running:
            try:
                # Wait a bit for data to accumulate
                await asyncio.sleep(30)
                
                ready_symbols = self.cache_manager.get_ready_symbols()
                
                for symbol in ready_symbols:
                    if symbol not in self.rsi_calculator.rsi_states:
                        stock_cache = self.cache_manager.get_stock_cache(symbol)
                        if stock_cache:
                            price_history = stock_cache.get_price_history()
                            if len(price_history) >= self.config['rsi']['rsi_period'] + 1:
                                self.rsi_calculator.initialize_rsi_state(symbol, price_history)
                                self.logger.info(f"Initialized RSI calculation for {symbol}")
                
            except Exception as e:
                self.logger.error(f"RSI initialization error: {e}")
                await asyncio.sleep(10)
    
    async def _monitoring_loop(self):
        """Main monitoring and signal detection loop"""
        last_update_time = datetime.now()
        
        while self.is_running:
            try:
                await asyncio.sleep(15)  # Check every 15 seconds
                
                current_time = datetime.now()
                
                # Update RSI calculations for symbols with new data
                await self._update_rsi_calculations()
                
                # Check for trading signals
                signals = self.rsi_calculator.get_signals(min_strength=0.3)
                
                if signals:
                    self._report_signals(signals)
                
                # Print periodic status updates
                if (current_time - last_update_time).seconds >= 60:  # Every minute
                    self._print_status_update()
                    last_update_time = current_time
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _update_rsi_calculations(self):
        """Update RSI calculations with latest price data"""
        for symbol in self.config['symbols']:
            if symbol in self.rsi_calculator.rsi_states:
                stock_cache = self.cache_manager.get_stock_cache(symbol)
                if stock_cache:
                    latest_prices = stock_cache.get_latest_price()
                    price_history = stock_cache.get_price_history()
                    
                    if len(price_history) >= 2:
                        current_price = price_history[-1]
                        previous_price = price_history[-2]
                        
                        # Update RSI incrementally
                        updated_state = self.rsi_calculator.update_rsi_incremental(
                            symbol, current_price, previous_price
                        )
                        
                        if updated_state and updated_state.signal != RSISignal.NO_SIGNAL:
                            self.logger.debug(f"RSI updated for {symbol}: {updated_state.current_rsi:.2f}")
    
    def _report_signals(self, signals: Dict):
        """Report trading signals"""
        print(f"\n{'='*50}")
        print(f"TRADING SIGNALS DETECTED - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        
        for symbol, state in signals.items():
            signal_type = state.signal.value.replace('_', ' ').title()
            print(f"{symbol:6} | RSI: {state.current_rsi:5.1f} | {signal_type:20} | Strength: {state.signal_strength:.2f}")
        
        print(f"{'='*50}\n")
    
    def _print_status_update(self):
        """Print system status update"""
        uptime = datetime.now() - self.start_time if self.start_time else None
        cache_stats = self.cache_manager.get_system_stats()
        rsi_stats = self.rsi_calculator.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"SYSTEM STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"Uptime: {uptime}")
        print(f"Data Updates: {cache_stats['total_updates']}")
        print(f"Symbols Tracked: {cache_stats['symbols_tracked']}")
        print(f"RSI Ready: {rsi_stats['symbols_tracked']}")
        print(f"Active Signals: {rsi_stats.get('active_signals', 0)}")
        
        if rsi_stats['symbols_tracked'] > 0:
            print(f"Average RSI: {rsi_stats['avg_rsi']:.1f}")
            print(f"RSI Range: {rsi_stats['min_rsi']:.1f} - {rsi_stats['max_rsi']:.1f}")
            print(f"Oversold Count: {rsi_stats['oversold_count']}")
            print(f"Overbought Count: {rsi_stats['overbought_count']}")
        
        print(f"{'='*60}\n")
    
    def get_current_positions(self) -> Dict:
        """Get current RSI positions and signals"""
        positions = {}
        
        for symbol, state in self.rsi_calculator.get_all_rsi_states().items():
            stock_cache = self.cache_manager.get_stock_cache(symbol)
            current_price = stock_cache.get_latest_price() if stock_cache else None
            
            positions[symbol] = {
                'current_price': current_price,
                'rsi': state.current_rsi,
                'signal': state.signal.value,
                'signal_strength': state.signal_strength,
                'timestamp': state.timestamp
            }
        
        return positions
    
    def get_system_health(self) -> Dict:
        """Get system health metrics"""
        return {
            'is_running': self.is_running,
            'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'cache_stats': self.cache_manager.get_system_stats(),
            'rsi_stats': self.rsi_calculator.get_statistics(),
            'market_feed_status': self.market_feed.is_running if self.market_feed else False
        }

# Configuration file template
def create_default_config(filename: str = "trading_config.json"):
    """Create a default configuration file"""
    config = {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX"],
        "market_data": {
            "update_interval": 15,
            "api_key": "YOUR_ALPHA_VANTAGE_API_KEY_HERE"
        },
        "rsi": {
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
            "extreme_oversold": 20,
            "extreme_overbought": 80,
            "signal_confirmation_periods": 2
        },
        "cache": {
            "max_prices": 200,
            "max_volumes": 50
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created default configuration file: {filename}")
    print("Please edit the file to add your API key and adjust settings as needed.")

# Main execution
async def main():
    """Main execution function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create trading system
        system = Phase1TradingSystem()
        
        # Initialize and start
        await system.initialize()
        
        logger.info("Phase 1 Trading System starting...")
        logger.info("Press Ctrl+C to stop")
        
        # Run the system
        await system.start()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        raise
    finally:
        logger.info("System cleanup complete")

if __name__ == "__main__":
    # Uncomment to create default config file
    # create_default_config()
    
    # Run the main system
    asyncio.run(main())