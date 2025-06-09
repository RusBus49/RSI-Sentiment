"""
Phase 1 Integration Test
Test the complete data flow: Yahoo Finance → Cache → RSI → Signals
"""

import asyncio
import logging
from datetime import datetime
import time

# Import the three simplified components
from typing import Dict
from market_data_feed import MarketDataFeed
from cache_manager import CacheManager
from rsi_calculator import RSICalculator

class Phase1IntegrationTest:
    def __init__(self):
        # Test configuration
        self.config = {
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
            'update_interval': 75,  # 75 seconds between market data updates
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'extreme_oversold': 20,
            'extreme_overbought': 80,
            'min_signal_strength': 0.3
        }
        
        # Components
        self.market_feed = None
        self.cache_manager = None
        self.rsi_calculator = None
        
        # Test metrics
        self.start_time = None
        self.signal_count = 0
        self.update_count = 0
        
        self.logger = logging.getLogger(__name__)
    
    async def setup(self):
        """Initialize all components"""
        print("🔧 Setting up Phase 1 components...")
        
        # Initialize cache manager
        self.cache_manager = CacheManager(self.config)
        print(f"✅ Cache manager initialized for {len(self.config['symbols'])} symbols")
        
        # Initialize RSI calculator
        self.rsi_calculator = RSICalculator(self.config)
        print(f"✅ RSI calculator initialized (period: {self.config['rsi_period']})")
        
        # Initialize market data feed
        self.market_feed = MarketDataFeed(self.config)
        
        # Connect market feed to cache manager
        self.market_feed.add_callback(self.cache_manager.update_price_data)
        self.market_feed.add_callback(self._track_updates)
        print("✅ Market data feed connected to cache manager")
        
        print("🚀 All components ready!")
    
    async def _track_updates(self, symbol: str, data: Dict):
        """Track updates for testing metrics"""
        self.update_count += 1
        
        # Update RSI calculations every few updates
        if self.update_count % len(self.config['symbols']) == 0:
            await self._process_rsi_signals()
    
    async def _process_rsi_signals(self):
        """Process RSI signals and display results"""
        # Update RSI calculations
        updated_states = self.rsi_calculator.update_from_cache(self.cache_manager)
        
        if not updated_states:
            return
        
        # Check for active signals
        active_signals = self.rsi_calculator.get_active_signals()
        
        if active_signals:
            self.signal_count += len(active_signals)
            print(f"\n🚨 SIGNALS DETECTED at {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)
            
            for symbol, state in active_signals.items():
                signal_emoji = "📈" if "long" in state.signal.value else "📉"
                print(f"{signal_emoji} {symbol}: {state.signal.value.upper()}")
                print(f"   RSI: {state.current_rsi:.1f} | Price: ${state.price:.2f} | Strength: {state.signal_strength:.2f}")
                print(f"   Volume: {state.volume:,} | Change: {state.price_change_percent:+.2f}%")
            
            print("=" * 60)
        
        # Display periodic summary
        cycle_num = self.update_count // len(self.config['symbols'])
        if cycle_num > 0 and cycle_num % 3 == 0:  # Every 3 cycles
            await self._display_summary()
    
    async def _display_summary(self):
        """Display periodic system summary"""
        print(f"\n📊 SYSTEM SUMMARY - Cycle {self.update_count // len(self.config['symbols'])}")
        print("-" * 50)
        
        # Cache stats
        cache_stats = self.cache_manager.get_system_stats()
        ready_symbols = self.cache_manager.get_ready_symbols()
        
        print(f"Data Updates: {cache_stats['total_updates']}")
        print(f"Symbols Ready: {len(ready_symbols)}/{cache_stats['symbols_tracked']}")
        
        # RSI stats
        rsi_stats = self.rsi_calculator.get_statistics()
        if rsi_stats['symbols_tracked'] > 0:
            print(f"Average RSI: {rsi_stats['avg_rsi']:.1f}")
            print(f"RSI Range: {rsi_stats['min_rsi']:.1f} - {rsi_stats['max_rsi']:.1f}")
        
        # Market feed stats
        feed_stats = self.market_feed.get_statistics()
        print(f"Feed Success Rate: {feed_stats['success_rate_percent']:.1f}%")
        print(f"Daily Requests: {feed_stats['requests_today']}")
        
        # Market overview
        market_overview = self.rsi_calculator.get_market_overview()
        print(f"Long Opportunities: {market_overview['long_opportunities']}")
        print(f"Short Opportunities: {market_overview['short_opportunities']}")
        
        print("-" * 50)
    
    async def run_test(self, duration_minutes: int = 10):
        """Run the integration test for specified duration"""
        self.start_time = datetime.now()
        
        print(f"🚀 Starting Phase 1 Integration Test")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Symbols: {self.config['symbols']}")
        print(f"Update interval: {self.config['update_interval']} seconds")
        print("=" * 60)
        print("Waiting for market data and RSI calculations...")
        print("(RSI requires 14+ data points per symbol)")
        print("=" * 60)
        
        try:
            # Start market data feed
            feed_task = asyncio.create_task(self.market_feed.start())
            
            # Wait for specified duration
            await asyncio.sleep(duration_minutes * 60)
            
        except asyncio.CancelledError:
            print("\n⏹️ Test cancelled by user")
        except Exception as e:
            print(f"\n❌ Test error: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        if self.market_feed:
            await self.market_feed.stop()
        
        # Final statistics
        await self._display_final_results()
    
    async def _display_final_results(self):
        """Display final test results"""
        test_duration = (datetime.now() - self.start_time).total_seconds() / 60
        
        print("\n" + "=" * 60)
        print("FINAL TEST RESULTS")
        print("=" * 60)
        
        print(f"Test Duration: {test_duration:.1f} minutes")
        print(f"Total Updates: {self.update_count}")
        print(f"Signals Generated: {self.signal_count}")
        
        # Component statistics
        if self.cache_manager:
            cache_stats = self.cache_manager.get_system_stats()
            print(f"\nCache Manager:")
            print(f"  Symbols with data: {cache_stats['symbols_with_data']}")
            print(f"  RSI ready: {cache_stats['symbols_ready_for_rsi']}")
            print(f"  Memory usage: {cache_stats['estimated_memory_mb']:.1f} MB")
        
        if self.rsi_calculator:
            rsi_stats = self.rsi_calculator.get_statistics()
            market_overview = self.rsi_calculator.get_market_overview()
            print(f"\nRSI Calculator:")
            print(f"  Symbols tracked: {rsi_stats['symbols_tracked']}")
            print(f"  Average RSI: {rsi_stats.get('avg_rsi', 0):.1f}")
            print(f"  Total signals: {rsi_stats['signals_generated']}")
            print(f"  Long opportunities: {market_overview['long_opportunities']}")
            print(f"  Short opportunities: {market_overview['short_opportunities']}")
        
        if self.market_feed:
            feed_stats = self.market_feed.get_statistics()
            print(f"\nMarket Data Feed:")
            print(f"  Success rate: {feed_stats['success_rate_percent']:.1f}%")
            print(f"  Total requests: {feed_stats['total_requests']}")
            print(f"  Daily usage: {feed_stats['requests_today']}/8000 ({feed_stats['requests_today']/80:.1f}%)")
            print(f"  Rate limited: {feed_stats['rate_limited_requests']}")
        
        # Test assessment
        print(f"\n🎯 TEST ASSESSMENT:")
        if self.signal_count > 0:
            print(f"✅ SUCCESS: Generated {self.signal_count} trading signals")
        else:
            print(f"⚠️ LIMITED: No signals generated (may need longer test or more volatile market)")
        
        if self.cache_manager and self.cache_manager.get_system_stats()['symbols_ready_for_rsi'] > 0:
            print(f"✅ SUCCESS: RSI calculations working")
        else:
            print(f"❌ ISSUE: RSI calculations not ready (need more data)")
        
        if self.market_feed and feed_stats['success_rate_percent'] > 90:
            print(f"✅ SUCCESS: Market data feed reliable")
        else:
            print(f"⚠️ ISSUE: Market data feed having issues")

# Quick test function
async def quick_test():
    """Run a quick 5-minute test"""
    test = Phase1IntegrationTest()
    await test.setup()
    await test.run_test(duration_minutes=5)

# Full test function  
async def full_test():
    """Run a comprehensive 15-minute test"""
    test = Phase1IntegrationTest()
    await test.setup()
    await test.run_test(duration_minutes=15)

# Simulation test (for when markets are closed)
async def simulation_test():
    """Run test with simulated data"""
    import random
    
    print("🎭 Running SIMULATION test (fake data)")
    print("=" * 60)
    
    config = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT'],
        'rsi_period': 14,
        'oversold_threshold': 30,
        'overbought_threshold': 70
    }
    
    # Initialize components
    cache_manager = CacheManager(config)
    rsi_calculator = RSICalculator(config)
    
    # Simulate data over time
    base_prices = {'AAPL': 150.0, 'GOOGL': 2500.0, 'MSFT': 300.0}
    
    print("Generating simulated market data...")
    
    for cycle in range(25):  # 25 cycles to ensure RSI calculation
        print(f"Cycle {cycle + 1}/25: ", end="")
        
        for symbol in config['symbols']:
            # Generate realistic price movement with some trending
            if cycle < 10:
                # Trending down (should trigger oversold)
                change_pct = random.uniform(-0.04, 0.01)
            elif cycle < 20:
                # Trending up (should trigger overbought)  
                change_pct = random.uniform(-0.01, 0.04)
            else:
                # Normal random walk
                change_pct = random.uniform(-0.02, 0.02)
            
            new_price = base_prices[symbol] * (1 + change_pct)
            base_prices[symbol] = new_price
            
            # Create realistic data
            data = {
                'symbol': symbol,
                'price': new_price,
                'volume': random.randint(1000000, 10000000),
                'timestamp': datetime.now(),
                'market_state': 'REGULAR',
                'source': 'simulation'
            }
            
            await cache_manager.update_price_data(symbol, data)
        
        # Update RSI and check for signals
        updated_states = rsi_calculator.update_from_cache(cache_manager)
        active_signals = rsi_calculator.get_active_signals()
        
        if updated_states:
            rsi_values = [f"{s}:{state.current_rsi:.0f}" for s, state in updated_states.items()]
            print(f"RSI: {', '.join(rsi_values)}")
            
            if active_signals:
                print(f"  🚨 {len(active_signals)} signals: ", end="")
                for symbol, state in active_signals.items():
                    print(f"{symbol}({state.signal.value}) ", end="")
                print()
        else:
            print("Building RSI data...")
        
        await asyncio.sleep(0.1)  # Small delay
    
    # Final results
    print(f"\n📊 SIMULATION RESULTS:")
    stats = rsi_calculator.get_statistics()
    market_overview = rsi_calculator.get_market_overview()
    
    print(f"Signals generated: {stats['signals_generated']}")
    print(f"Long opportunities: {market_overview['long_opportunities']}")
    print(f"Short opportunities: {market_overview['short_opportunities']}")
    print(f"Average RSI: {stats.get('avg_rsi', 0):.1f}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    import sys
    
    print("Phase 1 Integration Test Options:")
    print("1. Quick test (5 minutes, live data)")
    print("2. Full test (15 minutes, live data)")  
    print("3. Simulation test (2 minutes, fake data)")
    print("4. Exit")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting quick test...")
            asyncio.run(quick_test())
        elif choice == "2":
            print("\n🚀 Starting full test...")
            asyncio.run(full_test())
        elif choice == "3":
            print("\n🎭 Starting simulation test...")
            asyncio.run(simulation_test())
        elif choice == "4":
            print("👋 Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")