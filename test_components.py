"""
Test script to verify all components are working correctly
Run this before starting the main system
"""

import asyncio
import sys
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_imports():
    """Test all required imports"""
    print("\n=== Testing Imports ===")
    
    imports_status = []
    
    # Core imports
    core_imports = [
        ('asyncio', 'Core async library'),
        ('json', 'JSON parsing'),
        ('datetime', 'Date/time handling'),
        ('numpy', 'Numerical computing'),
        ('logging', 'Logging system')
    ]
    
    for module, description in core_imports:
        try:
            __import__(module)
            imports_status.append((module, description, True))
            print(f"✓ {module}: {description}")
        except ImportError:
            imports_status.append((module, description, False))
            print(f"✗ {module}: {description} - MISSING")
    
    # Trading system components
    print("\n=== Testing Trading Components ===")
    
    components = [
        ('market_data_feed', 'Market data feed'),
        ('cache_manager', 'Cache manager'),
        ('rsi_calculator', 'RSI calculator'),
        ('signal_generator', 'Signal generator'),
        ('risk_manager', 'Risk manager'),
        ('portfolio_manager', 'Portfolio manager'),
        ('order_manager', 'Order manager'),
        ('market_hours_helper', 'Market hours helper')
    ]
    
    for module, description in components:
        try:
            __import__(module)
            imports_status.append((module, description, True))
            print(f"✓ {module}: {description}")
        except ImportError as e:
            imports_status.append((module, description, False))
            print(f"✗ {module}: {description} - ERROR: {e}")
    
    # Optional imports
    print("\n=== Testing Optional Imports ===")
    
    optional_imports = [
        ('aiohttp', 'Web server', True),
        ('aiofiles', 'Async file operations', True),
        ('websockets', 'WebSocket support', True),
        ('psutil', 'System monitoring', True),
        ('GPUtil', 'GPU monitoring', False),
        ('pytz', 'Timezone support', True)
    ]
    
    for module, description, required in optional_imports:
        try:
            __import__(module)
            print(f"✓ {module}: {description}")
        except ImportError:
            if required:
                print(f"✗ {module}: {description} - REQUIRED BUT MISSING")
            else:
                print(f"⚠ {module}: {description} - Optional, not critical")
    
    # Summary
    failed = [item for item in imports_status if not item[2]]
    if failed:
        print(f"\n❌ {len(failed)} imports failed:")
        for module, desc, _ in failed:
            print(f"   - {module}: {desc}")
        return False
    else:
        print(f"\n✅ All core imports successful!")
        return True

async def test_config():
    """Test configuration file"""
    print("\n=== Testing Configuration ===")
    
    try:
        with open('trading_config.json', 'r') as f:
            config = json.load(f)
        
        print("✓ Config file loaded successfully")
        
        # Check required sections
        required_sections = ['symbols', 'rsi', 'system', 'market_data']
        for section in required_sections:
            if section in config:
                print(f"✓ Config section '{section}' found")
            else:
                print(f"✗ Config section '{section}' missing")
        
        # Display key settings
        print(f"\nKey Settings:")
        print(f"  Symbols: {config.get('symbols', [])[:5]}{'...' if len(config.get('symbols', [])) > 5 else ''}")
        print(f"  Update interval: {config.get('market_data', {}).get('update_interval', 'N/A')}s")
        print(f"  RSI period: {config.get('rsi', {}).get('rsi_period', 'N/A')}")
        print(f"  Market hours only: {config.get('system', {}).get('market_hours_only', 'N/A')}")
        
        return True
        
    except FileNotFoundError:
        print("✗ Config file 'trading_config.json' not found")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Config file has invalid JSON: {e}")
        return False

async def test_market_hours():
    """Test market hours checker"""
    print("\n=== Testing Market Hours ===")
    
    try:
        from market_hours_helper import create_market_hours_checker, get_market_status_summary
        
        checker = create_market_hours_checker()
        status = get_market_status_summary()
        
        print(f"Market Status: {status}")
        
        # Get detailed status
        detailed = checker.get_market_status()
        print(f"Current time (user): {detailed['current_time_user'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Market hours (user): {detailed['market_open_user'].strftime('%H:%M')} - {detailed['market_close_user'].strftime('%H:%M')}")
        
        should_run, reason = checker.should_system_run()
        if should_run:
            print(f"✓ System should run: {reason}")
        else:
            print(f"⚠ System should not run: {reason}")
        
        return True
        
    except Exception as e:
        print(f"✗ Market hours test failed: {e}")
        return False

async def test_cache_and_rsi():
    """Test cache manager and RSI calculator"""
    print("\n=== Testing Cache and RSI ===")
    
    try:
        from cache_manager import CacheManager
        from rsi_calculator import RSICalculator
        
        # Simple test config
        test_config = {
            'symbols': ['TEST'],
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70
        }
        
        cache = CacheManager(test_config)
        rsi_calc = RSICalculator(test_config)
        
        print("✓ Cache manager initialized")
        print("✓ RSI calculator initialized")
        
        # Add some test data
        for i in range(20):
            price = 100 + (i % 5) - 2  # Oscillating price
            data = {
                'symbol': 'TEST',
                'price': price,
                'volume': 1000000,
                'timestamp': datetime.now(),
                'market_state': 'REGULAR',
                'source': 'test'
            }
            await cache.update_price_data('TEST', data)
        
        # Update RSI
        rsi_calc.update_from_cache(cache)
        
        # Check if RSI was calculated
        rsi_state = rsi_calc.get_rsi_state('TEST')
        if rsi_state and rsi_state.current_rsi:
            print(f"✓ RSI calculated: {rsi_state.current_rsi:.2f}")
            return True
        else:
            print("✗ RSI calculation failed")
            return False
            
    except Exception as e:
        print(f"✗ Cache/RSI test failed: {e}")
        return False

async def test_system_resources():
    """Test system resources"""
    print("\n=== Testing System Resources ===")
    
    try:
        import psutil
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        print(f"CPU: {cpu_count} cores, {cpu_percent}% usage")
        
        # Memory
        memory = psutil.virtual_memory()
        print(f"Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available ({memory.percent}% used)")
        
        if memory.percent > 90:
            print("⚠ High memory usage detected!")
        
        # Disk
        disk = psutil.disk_usage('/')
        print(f"Disk: {disk.free / (1024**3):.1f} GB free ({disk.percent}% used)")
        
        # Check if running on Jetson
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                print(f"\n✓ Running on NVIDIA Jetson")
                print(f"  {f.read().strip()}")
        except:
            print("\n⚠ Not running on Jetson Nano")
        
        return True
        
    except ImportError:
        print("⚠ psutil not available - system monitoring limited")
        return True
    except Exception as e:
        print(f"✗ System resource test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("RSI-SENTIMENT TRADING SYSTEM - COMPONENT TEST")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Market Hours", test_market_hours),
        ("Cache & RSI", test_cache_and_rsi),
        ("System Resources", test_system_resources)
    ]
    
    for test_name, test_func in tests:
        try:
            passed = await test_func()
            if not passed:
                all_tests_passed = False
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED - System ready to run!")
        print("\nNext steps:")
        print("1. Review trading_config.json settings")
        print("2. Run: ./start_trading.sh")
        print("3. Open dashboard: http://localhost:8080")
    else:
        print("❌ SOME TESTS FAILED - Please fix issues before running")
        print("\nCommon fixes:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Ensure all .py files are in the same directory")
        print("3. Create trading_config.json if missing")
    print("=" * 60)

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher required")
        sys.exit(1)
    
    asyncio.run(main())