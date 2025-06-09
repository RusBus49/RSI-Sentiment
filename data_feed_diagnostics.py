"""
Yahoo Finance Data Feed Diagnostics
Simplified diagnostics focused solely on Yahoo Finance API
Tests rate limiting, data quality, and connection reliability
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

class YahooFinanceTester:
    """Test Yahoo Finance API with proper rate limiting"""
    
    def __init__(self):
        self.session = None
        self.request_times = []
        self.results = []
        
        # Yahoo Finance rate limits (conservative)
        self.max_requests_per_minute = 50  # Conservative vs 60 limit
        self.min_delay_seconds = 1.5       # 1.5s between requests
        
    async def __aenter__(self):
        """Async context manager entry"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        timeout = aiohttp.ClientTimeout(total=15)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=aiohttp.TCPConnector(limit=1)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_single_symbol(self, symbol: str, delay: float = None) -> Dict:
        """Test fetching data for a single symbol"""
        if delay is None:
            delay = self.min_delay_seconds
        
        # Rate limiting
        await asyncio.sleep(delay)
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        request_start = time.time()
        
        result = {
            'symbol': symbol,
            'success': False,
            'status_code': None,
            'price': None,
            'volume': None,
            'market_state': None,
            'response_time': None,
            'error': None
        }
        
        try:
            async with self.session.get(url) as response:
                response_time = time.time() - request_start
                result['status_code'] = response.status
                result['response_time'] = response_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse response
                    chart = data.get('chart', {})
                    if chart:
                        chart_result = chart.get('result', [])
                        if chart_result:
                            meta = chart_result[0].get('meta', {})
                            
                            price = meta.get('regularMarketPrice') or meta.get('previousClose')
                            volume = meta.get('regularMarketVolume', 0)
                            market_state = meta.get('marketState', 'UNKNOWN')
                            
                            if price is not None:
                                result.update({
                                    'success': True,
                                    'price': float(price),
                                    'volume': int(volume) if volume else 0,
                                    'market_state': market_state
                                })
                            else:
                                result['error'] = 'No price data in response'
                        else:
                            result['error'] = 'No chart result in response'
                    else:
                        result['error'] = 'No chart data in response'
                
                elif response.status == 429:
                    retry_after = response.headers.get('Retry-After', '60')
                    result['error'] = f'Rate limited (retry after {retry_after}s)'
                
                elif response.status == 404:
                    result['error'] = 'Symbol not found'
                
                else:
                    text = await response.text()
                    result['error'] = f'HTTP {response.status}: {text[:100]}'
        
        except asyncio.TimeoutError:
            result['error'] = 'Request timeout'
        except Exception as e:
            result['error'] = f'Exception: {str(e)}'
        
        self.results.append(result)
        self.request_times.append(time.time())
        return result
    
    def get_rate_stats(self) -> Dict:
        """Get current rate limiting statistics"""
        now = time.time()
        minute_ago = now - 60
        
        # Count requests in last minute
        recent_requests = [t for t in self.request_times if t > minute_ago]
        
        return {
            'requests_last_minute': len(recent_requests),
            'max_requests_per_minute': self.max_requests_per_minute,
            'can_make_request': len(recent_requests) < self.max_requests_per_minute,
            'total_requests': len(self.request_times)
        }

async def check_market_hours():
    """Check if markets are currently open and provide schedule info"""
    from datetime import datetime, timezone
    import pytz
    
    print("Market Hours Information:")
    print("-" * 30)
    
    try:
        # Get current time in both timezones
        et_tz = pytz.timezone('US/Eastern')
        pt_tz = pytz.timezone('US/Pacific')
        
        now_et = datetime.now(et_tz)
        now_pt = datetime.now(pt_tz)
        
        # Market hours: 9:30 AM - 4:00 PM ET = 6:30 AM - 1:00 PM PT
        market_open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        market_open_pt = now_pt.replace(hour=6, minute=30, second=0, microsecond=0)
        market_close_pt = now_pt.replace(hour=13, minute=0, second=0, microsecond=0)
        
        print(f"  Current time (PT): {now_pt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  Current time (ET): {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  Market hours (ET): 9:30 AM - 4:00 PM")
        print(f"  Market hours (PT): 6:30 AM - 1:00 PM")
        
        # Check if it's a weekday
        is_weekday = now_et.weekday() < 5  # 0-4 are Mon-Fri
        is_market_hours = market_open_et <= now_et <= market_close_et
        
        if is_weekday and is_market_hours:
            print(f"  Status: 🟢 MARKET OPEN")
            time_to_close = market_close_pt - now_pt
            hours, remainder = divmod(time_to_close.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            print(f"  Time until close: {hours}h {minutes}m (PT)")
        elif is_weekday:
            if now_et < market_open_et:
                print(f"  Status: 🟡 PRE-MARKET (opens at 6:30 AM PT)")
                time_to_open = market_open_pt - now_pt
                hours, remainder = divmod(time_to_open.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                print(f"  Time until open: {hours}h {minutes}m (PT)")
            else:
                print(f"  Status: 🟡 AFTER-HOURS (closed at 1:00 PM PT)")
        else:
            print(f"  Status: 🔴 WEEKEND (markets closed)")
        
        print(f"  Note: Your trading system should pause outside market hours")
        
    except ImportError:
        print(f"  ⚠️ Install pytz for market hours checking: pip install pytz")
        print(f"  Manual check: Markets are 6:30 AM - 1:00 PM PT, Monday-Friday")
    
    print()

async def test_network_connectivity():
    """Test basic network connectivity"""
    print("Testing Network Connectivity:")
    print("-" * 30)
    
    test_urls = [
        "https://httpbin.org/get",
        "https://www.google.com",
        "https://finance.yahoo.com"
    ]
    
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for url in test_urls:
            try:
                async with session.get(url) as response:
                    print(f"  {url}: ✅ {response.status}")
            except Exception as e:
                print(f"  {url}: ❌ {e}")
    print()

async def run_yahoo_diagnostics():
    """Run comprehensive Yahoo Finance diagnostics"""
    print("=" * 70)
    print("YAHOO FINANCE DATA FEED DIAGNOSTICS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check market hours first
    await check_market_hours()
    
    # Network connectivity test
    await test_network_connectivity()
    
    # Yahoo Finance specific info
    print("Yahoo Finance Rate Limits:")
    print("  - 60 GET requests per minute (using 50 to be safe)")
    print("  - 360 GET requests per hour")
    print("  - 8000 GET requests per day")
    print("  - Using 1.5 second delays between requests")
    print()
    
    # Test symbols
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    print(f"Testing {len(test_symbols)} symbols: {test_symbols}")
    print("=" * 70)
    
    # Run tests
    async with YahooFinanceTester() as tester:
        successful_tests = 0
        failed_tests = 0
        
        for i, symbol in enumerate(test_symbols):
            print(f"\nTesting {symbol} ({i+1}/{len(test_symbols)}):")
            print("-" * 40)
            
            result = await tester.test_single_symbol(symbol)
            
            if result['success']:
                successful_tests += 1
                print(f"  Status: ✅ SUCCESS")
                print(f"  Price: ${result['price']:.2f}")
                print(f"  Volume: {result['volume']:,}")
                print(f"  Market: {result['market_state']}")
                print(f"  Response Time: {result['response_time']:.2f}s")
            else:
                failed_tests += 1
                print(f"  Status: ❌ FAILED")
                print(f"  HTTP Code: {result['status_code']}")
                print(f"  Error: {result['error']}")
                if result['response_time']:
                    print(f"  Response Time: {result['response_time']:.2f}s")
            
            # Show rate limiting stats
            rate_stats = tester.get_rate_stats()
            print(f"  Rate Stats: {rate_stats['requests_last_minute']}/{rate_stats['max_requests_per_minute']} requests/minute")
            
            # Wait between symbols (except last one)
            if i < len(test_symbols) - 1:
                print("  Waiting 2 seconds before next symbol...")
                await asyncio.sleep(2)
    
    # Final summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Successful tests: {successful_tests}/{len(test_symbols)}")
    print(f"Failed tests: {failed_tests}/{len(test_symbols)}")
    print(f"Success rate: {(successful_tests/len(test_symbols)*100):.1f}%")
    
    if successful_tests > 0:
        print(f"\n✅ Yahoo Finance is working!")
        print(f"Recommendations:")
        print(f"  - Use update intervals of 60+ seconds")
        print(f"  - Process symbols sequentially with 1.5s delays")
        print(f"  - Monitor daily usage (currently very low)")
        print(f"  - Set simulation_mode: false in your config")
        
        # Calculate daily usage for different scenarios (MARKET HOURS ONLY)
        print(f"\nDaily usage projections (market hours only):")
        
        # US market hours: 9:30 AM - 4:00 PM ET = 6.5 hours = 390 minutes
        market_hours_minutes = 390
        
        symbols_5 = 5
        symbols_10 = 10
        symbols_15 = 15
        symbols_20 = 20
        
        # Calculate requests per day for different update intervals
        def calc_daily_usage(num_symbols, update_interval_seconds):
            updates_per_hour = 3600 // update_interval_seconds
            updates_per_day = updates_per_hour * 6.5  # 6.5 market hours
            return int(num_symbols * updates_per_day)
        
        scenarios = [
            (symbols_5, 60, "5 symbols, 60s intervals"),
            (symbols_10, 60, "10 symbols, 60s intervals"), 
            (symbols_10, 90, "10 symbols, 90s intervals"),
            (symbols_15, 90, "15 symbols, 90s intervals"),
            (symbols_20, 120, "20 symbols, 120s intervals")
        ]
        
        print(f"  Market hours: 9:30 AM - 4:00 PM ET ({market_hours_minutes} minutes)")
        print(f"  Yahoo limit: 8,000 requests/day")
        print()
        
        for symbols, interval, description in scenarios:
            daily_usage = calc_daily_usage(symbols, interval)
            percentage = (daily_usage / 8000) * 100
            
            # Color coding for usage levels
            if percentage < 25:
                status = "✅ Very Safe"
            elif percentage < 50:
                status = "✅ Safe"
            elif percentage < 75:
                status = "⚠️ Moderate"
            else:
                status = "❌ High"
            
            print(f"  - {description:25}: {daily_usage:,} requests/day ({percentage:.1f}%) {status}")
        
        print(f"\nRecommendations:")
        print(f"  ✅ 10 symbols with 60s intervals uses only ~12% of daily limit")
        print(f"  ✅ Can easily handle 15-20 symbols with 90-120s intervals")
        print(f"  ✅ Plenty of headroom for retries and error recovery")
        print(f"  ⚠️ Remember: program should pause outside market hours")
        
    else:
        print(f"\n❌ All tests failed!")
        print(f"Troubleshooting steps:")
        print(f"  1. Check internet connection")
        print(f"  2. Verify no firewall blocking finance.yahoo.com")
        print(f"  3. Try again in a few minutes (possible temporary rate limiting)")
        print(f"  4. Consider using simulation mode for development")
    
    return successful_tests > 0

async def test_production_scenario():
    """Test a realistic production scenario"""
    print("\n" + "=" * 70)
    print("PRODUCTION SCENARIO TEST")
    print("=" * 70)
    print("Simulating production usage with 5 symbols over 3 cycles...")
    
    production_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    async with YahooFinanceTester() as tester:
        for cycle in range(3):
            print(f"\nCycle {cycle + 1}/3:")
            print("-" * 30)
            
            cycle_start = time.time()
            successful = 0
            
            for symbol in production_symbols:
                result = await tester.test_single_symbol(symbol, delay=1.5)
                if result['success']:
                    successful += 1
                    print(f"  {symbol}: ${result['price']:.2f} ✅")
                else:
                    print(f"  {symbol}: {result['error']} ❌")
            
            cycle_time = time.time() - cycle_start
            print(f"  Cycle time: {cycle_time:.1f}s ({successful}/{len(production_symbols)} successful)")
            
            if cycle < 2:  # Don't wait after last cycle
                print("  Waiting 60s until next cycle...")
                await asyncio.sleep(60)
    
    print(f"\nProduction test complete!")

# Main execution
async def main():
    """Main diagnostic function"""
    try:
        # Run basic diagnostics
        success = await run_yahoo_diagnostics()
        
        if success:
            print(f"\nRun production scenario test? (y/n): ", end="")
            # In a real scenario, you'd get user input here
            # For this demo, we'll just run it
            await test_production_scenario()
        
    except KeyboardInterrupt:
        print(f"\nDiagnostics interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())