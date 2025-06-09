"""
Market Hours Helper
Handles timezone conversions and market hours checking
Optimized for West Coast users trading East Coast markets
"""

from datetime import datetime, time
from typing import Dict, Tuple, Optional
import pytz
import logging

class MarketHoursChecker:
    """Helper class for market hours and timezone management"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Timezone setup
        self.user_tz = pytz.timezone(config.get('user_timezone', 'US/Pacific'))
        self.market_tz = pytz.timezone(config.get('market_timezone', 'US/Eastern'))
        
        # Market hours (in market timezone)
        self.market_open_time = self._parse_time(config.get('market_open_time', '09:30'))
        self.market_close_time = self._parse_time(config.get('market_close_time', '16:00'))
        
        # Trading days (0=Monday, 6=Sunday)
        self.trading_days = [0, 1, 2, 3, 4]  # Mon-Fri
        
        self.logger = logging.getLogger(__name__)
    
    def _parse_time(self, time_str: str) -> time:
        """Parse time string like '09:30' into time object"""
        hour, minute = map(int, time_str.split(':'))
        return time(hour, minute)
    
    def is_market_open(self, check_time: Optional[datetime] = None) -> bool:
        """Check if market is currently open"""
        if check_time is None:
            check_time = datetime.now(self.market_tz)
        elif check_time.tzinfo is None:
            # Assume user timezone if no timezone info
            check_time = self.user_tz.localize(check_time)
        
        # Convert to market timezone
        market_time = check_time.astimezone(self.market_tz)
        
        # Check if it's a trading day
        if market_time.weekday() not in self.trading_days:
            return False
        
        # Check if within market hours
        current_time = market_time.time()
        return self.market_open_time <= current_time <= self.market_close_time
    
    def get_market_status(self) -> Dict:
        """Get comprehensive market status information"""
        now_user = datetime.now(self.user_tz)
        now_market = now_user.astimezone(self.market_tz)
        
        # Create market open/close times for today
        today_market = now_market.date()
        market_open_today = self.market_tz.localize(
            datetime.combine(today_market, self.market_open_time)
        )
        market_close_today = self.market_tz.localize(
            datetime.combine(today_market, self.market_close_time)
        )
        
        # Convert to user timezone
        market_open_user = market_open_today.astimezone(self.user_tz)
        market_close_user = market_close_today.astimezone(self.user_tz)
        
        is_open = self.is_market_open()
        is_trading_day = now_market.weekday() in self.trading_days
        
        status = {
            'is_open': is_open,
            'is_trading_day': is_trading_day,
            'current_time_user': now_user,
            'current_time_market': now_market,
            'market_open_user': market_open_user,
            'market_close_user': market_close_user,
            'market_open_market': market_open_today,
            'market_close_market': market_close_today,
            'user_timezone': str(self.user_tz),
            'market_timezone': str(self.market_tz)
        }
        
        # Calculate time until next open/close
        if is_open:
            status['time_until_close'] = market_close_user - now_user
            status['status_text'] = 'MARKET OPEN'
        elif is_trading_day and now_market < market_open_today:
            status['time_until_open'] = market_open_user - now_user
            status['status_text'] = 'PRE-MARKET'
        elif is_trading_day:
            status['status_text'] = 'AFTER-HOURS'
        else:
            status['status_text'] = 'WEEKEND/HOLIDAY'
        
        return status
    
    def get_next_market_open(self) -> datetime:
        """Get the next market open time in user timezone"""
        now_market = datetime.now(self.market_tz)
        today = now_market.date()
        
        # Try today first
        market_open_today = self.market_tz.localize(
            datetime.combine(today, self.market_open_time)
        )
        
        if (now_market < market_open_today and 
            now_market.weekday() in self.trading_days):
            return market_open_today.astimezone(self.user_tz)
        
        # Find next trading day
        days_ahead = 1
        while days_ahead <= 7:
            next_date = today + timedelta(days=days_ahead)
            if next_date.weekday() in self.trading_days:
                next_open = self.market_tz.localize(
                    datetime.combine(next_date, self.market_open_time)
                )
                return next_open.astimezone(self.user_tz)
            days_ahead += 1
        
        # Fallback (shouldn't happen)
        return market_open_today.astimezone(self.user_tz)
    
    def should_system_run(self) -> Tuple[bool, str]:
        """Determine if trading system should be running"""
        if not self.config.get('market_hours_only', True):
            return True, "Market hours checking disabled"
        
        status = self.get_market_status()
        
        if status['is_open']:
            return True, "Market is open"
        elif status['status_text'] == 'PRE-MARKET':
            # Could optionally run during pre-market
            return False, f"Pre-market: opens in {status.get('time_until_open', 'unknown')}"
        elif status['status_text'] == 'AFTER-HOURS':
            next_open = self.get_next_market_open()
            return False, f"After hours: next open at {next_open.strftime('%I:%M %p %Z')}"
        else:
            next_open = self.get_next_market_open()
            return False, f"Weekend: next open at {next_open.strftime('%a %I:%M %p %Z')}"
    
    def log_market_status(self):
        """Log current market status"""
        status = self.get_market_status()
        
        self.logger.info(f"Market Status: {status['status_text']}")
        self.logger.info(f"Current time (PT): {status['current_time_user'].strftime('%H:%M:%S')}")
        self.logger.info(f"Current time (ET): {status['current_time_market'].strftime('%H:%M:%S')}")
        self.logger.info(f"Market hours (PT): {status['market_open_user'].strftime('%H:%M')} - {status['market_close_user'].strftime('%H:%M')}")
        
        if 'time_until_close' in status:
            hours, remainder = divmod(status['time_until_close'].seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            self.logger.info(f"Time until close: {hours}h {minutes}m")
        elif 'time_until_open' in status:
            hours, remainder = divmod(status['time_until_open'].seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            self.logger.info(f"Time until open: {hours}h {minutes}m")

# Convenience functions for easy integration
def create_market_hours_checker(config_file: str = "trading_config.json") -> MarketHoursChecker:
    """Create market hours checker from config file"""
    import json
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        system_config = config.get('system', {})
        return MarketHoursChecker(system_config)
        
    except FileNotFoundError:
        # Default West Coast configuration
        default_config = {
            'user_timezone': 'US/Pacific',
            'market_timezone': 'US/Eastern',
            'market_open_time': '09:30',
            'market_close_time': '16:00',
            'market_hours_only': True
        }
        return MarketHoursChecker(default_config)

def is_market_open_now() -> bool:
    """Quick check if market is open right now"""
    checker = create_market_hours_checker()
    return checker.is_market_open()

def get_market_status_summary() -> str:
    """Get a human-readable market status summary"""
    checker = create_market_hours_checker()
    status = checker.get_market_status()
    
    if status['is_open']:
        time_left = status['time_until_close']
        hours, remainder = divmod(time_left.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"🟢 Market OPEN - closes in {hours}h {minutes}m (1:00 PM PT)"
    
    elif status['status_text'] == 'PRE-MARKET':
        time_until = status['time_until_open']
        hours, remainder = divmod(time_until.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"🟡 PRE-MARKET - opens in {hours}h {minutes}m (6:30 AM PT)"
    
    elif status['status_text'] == 'AFTER-HOURS':
        return f"🟡 AFTER-HOURS - market closed at 1:00 PM PT"
    
    else:
        next_open = checker.get_next_market_open()
        return f"🔴 CLOSED - next open: {next_open.strftime('%a 6:30 AM PT')}"

# Example usage and testing
if __name__ == "__main__":
    import json
    from datetime import timedelta
    
    # Test with West Coast configuration
    config = {
        'user_timezone': 'US/Pacific',
        'market_timezone': 'US/Eastern',
        'market_open_time': '09:30',
        'market_close_time': '16:00',
        'market_hours_only': True
    }
    
    checker = MarketHoursChecker(config)
    
    print("=== WEST COAST MARKET HOURS TEST ===")
    print()
    
    # Current status
    status = checker.get_market_status()
    print(f"Current Status: {status['status_text']}")
    print(f"Is Open: {status['is_open']}")
    print(f"Current Time (PT): {status['current_time_user'].strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
    print(f"Current Time (ET): {status['current_time_market'].strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
    print()
    
    print(f"Market Hours:")
    print(f"  Pacific Time: {status['market_open_user'].strftime('%I:%M %p')} - {status['market_close_user'].strftime('%I:%M %p')}")
    print(f"  Eastern Time: {status['market_open_market'].strftime('%I:%M %p')} - {status['market_close_market'].strftime('%I:%M %p')}")
    print()
    
    # System running recommendation
    should_run, reason = checker.should_system_run()
    print(f"Should trading system run: {should_run}")
    print(f"Reason: {reason}")
    print()
    
    # Quick status summary
    print(f"Summary: {get_market_status_summary()}")
    
    # Test at different times
    print(f"\n=== TESTING DIFFERENT TIMES ===")
    test_times = [
        "05:00",  # Before market
        "06:30",  # Market open (PT)
        "10:00",  # Mid-day
        "13:00",  # Market close (PT)
        "15:00",  # After market
    ]
    
    for test_time in test_times:
        hour, minute = map(int, test_time.split(':'))
        test_dt = datetime.now(checker.user_tz).replace(hour=hour, minute=minute, second=0, microsecond=0)
        is_open = checker.is_market_open(test_dt)
        print(f"  {test_time} PT: {'🟢 OPEN' if is_open else '🔴 CLOSED'}")
