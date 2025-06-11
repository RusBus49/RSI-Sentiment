#!/bin/bash
# Startup script for RSI-Sentiment Trading System on Jetson Nano

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}RSI-Sentiment Trading System${NC}"
echo -e "${GREEN}Jetson Nano Startup Script${NC}"
echo -e "${GREEN}================================${NC}"

# Check if running on Jetson
if [ -f /etc/nv_tegra_release ]; then
    echo -e "${GREEN}✓ Running on NVIDIA Jetson${NC}"
    # Set Jetson to max performance
    echo -e "${YELLOW}Setting Jetson to maximum performance...${NC}"
    sudo nvpmodel -m 0 2>/dev/null
    sudo jetson_clocks 2>/dev/null
else
    echo -e "${YELLOW}⚠ Not running on Jetson Nano${NC}"
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "Python version: ${PYTHON_VERSION}"

# Check if virtual environment exists
if [ ! -d "trading_env" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv trading_env
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source trading_env/bin/activate

# Function to check if package is installed
check_package() {
    if python -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}✓ $1 is installed${NC}"
        return 0
    else
        echo -e "${RED}✗ $1 is not installed${NC}"
        return 1
    fi
}

# Check required packages
echo -e "\n${YELLOW}Checking required packages...${NC}"
MISSING_PACKAGES=0

for package in aiohttp aiofiles websockets numpy psutil pytz; do
    if ! check_package $package; then
        MISSING_PACKAGES=$((MISSING_PACKAGES + 1))
    fi
done

# Install missing packages
if [ $MISSING_PACKAGES -gt 0 ]; then
    echo -e "\n${YELLOW}Installing missing packages...${NC}"
    pip install aiohttp aiofiles websockets numpy psutil pytz
fi

# Create necessary directories
echo -e "\n${YELLOW}Setting up directories...${NC}"
mkdir -p logs
mkdir -p data
mkdir -p web_dashboard

# Copy dashboard file if it doesn't exist
if [ ! -f "web_dashboard/index.html" ]; then
    echo -e "${YELLOW}Creating web dashboard...${NC}"
    # Create a simple placeholder if the dashboard HTML isn't available
    echo "<html><body><h1>Dashboard will be available here</h1></body></html>" > web_dashboard/index.html
fi

# Check if config file exists
if [ ! -f "trading_config.json" ]; then
    echo -e "${RED}Error: trading_config.json not found!${NC}"
    echo -e "${YELLOW}Please create a configuration file before running.${NC}"
    exit 1
fi

# Check available memory
AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}')
echo -e "\nAvailable memory: ${AVAILABLE_MEM} GB"

if (( $(echo "$AVAILABLE_MEM < 0.5" | bc -l) )); then
    echo -e "${YELLOW}⚠ Low memory warning! Consider adding swap space.${NC}"
fi

# Start the trading system
echo -e "\n${GREEN}Starting RSI-Sentiment Trading System...${NC}"
echo -e "${YELLOW}Dashboard: http://localhost:8080${NC}"
echo -e "${YELLOW}WebSocket: ws://localhost:8765${NC}"
echo -e "${YELLOW}Logs: tail -f trading_system.log${NC}"
echo -e "\nPress Ctrl+C to stop\n"

# Run with proper error handling
python3 main_controller.py 2>&1 | tee -a logs/startup_$(date +%Y%m%d_%H%M%S).log

# Cleanup on exit
deactivate
echo -e "\n${GREEN}Trading system stopped.${NC}"