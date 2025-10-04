#!/bin/bash

# Stock Analysis Dashboard Startup Script
# This script starts the complete dashboard system

echo "🚀 Stock Analysis Dashboard - Startup Script"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "Frontend/app.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Make sure you're in the Stock_Analysis_Prediction folder"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "   Please install Python 3.8 or higher"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is not installed"
    exit 1
fi

echo "✅ pip found: $(pip3 --version)"

# Navigate to Frontend directory
cd Frontend

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found in Frontend directory"
    exit 1
fi

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies"
    echo "   Please check your internet connection and try again"
    exit 1
fi

echo "✅ Dependencies installed successfully"

# Check if PostgreSQL is running
echo ""
echo "🗄️ Checking PostgreSQL connection..."

# Try to connect to PostgreSQL (adjust credentials as needed)
python3 -c "
import psycopg2
try:
    conn = psycopg2.connect(
        dbname='mydb',
        user='melvint',
        password='MelvinGeorgi',
        host='localhost',
        port='5432'
    )
    conn.close()
    print('✅ PostgreSQL connection successful')
except Exception as e:
    print(f'❌ PostgreSQL connection failed: {e}')
    print('Please ensure PostgreSQL is running and credentials are correct')
    exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Check if .env file exists
if [ ! -f "../.env" ]; then
    echo ""
    echo "⚠️ Warning: .env file not found"
    echo "   Creating a template .env file..."
    
    cat > ../.env << EOF
# Stock Analysis Dashboard Environment Variables
# Replace with your actual API keys

# Polygon.io API Key (for market data)
POLYGON_API_KEY=your_polygon_api_key_here

# NewsAPI Key (for sentiment analysis)
NEWS_API_KEY=your_news_api_key_here
EOF
    
    echo "✅ Created .env template file"
    echo "   Please edit .env file and add your API keys"
    echo "   Then run this script again"
    exit 1
fi

echo "✅ Environment file found"

# Start the server
echo ""
echo "🚀 Starting Stock Analysis Dashboard..."
echo "   Server will be available at: http://localhost:5000"
echo "   Press Ctrl+C to stop the server"
echo ""

# Run the startup script
python3 run_server.py

echo ""
echo "👋 Dashboard stopped. Thank you for using Stock Analysis Dashboard!"
