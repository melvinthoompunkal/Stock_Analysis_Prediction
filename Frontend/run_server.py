#!/usr/bin/env python3
"""
Startup script for the Stock Analysis Dashboard.
This script handles the complete startup process including dependency checks.
"""

import sys
import os
import subprocess
import time

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'flask', 'flask_cors', 'psycopg2', 'pandas', 'numpy', 
        'polygon', 'newsapi', 'textblob', 'yfinance'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'flask_cors':
                import flask_cors
            elif package == 'psycopg2':
                import psycopg2
            elif package == 'polygon':
                from polygon import RESTClient
            elif package == 'newsapi':
                from newsapi import NewsApiClient
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def check_postgresql():
    """Check if PostgreSQL is accessible."""
    try:
        import psycopg2
        
        # Try to connect with default credentials
        conn = psycopg2.connect(
            dbname="mydb",
            user="melvint",
            password="MelvinGeorgi",
            host="localhost",
            port="5432"
        )
        conn.close()
        print("✅ PostgreSQL connection successful")
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("Please ensure PostgreSQL is running and credentials are correct")
        return False

def check_api_keys():
    """Check if required API keys are available."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        polygon_key = os.getenv("POLYGON_API_KEY")
        news_key = os.getenv("NEWS_API_KEY")
        
        if not polygon_key:
            print("⚠️ POLYGON_API_KEY not found in .env file")
        else:
            print("✅ POLYGON_API_KEY found")
            
        if not news_key:
            print("⚠️ NEWS_API_KEY not found in .env file")
        else:
            print("✅ NEWS_API_KEY found")
            
        return True
        
    except Exception as e:
        print(f"⚠️ Could not check API keys: {e}")
        return True  # Don't block startup for missing API keys

def start_server():
    """Start the Flask server."""
    try:
        print("\n🚀 Starting Stock Analysis Dashboard...")
        print("=" * 50)
        
        # Import and run the Flask app
        from app import app
        
        port = int(os.getenv('PORT', '5500'))
        print(f"✅ Server starting on http://localhost:{port}")
        print("📊 Open your browser and navigate to the URL above")
        print("🔄 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        app.run(debug=True, host='0.0.0.0', port=port)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        return False
    
    return True

def main():
    """Main startup function."""
    print("Stock Analysis Dashboard - Startup Check")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        return False
    
    print("\n🗄️ Checking PostgreSQL...")
    if not check_postgresql():
        return False
    
    print("\n🔑 Checking API keys...")
    check_api_keys()
    
    print("\n" + "=" * 40)
    print("✅ All checks passed! Starting server...")
    
    # Start the server
    return start_server()

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Startup failed. Please fix the issues above and try again.")
        sys.exit(1)
