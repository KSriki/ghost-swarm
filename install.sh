#!/bin/bash

# Ghost Swarm Installation Script

set -e

echo "👻🐝 Ghost Swarm Installation"
echo "═══════════════════════════════════"
echo ""

# Check for UV
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ UV found: $(uv --version)"

# Create virtual environment only if it doesn't exist
echo ""
if [ -d ".venv" ]; then
    echo "✓ Virtual environment already exists"
else
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    echo "❌ Could not find virtual environment activation script"
    exit 1
fi

echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
uv pip install -e ".[dev]"

echo "✓ Dependencies installed"

# Check for .env file
echo ""
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating from template..."
    cp .env .env.example 2>/dev/null || echo "Please create a .env file with your API keys"
else
    echo "✓ .env file exists"
fi

# Check for Redis
echo ""
if command -v redis-cli &> /dev/null; then
    echo "✓ Redis found: $(redis-cli --version)"
else
    echo "⚠️  Redis not found. Install it or run with Docker:"
    echo "   docker run -d -p 6379:6379 redis:latest"
fi

echo ""
echo "═══════════════════════════════════"
echo "✓ Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Configure your .env file with API keys"
echo "  2. Start Redis if not running"
echo "  3. Run: python main.py"
echo ""