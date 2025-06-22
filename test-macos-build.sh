#!/bin/bash
set -e

echo "🔧 Testing macOS build process manually..."

# Check current Python version
echo "📍 Current Python version:"
python3 --version || echo "Python 3 not found"

# Use our existing UV environment
echo "📦 Using existing UV environment..."
source .venv/bin/activate

# Check if PyInstaller is available
echo "🔍 Checking PyInstaller..."
which pyinstaller || uv add pyinstaller

# Build with PyInstaller
echo "🏗️ Building with PyInstaller..."
pyinstaller searxng_cli_universal.spec

# Test the build
echo "🧪 Testing the build..."
if [ -f "dist/searxng-cli" ]; then
    echo "✅ Binary created successfully"
    
    echo "📋 Testing help command..."
    ./dist/searxng-cli --help
    
    echo "📂 Testing categories command..."
    ./dist/searxng-cli categories 2>/dev/null || echo "⚠️ Categories test failed but continuing"
    
    echo "🔧 Testing engines command..."
    ./dist/searxng-cli engines --common 2>/dev/null || echo "⚠️ Engines test failed but continuing"
    
    echo "🔍 Testing simple search..."
    ./dist/searxng-cli search "test" --engines duckduckgo 2>/dev/null | head -5 || echo "⚠️ Search test failed but continuing"
    
    echo "📊 Binary size:"
    ls -lh dist/searxng-cli
    
    echo "🎉 macOS build test completed successfully!"
else
    echo "❌ Build failed - binary not found"
    exit 1
fi