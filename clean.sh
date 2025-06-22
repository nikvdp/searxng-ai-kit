#!/bin/bash
# SearXNG CLI Build Cleanup Script

echo "🧹 Cleaning SearXNG CLI build artifacts..."

# Remove PyInstaller build directories
rm -rf build/
rm -rf dist/
rm -rf __pycache__/
rm -rf searx/__pycache__/

# Remove PyInstaller spec backups
rm -f *.spec.bak

# Remove any temporary release directories
rm -rf release/

# Remove Python cache files
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove UV cache (optional - uncomment if needed)
# rm -rf .venv/

echo "✅ Cleanup complete!"
echo ""
echo "Build directories cleaned:"
echo "  - build/"
echo "  - dist/"  
echo "  - __pycache__/"
echo "  - release/"
echo ""
echo "To rebuild: ./test-macos-build.sh or use GitHub Actions workflow"