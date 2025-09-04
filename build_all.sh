#!/bin/bash

set -e

echo "=== GenOCR Zero-Config Build ==="

# Get the script's directory (base project dir)
BASE_DIR=$(dirname "$(realpath "$0")")

# Stage 1: Build cpp_core
echo "Stage 1: Building C++ core..."
cd "$BASE_DIR/cpp_core"
mkdir -p build && cd build
cmake ..
cmake --build . -j

cd "$BASE_DIR"
echo "✓ C++ core built successfully"

# Stage 2: Install Python bindings with bundled libraries
echo "Stage 2: Installing Python bindings with bundled libraries..."
cd "$BASE_DIR/python_bindings"

# Uninstall previous version
pip uninstall genocr -y 2>/dev/null || true

# Install package with bundled libraries
pip install . --verbose --force-reinstall

echo ""
echo "Build completed with auto-loading libraries!"

# Test installation (should work without any manual configuration)
echo "Testing zero-config installation..."
python -c "
import sys
try:
    import genocr
    print('genocr imported successfully')
    print(f'Version: {genocr.__version__}')
    
    from genocr._genocr_core import GenOCR
    print('C++ core loaded successfully!')
    print('')
    print('SUCCESS: GenOCR ready for immediate use!')
    print('   No manual library configuration required!')
    print('')
    print('Usage:')
    print('   import genocr')
    print('   result = genocr.process_image(\"image.jpg\", \"models/\")')
    
except ImportError as e:
    print('Import failed:', e)
    print('')
    print('This may indicate missing system dependencies.')
    print('Try installing: sudo apt-get install libgomp1')
    sys.exit(1)
except Exception as e:
    print('Unexpected error:', e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo ""
echo "GenOCR is now ready for usage!"
