# PowerShell script: build_all.ps1
# Run this script in a PowerShell terminal (preferably as Administrator)

$ErrorActionPreference = "Stop"

Write-Host "=== GenOCR Zero-Config Build (Windows) ==="

# Get base directory of the script
$BASE_DIR = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition

# Stage 1: Build cpp_core
Write-Host "Stage 1: Building C++ core..."

Set-Location "$BASE_DIR\cpp_core"
New-Item -ItemType Directory -Force -Path "build" | Out-Null
Set-Location "build"

# Run CMake to configure and build
cmake .. 
cmake --build . --config Release

Set-Location $BASE_DIR
Write-Host "✓ C++ core built successfully"

# Stage 2: Install Python bindings with bundled libraries
Write-Host "Stage 2: Installing Python bindings with bundled libraries..."

Set-Location "$BASE_DIR\python_bindings"

# Uninstall previous genocr (ignore error if not installed)
try {
    pip uninstall genocr -y | Out-Null
} catch {
    # Ignore uninstall errors
}

# Install current version
pip install . --verbose --force-reinstall

Write-Host ""
Write-Host "Build completed with auto-loading libraries!"

# Test installation
Write-Host "Testing zero-config installation..."

$testScript = @"
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
    print('   result = genocr.process_image("image.jpg", "models/")')

except ImportError as e:
    print('Import failed:', e)
    print('')
    print('This may indicate missing system dependencies.')
    sys.exit(1)
except Exception as e:
    print('Unexpected error:', e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@

python -c "$testScript"

Write-Host ""
Write-Host "GenOCR is now ready for usage!"
