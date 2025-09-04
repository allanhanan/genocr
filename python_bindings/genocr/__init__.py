"""GenOCR - Multi-language OCR with AI correction"""

__version__ = "0.1.0"

import os
import sys
import ctypes
from pathlib import Path

def _setup_library_path():
    """Automatically configure library path for bundled libraries"""
    package_dir = Path(__file__).parent
    libs_dir = package_dir / "libs"
    
    if libs_dir.exists():
        # Pre-load shared libraries in CORRECT ORDER using ctypes
        try:
            # 1. Load ONNX Runtime first (base dependency)
            onnx_runtime_paths = list(libs_dir.glob("libonnxruntime.so*"))
            for onnx_runtime in sorted(onnx_runtime_paths):
                if onnx_runtime.is_file():
                    print(f"Loading: {onnx_runtime.name}")
                    ctypes.CDLL(str(onnx_runtime), mode=ctypes.RTLD_GLOBAL)
                    break
            
            # 2. Load ONNX GenAI second (depends on ONNX Runtime)
            onnx_genai_paths = list(libs_dir.glob("libonnxruntime-genai.so*"))
            for onnx_genai in sorted(onnx_genai_paths):
                if onnx_genai.is_file():
                    print(f"Loading: {onnx_genai.name}")
                    ctypes.CDLL(str(onnx_genai), mode=ctypes.RTLD_GLOBAL)
                    break
                    
        except OSError as e:
            print(f"Warning: Could not pre-load bundled libraries: {e}")
        
        # Fallback: set LD_LIBRARY_PATH
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if str(libs_dir) not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{libs_dir}:{current_ld_path}"

# Setup library path BEFORE importing C++ extension
_setup_library_path()

try:
    from ._genocr_core import GenOCR
    
    def process_image(image_path: str, models_dir: str) -> str:
        """Process a single image with OCR"""
        ocr = GenOCR(models_dir)
        return ocr.Run(image_path)
    
    def process_batch(image_paths: list, models_dir: str) -> list:
        """Process multiple images with OCR"""
        ocr = GenOCR(models_dir)
        return [ocr.Run(path) for path in image_paths]
        
    print("✓ GenOCR C++ core loaded successfully")
    
except ImportError as e:
    print(f"ERROR: Could not import C++ extension: {e}")
    print("This usually means:")
    print("1. The C++ core wasn't built properly")
    print("2. Library dependencies are missing")
    print("3. Version conflicts between libraries")
    
    def process_image(image_path: str, models_dir: str) -> str:
        return f"GenOCR C++ extension not available: {e}"
    
    def process_batch(image_paths: list, models_dir: str) -> list:
        return [f"GenOCR C++ extension not available: {e}"] * len(image_paths)
