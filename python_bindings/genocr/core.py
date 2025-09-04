"""High-level Python interface for GenOCR"""

import os
import tempfile
from pathlib import Path
from typing import List, Union, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ._genocr_core import GenOCR as _GenOCRCore
except ImportError as e:
    raise ImportError(
        "GenOCR C++ core not available. Please build the extension:\n"
        "  cd python_bindings && pip install -e .\n"
        f"Original error: {e}"
    ) from e


class GenOCR:
    """High-level GenOCR interface with Python conveniences"""
    
    def __init__(self, models_dir: str = "../models", use_cuda: bool = False):
        """
        Initialize GenOCR pipeline
        
        Args:
            models_dir: Path to models directory (default: ../models)
            use_cuda: Enable CUDA acceleration (default: False)
        """
        models_path = Path(models_dir).resolve()
        if not models_path.exists():
            raise FileNotFoundError(f"Models directory not found: {models_path}")
        
        # Check required model files
        required_models = [
            "detector.onnx",
            "ocr_detection_multilingual/inference.onnx",
            "ocr_recognition_multilingual/inference.onnx",
            "ocr_recognition_multilingual/ppocrv5_dict.txt"
        ]
        
        for model in required_models:
            model_path = models_path / model  
            if not model_path.exists():
                raise FileNotFoundError(f"Required model not found: {model_path}")
        
        self._core = _GenOCRCore(str(models_path), use_cuda)
        self.models_dir = models_path
        self.use_cuda = use_cuda
    
    def process(self, image: Union[str, Path, np.ndarray]) -> str:
        """
        Process an image and return OCR results
        
        Args:
            image: Image file path or numpy array
            
        Returns:
            OCR results with context detection and AI correction
        """
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            return self._core.run(str(image_path))
        
        elif isinstance(image, np.ndarray):
            if cv2 is None:
                raise ImportError("opencv-python is required for numpy array input")
            
            # Save numpy array to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                success = cv2.imwrite(tmp.name, image)
                if not success:
                    raise RuntimeError(f"Failed to save image to {tmp.name}")
                
                try:
                    return self._core.run(tmp.name)
                finally:
                    os.unlink(tmp.name)
        else:
            raise ValueError("Image must be file path (str/Path) or numpy array")
    
    def __repr__(self):
        return f"GenOCR(models_dir='{self.models_dir}', use_cuda={self.use_cuda})"


def process_image(image_path: Union[str, Path], 
                 models_dir: str = "../models", 
                 use_cuda: bool = False) -> str:
    """
    Convenience function to process a single image
    
    Args:
        image_path: Path to image file
        models_dir: Path to models directory
        use_cuda: Enable CUDA acceleration
        
    Returns:
        OCR results string
    """
    ocr = GenOCR(models_dir, use_cuda)
    return ocr.process(image_path)


def process_batch(image_paths: List[Union[str, Path]], 
                 models_dir: str = "../models", 
                 use_cuda: bool = False,
                 output_dir: Optional[str] = None) -> List[str]:
    """
    Process multiple images in batch
    
    Args:
        image_paths: List of image file paths
        models_dir: Path to models directory  
        use_cuda: Enable CUDA acceleration
        output_dir: Optional directory to save text results
        
    Returns:
        List of OCR result strings
    """
    ocr = GenOCR(models_dir, use_cuda)
    results = []
    
    for img_path in image_paths:
        try:
            result = ocr.process(img_path)
            results.append(result)
            
            # Save to file if output directory specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True)
                
                out_file = output_path / f"{Path(img_path).stem}.txt"
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(result)
                    
        except Exception as e:
            error_msg = f"Error processing {img_path}: {e}"
            print(error_msg)
            results.append(error_msg)
    
    return results


def check_installation() -> bool:
    """
    Check if GenOCR is properly installed
    
    Returns:
        True if installation is valid
    """
    try:
        from ._genocr_core import get_version
        print(f"GenOCR version: {get_version()}")
        return True
    except ImportError as e:
        print(f"GenOCR installation issue: {e}")
        return False
