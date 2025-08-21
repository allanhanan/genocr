import subprocess
import sys
from pathlib import Path
import os

def export_paddlex_correct_method():
    """
    Export PP-OCRv5 models using the correct PaddleX paddle2onnx method.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    models_dir = project_root / "models"
    
    detection_onnx_dir = models_dir / "ocr_detection_multilingual"
    recognition_onnx_dir = models_dir / "ocr_recognition_multilingual"
    
    detection_onnx_dir.mkdir(exist_ok=True, parents=True)
    recognition_onnx_dir.mkdir(exist_ok=True, parents=True)
    
    home_dir = os.path.expanduser("~")
    paddlex_cache = os.path.join(home_dir, ".paddlex", "official_models")
    
    # Use the CORRECT PaddleX model directories
    det_model_dir = os.path.join(paddlex_cache, "PP-OCRv5_server_det")
    rec_model_dir = os.path.join(paddlex_cache, "PP-OCRv5_server_rec")
    
    print("Converting PP-OCRv5 models using correct PaddleX method...")
    
    # Export detection model using the CORRECT syntax
    print("Converting detection model...")
    det_command = [
        "paddlex", "--paddle2onnx",
        "--paddle_model_dir", det_model_dir,
        "--onnx_model_dir", str(detection_onnx_dir),
        "--opset_version", "13"
    ]
    
    print(f"Running: {' '.join(det_command)}")
    result = subprocess.run(det_command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Detection conversion failed:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError("Detection model conversion failed")
    else:
        print("✓ Detection model converted successfully")
    
    # Export recognition model
    print("Converting recognition model...")
    rec_command = [
        "paddlex", "--paddle2onnx",
        "--paddle_model_dir", rec_model_dir,
        "--onnx_model_dir", str(recognition_onnx_dir), 
        "--opset_version", "13"
    ]
    
    print(f"Running: {' '.join(rec_command)}")
    result = subprocess.run(rec_command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Recognition conversion failed:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError("Recognition model conversion failed")
    else:
        print("✓ Recognition model converted successfully")
    
    print("\n=== PP-OCRv5 MULTILINGUAL EXPORT COMPLETE ===")
    
    # Find and verify the exported ONNX files
    det_onnx_files = list(detection_onnx_dir.glob("*.onnx"))
    rec_onnx_files = list(recognition_onnx_dir.glob("*.onnx"))
    
    if det_onnx_files:
        det_size = det_onnx_files[0].stat().st_size / (1024 * 1024)
        print(f"✓ Detection: {det_onnx_files[0].name} ({det_size:.1f} MB)")
    else:
        print("✗ Detection: No ONNX file found")
        
    if rec_onnx_files:
        rec_size = rec_onnx_files[0].stat().st_size / (1024 * 1024)
        print(f"✓ Recognition: {rec_onnx_files[0].name} ({rec_size:.1f} MB)")
    else:
        print("✗ Recognition: No ONNX file found")
    
    print("\nSUCCESS: General-purpose multilingual PP-OCRv5 models exported!")

if __name__ == "__main__":
    export_paddlex_correct_method()
