import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path

def test_detector_inference():
    """
    Verifies the working of detector.onnx model.
    """

    # Setup paths and parameters
    models_dir = Path(__file__).parent.parent.parent / "models"
    model_path = models_dir / "detector.onnx"
    image_path = Path(__file__).parent.parent / "test_assets/sample_document.png"
    input_shape = (640, 640)

    # Ensure the model file exists
    assert model_path.exists(), "Detector ONNX model not found."

    ort.set_default_logger_severity(3)

    # Load the ONNX session, preferring CUDA
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(str(model_path), providers=providers)

    print(f"ONNX Runtime is using provider: {session.get_providers()[0]}")
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image_resized = image.resize(input_shape)
    
    # Convert image to a CHW tensor and normalize
    # The model expects a float16 input as it was exported with half=True.
    image_data = np.array(image_resized).transpose(2, 0, 1)  # HWC to CHW
    input_tensor = np.expand_dims(image_data, axis=0).astype(np.float32) / 255.0

    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_tensor})

    # Sanity checks on the output
    assert outputs is not None, "Inference returned None."
    assert len(outputs) > 0, "Inference returned no outputs."
    
    # The output shape for YOLOv8 is (batch_size, 4 + num_classes, num_proposals)
    # For COCO (80 classes), which is (1, 84, 8400).
    assert outputs[0].shape[0] == 1, "Output batch size is not 1."
    assert outputs[0].shape[2] > 0, "Model produced zero proposals."

    print(f"Detector test passed. Output shape: {outputs[0].shape}")

if __name__ == "__main__":
    test_detector_inference()