from pathlib import Path
from ultralytics import YOLO

def main():
    """
    Exports a pre-trained YOLOv8n model to the ONNX format.

    Dynamic axes are enabled to allow for variable input
    batch sizes and image dimensions during inference.
    """
    # Define paths
    build_dir = Path(__file__).parent
    output_path = build_dir / "detector.onnx"
    
    # Load the pre trained model
    model = YOLO("yolov8n.pt")

# Define the final desired output path for the model
    output_dir = Path(__file__).parent
    output_path = output_dir / "detector.onnx"

    # Step 1: Export the model. It will be saved in the current working directory.
    model.export(
        format="onnx",
        opset=13,
        dynamic=True
    )

    # Step 2: Move the newly created ONNX file to its final destination.
    # The default name is derived from the model, e.g., 'yolov8n.onnx'.

    # Construct full path to the exported model (where YOLO actually saved it)
    model_name_on_disk = Path(model.ckpt_path).stem + ".onnx"
    exported_model_path = Path(model.ckpt_path).parent / model_name_on_disk

    # Define the final desired output path in the 'models/' directory
    output_dir = Path(__file__).parent.parent  # points to genocr/models
    output_path = output_dir / "detector.onnx"

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Safely move the file
    if output_path.exists():
        output_path.unlink()
    exported_model_path.rename(output_path)

    print(f"Detector model exported successfully to: {output_path}")


if __name__ == "__main__":
    main()