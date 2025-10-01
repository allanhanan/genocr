# GenOCR

GenOCR is a C++20 OCR pipeline that combines object context detection, multilingual text detection/recognition, and large‑context OCR correction in a single executable.
It uses ONNX Runtime for YOLOv8 and PaddleOCR, and ONNX Runtime GenAI for the Phi‑3 Vision corrector, with parallel execution and multiple deployment options including CLI, Python bindings, and HTTP API.

## Features

- Context object detection with YOLOv8 ONNX to surface scene cues used during correction.
- Multilingual text detection and recognition with PaddleOCR (detection + recognition + charset loading).
- OCR correction with a Phi‑3 Vision model via ONNX Runtime GenAI, handling large prompts and image inputs.
- Parallel YOLO and OCR execution, then single‑pass correction over the concatenated raw OCR text.
- Multiple deployment options: standalone CLI, Python library, and production HTTP API server.


## Requirements

- C++20 toolchain and CMake 3.18+; a system OpenCV installation discoverable by CMake (find_package OpenCV REQUIRED).
- Network access during first build to download ONNX Runtime and ONNX Runtime GenAI archives; system unzip available for extraction on Linux/macOS where used.
- Optional CUDA Toolkit: detected at configure time for GenAI package selection and enabled at runtime with the -c flag for standard inference providers.


## Models

- YOLOv8 detector: models/detector.onnx.
- PaddleOCR detection model: <models_dir>/ocr_detection_multilingual/inference.onnx (resolved from -m or default ../../models at runtime).
- PaddleOCR recognition model: <models_dir>/ocr_recognition_multilingual/inference.onnx (same resolution as above).
- PaddleOCR dictionary: <models_dir>/ocr_recognition_multilingual/ppocrv5_dict.txt (loaded at startup by the recognition path).
- Corrector (Phi‑3 Vision) directory: <models_dir>/corrector (passed to OgaModel::Create).

Note: When running from cpp_core/build, the default models directory is resolved as ../../models unless overridden via -m.

## Build

### C++ Core

The core builds from cpp_core using CMake, auto‑fetching ONNX Runtime, ONNX Runtime GenAI, and Clipper2 at configure time.
OpenCV must be available on the system and found by CMake; CUDA is optional and auto‑detected to select the GenAI package variant.

Build steps:

```bash
cd cpp_core
mkdir -p build && cd build
cmake .. && cmake --build . -j
```

This produces the genocr executable in cpp_core/build/ linked against the downloaded ONNX Runtime components and system OpenCV.

### Python Bindings

Python bindings provide a native Python interface to the C++ OCR pipeline via pybind11. The package bundles all required ONNX Runtime libraries for zero-configuration deployment.

Build and install:

```bash
# From project root, ensure cpp_core is built first
cd cpp_core && mkdir -p build && cd build
cmake .. && make -j
cd ../../

# Install Python package with bundled libraries
cd python_bindings
pip install .
```

The Python package automatically handles library loading and provides a simple API matching the C++ interface.

### API Server

The HTTP API server provides a production-ready REST interface using the Drogon C++ web framework. It wraps the C++ core with JSON endpoints for health checks, metrics, and OCR processing.

Build steps:

```bash
# Ensure cpp_core is built first
cd cpp_core && mkdir -p build && cd build
cmake .. && make -j
cd ../../

# Build API server
cd api_server && mkdir -p build && cd build
cmake .. && make -j
```

This produces the genocr_api_server executable with embedded Drogon framework.

## Run

### CLI Executable

The executable provides a simple CLI for single files, batch processing, and standby directory watching with live processing.
By default, the models directory is ../../models from the executable's working directory; override with -m if needed.

Usage:

```bash
# Show help
./genocr -h

# Process single file
./genocr document.png

# Process multiple files with output directory
./genocr -o results/ *.png

# Enable CUDA acceleration
./genocr -c -o gpu_results/ document.png

# Batch/standby mode – watch folder
./genocr -b watch_folder/ -o results/

# Custom models directory
./genocr -m /path/to/models document.png
```


### Python API

Python bindings provide seamless integration with Python applications:

```python
import genocr

# Process single image (returns formatted text output)
result = genocr.process_image("document.png", "models/")

# Parse structured output
lines = result.split('\n')
detected_objects = [line for line in lines if line.startswith('- ')]
final_text = result.split('--- Final Corrected Text ---\n')[^1]

print(f"Detected context: {detected_objects}")
print(f"OCR text: {final_text}")
```

The Python interface mirrors the C++ API and automatically handles library dependencies through bundled ONNX Runtime libraries.

### HTTP API Server

The Drogon-based API server provides RESTful endpoints for OCR processing with JSON request/response handling.

Start the server:

```bash
cd api_server/build
./genocr_api_server
# Server starts on http://localhost:8080
```

API endpoints:

**Health Check**

```bash
curl http://localhost:8080/api/v1/health
```

**OCR Processing** (POST multipart/form-data)

```bash
curl -X POST http://localhost:8080/api/v1/ocr/process \
  -F "image=@document.png"
```

Response format:

```json
{
  "success": true,
  "data": {
    "text": "Extracted and corrected text...",
    "detected_objects": ["laptop", "person"],
    "processing_time_ms": 2500,
    "request_id": "req_123..."
  },
  "timestamp": 1234567890
}
```

**Metrics**

```bash
curl http://localhost:8080/api/v1/metrics
```

Returns server statistics including request counts, success rates, and uptime.

Server configuration (config/config.json):

- Port: 8080 (default)
- Max body size: 50MB
- Thread pool: 4 workers
- Upload directory: ./uploads

Note: Currently supports PNG and BMP formats. JPEG support requires OpenCV built with libjpeg.

## Pipeline

- Stage 1: YOLOv8 context detection on the full image to collect scene/object clues for the corrector.
- Stage 2: PaddleOCR detection and recognition over perspective‑warped text quads, with chunking for wide crops and normalization in NCHW.
- Stage 3: Single‑pass correction using Phi‑3 Vision via ONNX Runtime GenAI, with prompt assembly, token constraints, and streaming generation.

YOLO and OCR run concurrently; their outputs are merged before correction to reduce latency and preserve document structure.


## Deployment Options

GenOCR provides three deployment modes to fit different use cases:

1. **Standalone CLI**: Direct executable for command-line batch processing and file watching
2. **Python Library**: Native Python integration via pip-installable package with bundled dependencies
3. **HTTP API**: Production REST API with Drogon framework for web service integration

All modes share the same C++ core pipeline, ensuring consistent OCR quality across deployment scenarios.

## Notes

- For PaddleOCR, ensure detection and recognition ONNX exports and the multilingual dictionary are placed under `<models_dir>/ocr_detection_multilingual` and `<models_dir>/ocr_recognition_multilingual` as referenced by the engine.
- For correction, place the Phi‑3 Vision model files under `<models_dir>/corrector`; the engine handles prompt construction and token budgeting internally.
- The HTTP API server requires absolute file paths due to worker thread execution context; automatic path resolution is handled internally.
- Python bindings include auto-loading of ONNX Runtime libraries, eliminating manual LD_LIBRARY_PATH configuration.


## License

Apache 2.0 License
See LICENSE at the project root for terms governing code and third‑party components.

send the full readme with it included as well

Here is the full updated README including the new Docker build, run, and test scripts section incorporated into the existing content:

***

# GenOCR

GenOCR is a C++20 OCR pipeline that combines object context detection, multilingual text detection/recognition, and large‑context OCR correction in a single executable.
It uses ONNX Runtime for YOLOv8 and PaddleOCR, and ONNX Runtime GenAI for the Phi‑3 Vision corrector, with parallel execution and multiple deployment options including CLI, Python bindings, and HTTP API.

## Features

- Context object detection with YOLOv8 ONNX to surface scene cues used during correction.
- Multilingual text detection and recognition with PaddleOCR (detection + recognition + charset loading).
- OCR correction with a Phi‑3 Vision model via ONNX Runtime GenAI, handling large prompts and image inputs.
- Parallel YOLO and OCR execution, then single‑pass correction over the concatenated raw OCR text.
- Multiple deployment options: standalone CLI, Python library, and production HTTP API server.


## Requirements

- C++20 toolchain and CMake 3.18+; a system OpenCV installation discoverable by CMake (find_package OpenCV REQUIRED).
- Network access during first build to download ONNX Runtime and ONNX Runtime GenAI archives; system unzip available for extraction on Linux/macOS where used.
- Optional CUDA Toolkit: detected at configure time for GenAI package selection and enabled at runtime with the -c flag for standard inference providers.


## Models

- YOLOv8 detector: models/detector.onnx.
- PaddleOCR detection model: `<models_dir>/ocr_detection_multilingual/inference.onnx` (resolved from -m or default ../../models at runtime).
- PaddleOCR recognition model: `<models_dir>/ocr_recognition_multilingual/inference.onnx` (same resolution as above).
- PaddleOCR dictionary: `<models_dir>/ocr_recognition_multilingual/ppocrv5_dict.txt` (loaded at startup by the recognition path).
- Corrector (Phi‑3 Vision) directory: `<models_dir>/corrector` (passed to OgaModel::Create).

*Note: When running from cpp_core/build, the default models directory is resolved as `../../models` unless overridden via `-m`.*

## Build

### C++ Core

The core builds from cpp_core using CMake, auto‑fetching ONNX Runtime, ONNX Runtime GenAI, and Clipper2 at configure time.
OpenCV must be available on the system and found by CMake; CUDA is optional and auto‑detected to select the GenAI package variant.

Build steps:

```bash
cd cpp_core
mkdir -p build && cd build
cmake .. && cmake --build . -j
```

This produces the `genocr` executable in `cpp_core/build/` linked against the downloaded ONNX Runtime components and system OpenCV.

### Python Bindings

Python bindings provide a native Python interface to the C++ OCR pipeline via pybind11. The package bundles all required ONNX Runtime libraries for zero-configuration deployment.

Build and install:

```bash
# From project root, ensure cpp_core is built first
cd cpp_core && mkdir -p build && cd build
cmake .. && make -j
cd ../../

# Install Python package with bundled libraries
cd python_bindings
pip install .
```

The Python package automatically handles library loading and provides a simple API matching the C++ interface.

### API Server

The HTTP API server provides a production-ready REST interface using the Drogon C++ web framework. It wraps the C++ core with JSON endpoints for health checks, metrics, and OCR processing.

Build steps:

```bash
# Ensure cpp_core is built first
cd cpp_core && mkdir -p build && cd build
cmake .. && make -j
cd ../../

# Build API server
cd api_server && mkdir -p build && cd build
cmake .. && make -j
```

This produces the `genocr_api_server` executable with embedded Drogon framework.

***

## Docker Deployment

GenOCR provides convenience scripts for building and running inside Docker containers, simplifying environment setup and deployment.

### Scripts

- **docker-build.sh**

Builds the Docker image for CPU or GPU mode.

Usage:

```bash
./scripts/docker-build.sh [cpu|gpu] [tag]
```

Example:

```bash
./scripts/docker-build.sh cpu latest
```

- **docker-run.sh**

Runs the GenOCR API server container, waits for server readiness by polling the health endpoint.

Usage:

```bash
./scripts/docker-run.sh [cpu|gpu]
```

Features:
    - Starts container with CPU or GPU mode.
    - Waits up to 2 minutes for server to become healthy.
    - Progress indicator shown during wait.
    - Prints example curl commands and endpoints on success.
- **docker-test.sh**

Runs integration tests against the running Docker container.

Usage:

```bash
./scripts/docker-test.sh
```


### Docker Compose

The provided `docker-compose.yml` builds and runs GenOCR in a container exposing port 8080, mounting models and uploads, with resource limits for CPU and memory.

***

## Run

### CLI Executable

The executable provides a simple CLI for single files, batch processing, and standby directory watching with live processing.
By default, the models directory is `../../models` from the executable's working directory; override with `-m` if needed.

Usage:

```bash
# Show help
./genocr -h

# Process single file
./genocr document.png

# Process multiple files with output directory
./genocr -o results/ *.png

# Enable CUDA acceleration
./genocr -c -o gpu_results/ document.png

# Batch/standby mode – watch folder
./genocr -b watch_folder/ -o results/

# Custom models directory
./genocr -m /path/to/models document.png
```


### Python API

Python bindings provide seamless integration with Python applications:

```python
import genocr

# Process single image (returns formatted text output)
result = genocr.process_image("document.png", "models/")

# Parse structured output
lines = result.split('\n')
detected_objects = [line for line in lines if line.startswith('- ')]
final_text = result.split('--- Final Corrected Text ---\n')[-1]

print(f"Detected context: {detected_objects}")
print(f"OCR text: {final_text}")
```

The Python interface mirrors the C++ API and automatically handles library dependencies through bundled ONNX Runtime libraries.

### HTTP API Server

The Drogon-based API server provides RESTful endpoints for OCR processing with JSON request/response handling.

Start the server:

```bash
cd api_server/build
./genocr_api_server
# Server starts on http://localhost:8080
```

API endpoints:

- **Health Check**

```bash
curl http://localhost:8080/api/v1/health
```

- **OCR Processing** (POST multipart/form-data)

```bash
curl -X POST http://localhost:8080/api/v1/ocr/process \
  -F "image=@document.png"
```

Response sample:

```json
{
  "success": true,
  "data": {
    "text": "Extracted and corrected text...",
    "detected_objects": ["laptop", "person"],
    "processing_time_ms": 2500,
    "request_id": "req_123..."
  },
  "timestamp": 1234567890
}
```

- **Metrics**

```bash
curl http://localhost:8080/api/v1/metrics
```


Returns server statistics including request counts, success rates, and uptime.

Server configuration (`config/config.json`):

- Port: 8080 (default)
- Max body size: 50MB
- Thread pool: 4 workers
- Upload directory: ./uploads

Currently supports PNG and BMP formats. JPEG requires system OpenCV built with libjpeg.

***

## Pipeline

- Stage 1: YOLOv8 context detection on the full image to collect scene/object clues for the corrector.
- Stage 2: PaddleOCR detection and recognition over perspective-warped text quads, with chunking for wide crops and normalization in NCHW.
- Stage 3: Single-pass correction using Phi-3 Vision via ONNX Runtime GenAI, with prompt assembly, token constraints, and streaming generation.

YOLO and OCR run concurrently; their outputs are merged before correction to reduce latency and maintain document structure.

***

## Project Layout
```
.
├── api_server/                     # REST API server built with Drogon framework
│   ├── CMakeLists.txt              # Build configuration
│   ├── config/                     # Server config files
│   │   └── config.json             # JSON config for server (port, threads, etc.)
│   ├── include/                    # Header files for API endpoints
│   │   ├── json_response.hpp
│   │   ├── ocr_controller.hpp
│   │   └── ocr_service.hpp
│   └── src/                        # Source files for server
│       ├── json_response.cpp
│       ├── main.cpp
│       ├── ocr_controller.cpp
│       └── ocr_service.cpp
├── cpp_core/                        # Core OCR pipeline (C++20 code)
│   ├── CMakeLists.txt              # Build system configuration
│   ├── include/                    # Headers for core engine
│   │   ├── corrector_engine.hpp
│   │   ├── genocr.hpp
│   │   ├── paddle_ocr_engine.hpp
│   │   ├── paddle_utils.hpp
│   │   └── yolo_engine.hpp
│   └── src/                        # Implementation files
│       ├── corrector_engine.cpp
│       ├── genocr.cpp
│       ├── main.cpp
│       ├── paddle_ocr_engine.cpp
│       ├── paddle_utils.cpp
│       └── yolo_engine.cpp
├── models/                          # ML models for detection and recognition
│   ├── detector.onnx
│   ├── ocr_detection_multilingual/
│   │   ├── inference.onnx
│   │   └── inference.yml
│   ├── ocr_recognition_multilingual/
│   │   ├── inference.onnx
│   │   ├── inference.yml
│   │   └── ppocrv5_dict.txt
│   └── corrector/                   # Phi-3 Vision models directory
├── python_bindings/                 # Python package with pybind11 bindings
│   ├── setup.py
│   ├── CMakeLists.txt
│   └── src/                         # Native Python bindings source
│       └── python_bindings.cpp
├── tests/                           # Unit tests for models and bindings
│   ├── models/
│   └── test_assets/                 # Test assets like images and files
├── scripts/                         # Scripts for build, docker, etc.
│   ├── build_all.ps1
│   ├── build_all.sh
│   ├── docker-build.sh             # Docker image build script
│   ├── docker-run.sh               # Docker container run script
│   └── docker-test.sh              # Docker test script
├── README.md                        # Project overview and build instructions
├── LICENSE                          # Licensing info
├── yolov8n.pt                       # YOLOv8 model weights
├── requirements.txt                 # Python dependencies
└── pytest.ini                       # pytest config
```


***

## License

Apache 2.0 License
See LICENSE at the project root for terms governing code and third‑party components.

***

