# GenOCR

GenOCR is a C++20 OCR pipeline that combines object context detection, multilingual text detection/recognition, and large‑context OCR correction in a single executable.
It uses ONNX Runtime for YOLOv8 and PaddleOCR, and ONNX Runtime GenAI for the Phi‑3 Vision corrector, with parallel execution and a practical CLI for single files or standby batch mode.

## Features

- Context object detection with YOLOv8 ONNX to surface scene cues used during correction.
- Multilingual text detection and recognition with PaddleOCR (detection + recognition + charset loading).
- OCR correction with a Phi‑3 Vision model via ONNX Runtime GenAI, handling large prompts and image inputs.
- Parallel YOLO and OCR execution, then single‑pass correction over the concatenated raw OCR text.
- CLI with CUDA toggle, custom models directory, output directory selection, and directory watcher standby mode.


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

The core builds from cpp_core using CMake, auto‑fetching ONNX Runtime, ONNX Runtime GenAI, and Clipper2 at configure time.
OpenCV must be available on the system and found by CMake; CUDA is optional and auto‑detected to select the GenAI package variant.

- Build steps:
    - cd cpp_core.
    - mkdir -p build \&\& cd build.
    - cmake .. \&\& cmake --build . -j.

This produces the genocr executable in cpp_core/build/ linked against the downloaded ONNX Runtime components and system OpenCV.

## Run

The executable provides a simple CLI for single files, batch processing, and standby directory watching with live processing.
By default, the models directory is ../../models from the executable’s working directory; override with -m if needed.

### Usage

- Show help and options: run with no arguments or -h/--help to see flags and examples emitted by the program itself.


### Examples

- Process single file :

```bash
./genocr document.png
```

- Process multiple files with output directory :

```bash
./genocr -o results/ *.png *.jpg
```

- Enable CUDA acceleration (if built with CUDA and supported at runtime) :

```bash
./genocr -c -o gpu_results/ document.png
```

- Batch/standby mode – watch folder and process new files automatically :

```bash
./genocr -b watch_folder/ -o results/
```

- Custom models directory :

```bash
./genocr -m /path/to/models -c document.png
```


## Pipeline

- Stage 1: YOLOv8 context detection on the full image to collect scene/object clues for the corrector.
- Stage 2: PaddleOCR detection and recognition over perspective‑warped text quads, with chunking for wide crops and normalization in NCHW.
- Stage 3: Single‑pass correction using Phi‑3 Vision via ONNX Runtime GenAI, with prompt assembly, token constraints, and streaming generation.

YOLO and OCR run concurrently; their outputs are merged before correction to reduce latency and preserve document structure.

## Project layout

```
.
├── api_server
├── cpp_core
│   ├── CMakeLists.txt
│   ├── include
│   │   ├── corrector_engine.hpp
│   │   ├── genocr.hpp
│   │   ├── paddle_ocr_engine.hpp
│   │   ├── paddle_utils.hpp
│   │   └── yolo_engine.hpp
│   └── src
│       ├── corrector_engine.cpp
│       ├── genocr.cpp
│       ├── main.cpp
│       ├── paddle_ocr_engine.cpp
│       ├── paddle_utils.cpp
│       └── yolo_engine.cpp
├── LICENSE
├── models
│   ├── detector.onnx
│   └── scripts
│       ├── exportPaddleOCR.py
│       ├── exportPhi-3.py
│       └── exportYoLo.py
├── ppocrv5_dict.txt
├── python_bindings
├── README.md
├── requirements.txt
├── tests
│   ├── models
│   │   ├── test_corrector.py
│   │   ├── test_detector.py
│   │   └── test_ocr.py
│   └── test_assets
│       ├── handwritingHard.png
│       ├── ocrTTYHard.jpeg
│       ├── ocrTTYHard.png
│       ├── ocrTTYMid.jpeg
│       ├── ocrTTYMid.png
│       ├── sample_document.jpg
│       ├── sample_document.png
│       ├── sample_document.txt
│       └── sample_line_crop.png
└── yolov8n.pt
```

This structure aligns with the C++ core sources, CMake build, and runtime model expectations encoded in the engines and CLI.

## Notes

- For PaddleOCR, ensure detection and recognition ONNX exports and the multilingual dictionary are placed under <models_dir>/ocr_detection_multilingual and <models_dir>/ocr_recognition_multilingual as referenced by the engine.
- For correction, place the Phi‑3 Vision model files under <models_dir>/corrector; the engine handles prompt construction and token budgeting internally.


## Roadmap

API server, Docker packaging, and Python bindings are planned to complement the C++ core and provide additional integration options.

## License

See LICENSE at the project root for terms governing code and third‑party components.
