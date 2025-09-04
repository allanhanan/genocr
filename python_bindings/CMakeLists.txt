cmake_minimum_required(VERSION 3.18)

# Get cpp_core build artifacts
set(CPP_CORE_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build/cpp_core)
set(CPP_CORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_core)

# Create the Python module
pybind11_add_module(_genocr_core src/python_bindings.cpp)

# Link against the pre-built genocr_core library
target_link_libraries(_genocr_core PRIVATE 
    ${CPP_CORE_BUILD_DIR}/libgenocr_core.a
    ${CPP_CORE_BUILD_DIR}/_deps/clipper2-build/libclipper.a
)

# Include directories from cpp_core
target_include_directories(_genocr_core PRIVATE 
    ${CPP_CORE_SOURCE_DIR}/include
    ${CPP_CORE_BUILD_DIR}/_deps/clipper2-src/CPP/Clipper2Lib/include
    ${CPP_CORE_BUILD_DIR}/genai_compat
)

# Find OpenCV and ONNX Runtime paths dynamically
find_package(OpenCV REQUIRED)
target_include_directories(_genocr_core PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(_genocr_core PRIVATE ${OpenCV_LIBS})

# Find ONNX Runtime libraries built by cpp_core
file(GLOB ORT_LIBS "${CPP_CORE_BUILD_DIR}/../cpp_core/src/third_party/*/lib/*.so")
target_link_libraries(_genocr_core PRIVATE ${ORT_LIBS})
