cmake_minimum_required(VERSION 3.18)
project(GenOCRCore CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- Find System Dependencies ---
find_package(OpenCV REQUIRED)

set(ORT_STANDARD_URL "")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(ORT_STANDARD_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(ORT_STANDARD_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ORT_STANDARD_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-universal2-1.22.0.tgz")
endif()

if(ORT_STANDARD_URL STREQUAL "")
    message(FATAL_ERROR "Could not determine a suitable ONNX Runtime package for this system.")
endif()

set(ORT_STANDARD_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_standard_archive")
set(ORT_STANDARD_EXTRACT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/onnxruntime_standard")

file(DOWNLOAD
    ${ORT_STANDARD_URL}
    ${ORT_STANDARD_ARCHIVE}
    SHOW_PROGRESS
)

file(MAKE_DIRECTORY ${ORT_STANDARD_EXTRACT_DIR})

if(ORT_STANDARD_URL MATCHES "\\.zip$")
    execute_process(
        COMMAND unzip -o ${ORT_STANDARD_ARCHIVE}
        WORKING_DIRECTORY ${ORT_STANDARD_EXTRACT_DIR}
        RESULT_VARIABLE extract_result
    )
else()
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${ORT_STANDARD_ARCHIVE}
        WORKING_DIRECTORY ${ORT_STANDARD_EXTRACT_DIR}
        RESULT_VARIABLE extract_result
    )
endif()

if(NOT extract_result EQUAL 0)
    message(FATAL_ERROR "Failed to extract ONNX Runtime standard archive")
endif()


# Try common nested paths
set(ORT_STANDARD_CAND1 "${ORT_STANDARD_EXTRACT_DIR}/onnxruntime-linux-x64-1.22.0")
set(ORT_STANDARD_CAND2 "${ORT_STANDARD_EXTRACT_DIR}/onnxruntime-win-x64-1.22.0")
set(ORT_STANDARD_CAND3 "${ORT_STANDARD_EXTRACT_DIR}/onnxruntime-osx-universal2-1.22.0")

foreach(cand IN LISTS ORT_STANDARD_CAND1 ORT_STANDARD_CAND2 ORT_STANDARD_CAND3)
    if(EXISTS "${cand}/include/onnxruntime_cxx_api.h")
        set(ONNXRuntime_INCLUDE_DIRS "${cand}/include")
        set(ONNXRuntime_LIBRARIES "${cand}/lib/libonnxruntime.so") # or .dll/.dylib based on platform
        break()
    endif()
endforeach()

if(NOT ONNXRuntime_INCLUDE_DIRS OR NOT ONNXRuntime_LIBRARIES)
    message(FATAL_ERROR "Could not find ONNX Runtime headers or library after extraction")
endif()
message(STATUS "Using ONNX Runtime from: ${ONNXRuntime_LIBRARIES}")


# --- Detect Platform and Select ONNX GenAI Package ---
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA Toolkit found. Build will include CUDA support.")
else()
    message(STATUS "CUDA Toolkit not found. Configuring for CPU-only build.")
endif()

set(ORT_GENAI_URL "")
set(ORT_GENAI_HASH "")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CUDAToolkit_FOUND)
        set(ORT_GENAI_URL "https://github.com/microsoft/onnxruntime-genai/releases/download/v0.9.0/onnxruntime-genai-linux-cuda-x64-capi.zip")
        set(ORT_GENAI_HASH "SHA256=23bcbc78ebcb2b27dc5c52d106670ac4a0e4e978c7c6332a803b33be06722ffb")
    else()
        set(ORT_GENAI_URL "https://github.com/microsoft/onnxruntime-genai/releases/download/v0.9.0/onnxruntime-genai-linux-cpu-x64-capi.zip")
        set(ORT_GENAI_HASH "SHA256=8cd98a529606ab5025f6819468d25904cbd995ed827ea17c4a3934a4002ba6d0")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(CUDAToolkit_FOUND)
        set(ORT_GENAI_URL "https://github.com/microsoft/onnxruntime-genai/releases/download/v0.9.0/onnxruntime-genai-win-cuda-x64-capi.zip")
        set(ORT_GENAI_HASH "SHA256=09ed6abb06b9d5f344c84b5e161b24508c3af9eede06fb1beca7c157e6265714")
    else()
        set(ORT_GENAI_URL "https://github.com/microsoft/onnxruntime-genai/releases/download/v0.9.0/onnxruntime-genai-win-cpu-x64-capi.zip")
        set(ORT_GENAI_HASH "SHA256=bce06fe2694fac75ba989db38ccb3a24715da547ee9a6bb54a548a7c41eb3a32")
    endif()
endif()

if(ORT_GENAI_URL STREQUAL "")
    message(FATAL_ERROR "Could not determine a suitable onnxruntime-genai package for this system.")
endif()

# --- Download and Extract ONNX GenAI ---
set(ORT_GENAI_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_genai.zip")
set(ORT_GENAI_EXTRACT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/onnxruntime_genai")

file(DOWNLOAD
    ${ORT_GENAI_URL}
    ${ORT_GENAI_ARCHIVE}
    EXPECTED_HASH ${ORT_GENAI_HASH}
    SHOW_PROGRESS
)

file(MAKE_DIRECTORY ${ORT_GENAI_EXTRACT_DIR})

execute_process(
    COMMAND unzip -o ${ORT_GENAI_ARCHIVE}
    WORKING_DIRECTORY ${ORT_GENAI_EXTRACT_DIR}
    RESULT_VARIABLE extract_result
)
if(NOT extract_result EQUAL 0)
    message(FATAL_ERROR "Failed to extract ONNXRuntime GenAI archive using system unzip")
endif()
message(STATUS "Extracted ONNXRuntime GenAI to: ${ORT_GENAI_EXTRACT_DIR}")

# Extract inner tar.gz
set(ORT_GENAI_TAR        "${ORT_GENAI_EXTRACT_DIR}/onnxruntime-genai-linux-cpu-x64-capi/onnxruntime-genai-0.9.0-linux-x64.tar.gz")
set(ORT_GENAI_INNER_ROOT "${ORT_GENAI_EXTRACT_DIR}/onnxruntime-genai-linux-cpu-x64-capi")
execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${ORT_GENAI_TAR}
    WORKING_DIRECTORY ${ORT_GENAI_INNER_ROOT}
    RESULT_VARIABLE untar_result
)
if(NOT untar_result EQUAL 0)
    message(FATAL_ERROR "Failed to extract inner ONNXRuntime GenAI tar.gz archive")
endif()

# Resolve actual include/lib dir (handles double-nesting)
set(ORT_GENAI_CAND1 "${ORT_GENAI_INNER_ROOT}/onnxruntime-genai-0.9.0-linux-x64")
set(ORT_GENAI_CAND2 "${ORT_GENAI_CAND1}/onnxruntime-genai-0.9.0-linux-x64")
if(EXISTS "${ORT_GENAI_CAND1}/include/ort_genai.h")
    set(ORT_GENAI_PACKAGE_DIR "${ORT_GENAI_CAND1}")
elseif(EXISTS "${ORT_GENAI_CAND2}/include/ort_genai.h")
    set(ORT_GENAI_PACKAGE_DIR "${ORT_GENAI_CAND2}")
else()
    message(FATAL_ERROR "Could not find ort_genai.h after extraction under: ${ORT_GENAI_INNER_ROOT}")
endif()
message(STATUS "Using ONNX GenAI package dir: ${ORT_GENAI_PACKAGE_DIR}")

# --- ONNX GenAI Interface Library ---
add_library(onnxruntime_genai INTERFACE)
target_include_directories(onnxruntime_genai INTERFACE ${ORT_GENAI_PACKAGE_DIR}/include)
target_link_directories(onnxruntime_genai INTERFACE ${ORT_GENAI_PACKAGE_DIR}/lib)
target_link_libraries(onnxruntime_genai INTERFACE ${ORT_GENAI_PACKAGE_DIR}/lib/libonnxruntime-genai.so)

# Create a compatibility header so existing includes work: onnxruntime_genai.h -> ort_genai.h
set(GENAI_COMPAT_DIR "${CMAKE_CURRENT_BINARY_DIR}/genai_compat")
file(MAKE_DIRECTORY "${GENAI_COMPAT_DIR}")
file(WRITE "${GENAI_COMPAT_DIR}/onnxruntime_genai.h"
"#pragma once
#include \"ort_genai.h\"
")

# --- Fetch and Build Clipper2 ---
include(FetchContent)
FetchContent_Declare(
    clipper2
    GIT_REPOSITORY https://github.com/AngusJohnson/Clipper2.git
    GIT_TAG        Clipper2_1.5.4
)
FetchContent_MakeAvailable(clipper2)

add_library(clipper STATIC 
    ${clipper2_SOURCE_DIR}/CPP/Clipper2Lib/src/clipper.engine.cpp
    ${clipper2_SOURCE_DIR}/CPP/Clipper2Lib/src/clipper.offset.cpp
    ${clipper2_SOURCE_DIR}/CPP/Clipper2Lib/src/clipper.rectclip.cpp
)
target_include_directories(clipper PUBLIC ${clipper2_SOURCE_DIR}/CPP/Clipper2Lib/include)

# --- Build Core Library ---
add_library(genocr_core STATIC
    src/genocr.cpp
    src/yolo_engine.cpp
    src/paddle_ocr_engine.cpp
    src/corrector_engine.cpp
    src/paddle_utils.cpp 
)
# Put the compat header dir FIRST so #include <onnxruntime_genai.h> resolves
target_include_directories(genocr_core PUBLIC 
    ${GENAI_COMPAT_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRuntime_INCLUDE_DIRS}
    ${ORT_GENAI_PACKAGE_DIR}/include
)
target_link_libraries(genocr_core PRIVATE 
    ${OpenCV_LIBS}
    ${ONNXRuntime_LIBRARIES}
    onnxruntime_genai
    clipper
)

# --- Build Test Executable ---
add_executable(genocr src/main.cpp)
target_link_libraries(genocr PRIVATE genocr_core)
