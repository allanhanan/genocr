cmake_minimum_required(VERSION 3.18)
project(GenOCRCore CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
        set(ONNXRuntime_LIBRARIES "${cand}/lib/libonnxruntime.so")
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
        set(ORT_GENAI_URL "https://github.com/microsoft/onnxruntime-genai/releases/download/v0.9.0/onnxruntime-genai-0.9.0-linux-x64-cuda.tar.gz")
        set(ORT_GENAI_HASH "SHA256=4cdb4b0d311b0cafa741f5cfa509ae4ac27550e9b03f566fe4cd10b2e515d231")
    else()
        set(ORT_GENAI_URL "https://github.com/microsoft/onnxruntime-genai/releases/download/v0.9.0/onnxruntime-genai-0.9.0-linux-x64.tar.gz")
        set(ORT_GENAI_HASH "SHA256=ac2b1adbbfdb7233f9d8fa641a41cc2b28c87f81d8a7e55566e0c589e8115081")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(CUDAToolkit_FOUND)
        set(ORT_GENAI_URL "https://github.com/microsoft/onnxruntime-genai/releases/download/v0.9.0/onnxruntime-genai-0.9.0-win-x64-cuda.zip")
        set(ORT_GENAI_HASH "SHA256=eefc6cb42bc136d269239f005a61e26277abd3aabef789ac96929b587062def7")
    else()
        set(ORT_GENAI_URL "https://github.com/microsoft/onnxruntime-genai/releases/download/v0.9.0/onnxruntime-genai-0.9.0-win-x64.zip")
        set(ORT_GENAI_HASH "SHA256=3cd1fccd7596c3660f5c6fe1bc2f1087c17c9af02134267a184a4ca42ce59300")
    endif()
endif()

if(ORT_GENAI_URL STREQUAL "")
    message(FATAL_ERROR "Could not determine a suitable onnxruntime-genai package for this system.")
endif()

# --- Download and Extract ONNX GenAI ---
set(ORT_GENAI_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_genai_archive")
set(ORT_GENAI_EXTRACT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/onnxruntime_genai")

file(DOWNLOAD
    ${ORT_GENAI_URL}
    ${ORT_GENAI_ARCHIVE}
    EXPECTED_HASH ${ORT_GENAI_HASH}
    SHOW_PROGRESS
)

file(MAKE_DIRECTORY ${ORT_GENAI_EXTRACT_DIR})

# Extract based on file type
if(ORT_GENAI_URL MATCHES "\\.zip$")
    execute_process(
        COMMAND unzip -o ${ORT_GENAI_ARCHIVE}
        WORKING_DIRECTORY ${ORT_GENAI_EXTRACT_DIR}
        RESULT_VARIABLE extract_result
    )
else()
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${ORT_GENAI_ARCHIVE}
        WORKING_DIRECTORY ${ORT_GENAI_EXTRACT_DIR}
        RESULT_VARIABLE extract_result
    )
endif()

if(NOT extract_result EQUAL 0)
    message(FATAL_ERROR "Failed to extract ONNXRuntime GenAI archive")
endif()
message(STATUS "Extracted ONNXRuntime GenAI to: ${ORT_GENAI_EXTRACT_DIR}")

# Find the extracted directory (new structure is simpler)
set(ORT_GENAI_CAND1 "${ORT_GENAI_EXTRACT_DIR}/onnxruntime-genai-0.9.0-linux-x64")
set(ORT_GENAI_CAND2 "${ORT_GENAI_EXTRACT_DIR}/onnxruntime-genai-0.9.0-linux-x64-cuda")
set(ORT_GENAI_CAND3 "${ORT_GENAI_EXTRACT_DIR}/onnxruntime-genai-0.9.0-win-x64")
set(ORT_GENAI_CAND4 "${ORT_GENAI_EXTRACT_DIR}/onnxruntime-genai-0.9.0-win-x64-cuda")

foreach(cand IN LISTS ORT_GENAI_CAND1 ORT_GENAI_CAND2 ORT_GENAI_CAND3 ORT_GENAI_CAND4)
    if(EXISTS "${cand}/include/ort_genai.h")
        set(ORT_GENAI_PACKAGE_DIR "${cand}")
        break()
    endif()
endforeach()

if(NOT ORT_GENAI_PACKAGE_DIR)
    message(FATAL_ERROR "Could not find ort_genai.h after extraction under: ${ORT_GENAI_EXTRACT_DIR}")
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
