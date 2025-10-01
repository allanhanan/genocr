cmake_minimum_required(VERSION 3.18)
project(GenOCRAPIServer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# === Paths ===
set(CPP_CORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_core)
set(CPP_CORE_BUILD_DIR ${CPP_CORE_SOURCE_DIR}/build)

# === Ensure cpp_core is built ===
if(NOT EXISTS ${CPP_CORE_BUILD_DIR}/libgenocr_core.a)
    message(FATAL_ERROR 
        "cpp_core not built! Run: \n"
        "cd ../cpp_core && mkdir build && cd build && cmake .. && make -j")
endif()
message(STATUS "Using pre-built cpp_core: ${CPP_CORE_BUILD_DIR}/libgenocr_core.a")

# === Fetch Drogon ===
include(FetchContent)
FetchContent_Declare(
    drogon
    GIT_REPOSITORY https://github.com/drogonframework/drogon.git
    GIT_TAG        v1.9.7
)
FetchContent_MakeAvailable(drogon)

# === Find OpenCV ===
find_package(OpenCV REQUIRED)

# === Find ONNX Runtime ===
set(ORT_STANDARD_EXTRACT_DIR "${CPP_CORE_SOURCE_DIR}/src/third_party/onnxruntime_standard")
set(ORT_GENAI_EXTRACT_DIR "${CPP_CORE_SOURCE_DIR}/src/third_party/onnxruntime_genai")

# Standard ONNX Runtime
set(ORT_STANDARD_CAND1 "${ORT_STANDARD_EXTRACT_DIR}/onnxruntime-linux-x64-1.22.0")
set(ORT_STANDARD_CAND2 "${ORT_STANDARD_EXTRACT_DIR}/onnxruntime-win-x64-1.22.0")
set(ORT_STANDARD_CAND3 "${ORT_STANDARD_EXTRACT_DIR}/onnxruntime-osx-universal2-1.22.0")

foreach(cand IN LISTS ORT_STANDARD_CAND1 ORT_STANDARD_CAND2 ORT_STANDARD_CAND3)
    if(EXISTS "${cand}/include/onnxruntime_cxx_api.h")
        set(ONNXRuntime_INCLUDE_DIRS "${cand}/include")
        set(ONNXRuntime_LIBRARIES "${cand}/lib/libonnxruntime.so")
        message(STATUS "Found ONNX Runtime: ${ONNXRuntime_LIBRARIES}")
        break()
    endif()
endforeach()

# ONNX GenAI
set(ORT_GENAI_CAND1 "${ORT_GENAI_EXTRACT_DIR}/onnxruntime-genai-0.9.0-linux-x64")
set(ORT_GENAI_CAND2 "${ORT_GENAI_EXTRACT_DIR}/onnxruntime-genai-0.9.0-linux-x64-cuda")
set(ORT_GENAI_CAND3 "${ORT_GENAI_EXTRACT_DIR}/onnxruntime-genai-0.9.0-win-x64")
set(ORT_GENAI_CAND4 "${ORT_GENAI_EXTRACT_DIR}/onnxruntime-genai-0.9.0-win-x64-cuda")

foreach(cand IN LISTS ORT_GENAI_CAND1 ORT_GENAI_CAND2 ORT_GENAI_CAND3 ORT_GENAI_CAND4)
    if(EXISTS "${cand}/include/ort_genai.h")
        set(ORT_GENAI_PACKAGE_DIR "${cand}")
        message(STATUS "Found ONNX GenAI: ${ORT_GENAI_PACKAGE_DIR}")
        break()
    endif()
endforeach()

# === API Server Executable ===
add_executable(genocr_api_server
    src/main.cpp
    src/ocr_controller.cpp
    src/ocr_service.cpp
    src/json_response.cpp
)

# === Link Libraries ===
target_link_libraries(genocr_api_server PRIVATE 
    ${CPP_CORE_BUILD_DIR}/libgenocr_core.a
    ${CPP_CORE_BUILD_DIR}/libclipper.a
    drogon
    ${OpenCV_LIBS}
)

# ONNX Runtime
if(ONNXRuntime_LIBRARIES)
    target_link_libraries(genocr_api_server PRIVATE ${ONNXRuntime_LIBRARIES})
endif()

# ONNX GenAI
if(ORT_GENAI_PACKAGE_DIR AND EXISTS ${ORT_GENAI_PACKAGE_DIR}/lib/libonnxruntime-genai.so)
    target_link_libraries(genocr_api_server PRIVATE ${ORT_GENAI_PACKAGE_DIR}/lib/libonnxruntime-genai.so)
endif()

# === Include Directories ===
target_include_directories(genocr_api_server PRIVATE
    include
    ${CPP_CORE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

if(ONNXRuntime_INCLUDE_DIRS)
    target_include_directories(genocr_api_server PRIVATE ${ONNXRuntime_INCLUDE_DIRS})
endif()

if(ORT_GENAI_PACKAGE_DIR)
    target_include_directories(genocr_api_server PRIVATE ${ORT_GENAI_PACKAGE_DIR}/include)
endif()

# Include compat header (onnxruntime_genai.h)
if(EXISTS ${CPP_CORE_BUILD_DIR}/genai_compat)
    target_include_directories(genocr_api_server PRIVATE ${CPP_CORE_BUILD_DIR}/genai_compat)
endif()

# Include Clipper2 headers from cpp_core
set(CLIPPER2_INCLUDE_DIR ${CPP_CORE_BUILD_DIR}/_deps/clipper2-src/CPP/Clipper2Lib/include)
if(EXISTS ${CLIPPER2_INCLUDE_DIR})
    target_include_directories(genocr_api_server PRIVATE ${CLIPPER2_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Clipper2 include directory not found: ${CLIPPER2_INCLUDE_DIR}")
endif()

# === Copy config file to build dir ===
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config/config.json 
               ${CMAKE_CURRENT_BINARY_DIR}/config.json COPYONLY)
