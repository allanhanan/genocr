import os
import sys
import shutil
from pathlib import Path

def get_cpp_artifacts():
    """Find built C++ artifacts and copy runtime libraries"""
    project_root = Path(__file__).parent.parent
    cpp_core_build = project_root / "cpp_core" / "build"
    
    if not cpp_core_build.exists():
        print("ERROR: cpp_core not built! Run: cd cpp_core && mkdir build && cd build && cmake .. && make")
        sys.exit(1)

    # Find main library
    lib = cpp_core_build / "libgenocr_core.a"
    if not lib.exists():
        print(f"ERROR: {lib} not found")
        sys.exit(1)

    extra_libs = []
    
    # Find clipper libraries
    clipper_libs = list(cpp_core_build.glob("**/libclipper.a"))
    extra_libs.extend([str(lib) for lib in clipper_libs])
    
    # Find the PRIMARY ONNX Runtime library (avoid duplicates)
    onnx_runtime_libs = []
    onnx_genai_libs = []
    
    # Look for the EXACT GenAI library built by CMake
    genai_lib_path = (cpp_core_build.parent / "src" / "third_party" / "onnxruntime_genai" / 
                      "onnxruntime-genai-0.9.0-linux-x64" / "lib" / "libonnxruntime-genai.so")
    
    if genai_lib_path.exists():
        onnx_genai_libs.append(genai_lib_path)
    
    # Look for the GenAI's bundled ONNX Runtime (avoid the standard one)
    genai_runtime_lib = (cpp_core_build.parent / "src" / "third_party" / "onnxruntime_genai" / 
                         "onnxruntime-genai-0.9.0-linux-x64" / "lib" / "libonnxruntime.so")
    
    if genai_runtime_lib.exists():
        onnx_runtime_libs.append(genai_runtime_lib)
    else:
        # Fallback to standard ONNX Runtime if GenAI version not found
        std_runtime_libs = list((cpp_core_build.parent / "src" / "third_party" / "onnxruntime_standard").glob("**/lib/libonnxruntime.so"))
        if std_runtime_libs:
            onnx_runtime_libs.append(std_runtime_libs[0])  # Take only the first one
    
    print(f"✓ Found ONNX Runtime libs: {len(onnx_runtime_libs)}")
    print(f"✓ Found ONNX GenAI libs: {len(onnx_genai_libs)}")
    
    if not onnx_runtime_libs or not onnx_genai_libs:
        print("ERROR: Required ONNX libraries not found.")
        print("Available .so files:")
        all_sos = list((cpp_core_build.parent / "src" / "third_party").glob("**/*.so*"))
        for so_file in all_sos[:10]:  # Show first 10
            print(f"  {so_file}")
        sys.exit(1)
    
    # Create libs directory
    libs_dir = Path(__file__).parent / "genocr" / "libs"
    libs_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean old libraries
    for old_lib in libs_dir.glob("*.so*"):
        old_lib.unlink()
    
    bundled_libs = []
    
    # Copy ONNX Runtime first
    for lib_path in onnx_runtime_libs:
        dest = libs_dir / lib_path.name
        shutil.copy2(str(lib_path), str(dest))
        bundled_libs.append(str(dest))
        print(f"✓ Bundled: {lib_path.name}")
    
    # Copy GenAI second
    for lib_path in onnx_genai_libs:
        dest = libs_dir / lib_path.name
        shutil.copy2(str(lib_path), str(dest))
        bundled_libs.append(str(dest))
        print(f"✓ Bundled: {lib_path.name}")
    
    # Find includes
    includes = [
        str(project_root / "cpp_core" / "include"),
        str(cpp_core_build / "genai_compat"),
    ]
    
    # Add built library includes
    for pattern in ["onnxruntime_cxx_api.h", "clipper.h", "ort_genai.h"]:
        headers = list(cpp_core_build.parent.glob(f"**/include/{pattern}"))
        includes.extend([str(h.parent) for h in headers])
    
    # Find OpenCV includes - try multiple approaches
    opencv_includes = []
    
    # Try pkg-config first
    try:
        import subprocess
        result = subprocess.run(['pkg-config', '--cflags-only-I', 'opencv4'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            opencv_flags = result.stdout.strip().split()
            opencv_includes = [flag[2:] for flag in opencv_flags if flag.startswith('-I')]
        else:
            # Try opencv instead of opencv4
            result = subprocess.run(['pkg-config', '--cflags-only-I', 'opencv'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                opencv_flags = result.stdout.strip().split()
                opencv_includes = [flag[2:] for flag in opencv_flags if flag.startswith('-I')]
    except:
        pass
    
    # Try cv2 Python module
    if not opencv_includes:
        try:
            import cv2
            cv2_includes = cv2.includes()
            if cv2_includes:
                opencv_includes.extend(cv2_includes)
        except ImportError:
            pass
    
    # Common system paths as fallback
    if not opencv_includes:
        common_paths = [
            '/usr/include/opencv4',
            '/usr/include/opencv2',
            '/usr/include',
            '/usr/local/include/opencv4',
            '/usr/local/include/opencv2',
            '/usr/local/include'
        ]
        for path in common_paths:
            if Path(path).exists():
                opencv_includes.append(path)
    
    includes.extend(opencv_includes)
    print(f"✓ OpenCV includes: {opencv_includes}")
    
    return str(lib), extra_libs, list(set(includes)), bundled_libs

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from setuptools import setup, find_packages
    
    main_lib, extra_libs, includes, bundled_libs = get_cpp_artifacts()
    
    ext_modules = [
        Pybind11Extension(
            "_genocr_core",
            ["src/python_bindings.cpp"],
            include_dirs=includes,
            extra_objects=[main_lib] + extra_libs,
            libraries=["opencv_core", "opencv_imgproc", "opencv_imgcodecs", "opencv_dnn"],
            cxx_std=20,
            define_macros=[("VERSION_INFO", '"dev"')],
        ),
    ]
    
    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        package_data={
            'genocr': ['libs/*.so*'],
        },
        include_package_data=True,
    )
    
except ImportError as e:
    print(f"Error: {e}")
    print("Install pybind11: pip install pybind11")
    sys.exit(1)
