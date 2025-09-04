#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Include the C++ headers (these should handle OpenCV internally)
#include "genocr.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_genocr_core, m) {
    m.doc() = "GenOCR - Multi-language OCR with AI correction";
    
    // GenOCR class
    py::class_<GenOCR>(m, "GenOCR")
        .def(py::init<const std::string&, bool>(), 
             py::arg("models_directory"), py::arg("use_cuda") = false)
        .def("run", &GenOCR::Run, py::arg("image_path"));
    
    // Utility functions
    m.def("process_image", [](const std::string& image_path, const std::string& models_dir) {
        GenOCR ocr(models_dir);
        return ocr.Run(image_path);
    }, "Process a single image", py::arg("image_path"), py::arg("models_dir"));
    
    m.def("process_batch", [](const std::vector<std::string>& image_paths, 
                              const std::string& models_dir) {
        GenOCR ocr(models_dir);
        std::vector<std::string> results;
        
        for (const auto& path : image_paths) {
            results.push_back(ocr.Run(path));
        }
        
        return results;
    }, "Process multiple images", 
       py::arg("image_paths"), py::arg("models_dir"));
    
    m.def("check_installation", []() {
        return "GenOCR installation OK";
    });
    
    m.attr("__version__") = "0.1.0";
}
