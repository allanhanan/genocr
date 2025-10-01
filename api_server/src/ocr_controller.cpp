#include "ocr_controller.hpp"
#include "json_response.hpp"
#include <drogon/MultiPart.h>
#include <filesystem>
#include <chrono>
#include <unistd.h>  // for getcwd
#include <limits.h>  // for PATH_MAX

OCRController::OCRController() {
    try {
        std::string models_dir = "../../models";
        ocrService_ = std::make_unique<OCRService>(models_dir);
        LOG_INFO << "OCR Controller initialized successfully";
    } catch (const std::exception& e) {
        LOG_ERROR << "Failed to initialize OCR Controller: " << e.what();
        throw;
    }
}

void OCRController::processImage(const HttpRequestPtr &req,
                                std::function<void(const HttpResponsePtr &)> &&callback) {
    requestCount_++;
    
    try {
        // Validate content type
        auto contentType = req->getHeader("content-type");
        if (contentType.find("multipart/form-data") == std::string::npos) {
            errorCount_++;
            callback(JSONResponse::validationError("content-type", "Expected multipart/form-data"));
            return;
        }
        
        // Parse multipart data
        drogon::MultiPartParser fileUpload;
        if (fileUpload.parse(req) != 0) {
            errorCount_++;
            callback(JSONResponse::error("Failed to parse multipart data", 400));
            return;
        }
        
        // Check if any files were uploaded
        auto files = fileUpload.getFiles();
        if (files.empty()) {
            errorCount_++;
            callback(JSONResponse::validationError("image", "No image file provided"));
            return;
        }
        
        // Get the first uploaded file
        auto &file = files[0];
        
        // Validate file type
        std::string filename = file.getFileName();
        std::string ext = std::filesystem::path(filename).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp") {
            errorCount_++;
            callback(JSONResponse::validationError("image", "Unsupported file format. Use JPG, PNG, or BMP"));
            return;
        }
        
        // Validate file size
        if (file.fileLength() > 50 * 1024 * 1024) {  // 50MB limit
            errorCount_++;
            callback(JSONResponse::validationError("image", "File size exceeds 50MB limit"));
            return;
        }
        
        if (file.fileLength() == 0) {
            errorCount_++;
            callback(JSONResponse::validationError("image", "Empty file provided"));
            return;
        }
        
        // FIXED: Create absolute path to ensure OpenCV can find the file
        auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
        std::string filename_only = "upload_" + std::to_string(timestamp) + ext;
        
        // Get current working directory
        char cwd[PATH_MAX];
        if (getcwd(cwd, sizeof(cwd)) == nullptr) {
            errorCount_++;
            callback(JSONResponse::error("Failed to get working directory", 500));
            return;
        }
        
        // Build absolute path
        std::string saved_filename = std::string(cwd) + "/uploads/" + filename_only;
        
        LOG_DEBUG << "Saving file to: " << saved_filename;
        
        try {
            // Save file with absolute path
            file.saveAs(saved_filename);
            
            // Verify file was saved and is readable
            if (!std::filesystem::exists(saved_filename)) {
                errorCount_++;
                LOG_ERROR << "File not saved: " << saved_filename;
                callback(JSONResponse::error("Failed to save uploaded file", 500));
                return;
            }
            
            LOG_DEBUG << "File saved successfully, size: " << std::filesystem::file_size(saved_filename) << " bytes";
            
            // Process the image with absolute path
            OCRResult result = ocrService_->processImageFile(saved_filename);
            
            // Cleanup temporary file
            try {
                std::filesystem::remove(saved_filename);
                LOG_DEBUG << "Temporary file cleaned up: " << saved_filename;
            } catch (const std::exception& cleanup_error) {
                LOG_WARN << "Failed to cleanup temp file: " << cleanup_error.what();
            }
            
            if (result.success) {
                successCount_++;
                callback(JSONResponse::success(result));
            } else {
                errorCount_++;
                callback(JSONResponse::error(result.error_message, 500, result.request_id));
            }
            
        } catch (const std::exception& e) {
            // Cleanup on error
            if (std::filesystem::exists(saved_filename)) {
                try {
                    std::filesystem::remove(saved_filename);
                } catch (...) {
                    // Ignore cleanup errors
                }
            }
            errorCount_++;
            LOG_ERROR << "Processing exception: " << e.what();
            callback(JSONResponse::error("Processing failed: " + std::string(e.what()), 500));
        }
        
    } catch (const std::exception& e) {
        errorCount_++;
        LOG_ERROR << "OCR processing error: " << e.what();
        callback(JSONResponse::error("Internal server error", 500));
    }
}

void OCRController::healthCheck(const HttpRequestPtr &req,
                               std::function<void(const HttpResponsePtr &)> &&callback) {
    bool healthy = ocrService_ && ocrService_->isHealthy();
    callback(JSONResponse::health(healthy, ocrService_ ? ocrService_->getVersion() : "unknown"));
}

void OCRController::getMetrics(const HttpRequestPtr &req,
                              std::function<void(const HttpResponsePtr &)> &&callback) {
    callback(JSONResponse::metrics(requestCount_.load(), successCount_.load(), errorCount_.load()));
}
