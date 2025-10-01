#pragma once

#include <drogon/HttpController.h>
#include "ocr_service.hpp"

using namespace drogon;

class OCRController : public drogon::HttpController<OCRController>
{
public:
    METHOD_LIST_BEGIN
    // OCR processing endpoint
    ADD_METHOD_TO(OCRController::processImage, "/api/v1/ocr/process", Post);
    
    // Health check endpoint
    ADD_METHOD_TO(OCRController::healthCheck, "/api/v1/health", Get);
    
    // Metrics endpoint
    ADD_METHOD_TO(OCRController::getMetrics, "/api/v1/metrics", Get);
    METHOD_LIST_END

    // Constructor - initialize service
    OCRController();

    // Endpoint handlers
    void processImage(const HttpRequestPtr &req,
                     std::function<void(const HttpResponsePtr &)> &&callback);
                     
    void healthCheck(const HttpRequestPtr &req,
                    std::function<void(const HttpResponsePtr &)> &&callback);
                    
    void getMetrics(const HttpRequestPtr &req,
                   std::function<void(const HttpResponsePtr &)> &&callback);

private:
    std::unique_ptr<OCRService> ocrService_;
    std::atomic<int> requestCount_{0};
    std::atomic<int> successCount_{0};
    std::atomic<int> errorCount_{0};
};
