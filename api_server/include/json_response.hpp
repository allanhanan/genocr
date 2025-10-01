#pragma once

#include <drogon/drogon.h>
#include "ocr_service.hpp"

class JSONResponse {
public:
    static drogon::HttpResponsePtr success(const OCRResult& result);
    static drogon::HttpResponsePtr error(const std::string& message, 
                                       int status_code = 400, 
                                       const std::string& request_id = "");
    static drogon::HttpResponsePtr health(bool is_healthy, const std::string& version);
    static drogon::HttpResponsePtr metrics(int total_requests, int success_count, int error_count);
    static drogon::HttpResponsePtr validationError(const std::string& field, const std::string& message);
};
