#include "json_response.hpp"
#include <drogon/drogon.h>

drogon::HttpResponsePtr JSONResponse::success(const OCRResult& result) {
    Json::Value response;
    response["success"] = result.success;
    response["request_id"] = result.request_id;
    response["data"]["text"] = result.text;
    response["data"]["processing_time_ms"] = result.processing_time_ms;
    
    Json::Value objects(Json::arrayValue);
    for (const auto& obj : result.detected_objects) {
        objects.append(obj);
    }
    response["data"]["detected_objects"] = objects;
    response["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    resp->setStatusCode(drogon::k200OK);
    resp->addHeader("Content-Type", "application/json");
    return resp;
}

drogon::HttpResponsePtr JSONResponse::error(const std::string& message, int status_code, const std::string& request_id) {
    Json::Value response;
    response["success"] = false;
    response["error"]["message"] = message;
    response["error"]["code"] = status_code;
    if (!request_id.empty()) {
        response["request_id"] = request_id;
    }
    response["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    resp->setStatusCode(static_cast<drogon::HttpStatusCode>(status_code));
    resp->addHeader("Content-Type", "application/json");
    return resp;
}

drogon::HttpResponsePtr JSONResponse::health(bool is_healthy, const std::string& version) {
    Json::Value response;
    response["healthy"] = is_healthy;
    response["status"] = is_healthy ? "OK" : "ERROR";
    response["version"] = version;
    response["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    resp->setStatusCode(is_healthy ? drogon::k200OK : drogon::k503ServiceUnavailable);
    resp->addHeader("Content-Type", "application/json");
    return resp;
}

drogon::HttpResponsePtr JSONResponse::metrics(int total_requests, int success_count, int error_count) {
    Json::Value response;
    response["metrics"]["total_requests"] = total_requests;
    response["metrics"]["successful_requests"] = success_count;
    response["metrics"]["failed_requests"] = error_count;
    response["metrics"]["success_rate"] = total_requests > 0 ? 
        static_cast<double>(success_count) / total_requests * 100.0 : 0.0;
    response["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    resp->setStatusCode(drogon::k200OK);
    resp->addHeader("Content-Type", "application/json");
    return resp;
}

drogon::HttpResponsePtr JSONResponse::validationError(const std::string& field, const std::string& message) {
    Json::Value response;
    response["success"] = false;
    response["error"]["type"] = "validation_error";
    response["error"]["field"] = field;
    response["error"]["message"] = message;
    response["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    resp->setStatusCode(drogon::k400BadRequest);
    resp->addHeader("Content-Type", "application/json");
    return resp;
}
