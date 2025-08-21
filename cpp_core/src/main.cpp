#include "genocr.hpp"
#include <iostream>

int main() {
    std::string models_dir = "../../models";
    std::string image_path = "../../tests/test_assets/ocrTTYMid.png";

    try {
        GenOCR pipeline(models_dir, false);
        pipeline.Run(image_path);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}