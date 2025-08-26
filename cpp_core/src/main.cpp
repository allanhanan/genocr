#include "genocr.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <map>
#include <mutex>
#include <atomic>
#include <set>

class OCRProcessor {
private:
    std::unique_ptr<GenOCR> pipeline_;
    std::mutex process_mutex_;
    std::map<std::string, std::string> active_processes_;
    std::atomic<int> processed_count_{0};
    
public:
    OCRProcessor(const std::string& models_dir, bool use_cuda = false) {
        std::cout << "Initializing OCR Pipeline..." << std::endl;
        pipeline_ = std::make_unique<GenOCR>(models_dir, use_cuda);
        std::cout << "OCR Pipeline ready!" << std::endl;
    }
    
    std::string processFile(const std::string& image_path) {
        std::lock_guard<std::mutex> lock(process_mutex_);
        active_processes_[image_path] = "Processing...";
        
        // Capture console output
        std::ostringstream buffer;
        std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
        
        try {
            std::string result = pipeline_->Run(image_path);
            std::cout.rdbuf(old);
            active_processes_[image_path] = "Completed";
            processed_count_++;
            return result;
        } catch (const std::exception& e) {
            std::cout.rdbuf(old);
            active_processes_[image_path] = "Error: " + std::string(e.what());
            return "Error processing " + image_path + ": " + e.what();
        }
    }
    
    void showStatus() {
        std::lock_guard<std::mutex> lock(process_mutex_);
        std::cout << "\n=== OCR Processor Status ===" << std::endl;
        std::cout << "Total processed: " << processed_count_ << std::endl;
        std::cout << "Active processes:" << std::endl;
        
        for (const auto& [file, status] : active_processes_) {
            std::cout << "  " << std::filesystem::path(file).filename().string() 
                      << " -> " << status << std::endl;
        }
        std::cout << "===========================" << std::endl;
    }
    
    int getProcessedCount() const { return processed_count_; }
};

void saveResult(const std::string& result, const std::string& image_path, const std::string& output_dir) {
    std::filesystem::path out_path;
    
    if (output_dir.empty()) {
        out_path = std::filesystem::path(image_path).replace_extension(".txt");
    } else {
        std::filesystem::create_directories(output_dir);
        out_path = std::filesystem::path(output_dir) / 
                   std::filesystem::path(image_path).filename().replace_extension(".txt");
    }
    
    std::ofstream ofs(out_path);
    if (ofs.is_open()) {
        ofs << result;
        std::cout << "✓ Saved: " << out_path << std::endl;
    } else {
        std::cerr << "✗ Failed to save: " << out_path << std::endl;
    }
}

bool isImageFile(const std::string& filepath) {
    std::string ext = std::filesystem::path(filepath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tiff");
}

void processSingleFiles(OCRProcessor& processor, const std::vector<std::string>& files, const std::string& output_dir) {
    std::cout << "\n=== Processing " << files.size() << " files ===" << std::endl;
    
    for (size_t i = 0; i < files.size(); ++i) {
        const auto& file = files[i];
        
        if (!std::filesystem::exists(file)) {
            std::cerr << "✗ File not found: " << file << std::endl;
            continue;
        }
        
        if (!isImageFile(file)) {
            std::cerr << "✗ Not an image file: " << file << std::endl;
            continue;
        }
        
        std::cout << "\n[" << (i+1) << "/" << files.size() << "] Processing: " 
                  << std::filesystem::path(file).filename() << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        std::string result = processor.processFile(file);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "⏱ Processed in " << duration.count() << "s" << std::endl;
        
        saveResult(result, file, output_dir);
    }
    
    std::cout << "\n ✓ Completed processing " << files.size() << " files!" << std::endl;
}

void standbyMode(OCRProcessor& processor, const std::string& watch_dir, const std::string& output_dir) {
    std::cout << "\n=== Standby Mode Active ===" << std::endl;
    std::cout << "Watching: " << watch_dir << std::endl;
    std::cout << "Output: " << (output_dir.empty() ? "Same directory as input" : output_dir) << std::endl;
    std::cout << "Commands: 'status', 'quit'" << std::endl;
    std::cout << "============================\n" << std::endl;
    
    std::filesystem::path watch_path(watch_dir);
    if (!std::filesystem::exists(watch_path)) {
        std::filesystem::create_directories(watch_path);
    }
    
    std::set<std::string> processed_files;
    
    // Start background thread for file watching
    std::atomic<bool> should_quit{false};
    std::thread watcher([&]() {
        while (!should_quit) {
            try {
                for (const auto& entry : std::filesystem::directory_iterator(watch_path)) {
                    if (entry.is_regular_file()) {
                        std::string filepath = entry.path().string();
                        
                        if (isImageFile(filepath) && processed_files.find(filepath) == processed_files.end()) {
                            processed_files.insert(filepath);
                            
                            std::cout << "New file detected: " << entry.path().filename() << std::endl;
                            
                            auto start = std::chrono::high_resolution_clock::now();
                            std::string result = processor.processFile(filepath);
                            auto end = std::chrono::high_resolution_clock::now();
                            
                            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
                            std::cout << "⏱ Processed in " << duration.count() << "s" << std::endl;
                            
                            saveResult(result, filepath, output_dir);
                            
                            // Optionally move processed file to avoid reprocessing
                            std::filesystem::path processed_dir = watch_path / "processed";
                            std::filesystem::create_directories(processed_dir);
                            std::filesystem::rename(filepath, processed_dir / entry.path().filename());
                        }
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in file watcher: " << e.what() << std::endl;
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    });
    
    // Command loop
    std::string command;
    while (std::getline(std::cin, command)) {
        if (command == "quit" || command == "q") {
            should_quit = true;
            break;
        } else if (command == "status" || command == "s") {
            processor.showStatus();
        } else if (command == "help" || command == "h") {
            std::cout << "Commands:\n"
                      << "  status (s) - Show processing status\n"
                      << "  quit (q)   - Exit standby mode\n"
                      << "  help (h)   - Show this help\n" << std::endl;
        } else if (!command.empty()) {
            std::cout << "Unknown command: " << command << std::endl;
        }
    }
    
    watcher.join();
    std::cout << "Standby mode ended." << std::endl;
}

void showUsage(const char* program_name) {
    std::cout << "GenOCR - Multi-language OCR with AI Correction\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << program_name << " [OPTIONS] <image_files...>\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help              Show this help message\n";
    std::cout << "  -o, --output DIR        Output directory for results\n";
    std::cout << "  -m, --models DIR        Models directory (default: ../../models)\n";
    std::cout << "  -c, --cuda              Enable CUDA acceleration\n";
    std::cout << "  -b, --batch DIR         Batch/standby mode - watch directory for new files\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " image1.png image2.jpg         # Process specific files\n";
    std::cout << "  " << program_name << " -o results/ *.png             # Process all PNGs, save to results/\n";
    std::cout << "  " << program_name << " -b watch_folder/               # Standby mode - watch for new files\n";
    std::cout << "  " << program_name << " -c -b watch/ -o results/       # CUDA + batch mode with output dir\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        showUsage(argv[0]);
        return 1;
    }
    
    std::vector<std::string> input_files;
    std::string output_dir;
    std::string models_dir = "../../models";
    std::string batch_dir;
    bool use_cuda = false;
    bool batch_mode = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            showUsage(argv[0]);
            return 0;
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_dir = argv[++i];
        } else if ((arg == "-m" || arg == "--models") && i + 1 < argc) {
            models_dir = argv[++i];
        } else if ((arg == "-b" || arg == "--batch") && i + 1 < argc) {
            batch_mode = true;
            batch_dir = argv[++i];
        } else if (arg == "-c" || arg == "--cuda") {
            use_cuda = true;
        } else if (arg.length() > 0 && arg[0] == '-') {  // Fixed C++17 compatible check
            std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        } else {
            input_files.push_back(arg);
        }
    }
    
    // Validate arguments
    if (!batch_mode && input_files.empty()) {
        std::cerr << "Error: No input files specified and not in batch mode." << std::endl;
        showUsage(argv[0]);
        return 1;
    }
    
    try {
        // Initialize OCR processor (this is where the 50s initialization happens)
        OCRProcessor processor(models_dir, use_cuda);
        
        if (batch_mode) {
            standbyMode(processor, batch_dir, output_dir);
        } else {
            processSingleFiles(processor, input_files, output_dir);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
