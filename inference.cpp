#include "FindTemplateV1.h"
#include "Timer.h"
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {
void usage(const char* p) {
    std::cout << "Usage: " << p << " [search_image] [model1.json model2.json ...]"
              << " [--scale-min N] [--scale-max N] [--scale-step N]"
              << " [--min-visible-ratio N] [--output FILE]\n";
}
bool number(int& i, int argc, const char* argv[], double& out) {
    if (i + 1 >= argc) return false;
    char* end = nullptr;
    out = std::strtod(argv[++i], &end);
    return end && *end == '\0';
}
}

int main(int argc, const char* argv[]) {
    std::string imagePath = "../assert/src.bmp";
    std::vector<std::string> modelPaths;
    std::string outputPath = "result.png";
    T_T::ScaleSearchCfg scaleCfg;

    int i = 1;
    if (i < argc && argv[i][0] != '-') imagePath = argv[i++];
    while (i < argc) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
        if (arg == "--scale-min") {
            if (!number(i, argc, argv, scaleCfg.scale_min)) { usage(argv[0]); return 2; }
        } else if (arg == "--scale-max") {
            if (!number(i, argc, argv, scaleCfg.scale_max)) { usage(argv[0]); return 2; }
        } else if (arg == "--scale-step") {
            if (!number(i, argc, argv, scaleCfg.scale_step)) { usage(argv[0]); return 2; }
        } else if (arg == "--min-visible-ratio") {
            if (!number(i, argc, argv, scaleCfg.min_visible_ratio)) { usage(argv[0]); return 2; }
        } else if (arg == "--output") {
            if (++i >= argc) { usage(argv[0]); return 2; }
            outputPath = argv[i];
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << '\n'; usage(argv[0]); return 2;
        } else {
            modelPaths.push_back(arg);
        }
        ++i;
    }
    if (modelPaths.empty()) modelPaths.push_back("./model.json");

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) { std::cerr << "Failed to read search image: " << imagePath << '\n'; return 1; }

    SM_V1::SearchTemplate matcher;
    std::vector<T_T::Template::Ptr> models;
    for (const auto& path : modelPaths) {
        auto model = matcher.loadModelFileFromJson(path);
        if (!model) { std::cerr << "Failed to load model: " << path << '\n'; return 1; }
        models.push_back(model);
    }

    Timer timer(TimerMethod::HighResolutionClock);
    std::vector<T_T::MatchResult> results;
    timer.start();
    const bool ok = matcher.searchTemplate(image, cv::Mat(), models, -180, 180, 0.7f, 200,
                                           0.5f, -1, 0.9f, true, scaleCfg, results);
    timer.record("Template matching");
    if (!ok) { std::cerr << "Template matching failed; check search parameters.\n"; return 1; }

    std::cout << std::fixed << std::setprecision(6);
    for (size_t n = 0; n < results.size(); ++n) {
        const auto& r = results[n];
        std::cout << '[' << n << "] template_id=" << r.template_id
                  << " x=" << r.pose.x << " y=" << r.pose.y << " angle=" << r.pose.angle
                  << " score=" << r.score << " scale=" << r.scale
                  << " visible=" << r.visible_ratio << '\n';
    }

    cv::Mat color;
    cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
    matcher.drawMatchResults(color, results, models);
    cv::putText(color, "Time: " + std::to_string(timer.get("Template matching")) +
                " ms  Matches: " + std::to_string(results.size()), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    if (!cv::imwrite(outputPath, color)) {
        std::cerr << "Failed to write result image: " << outputPath << '\n'; return 1;
    }
    std::cout << "Result image: " << outputPath << '\n';
    timer.report();
    return 0;
}
