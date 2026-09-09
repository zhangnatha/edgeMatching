#include "FindTemplateV1.h"
#include "Timer.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <set>

namespace {
void usage(const char* p) {
    std::cout << "Usage: " << p << " [search_image] [model1.json model2.json ...]"
              << " [--min-score N] [--max-overlap N]"
              << " [--angle-start DEG] [--angle-end DEG]"
              << " [--scale-min N] [--scale-max N] [--scale-step N]"
              << " [--min-visible-ratio N] [--min-contrast N]"
              << " [--metric use-polarity|ignore-global-polarity|ignore-local-polarity]"
              << " [--subpixel] [--output FILE]\n";
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
    double minScore = 0.7;
    double maxOverlap = 0.5;
    double angleStart = -180.0;
    double angleEnd = 180.0;

    int i = 1;
    if (i < argc && argv[i][0] != '-') imagePath = argv[i++];
    while (i < argc) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
        if (arg == "--min-score") {
            if (!number(i, argc, argv, minScore)) { usage(argv[0]); return 2; }
        } else if (arg == "--max-overlap") {
            if (!number(i, argc, argv, maxOverlap)) { usage(argv[0]); return 2; }
        } else if (arg == "--angle-start") {
            if (!number(i, argc, argv, angleStart)) { usage(argv[0]); return 2; }
        } else if (arg == "--angle-end") {
            if (!number(i, argc, argv, angleEnd)) { usage(argv[0]); return 2; }
        } else if (arg == "--scale-min") {
            if (!number(i, argc, argv, scaleCfg.scale_min)) { usage(argv[0]); return 2; }
        } else if (arg == "--scale-max") {
            if (!number(i, argc, argv, scaleCfg.scale_max)) { usage(argv[0]); return 2; }
        } else if (arg == "--scale-step") {
            if (!number(i, argc, argv, scaleCfg.scale_step)) { usage(argv[0]); return 2; }
        } else if (arg == "--min-visible-ratio") {
            if (!number(i, argc, argv, scaleCfg.min_visible_ratio)) { usage(argv[0]); return 2; }
        } else if (arg == "--min-contrast") {
            double value = 0.0;
            if (!number(i, argc, argv, value) || !std::isfinite(value) ||
                std::floor(value) != value || value < 0.0 || value > 361.0) {
                std::cerr << "--min-contrast must be an integer in [0, 361].\n";
                return 2;
            }
            scaleCfg.min_contrast = static_cast<int>(value);
        } else if (arg == "--metric") {
            if (++i >= argc) { usage(argv[0]); return 2; }
            const std::string value = argv[i];
            if (value == "use-polarity") scaleCfg.metric = I_I::USE_POLARITY;
            else if (value == "ignore-local-polarity") scaleCfg.metric = I_I::IGNORE_LOCAL_POLARITY;
            else if (value == "ignore-global-polarity") scaleCfg.metric = I_I::IGNORE_GLOBAL_POLARITY;
            else { std::cerr << "Invalid --metric value: " << value << '\n'; return 2; }
        } else if (arg == "--subpixel") {
            scaleCfg.subpixel_refine = true;
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
    if (!std::isfinite(minScore) || minScore < 0.0 || minScore > 1.0 ||
        !std::isfinite(maxOverlap) || maxOverlap < 0.0 || maxOverlap > 1.0) {
        std::cerr << "--min-score and --max-overlap must be in [0, 1].\n";
        return 2;
    }
    if (!std::isfinite(angleStart) || !std::isfinite(angleEnd) ||
        angleStart < -180.0 || angleEnd > 180.0 || angleStart > angleEnd ||
        std::floor(angleStart) != angleStart || std::floor(angleEnd) != angleEnd) {
        std::cerr << "Angles must be integer degrees in [-180, 180] with start <= end.\n";
        return 2;
    }
    if (modelPaths.empty()) modelPaths.push_back("./model.json");

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) { std::cerr << "Failed to read search image: " << imagePath << '\n'; return 1; }

    SM_V1::SearchTemplate matcher;
    std::vector<T_T::Template::Ptr> models;
    std::set<int> usedIds;
    int nextId = 1;
    for (const auto& path : modelPaths) {
        auto model = matcher.loadModelFileFromJson(path);
        if (!model) { std::cerr << "Failed to load model: " << path << '\n'; return 1; }
        int id = model->template_cfg.id;
        if (id <= 0 || usedIds.count(id) != 0) {
            while (usedIds.count(nextId) != 0) ++nextId;
            id = nextId++;
            model->template_cfg.id = id;
            std::cout << "Assigned unique template ID " << id << " to " << path << '\n';
        }
        usedIds.insert(id);
        models.push_back(model);
    }

    Timer timer(TimerMethod::HighResolutionClock);
    std::vector<T_T::MatchResult> results;
    timer.start();
    const bool ok = matcher.searchTemplate(image, cv::Mat(), models,
                                           static_cast<int>(angleStart),
                                           static_cast<int>(angleEnd),
                                           static_cast<float>(minScore), 200,
                                           static_cast<float>(maxOverlap), -1, 0.9f,
                                           true, scaleCfg, results);
    timer.record("Template matching");
    if (!ok) { std::cerr << "Template matching failed; check search parameters.\n"; return 1; }

    std::cout << std::fixed << std::setprecision(6);
    for (size_t n = 0; n < results.size(); ++n) {
        const auto& r = results[n];
        std::cout << '[' << n << "] template_id=" << r.template_id
                  << " x=" << r.pose.x << " y=" << r.pose.y << " angle=" << r.pose.angle
                  << " score=" << r.score << " scale=" << r.scale
                  << " visible=" << r.visible_ratio
                  << " matched=" << r.matched_ratio << '\n';
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
