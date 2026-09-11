#include "FindTemplateV1.h"
#include "Timer.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
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

std::string matchDetail(size_t index, const T_T::MatchResult& result)
{
    const int templateId = result.template_id > 0 ? result.template_id : 0;
    std::ostringstream detail;
    detail << '#' << (index + 1)
           << " T" << templateId
           << std::fixed << std::setprecision(3)
           << " S" << result.score
           << std::setprecision(2)
           << " M" << result.scale
           << std::setprecision(1)
           << " C" << result.pose.x << ',' << result.pose.y << ',' << result.pose.angle;
    return detail.str();
}

void drawInfoPanel(cv::Mat& image, double elapsedMs,
                   const std::vector<T_T::MatchResult>& results)
{
    if (image.empty()) return;

    // Keep the panel legible on both small fixtures and large camera frames.
    const int shortSide = std::max(1, std::min(image.cols, image.rows));
    const int pad = std::max(10, static_cast<int>(std::round(shortSide * 0.025)));
    const int lineGap = std::max(4, static_cast<int>(std::round(shortSide * 0.010)));
    double fontScale = std::max(0.25, std::min(0.80, shortSide / 900.0));
    const std::string timeLine = "Time: " + [&]() {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3) << elapsedMs << " ms";
        return ss.str();
    }();
    const std::string countLine = "Matches: " + std::to_string(results.size());
    std::vector<std::string> details;
    for (size_t index = 0; index < results.size(); ++index)
        details.push_back(matchDetail(index, results[index]));

    int thickness = 1;
    int baseline = 0;
    cv::Size sample;
    int lineHeight = 0;
    // Preserve the source image height. If many matches exist, details flow
    // into additional columns in the left panel instead of extending the
    // canvas downward or shrinking into unreadable text.
    while (fontScale > 0.12)
    {
        thickness = std::max(1, static_cast<int>(std::round(fontScale * 1.7)));
        sample = cv::getTextSize("Ag", cv::FONT_HERSHEY_SIMPLEX,
                                 fontScale, thickness, &baseline);
        lineHeight = sample.height + baseline + lineGap;
        if (2 * pad + 3 * lineHeight <= image.rows) break;
        fontScale *= 0.88;
    }
    const int detailTop = pad + 2 * lineHeight + lineGap;
    const int rowsPerColumn = std::max(1, (image.rows - detailTop - pad) / lineHeight);
    const int columnCount = details.empty() ? 0 :
        (static_cast<int>(details.size()) + rowsPerColumn - 1) / rowsPerColumn;
    const int columnGap = std::max(pad, lineGap * 2);
    std::vector<int> columnWidths(columnCount, 0);
    for (size_t index = 0; index < details.size(); ++index)
    {
        const int column = static_cast<int>(index) / rowsPerColumn;
        columnWidths[column] = std::max(columnWidths[column],
            cv::getTextSize(details[index], cv::FONT_HERSHEY_SIMPLEX,
                            fontScale, thickness, &baseline).width);
    }
    int detailWidth = 0;
    for (const int width : columnWidths) detailWidth += width;
    if (columnCount > 1) detailWidth += (columnCount - 1) * columnGap;
    const int headerWidth = std::max(
        cv::getTextSize(timeLine, cv::FONT_HERSHEY_SIMPLEX,
                        fontScale, thickness, &baseline).width,
        cv::getTextSize(countLine, cv::FONT_HERSHEY_SIMPLEX,
                        fontScale, thickness, &baseline).width);
    const int panelWidth = std::max(150, 2 * pad + std::max(headerWidth, detailWidth));

    cv::Mat expanded(image.rows, image.cols + panelWidth, image.type(),
                     cv::Scalar(26, 31, 42));
    image.copyTo(expanded(cv::Rect(panelWidth, 0, image.cols, image.rows)));
    // A subtle divider keeps the panel visually separate without competing
    // with the colored result frames in the image area.
    cv::line(expanded, cv::Point(panelWidth, 0), cv::Point(panelWidth, image.rows - 1),
             cv::Scalar(70, 78, 92), 1, cv::LINE_AA);

    int y = pad + sample.height;
    cv::putText(expanded, timeLine, cv::Point(pad, y), cv::FONT_HERSHEY_SIMPLEX,
                fontScale, cv::Scalar(238, 242, 248), thickness, cv::LINE_AA);
    y += lineHeight;
    cv::putText(expanded, countLine, cv::Point(pad, y), cv::FONT_HERSHEY_SIMPLEX,
                fontScale, cv::Scalar(238, 242, 248), thickness, cv::LINE_AA);
    cv::line(expanded, cv::Point(pad, detailTop - lineGap),
             cv::Point(panelWidth - pad, detailTop - lineGap),
             cv::Scalar(70, 78, 92), 1, cv::LINE_AA);

    int columnX = pad;
    for (int column = 0; column < columnCount; ++column)
    {
        const int first = column * rowsPerColumn;
        const int last = std::min(static_cast<int>(details.size()), first + rowsPerColumn);
        int rowY = detailTop + sample.height;
        for (int index = first; index < last; ++index)
        {
            cv::putText(expanded, details[index], cv::Point(columnX, rowY),
                        cv::FONT_HERSHEY_SIMPLEX, fontScale,
                        cv::Scalar(238, 242, 248), thickness, cv::LINE_AA);
            rowY += lineHeight;
        }
        columnX += columnWidths[column] + columnGap;
    }
    image = expanded;
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
    drawInfoPanel(color, timer.get("Template matching"), results);
    if (!cv::imwrite(outputPath, color)) {
        std::cerr << "Failed to write result image: " << outputPath << '\n'; return 1;
    }
    std::cout << "Result image: " << outputPath << '\n';
    timer.report();
    return 0;
}
