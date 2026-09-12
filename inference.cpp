#include "FindTemplateV1.h"
#include "Timer.h"
#include "ModelIdNormalization.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::string getCpuInfo()
{
    static std::string cachedCpu;
    if (!cachedCpu.empty()) return cachedCpu;

    std::ifstream file("/proc/cpuinfo");
    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line))
        {
            if (line.compare(0, 10, "model name") == 0 ||
                line.compare(0, 9, "Processor") == 0 ||
                line.compare(0, 8, "Hardware") == 0 ||
                line.compare(0, 7, "Model") == 0)
            {
                size_t colon = line.find(':');
                if (colon != std::string::npos)
                {
                    size_t start = line.find_first_not_of(" \t", colon + 1);
                    if (start != std::string::npos)
                    {
                        std::string raw = line.substr(start);
                        while (!raw.empty() && (raw.back() == '\r' || raw.back() == '\n' ||
                                                raw.back() == ' ' || raw.back() == '\t'))
                            raw.pop_back();
                        std::string clean;
                        bool inSpace = false;
                        for (char c : raw)
                        {
                            if (c == ' ' || c == '\t')
                            {
                                if (!inSpace) { clean.push_back(' '); inSpace = true; }
                            }
                            else
                            {
                                clean.push_back(c);
                                inSpace = false;
                            }
                        }
                        if (!clean.empty())
                        {
                            cachedCpu = clean;
                            return cachedCpu;
                        }
                    }
                }
            }
        }
    }
    const int ncpus = cv::getNumberOfCPUs();
    cachedCpu = "CPU (" + std::to_string(ncpus) + " cores)";
    return cachedCpu;
}

void usage(const char* p) {
    std::cout << "Usage: " << p << " [search_image] [model1.json model2.json ...]"
              << " [--min-score N] [--max-overlap N]"
              << " [--angle-start DEG] [--angle-end DEG]"
              << " [--scale-min N] [--scale-max N] [--scale-step N]"
              << " [--min-visible-ratio N] [--min-contrast N]"
              << " [--metric use-polarity|ignore-global-polarity|ignore-local-polarity]"
              << " [--subpixel] [--simd] [--output FILE]\n";
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

    // 在小尺寸样例和大幅相机图像上都保持信息栏清晰可读。
    const int shortSide = std::max(1, std::min(image.cols, image.rows));
    const int pad = std::max(10, static_cast<int>(std::round(shortSide * 0.025)));
    const int lineGap = std::max(4, static_cast<int>(std::round(shortSide * 0.010)));
    double fontScale = std::max(0.25, std::min(0.80, shortSide / 900.0));
    const std::string resolutionLine = "Resolution: " + std::to_string(image.cols) + "x" + std::to_string(image.rows);
    const std::string cpuLine = "CPU: " + getCpuInfo();
    const std::string timeLine = "Cost Time: " + [&]() {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3) << elapsedMs << " ms";
        return ss.str();
    }();
    const std::string countLine = "Matches: " + std::to_string(results.size());
    const std::vector<std::string> headerLines = {
        resolutionLine,
        cpuLine,
        timeLine,
        countLine
    };
    const int headerLineCount = static_cast<int>(headerLines.size());

    std::vector<std::string> details;
    for (size_t index = 0; index < results.size(); ++index)
        details.push_back(matchDetail(index, results[index]));

    int thickness = 1;
    int baseline = 0;
    cv::Size sample;
    int lineHeight = 0;
    // 保持原图高度不变；结果较多时在左侧信息栏增加列，而不是向下扩展画布
    // 或缩小文字导致无法阅读。
    while (fontScale > 0.12)
    {
        thickness = std::max(1, static_cast<int>(std::round(fontScale * 1.7)));
        sample = cv::getTextSize("Ag", cv::FONT_HERSHEY_SIMPLEX,
                                 fontScale, thickness, &baseline);
        lineHeight = sample.height + baseline + lineGap;
        if (2 * pad + (headerLineCount + 1) * lineHeight <= image.rows) break;
        fontScale *= 0.88;
    }
    const int detailTop = pad + headerLineCount * lineHeight + lineGap;
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
    int headerWidth = 0;
    for (const auto& hline : headerLines)
    {
        headerWidth = std::max(headerWidth,
            cv::getTextSize(hline, cv::FONT_HERSHEY_SIMPLEX,
                            fontScale, thickness, &baseline).width);
    }
    const int panelWidth = std::max(150, 2 * pad + std::max(headerWidth, detailWidth));

    cv::Mat expanded(image.rows, image.cols + panelWidth, image.type(),
                     cv::Scalar(26, 31, 42));
    image.copyTo(expanded(cv::Rect(panelWidth, 0, image.cols, image.rows)));
    // 使用细分隔线区分信息栏和图像区域，同时避免与彩色结果框争夺视觉重点。
    cv::line(expanded, cv::Point(panelWidth, 0), cv::Point(panelWidth, image.rows - 1),
             cv::Scalar(70, 78, 92), 1, cv::LINE_AA);

    int y = pad + sample.height;
    for (const auto& hline : headerLines)
    {
        cv::putText(expanded, hline, cv::Point(pad, y), cv::FONT_HERSHEY_SIMPLEX,
                    fontScale, cv::Scalar(238, 242, 248), thickness, cv::LINE_AA);
        y += lineHeight;
    }
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

std::string getJsonPathFromImagePath(const std::string& imagePath)
{
    const size_t dot = imagePath.find_last_of('.');
    const size_t slash = imagePath.find_last_of("/\\");
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
    {
        return imagePath.substr(0, dot) + ".json";
    }
    return imagePath + ".json";
}

bool saveResultsToJson(const std::string& jsonPath,
                       const std::string& imagePath,
                       int imageWidth, int imageHeight,
                       const T_T::ScaleSearchCfg& scaleCfg,
                       double angleStart, double angleEnd,
                       double minScore, double maxOverlap,
                       double elapsedMs,
                       const std::string& cpuInfo,
                       const std::vector<T_T::MatchResult>& results)
{
    std::ofstream out(jsonPath);
    if (!out.is_open())
    {
        std::cerr << "Failed to open output json file: " << jsonPath << '\n';
        return false;
    }

    auto escapeJson = [](const std::string& s) {
        std::ostringstream ss;
        for (char c : s)
        {
            if (c == '"') ss << "\\\"";
            else if (c == '\\') ss << "\\\\";
            else if (c == '\b') ss << "\\b";
            else if (c == '\f') ss << "\\f";
            else if (c == '\n') ss << "\\n";
            else if (c == '\r') ss << "\\r";
            else if (c == '\t') ss << "\\t";
            else ss << c;
        }
        return ss.str();
    };

    std::string metricStr = "use-polarity";
    if (scaleCfg.metric == I_I::IGNORE_LOCAL_POLARITY) metricStr = "ignore-local-polarity";
    else if (scaleCfg.metric == I_I::IGNORE_GLOBAL_POLARITY) metricStr = "ignore-global-polarity";

    out << "{\n";
    out << "  \"image\": {\n";
    out << "    \"path\": \"" << escapeJson(imagePath) << "\",\n";
    out << "    \"width\": " << imageWidth << ",\n";
    out << "    \"height\": " << imageHeight << "\n";
    out << "  },\n";

    out << "  \"search_config\": {\n";
    out << "    \"scale_min\": " << std::fixed << std::setprecision(4) << scaleCfg.scale_min << ",\n";
    out << "    \"scale_max\": " << std::fixed << std::setprecision(4) << scaleCfg.scale_max << ",\n";
    out << "    \"scale_step\": " << std::fixed << std::setprecision(4) << scaleCfg.scale_step << ",\n";
    out << "    \"angle_start\": " << std::fixed << std::setprecision(2) << angleStart << ",\n";
    out << "    \"angle_end\": " << std::fixed << std::setprecision(2) << angleEnd << ",\n";
    out << "    \"min_score\": " << std::fixed << std::setprecision(4) << minScore << ",\n";
    out << "    \"max_overlap\": " << std::fixed << std::setprecision(4) << maxOverlap << ",\n";
    out << "    \"min_visible_ratio\": " << std::fixed << std::setprecision(4) << scaleCfg.min_visible_ratio << ",\n";
    out << "    \"min_contrast\": " << scaleCfg.min_contrast << ",\n";
    out << "    \"metric\": \"" << metricStr << "\",\n";
    out << "    \"subpixel_refine\": " << (scaleCfg.subpixel_refine ? "true" : "false") << ",\n";
    out << "    \"simd_requested\": " << (scaleCfg.use_simd ? "true" : "false") << ",\n";
    out << "    \"simd_actual\": " << (scaleCfg.use_simd && SM_V1::SearchTemplate::isSimdAvailable() ? "true" : "false") << "\n";
    out << "  },\n";

    out << "  \"performance\": {\n";
    out << "    \"cpu\": \"" << escapeJson(cpuInfo) << "\",\n";
    out << "    \"cost_time_ms\": " << std::fixed << std::setprecision(3) << elapsedMs << "\n";
    out << "  },\n";

    out << "  \"summary\": {\n";
    out << "    \"total_matches\": " << results.size() << "\n";
    out << "  },\n";

    out << "  \"results\": [\n";
    for (size_t n = 0; n < results.size(); ++n)
    {
        const auto& r = results[n];
        out << "    {\n";
        out << "      \"index\": " << (n + 1) << ",\n";
        out << "      \"template_id\": " << r.template_id << ",\n";
        out << "      \"score\": " << std::fixed << std::setprecision(6) << r.score << ",\n";
        out << "      \"scale\": " << std::fixed << std::setprecision(4) << r.scale << ",\n";
        out << "      \"pose\": {\n";
        out << "        \"x\": " << std::fixed << std::setprecision(6) << r.pose.x << ",\n";
        out << "        \"y\": " << std::fixed << std::setprecision(6) << r.pose.y << ",\n";
        out << "        \"angle\": " << std::fixed << std::setprecision(6) << r.pose.angle << "\n";
        out << "      },\n";
        out << "      \"visible_ratio\": " << std::fixed << std::setprecision(6) << r.visible_ratio << ",\n";
        out << "      \"matched_ratio\": " << std::fixed << std::setprecision(6) << r.matched_ratio << "\n";
        out << "    }" << (n + 1 < results.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    return true;
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
        } else if (arg == "--simd") {
            scaleCfg.use_simd = true;
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
    for (const auto& path : modelPaths) {
        const bool binary = path.size() >= 4 &&
            (path.substr(path.size() - 4) == ".bin" || path.substr(path.size() - 4) == ".BIN");
        auto model = binary ? matcher.loadModelFileFromBinary(path)
                            : matcher.loadModelFileFromJson(path);
        if (!model) { std::cerr << "Failed to load model: " << path << '\n'; return 1; }
        models.push_back(model);
    }
    std::vector<SM_V1::ModelIdAssignment> assignments;
    std::string normalizationError;
    if (!SM_V1::normalizeTemplateIds(models, assignments, &normalizationError)) {
        std::cerr << "Template ID normalization failed: " << normalizationError << '\n';
        return 1;
    }
    for (size_t i = 0; i < assignments.size(); ++i)
        std::cout << "Template ID mapping: original=" << assignments[i].original_id
                  << " runtime=" << assignments[i].runtime_id
                  << " file=" << modelPaths[i] << '\n';

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

    const std::string jsonPath = getJsonPathFromImagePath(outputPath);
    if (!saveResultsToJson(jsonPath, imagePath, image.cols, image.rows,
                           scaleCfg, angleStart, angleEnd, minScore, maxOverlap,
                           timer.get("Template matching"), getCpuInfo(), results)) {
        std::cerr << "Failed to write result json: " << jsonPath << '\n'; return 1;
    }
    std::cout << "Result json: " << jsonPath << '\n';

    timer.report();
    return 0;
}
