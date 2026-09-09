#include "FindTemplateV1.h"

#include <opencv2/imgcodecs.hpp>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {
void usage(const char* program)
{
    std::cerr << "Usage: " << program << " IMAGE MODEL [MODEL ...]"
              << " [--warmup N] [--iterations N] [--min-score X]"
              << " [--min-visible-ratio X] [--angle-start N] [--angle-end N]\n";
}

bool parseInt(const char* text, int& value)
{
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (!end || *end != '\0') return false;
    value = static_cast<int>(parsed);
    return true;
}

bool parseDouble(const char* text, double& value)
{
    char* end = nullptr;
    value = std::strtod(text, &end);
    return end && *end == '\0' && std::isfinite(value);
}

double percentile(std::vector<double> values, double quantile)
{
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(
        std::ceil(quantile * values.size()) - 1.0);
    return values[std::min(index, values.size() - 1)];
}

void hashBytes(uint64_t& hash, const void* data, size_t size)
{
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i)
    {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
}

uint64_t resultHash(const std::vector<T_T::MatchResult>& results)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    const uint64_t count = results.size();
    hashBytes(hash, &count, sizeof(count));
    for (const auto& result : results)
    {
        hashBytes(hash, &result.pose.x, sizeof(result.pose.x));
        hashBytes(hash, &result.pose.y, sizeof(result.pose.y));
        hashBytes(hash, &result.pose.angle, sizeof(result.pose.angle));
        hashBytes(hash, &result.score, sizeof(result.score));
        hashBytes(hash, &result.scale, sizeof(result.scale));
        hashBytes(hash, &result.template_id, sizeof(result.template_id));
        hashBytes(hash, &result.visible_ratio, sizeof(result.visible_ratio));
        hashBytes(hash, &result.matched_ratio, sizeof(result.matched_ratio));
    }
    return hash;
}
}

int main(int argc, char* argv[])
{
    if (argc < 3) { usage(argv[0]); return 2; }
    const std::string imagePath = argv[1];
    std::vector<std::string> modelPaths;
    int warmup = 5, iterations = 30;
    int angleStart = -180, angleEnd = 180;
    double minScore = 0.7, minVisible = 1.0;
    for (int i = 2; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument == "--warmup" || argument == "--iterations" ||
            argument == "--angle-start" || argument == "--angle-end")
        {
            if (++i >= argc) { usage(argv[0]); return 2; }
            int value = 0;
            if (!parseInt(argv[i], value)) { usage(argv[0]); return 2; }
            if (argument == "--warmup") warmup = value;
            else if (argument == "--iterations") iterations = value;
            else if (argument == "--angle-start") angleStart = value;
            else angleEnd = value;
        }
        else if (argument == "--min-score" || argument == "--min-visible-ratio")
        {
            if (++i >= argc) { usage(argv[0]); return 2; }
            double value = 0.0;
            if (!parseDouble(argv[i], value)) { usage(argv[0]); return 2; }
            if (argument == "--min-score") minScore = value;
            else minVisible = value;
        }
        else if (!argument.empty() && argument[0] == '-')
        {
            std::cerr << "Unknown option: " << argument << '\n';
            return 2;
        }
        else modelPaths.push_back(argument);
    }
    if (modelPaths.empty() || warmup < 0 || iterations < 1 ||
        minScore < 0.0 || minScore > 1.0 || minVisible <= 0.0 || minVisible > 1.0 ||
        angleStart < -180 || angleEnd > 180 || angleStart > angleEnd)
    {
        usage(argv[0]);
        return 2;
    }

    const cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) { std::cerr << "Cannot load image: " << imagePath << '\n'; return 1; }
    SM_V1::SearchTemplate matcher;
    std::vector<T_T::Template::Ptr> models;
    std::set<int> ids;
    for (const auto& path : modelPaths)
    {
        auto model = matcher.loadModelFileFromJson(path);
        if (!model || model->template_cfg.id <= 0 || !ids.insert(model->template_cfg.id).second)
        {
            std::cerr << "Invalid model or duplicate ID: " << path << '\n';
            return 1;
        }
        models.push_back(model);
    }

    std::vector<double> samples;
    size_t expectedCount = 0;
    uint64_t expectedHash = 0;
    for (int run = -warmup; run < iterations; ++run)
    {
        std::vector<T_T::MatchResult> results;
        const auto begin = std::chrono::steady_clock::now();
        const bool ok = matcher.searchTemplate(
            image, cv::Mat(), models, angleStart, angleEnd,
            static_cast<float>(minScore), 200, 0.5f, -1, 0.9f, true,
            T_T::ScaleSearchCfg(1.0, 1.0, 1.0, minVisible), results);
        const auto end = std::chrono::steady_clock::now();
        if (!ok) { std::cerr << "searchTemplate failed\n"; return 1; }
        const uint64_t hash = resultHash(results);
        if (run == -warmup) { expectedCount = results.size(); expectedHash = hash; }
        else if (results.size() != expectedCount || hash != expectedHash)
        {
            std::cerr << "Non-deterministic result at run " << run
                      << ": count=" << results.size() << " hash=0x" << std::hex << hash << '\n';
            return 3;
        }
        if (run >= 0)
            samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }

    const double median = percentile(samples, 0.50);
    std::vector<double> deviations;
    deviations.reserve(samples.size());
    for (double sample : samples) deviations.push_back(std::abs(sample - median));
    std::cout << std::fixed << std::setprecision(3)
              << "warmup=" << warmup << " iterations=" << iterations
              << " omp_max_threads=" << omp_get_max_threads()
              << " hardware_threads=" << std::thread::hardware_concurrency() << '\n'
              << "median_ms=" << median
              << " p90_ms=" << percentile(samples, 0.90)
              << " p95_ms=" << percentile(samples, 0.95)
              << " mad_ms=" << percentile(deviations, 0.50) << '\n'
              << "count=" << expectedCount << " hash=0x" << std::hex << expectedHash << '\n';
    return 0;
}
