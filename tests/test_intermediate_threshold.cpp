#include "FindTemplateV1.h"
#include "MakeTemplateV1.h"
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#ifndef ASSERT_DIR
#define ASSERT_DIR "../assert"
#endif

int main()
{
    const std::string root = ASSERT_DIR;
    const std::string imagePath = root + "/src9_2.png";
    const std::string modelPaths[] = {
        "/tmp/edgeMatching_threshold_model_9.json",
        "/tmp/edgeMatching_threshold_model_10.json",
        "/tmp/edgeMatching_threshold_model_11.json"};
    const std::string templatePaths[] = {
        root + "/m9.bmp", root + "/m9_1.bmp", root + "/m9_2.bmp"};

    SM_V1::CreateTemplate trainer;
    for (int i = 0; i < 3; ++i)
    {
        const cv::Mat image = cv::imread(templatePaths[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) return 1;
        const cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));
        auto model = std::make_shared<T_T::Template>();
        model->template_cfg.id = 9 + i;
        if (!trainer.createTemplate(image, mask, -1, -180, 180, 1.0,
                                    false, 25, 100, model) ||
            !trainer.saveModelFile2Json(model, modelPaths[i])) return 2;
    }

    const cv::Mat scene = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (scene.empty()) return 3;

    SM_V1::SearchTemplate matcher;
    std::vector<T_T::Template::Ptr> models;
    for (const auto& path : modelPaths)
    {
        auto model = matcher.loadModelFileFromJson(path);
        if (!model) return 4;
        models.push_back(model);
    }

    T_T::ScaleSearchCfg cfg(1.0, 1.0, 1.0, 1.0, false, 0,
                            I_I::IGNORE_LOCAL_POLARITY);
    std::vector<T_T::MatchResult> results;
    const bool ok = matcher.searchTemplate(scene, cv::Mat(), models,
                                           -180, 180, 0.90f, 200,
                                           0.5f, -1, 0.9f, true,
                                           cfg, results);
    // src9_2 has eight true instances at 0.90 once the correct angular modes
    // survive coarse-to-fine propagation.  Before the intermediate gate fix,
    // two proposals were discarded before L0; before angle-mode retention, a
    // false T11 pose displaced the occluded T9 instance.
    if (!ok || results.size() != 8)
    {
        std::cerr << "intermediate threshold regression: got "
                  << results.size() << " results, expected 8\n";
        return 5;
    }

    const auto angleDistance = [](double lhs, double rhs) {
        double distance = std::fmod(std::abs(lhs - rhs), 360.0);
        return distance > 180.0 ? 360.0 - distance : distance;
    };
    const auto hasPose = [&](const std::vector<T_T::MatchResult>& found, int id,
                             double x, double y, double angle) {
        for (const auto& result : found)
            if (result.template_id == id && std::hypot(result.pose.x - x, result.pose.y - y) <= 6.0 &&
                angleDistance(result.pose.angle, angle) <= 5.0) return true;
        return false;
    };
    if (!hasPose(results, 9, 325, 315, 134)) return 6;

    const auto searchScene = [&](const std::string& filename, float score,
                                 std::vector<T_T::MatchResult>& found) {
        const cv::Mat input = cv::imread(root + "/" + filename, cv::IMREAD_GRAYSCALE);
        found.clear();
        return !input.empty() && matcher.searchTemplate(input, cv::Mat(), models,
            -180, 180, score, 200, 0.5f, -1, 0.9f, true, cfg, found);
    };

    if (!searchScene("src9_3.png", 0.90f, results) || results.size() != 5) return 7;
    if (!searchScene("src9_4.png", 0.89f, results) || results.size() != 4 ||
        !hasPose(results, 9, 263, 160, 103) ||
        !hasPose(results, 10, 214, 242, 177) ||
        !hasPose(results, 11, 327, 216, -35) ||
        !hasPose(results, 9, 278, 310, -59)) return 8;
    if (!searchScene("src9_5.png", 0.90f, results) || results.size() != 3 ||
        !hasPose(results, 9, 359, 209, 136) ||
        !hasPose(results, 9, 213, 263, -133) ||
        !hasPose(results, 9, 362, 308, -44)) return 9;

    for (const auto& path : modelPaths) std::remove(path.c_str());
    return 0;
}
