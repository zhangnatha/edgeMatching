#include "FindTemplateV1.h"
#include "MakeTemplateV1.h"
#include <opencv2/opencv.hpp>
#include <cstdio>
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
    for (const auto& path : modelPaths) std::remove(path.c_str());

    // src9_2 has seven high-quality candidates at this threshold after the
    // intermediate pyramid gate is relaxed. Before the fix, two of those
    // proposals were discarded before reaching L0 and only five remained.
    if (!ok || results.size() < 7)
    {
        std::cerr << "intermediate threshold regression: got "
                  << results.size() << " results, expected at least 7\n";
        return 5;
    }
    return 0;
}
