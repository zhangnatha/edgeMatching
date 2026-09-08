#include "FindTemplateV1.h"
#include "MakeTemplateV1.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <vector>

namespace {
cv::Mat makePattern(bool alternate) {
    cv::Mat image(64, 80, CV_8UC1, cv::Scalar(20));
    cv::rectangle(image, cv::Rect(5, 5, 70, 54), cv::Scalar(230), 3);
    if (alternate) {
        std::vector<cv::Point> triangle{{12, 52}, {40, 10}, {68, 52}};
        cv::polylines(image, triangle, true, cv::Scalar(230), 4);
        cv::circle(image, cv::Point(40, 38), 8, cv::Scalar(230), 3);
    } else {
        cv::line(image, cv::Point(12, 16), cv::Point(67, 48), cv::Scalar(230), 4);
        cv::line(image, cv::Point(12, 48), cv::Point(67, 16), cv::Scalar(230), 4);
        cv::circle(image, cv::Point(40, 32), 11, cv::Scalar(230), 3);
    }
    return image;
}

T_T::Template::Ptr train(const cv::Mat& image, int id) {
    auto model = std::make_shared<T_T::Template>();
    model->template_cfg.id = id;
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));
    SM_V1::CreateTemplate creator;
    if (!creator.createTemplate(image, mask, 0, 0, 0, 1.0, false, 15, 60, model)) return nullptr;
    return model;
}

cv::Mat sceneWith(const cv::Mat& pattern, double scale, int centerX, int centerY) {
    cv::Mat scene(180, 300, CV_8UC1, cv::Scalar(20));
    cv::Mat resized;
    cv::resize(pattern, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);
    const cv::Rect wanted(centerX - resized.cols / 2, centerY - resized.rows / 2,
                          resized.cols, resized.rows);
    const cv::Rect visible = wanted & cv::Rect(0, 0, scene.cols, scene.rows);
    if (!visible.empty()) {
        resized(cv::Rect(visible.x - wanted.x, visible.y - wanted.y,
                         visible.width, visible.height)).copyTo(scene(visible));
    }
    return scene;
}

bool nearResult(const std::vector<T_T::MatchResult>& results, int id,
                double scale, int x, int y, double tolerance) {
    for (const auto& r : results) {
        if (r.template_id == id && std::abs(r.scale - scale) < 0.02 &&
            std::hypot(r.pose.x - x, r.pose.y - y) <= tolerance) return true;
    }
    return false;
}
}

int main() {
    const cv::Mat patternA = makePattern(false);
    const cv::Mat patternB = makePattern(true);
    auto modelA = train(patternA, 101);
    auto modelB = train(patternB, 202);
    if (!modelA || !modelB) return 1;
    if (modelA->template_cfg.id != 101 || modelB->template_cfg.id != 202) return 2;
    if (modelA->templates.size() != 1 || modelA->templates[0]->shape_angle.size() != 1) return 2;

    auto defaultModel = std::make_shared<T_T::Template>();
    cv::Mat defaultMask(patternA.size(), CV_8UC1, cv::Scalar(255));
    SM_V1::CreateTemplate creator;
    if (defaultModel->template_cfg.id != 1 ||
        !creator.createTemplate(patternA, defaultMask, 0, 0, 0, 1.0, false, 15, 60, defaultModel) ||
        defaultModel->template_cfg.id != 1 || !defaultModel->is_inited) return 2;

    SM_V1::SearchTemplate matcher;
    const double scales[] = {0.8, 1.0, 1.2};
    for (double scale : scales) {
        const cv::Mat scene = sceneWith(patternA, scale, 145, 90);
        std::vector<T_T::MatchResult> results;
        const T_T::ScaleSearchCfg cfg(scale, scale, 0.1, 1.0);
        if (!matcher.searchTemplate(scene, cv::Mat(), modelA, 0, 0, 0.55f, 5,
                                    0.4f, 0, 0.8f, true, cfg, results)) return 3;
        if (!nearResult(results, 101, scale, 145, 90, 8.0)) {
            std::cerr << "scale regression failed at " << scale << '\n';
            return 4;
        }
    }

    std::vector<T_T::MatchResult> crossScaleResults;
    if (!matcher.searchTemplate(sceneWith(patternA, 1.0, 145, 90), cv::Mat(), modelA,
                                0, 0, 0.55f, 10, 0.4f, 0, 0.8f, true,
                                T_T::ScaleSearchCfg(0.8, 1.2, 0.1, 1.0), crossScaleResults) ||
        crossScaleResults.size() != 1 || !nearResult(crossScaleResults, 101, 1.0, 145, 90, 8.0)) return 4;

    const cv::Mat partial = sceneWith(patternA, 1.0, 285, 90);
    std::vector<T_T::MatchResult> partialResults;
    if (!matcher.searchTemplate(partial, cv::Mat(), modelA, 0, 0, 0.50f, 5,
                                0.4f, 0, 0.8f, true,
                                T_T::ScaleSearchCfg(1.0, 1.0, 1.0, 0.55), partialResults)) return 5;
    if (!nearResult(partialResults, 101, 1.0, 285, 90, 10.0)) return 6;

    cv::Mat multiScene(180, 300, CV_8UC1, cv::Scalar(20));
    sceneWith(patternA, 1.0, 75, 90).copyTo(multiScene);
    cv::Mat bScene = sceneWith(patternB, 1.0, 220, 90);
    cv::max(multiScene, bScene, multiScene);
    std::vector<T_T::MatchResult> multiResults;
    std::vector<T_T::Template::Ptr> models{modelA, modelB};
    if (!matcher.searchTemplate(multiScene, cv::Mat(), models, 0, 0, 0.55f, 10,
                                0.4f, 0, 0.8f, true, T_T::ScaleSearchCfg(), multiResults)) return 7;
    if (!nearResult(multiResults, 101, 1.0, 75, 90, 8.0) ||
        !nearResult(multiResults, 202, 1.0, 220, 90, 8.0)) return 8;

    if (matcher.searchTemplate(multiScene, cv::Mat(),
                               std::vector<T_T::Template::Ptr>{modelA, nullptr},
                               0, 0, 0.55f, 10, 0.4f, 0, 0.8f, true,
                               T_T::ScaleSearchCfg(), multiResults)) return 10;
    if (matcher.searchTemplate(multiScene, cv::Mat(),
                               std::vector<T_T::Template::Ptr>{modelA, modelA},
                               0, 0, 0.55f, 10, 0.4f, 0, 0.8f, true,
                               T_T::ScaleSearchCfg(), multiResults)) return 11;

    cv::Mat color;
    cv::cvtColor(partial, color, cv::COLOR_GRAY2BGR);
    matcher.drawMatchResults(color, partialResults, models);
    if (color.empty() || color.size() != partial.size()) return 9;
    return 0;
}
