#include "FindTemplateV1.h"
#include "MakeTemplateV1.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
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

bool hasFractionalFeature(const T_T::Template::Ptr& model) {
    if (!model || model->templates.empty() || model->templates[0]->shape_angle.empty()) return false;
    for (const auto& point : model->templates[0]->shape_angle[0]->shape_point) {
        if (std::abs(point.x - std::round(point.x)) > 1e-4 ||
            std::abs(point.y - std::round(point.y)) > 1e-4) return true;
    }
    return false;
}

double rotationError(double measured, double expectedMagnitude) {
    return std::min(std::abs(measured - expectedMagnitude),
                    std::abs(measured + expectedMagnitude));
}

bool gradientValid(float magnitude, int minContrast) {
    return magnitude > 1e-6f && magnitude >= static_cast<float>(minContrast);
}
}

int main() {
    // Exact threshold boundary shared by scalar, SIMD and subpixel scorers.
    if (gradientValid(0.0f, 0) ||
        !gradientValid(1.0f, 0) || !gradientValid(1.0f, 1) ||
        !gradientValid(5.0f, 0) || !gradientValid(5.0f, 1) ||
        !gradientValid(5.0f, 5) || gradientValid(5.0f, 6)) return 30;
    const cv::Mat patternA = makePattern(false);
    const cv::Mat patternB = makePattern(true);
    auto modelA = train(patternA, 101);
    auto modelB = train(patternB, 202);
    if (!modelA || !modelB) return 1;
    if (modelA->template_cfg.id != 101 || modelB->template_cfg.id != 202) return 2;
    if (!hasFractionalFeature(modelA) || !hasFractionalFeature(modelB)) return 14;
    if (modelA->templates.size() != 1 || modelA->templates[0]->shape_angle.size() != 1) return 2;

    auto defaultModel = std::make_shared<T_T::Template>();
    cv::Mat defaultMask(patternA.size(), CV_8UC1, cv::Scalar(255));
    SM_V1::CreateTemplate creator;
    if (defaultModel->template_cfg.id != 1 ||
        !creator.createTemplate(patternA, defaultMask, 0, 0, 0, 1.0, false, 15, 60, defaultModel) ||
        defaultModel->template_cfg.id != 1 || !defaultModel->is_inited) return 2;

    auto pyramidModel = std::make_shared<T_T::Template>();
    if (!creator.createTemplate(patternA, defaultMask, 2, 0, 0, 1.0, false, 15, 60,
                                pyramidModel)) return 17;
    cv::Mat pyramidVisualization;
    if (!creator.drawPyramidFeatures(patternA, pyramidModel, pyramidVisualization) ||
        pyramidVisualization.type() != CV_8UC3 ||
        pyramidVisualization.cols <= patternA.cols ||
        pyramidVisualization.rows <= patternA.rows) return 18;
    std::vector<cv::Mat> visualizationChannels;
    cv::split(pyramidVisualization, visualizationChannels);
    cv::Mat yellowPixels = (visualizationChannels[1] > 240) &
                           (visualizationChannels[2] > 240) &
                           (visualizationChannels[0] < 40);
    if (cv::countNonZero(yellowPixels) == 0) return 19;

    SM_V1::SearchTemplate matcher;
    // The original five-argument constructor remains source-compatible.
    const T_T::ScaleSearchCfg legacyCfg(1.0, 1.0, 1.0, 1.0, false);
    const cv::Mat compatibilityScene = sceneWith(patternA, 1.0, 145, 90);
    std::vector<T_T::MatchResult> legacyResults, explicitResults;
    if (!matcher.searchTemplate(compatibilityScene, cv::Mat(), modelA, 0, 0, 0.55f, 5,
                                0.4f, 0, 0.8f, true, legacyCfg, legacyResults) ||
        !matcher.searchTemplate(compatibilityScene, cv::Mat(), modelA, 0, 0, 0.55f, 5,
                                0.4f, 0, 0.8f, true,
                                T_T::ScaleSearchCfg(1.0, 1.0, 1.0, 1.0, false, 0,
                                                    I_I::USE_POLARITY), explicitResults) ||
        legacyResults.size() != explicitResults.size()) return 20;
    std::vector<T_T::MatchResult> contrastResults;
    if (!matcher.searchTemplate(compatibilityScene, cv::Mat(), modelA, 0, 0, 0.20f, 5,
                                0.4f, 0, 0.8f, true,
                                T_T::ScaleSearchCfg(1, 1, 1, 1, false, 100,
                                                    I_I::USE_POLARITY), contrastResults) ||
        contrastResults.empty()) return 25;
    const auto contrastBest = std::max_element(contrastResults.begin(), contrastResults.end(),
        [](const T_T::MatchResult& a, const T_T::MatchResult& b) { return a.score < b.score; });
    const auto legacyBest = std::max_element(legacyResults.begin(), legacyResults.end(),
        [](const T_T::MatchResult& a, const T_T::MatchResult& b) { return a.score < b.score; });
    if (contrastBest->matched_ratio >= 0.999 ||
        std::abs(contrastBest->visible_ratio - legacyBest->visible_ratio) > 1e-6 ||
        contrastBest->score >= legacyBest->score) return 26;
    for (size_t i = 0; i < legacyResults.size(); ++i)
        if (std::abs(legacyResults[i].pose.x - explicitResults[i].pose.x) > 1e-6 ||
            std::abs(legacyResults[i].pose.y - explicitResults[i].pose.y) > 1e-6 ||
            std::abs(legacyResults[i].pose.angle - explicitResults[i].pose.angle) > 1e-6 ||
            std::abs(legacyResults[i].score - explicitResults[i].score) > 1e-6) return 21;
    cv::Mat invertedScene;
    cv::bitwise_not(compatibilityScene, invertedScene);
    std::vector<T_T::MatchResult> useInverted, globalInverted, localInverted;
    matcher.searchTemplate(invertedScene, cv::Mat(), modelA, 0, 0, 0.8f, 5,
                           0.4f, 0, 0.8f, true,
                           T_T::ScaleSearchCfg(1, 1, 1, 1, false, 0,
                                               I_I::USE_POLARITY), useInverted);
    if (!matcher.searchTemplate(invertedScene, cv::Mat(), modelA, 0, 0, 0.8f, 5,
                                0.4f, 0, 0.8f, true,
                                T_T::ScaleSearchCfg(1, 1, 1, 1, false, 0,
                                                    I_I::IGNORE_GLOBAL_POLARITY), globalInverted) ||
        !matcher.searchTemplate(invertedScene, cv::Mat(), modelA, 0, 0, 0.8f, 5,
                                0.4f, 0, 0.8f, true,
                                T_T::ScaleSearchCfg(1, 1, 1, 1, false, 0,
                                                    I_I::IGNORE_LOCAL_POLARITY), localInverted) ||
        nearResult(useInverted, 101, 1.0, 145, 90, 2.0) ||
        !nearResult(globalInverted, 101, 1.0, 145, 90, 2.0) ||
        !nearResult(localInverted, 101, 1.0, 145, 90, 2.0)) return 24;

    cv::Mat partlyInverted = compatibilityScene.clone();
    cv::Mat leftObjectHalf = partlyInverted(cv::Rect(105, 58, 40, 64));
    cv::bitwise_not(leftObjectHalf, leftObjectHalf);
    std::vector<T_T::MatchResult> globalPartialPolarity, localPartialPolarity;
    matcher.searchTemplate(partlyInverted, cv::Mat(), modelA, 0, 0, 0.75f, 5,
                           0.4f, 0, 0.8f, true,
                           T_T::ScaleSearchCfg(1, 1, 1, 1, false, 0,
                                               I_I::IGNORE_GLOBAL_POLARITY),
                           globalPartialPolarity);
    if (!matcher.searchTemplate(partlyInverted, cv::Mat(), modelA, 0, 0, 0.75f, 5,
                                0.4f, 0, 0.8f, true,
                                T_T::ScaleSearchCfg(1, 1, 1, 1, false, 0,
                                                    I_I::IGNORE_LOCAL_POLARITY),
                                localPartialPolarity) ||
        nearResult(globalPartialPolarity, 101, 1.0, 145, 90, 2.0) ||
        !nearResult(localPartialPolarity, 101, 1.0, 145, 90, 2.0)) return 30;
    std::vector<T_T::MatchResult> invalidResults;
    if (matcher.searchTemplate(compatibilityScene, cv::Mat(), modelA, 0, 0, 0.55f, 5,
                               0.4f, 0, 0.8f, true,
                               T_T::ScaleSearchCfg(1, 1, 1, 1, false, -1), invalidResults) ||
        ([&]() {
            T_T::ScaleSearchCfg invalidMetric;
            invalidMetric.metric = 99;
            return matcher.searchTemplate(compatibilityScene, cv::Mat(), modelA, 0, 0,
                                          0.55f, 5, 0.4f, 0, 0.8f, true,
                                          invalidMetric, invalidResults);
        })()) return 22;
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

    const double targetScale = 1.03;
    const cv::Mat scaledScene = sceneWith(patternA, targetScale, 145, 90);
    std::vector<T_T::MatchResult> discreteScaleResults, refinedScaleResults;
    if (!matcher.searchTemplate(scaledScene, cv::Mat(), modelA, 0, 0, 0.30f, 1,
                                0.4f, 0, 0.8f, true,
                                T_T::ScaleSearchCfg(0.95, 1.15, 0.05, 1.0),
                                discreteScaleResults) ||
        !matcher.searchTemplate(scaledScene, cv::Mat(), modelA, 0, 0, 0.30f, 1,
                                0.4f, 0, 0.8f, true,
                                T_T::ScaleSearchCfg(0.95, 1.15, 0.05, 1.0, true),
                                refinedScaleResults) ||
        discreteScaleResults.empty() || refinedScaleResults.empty()) return 31;
    const double discreteScaleError = std::abs(discreteScaleResults[0].scale - targetScale);
    const double refinedScaleError = std::abs(refinedScaleResults[0].scale - targetScale);
    if (refinedScaleError + 0.005 >= discreteScaleError ||
        std::abs(refinedScaleResults[0].scale * 20.0 -
                 std::round(refinedScaleResults[0].scale * 20.0)) < 0.01) return 32;

    cv::Mat subpixelScene;
    const cv::Mat subpixelTransform = (cv::Mat_<double>(2, 3) <<
        1.0, 0.0, 0.35, 0.0, 1.0, -0.40);
    cv::warpAffine(sceneWith(patternA, 1.0, 145, 90), subpixelScene,
                   subpixelTransform,
                   cv::Size(300, 180), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(20));
    std::vector<T_T::MatchResult> subpixelResults;
    if (!matcher.searchTemplate(subpixelScene, cv::Mat(), modelA, 0, 0, 0.55f, 5,
                                0.4f, 0, 0.8f, true,
                                T_T::ScaleSearchCfg(1.0, 1.0, 1.0, 1.0, true),
                                subpixelResults) || subpixelResults.empty()) return 15;
    const auto& subpixelBest = *std::max_element(subpixelResults.begin(), subpixelResults.end(),
        [](const T_T::MatchResult& lhs, const T_T::MatchResult& rhs) { return lhs.score < rhs.score; });
    if (std::hypot(subpixelBest.pose.x - 145.35, subpixelBest.pose.y - 89.60) > 0.8 ||
        (std::abs(subpixelBest.pose.x - std::round(subpixelBest.pose.x)) < 0.05 &&
         std::abs(subpixelBest.pose.y - std::round(subpixelBest.pose.y)) < 0.05)) return 16;

    // Interpolation refines orientation as well as position. Load the model once
    // to expand its canonical features into the configured discrete angle cache.
    auto angleModel = std::make_shared<T_T::Template>();
    angleModel->template_cfg.id = 303;
    if (!creator.createTemplate(patternA, defaultMask, 0, -20, 20, 1.0, false,
                                15, 60, angleModel)) return 27;
    const std::string angleModelPath = cv::tempfile(".json");
    if (!creator.saveModelFile2Json(angleModel, angleModelPath)) return 27;
    auto loadedAngleModel = matcher.loadModelFileFromJson(angleModelPath);
    std::remove(angleModelPath.c_str());
    if (!loadedAngleModel) return 27;
    const double targetAngle = 12.35;
    cv::Mat rotatedPattern;
    cv::warpAffine(patternA, rotatedPattern,
                   cv::getRotationMatrix2D(cv::Point2f(patternA.cols * 0.5f,
                                                       patternA.rows * 0.5f),
                                           targetAngle, 1.0),
                   patternA.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar(20));
    const cv::Mat angleScene = sceneWith(rotatedPattern, 1.0, 145, 90);
    std::vector<T_T::MatchResult> discreteAngleResults, refinedAngleResults;
    if (!matcher.searchTemplate(angleScene, cv::Mat(), loadedAngleModel, -20, 20,
                                0.55f, 1, 0.4f, 0, 0.8f, true,
                                T_T::ScaleSearchCfg(), discreteAngleResults) ||
        !matcher.searchTemplate(angleScene, cv::Mat(), loadedAngleModel, -20, 20,
                                0.55f, 1, 0.4f, 0, 0.8f, true,
                                T_T::ScaleSearchCfg(1, 1, 1, 1, true), refinedAngleResults) ||
        discreteAngleResults.empty() || refinedAngleResults.empty()) return 28;
    const double discreteError = rotationError(discreteAngleResults[0].pose.angle, targetAngle);
    const double refinedError = rotationError(refinedAngleResults[0].pose.angle, targetAngle);
    if (discreteError > 1.5 || refinedError > 0.9 ||
        refinedError > discreteError * 0.70 ||
        std::abs(refinedAngleResults[0].pose.angle -
                 std::round(refinedAngleResults[0].pose.angle)) < 0.02 ||
        refinedAngleResults[0].score + 1e-6 < discreteAngleResults[0].score) return 29;

    const cv::Mat partial = sceneWith(patternA, 1.0, 285, 90);
    std::vector<T_T::MatchResult> partialResults;
    if (!matcher.searchTemplate(partial, cv::Mat(), modelA, 0, 0, 0.50f, 5,
                                0.4f, 0, 0.8f, true,
                                T_T::ScaleSearchCfg(1.0, 1.0, 1.0, 0.55), partialResults)) return 5;
    if (!nearResult(partialResults, 101, 1.0, 285, 90, 10.0)) return 6;
    for (const auto& result : partialResults)
        if (result.visible_ratio < 0.0 || result.visible_ratio > 1.0 ||
            result.matched_ratio < 0.0 || result.matched_ratio > 1.0) return 23;

    // The center of a legitimate border target may lie outside the image.
    const cv::Mat outsideCenter = sceneWith(patternA, 1.0, 301, 90);
    std::vector<T_T::MatchResult> outsideCenterResults;
    if (!matcher.searchTemplate(outsideCenter, cv::Mat(), modelA, 0, 0, 0.45f, 5,
                                0.4f, 0, 0.8f, true,
                                T_T::ScaleSearchCfg(1.0, 1.0, 1.0, 0.40),
                                outsideCenterResults)) return 12;
    if (!nearResult(outsideCenterResults, 101, 1.0, 301, 90, 12.0)) return 13;

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
