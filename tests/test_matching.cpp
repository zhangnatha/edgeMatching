#include <mutex>
#include "Type.h"
#include "ROI.h"
#include <opencv2/opencv.hpp>
#define private public
#include "FindTemplateV1.h"
#undef private
#include "MakeTemplateV1.h"
#include "ModelIdNormalization.h"
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

T_T::Template::Ptr train(const cv::Mat& image, int id,
                         T_T::EdgeMethod method = T_T::EDGE_CURRENT) {
    auto model = std::make_shared<T_T::Template>();
    model->template_cfg.id = id;
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));
    SM_V1::CreateTemplate creator;
    if (!creator.createTemplate(image, mask, 0, 0, 0, 1.0, false, 15, 60, model, method)) return nullptr;
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

bool sameResults(std::vector<T_T::MatchResult> a, std::vector<T_T::MatchResult> b) {
    const auto order = [](const T_T::MatchResult& lhs, const T_T::MatchResult& rhs) {
        if (lhs.template_id != rhs.template_id) return lhs.template_id < rhs.template_id;
        if (lhs.pose.y != rhs.pose.y) return lhs.pose.y < rhs.pose.y;
        if (lhs.pose.x != rhs.pose.x) return lhs.pose.x < rhs.pose.x;
        return lhs.score > rhs.score;
    };
    std::sort(a.begin(), a.end(), order);
    std::sort(b.begin(), b.end(), order);
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].template_id != b[i].template_id ||
            std::abs(a[i].pose.x - b[i].pose.x) > 1e-6 ||
            std::abs(a[i].pose.y - b[i].pose.y) > 1e-6 ||
            std::abs(a[i].pose.angle - b[i].pose.angle) > 1e-6 ||
            std::abs(a[i].scale - b[i].scale) > 1e-9 ||
            std::abs(a[i].score - b[i].score) > 1e-5 ||
            std::abs(a[i].visible_ratio - b[i].visible_ratio) > 1e-6 ||
            std::abs(a[i].matched_ratio - b[i].matched_ratio) > 1e-6) return false;
    }
    return true;
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
    auto modelDevernay = train(patternA, 203, T_T::EDGE_DEVERNAY);
    auto modelPixel = train(patternA, 204, T_T::EDGE_CANNY_PIXEL);
    if (!modelA || !modelB || !modelDevernay || !modelPixel) return 1;
    if (modelA->template_cfg.id != 101 || modelB->template_cfg.id != 202) return 2;
    if (!hasFractionalFeature(modelA) || !hasFractionalFeature(modelB)) return 14;
    if (modelA->template_cfg.edge_method != T_T::EDGE_CURRENT ||
        modelDevernay->template_cfg.edge_method != T_T::EDGE_DEVERNAY ||
        !hasFractionalFeature(modelDevernay)) return 15;
    if (modelPixel->template_cfg.edge_method != T_T::EDGE_CANNY_PIXEL ||
        hasFractionalFeature(modelPixel)) return 16;
    const std::string pixelModelPath = "/tmp/edge_matching_pixel.json";
    SM_V1::CreateTemplate pixelSaver;
    if (!pixelSaver.saveModelFile2Json(modelPixel, pixelModelPath)) return 50;
    SM_V1::SearchTemplate pixelLoader;
    const T_T::Template::Ptr loadedPixel = pixelLoader.loadModelFileFromJson(pixelModelPath);
    if (!loadedPixel || loadedPixel->template_cfg.edge_method != T_T::EDGE_CANNY_PIXEL) return 51;
    if (modelA->templates.size() != 1 || modelA->templates[0]->shape_angle.size() != 1) return 2;

    // Exercise the hybrid NMS policy directly.  The synthetic model has two
    // sparse support points inside a large box: at a 30 px displacement the
    // boxes overlap by 70%, but the support bands are disjoint.
    auto nmsModel = std::make_shared<T_T::Template>();
    nmsModel->template_cfg.id = 303;
    nmsModel->template_cfg.image_width = 100;
    nmsModel->template_cfg.image_height = 100;
    auto nmsShapeInfo = std::make_shared<T_T::ShapeInfo>();
    auto nmsShapeAngle = std::make_shared<T_T::ShapeAngle>();
    nmsShapeAngle->angle = 0.0;
    nmsShapeAngle->shape_point.push_back(T_T::ShapePoint{-45.0, 0.0, 1.0f, 0.0f});
    nmsShapeAngle->shape_point.push_back(T_T::ShapePoint{45.0, 0.0, 1.0f, 0.0f});
    nmsShapeInfo->shape_angle.push_back(nmsShapeAngle);
    nmsModel->templates.push_back(nmsShapeInfo);
    SM_V1::SearchTemplate nmsMatcher;
    const auto nmsResult = [](double x, double score) {
        return T_T::MatchResult(T_T::Pose2d(x, 90.0, 0.0), score, 1.0, 303);
    };
    const std::vector<T_T::MatchResult> highSupportDistinct{
        nmsResult(100.0, 0.95), nmsResult(130.0, 0.94)};
    if (nmsMatcher._filterMaxOverLapCandidates(highSupportDistinct, 0.4f,
                                               nmsModel, true).size() != 2) return 34;
    const std::vector<T_T::MatchResult> sameSupportDuplicate{
        nmsResult(100.0, 0.95), nmsResult(100.0, 0.94)};
    if (nmsMatcher._filterMaxOverLapCandidates(sameSupportDuplicate, 0.4f,
                                               nmsModel, true).size() != 1) return 35;
    const std::vector<T_T::MatchResult> lowBoxDuplicate{
        nmsResult(100.0, 0.80), nmsResult(130.0, 0.79)};
    if (nmsMatcher._filterMaxOverLapCandidates(lowBoxDuplicate, 0.4f,
                                               nmsModel, true).size() != 1) return 36;
    const std::vector<T_T::MatchResult> highNearCenterDuplicate{
        nmsResult(100.0, 0.95),
        T_T::MatchResult(T_T::Pose2d(103.0, 90.0, 180.0), 0.94, 1.0, 303)};
    if (nmsMatcher._filterMaxOverLapCandidates(highNearCenterDuplicate, 0.4f,
                                               nmsModel, true).size() != 1) return 39;
    const std::vector<T_T::MatchResult> coarseAngleModes{
        nmsResult(100.0, 0.95),
        T_T::MatchResult(T_T::Pose2d(101.0, 91.0, 5.0), 0.94, 1.0, 303),
        T_T::MatchResult(T_T::Pose2d(100.0, 90.0, 180.0), 0.93, 1.0, 303),
        T_T::MatchResult(T_T::Pose2d(101.0, 91.0, 175.0), 0.92, 1.0, 303)};
    const auto retainedAngleModes = nmsMatcher._filterNearCandidates(coarseAngleModes);
    if (retainedAngleModes.size() != 2 ||
        std::abs(retainedAngleModes[0].pose.angle) > 1e-9 ||
        std::abs(retainedAngleModes[1].pose.angle - 180.0) > 1e-9) return 40;

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
    cv::Mat greenPixels = (visualizationChannels[1] > 240) &
                          (visualizationChannels[2] < 40) &
                          (visualizationChannels[0] < 40);
    if (cv::countNonZero(greenPixels) == 0) return 19;
    std::vector<cv::Size> pyramidSizes(1, patternA.size());
    for (int level = 1; level <= pyramidModel->template_cfg.num_levels; ++level)
        pyramidSizes.push_back(cv::Size(pyramidSizes.back().width / 2,
                                        pyramidSizes.back().height / 2));
    int tileX = 18, tileY = 18;
    for (int level = pyramidModel->template_cfg.num_levels; level >= 0; --level)
    {
        const cv::Rect tile(tileX, tileY, pyramidSizes[level].width,
                            pyramidSizes[level].height);
        if ((tile & cv::Rect(0, 0, greenPixels.cols, greenPixels.rows)) != tile ||
            cv::countNonZero(greenPixels(tile)) == 0) return 31;
        tileX += pyramidSizes[level].width + 24;
        tileY += pyramidSizes[level].height + 24;
    }

    SM_V1::SearchTemplate matcher;
    // The optional AVX2 extractor must remain numerically equivalent to the
    // portable implementation.  In a scalar build both calls intentionally
    // select the same fallback path.
    std::vector<float> scalarGradX(patternA.total()), scalarGradY(patternA.total()), scalarMagnitude;
    std::vector<float> simdGradX(patternA.total()), simdGradY(patternA.total()), simdMagnitude;
    matcher._getFeature(patternA, defaultMask, patternA.cols, patternA.rows,
                        scalarGradX, scalarGradY, scalarMagnitude, false);
    matcher._getFeature(patternA, defaultMask, patternA.cols, patternA.rows,
                        simdGradX, simdGradY, simdMagnitude, true);
    if (scalarGradX.size() != simdGradX.size() || scalarGradY.size() != simdGradY.size() ||
        scalarMagnitude.size() != simdMagnitude.size()) return 42;
    for (size_t i = 0; i < scalarGradX.size(); ++i)
    {
        if (std::abs(scalarGradX[i] - simdGradX[i]) > 2e-5f ||
            std::abs(scalarGradY[i] - simdGradY[i]) > 2e-5f ||
            std::abs(scalarMagnitude[i] - simdMagnitude[i]) > 2e-5f) return 43;
    }
    std::vector<T_T::MatchResult> requestedSimdResults;
    if (!matcher.searchTemplate(patternA, cv::Mat(), modelA, 0, 0, 0.30f, 1,
                                0.4f, 0, 0.8f, true,
                                T_T::ScaleSearchCfg(1.0, 1.0, 1.0, 1.0, false, 0,
                                                    I_I::USE_POLARITY, true),
                                requestedSimdResults) || requestedSimdResults.empty()) return 44;
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

    // Subpixel-only scale, position, and angle assertions are compiled in and
    // exercised only by the feature-enabled build.  The disabled build still
    // verifies that requesting the API flag has deterministic legacy behavior.
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

    // Different templates whose rectangular boxes overlap substantially must
    // survive when their feature support bands are distinct.  The two boxes
    // here overlap by half their width, while the two contour layouts remain
    // separable enough for support IoU at max_overlap=0.4.
    cv::Mat overlappingScene(180, 300, CV_8UC1, cv::Scalar(20));
    cv::max(overlappingScene, sceneWith(patternA, 1.0, 110, 90), overlappingScene);
    cv::max(overlappingScene, sceneWith(patternB, 1.0, 150, 90), overlappingScene);
    std::vector<T_T::MatchResult> overlappingResults;
    if (!matcher.searchTemplate(overlappingScene, cv::Mat(), models, 0, 0, 0.55f, 10,
                                0.4f, 0, 0.8f, true, T_T::ScaleSearchCfg(),
                                overlappingResults) ||
        !nearResult(overlappingResults, 101, 1.0, 110, 90, 8.0) ||
        !nearResult(overlappingResults, 202, 1.0, 150, 90, 8.0)) return 33;

    // A near-center cross-template duplicate is suppressed even when the
    // templates' sparse supports do not produce a high IoU.  This exception
    // is intentionally limited to very close centers; distinct overlapping
    // objects above must remain independently detectable.
    cv::Mat nearCenterScene(180, 300, CV_8UC1, cv::Scalar(20));
    cv::max(nearCenterScene, sceneWith(patternA, 1.0, 110, 90), nearCenterScene);
    cv::max(nearCenterScene, sceneWith(patternB, 1.0, 115, 90), nearCenterScene);
    std::vector<T_T::MatchResult> nearCenterResults;
    if (!matcher.searchTemplate(nearCenterScene, cv::Mat(), models, 0, 0, 0.55f, 10,
                                0.4f, 0, 0.8f, true, T_T::ScaleSearchCfg(),
                                nearCenterResults)) return 37;
    if (nearResult(nearCenterResults, 101, 1.0, 110, 90, 8.0) &&
        nearResult(nearCenterResults, 202, 1.0, 115, 90, 8.0)) return 38;

    // The batch path shares ROI/scale preparation but must remain equivalent to
    // independent per-model searches, including multi-scale result metadata.
    const T_T::ScaleSearchCfg batchCfg(0.9, 1.1, 0.1, 1.0);
    std::vector<T_T::MatchResult> batchResults, sequentialResults;
    if (!matcher.searchTemplate(multiScene, cv::Mat(), models, 0, 0, 0.55f, -1,
                                0.4f, 0, 0.8f, true, batchCfg, batchResults)) return 24;
    for (const auto& model : models) {
        std::vector<T_T::MatchResult> current;
        if (!matcher.searchTemplate(multiScene, cv::Mat(), model, 0, 0, 0.55f, -1,
                                    0.4f, 0, 0.8f, true, batchCfg, current)) return 25;
        sequentialResults.insert(sequentialResults.end(), current.begin(), current.end());
    }
    if (batchResults.size() < 2 ||
        !nearResult(batchResults, 101, 1.0, 75, 90, 8.0) ||
        !nearResult(batchResults, 202, 1.0, 220, 90, 8.0)) return 26;
    for (const auto& result : batchResults) {
        if (!std::isfinite(result.pose.x) || !std::isfinite(result.pose.y) ||
            result.pose.x < 0.0 || result.pose.x >= multiScene.cols ||
            result.pose.y < 0.0 || result.pose.y >= multiScene.rows) return 32;
    }

    const size_t resultCountBeforeInvalidCall = multiResults.size();
    if (matcher.searchTemplate(multiScene, cv::Mat(),
                               std::vector<T_T::Template::Ptr>{modelA, nullptr},
                               0, 0, 0.55f, 10, 0.4f, 0, 0.8f, true,
                               T_T::ScaleSearchCfg(), multiResults)) return 10;
    if (matcher.searchTemplate(multiScene, cv::Mat(),
                               std::vector<T_T::Template::Ptr>{modelA, modelA},
                               0, 0, 0.55f, 10, 0.4f, 0, 0.8f, true,
                               T_T::ScaleSearchCfg(), multiResults)) return 11;
    if (multiResults.size() != resultCountBeforeInvalidCall) return 27;

    cv::Mat color;
    cv::cvtColor(partial, color, cv::COLOR_GRAY2BGR);
    matcher.drawMatchResults(color, partialResults, models);
    if (color.empty() || color.size() != partial.size()) return 9;

    // UI loads colour files for display but must search the same grayscale
    // pixels as the CLI.  The core entry point must therefore be invariant to
    // an equivalent BGR wrapper.
    cv::Mat multiSceneColor;
    cv::cvtColor(multiScene, multiSceneColor, cv::COLOR_GRAY2BGR);
    std::vector<T_T::MatchResult> grayInputResults, colorInputResults;
    if (!matcher.searchTemplate(multiScene, cv::Mat(), models, 0, 0, 0.55f, 10,
                                0.4f, 0, 0.8f, true, T_T::ScaleSearchCfg(), grayInputResults) ||
        !matcher.searchTemplate(multiSceneColor, cv::Mat(), models, 0, 0, 0.55f, 10,
                                0.4f, 0, 0.8f, true, T_T::ScaleSearchCfg(), colorInputResults) ||
        !sameResults(grayInputResults, colorInputResults)) return 39;

    // Duplicate/invalid IDs are normalized deterministically on the loaded
    // in-memory copies, matching the UI/CLI policy without touching files.
    auto duplicateA = std::make_shared<T_T::Template>(*modelA);
    auto duplicateB = std::make_shared<T_T::Template>(*modelB);
    duplicateA->template_cfg.id = 501;
    duplicateB->template_cfg.id = 501;
    std::vector<T_T::Template::Ptr> duplicateModels{duplicateA, duplicateB};
    std::vector<SM_V1::ModelIdAssignment> assignments;
    std::string normalizationError;
    if (!SM_V1::normalizeTemplateIds(duplicateModels, assignments, &normalizationError) ||
        assignments.size() != 2 || assignments[0].original_id != 501 ||
        assignments[0].runtime_id != 501 || assignments[1].original_id != 501 ||
        assignments[1].runtime_id != 1 || duplicateModels[0]->template_cfg.id != 501 ||
        duplicateModels[1]->template_cfg.id != 1) return 40;
    std::vector<T_T::MatchResult> normalizedResults;
    if (!matcher.searchTemplate(multiScene, cv::Mat(), duplicateModels, 0, 0, 0.55f, 10,
                                0.4f, 0, 0.8f, true, T_T::ScaleSearchCfg(), normalizedResults) ||
        !nearResult(normalizedResults, 501, 1.0, 75, 90, 8.0) ||
        !nearResult(normalizedResults, 1, 1.0, 220, 90, 8.0)) return 41;
    return 0;
}
