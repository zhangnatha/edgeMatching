#include "MakeTemplateV1.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace
{
cv::Mat makeSlantedEdge()
{
    const int width = 128;
    const int height = 128;
    const double slope = 0.37;
    const double intercept = 55.35;
    const double normalizer = std::sqrt(1.0 + slope * slope);
    cv::Mat image(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const double distance = (x + slope * y - intercept) / normalizer;
            const double intensity = 20.0 + 220.0 / (1.0 + std::exp(-distance / 1.35));
            image.at<unsigned char>(y, x) = static_cast<unsigned char>(
                std::max(0.0, std::min(255.0, intensity)));
        }
    return image;
}

cv::Mat makeCircleEdge()
{
    const int width = 128;
    const int height = 128;
    const double centerX = 63.35;
    const double centerY = 61.70;
    const double radius = 28.40;
    cv::Mat image(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const double distance = std::hypot(x - centerX, y - centerY) - radius;
            const double intensity = 20.0 + 220.0 / (1.0 + std::exp(-distance / 1.35));
            image.at<unsigned char>(y, x) = static_cast<unsigned char>(
                std::max(0.0, std::min(255.0, intensity)));
        }
    return image;
}
}

int main()
{
    const cv::Mat image = makeSlantedEdge();
    const cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));
    auto model = std::make_shared<T_T::Template>();
    SM_V1::CreateTemplate creator;
    if (!creator.createTemplate(image, mask, 0, 0, 0, 1.0, false, 5, 20, model,
                                T_T::EDGE_DEVERNAY) ||
        !model || model->templates.empty() || !model->templates[0] ||
        model->templates[0]->shape_angle.empty()) return 1;

    const auto& points = model->templates[0]->shape_angle[0]->shape_point;
    if (points.size() < 20) return 2;

    const double slope = 0.37;
    const double intercept = 55.35;
    const double normalizer = std::sqrt(1.0 + slope * slope);
    std::vector<double> localizedErrors;
    std::vector<double> integerErrors;
    localizedErrors.reserve(points.size());
    integerErrors.reserve(points.size());
    for (const auto& point : points)
    {
        const double x = point.x + image.cols * 0.5;
        const double y = point.y + image.rows * 0.5;
        const double nx = point.edge_dx;
        const double ny = point.edge_dy;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(nx) ||
            !std::isfinite(ny)) return 3;
        const double norm = std::hypot(nx, ny);
        if (std::abs(norm - 1.0) > 1e-3) return 4;

        // Recover the integer support pixel used by the bounded quadratic
        // fit.  The perpendicular residual guards against choosing a nearby
        // integer pixel on a slanted edge.
        double bestDistance = std::numeric_limits<double>::infinity();
        double bestOffset = 0.0;
        for (int iy = static_cast<int>(std::floor(y)) - 1;
             iy <= static_cast<int>(std::ceil(y)) + 1; ++iy)
            for (int ix = static_cast<int>(std::floor(x)) - 1;
                 ix <= static_cast<int>(std::ceil(x)) + 1; ++ix)
            {
                const double dx = x - ix;
                const double dy = y - iy;
                const double tangentResidual = std::abs(-ny * dx + nx * dy);
                const double distance = tangentResidual * tangentResidual + dx * dx + dy * dy;
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestOffset = nx * dx + ny * dy;
                }
            }
        if (!std::isfinite(bestOffset) || std::abs(bestOffset) > 0.5 + 1e-4) return 5;

        // Ignore the image margins where the finite-difference support is
        // clipped, then compare against the analytic edge locus.
        if (y < 12.0 || y > image.rows - 13.0) continue;
        const double error = std::abs(x + slope * y - intercept) / normalizer;
        if (error < 1.0)
        {
            localizedErrors.push_back(error);
            const double integerX = std::round(x);
            const double integerY = std::round(y);
            integerErrors.push_back(std::abs(integerX + slope * integerY - intercept) /
                                    normalizer);
        }
    }
    if (localizedErrors.size() < 20 || localizedErrors.size() != integerErrors.size()) return 6;
    std::sort(localizedErrors.begin(), localizedErrors.end());
    std::sort(integerErrors.begin(), integerErrors.end());
    const double localizedMedian = localizedErrors[localizedErrors.size() / 2];
    const double integerMedian = integerErrors[integerErrors.size() / 2];
    if (!(localizedMedian < integerMedian - 0.01))
    {
        std::cerr << "Devernay median error " << localizedMedian
                  << " is not below integer baseline " << integerMedian << std::endl;
        return 7;
    }

    const cv::Mat circle = makeCircleEdge();
    const cv::Mat circleMask(circle.size(), CV_8UC1, cv::Scalar(255));
    auto circleModel = std::make_shared<T_T::Template>();
    if (!creator.createTemplate(circle, circleMask, 0, 0, 0, 1.0, false, 5, 20,
                                circleModel, T_T::EDGE_DEVERNAY) || !circleModel || circleModel->templates.empty() ||
        !circleModel->templates[0] || circleModel->templates[0]->shape_angle.empty()) return 8;
    const auto& circlePoints = circleModel->templates[0]->shape_angle[0]->shape_point;
    if (circlePoints.size() < 30) return 9;
    std::vector<double> circleLocalized;
    std::vector<double> circleInteger;
    for (const auto& point : circlePoints)
    {
        const double x = point.x + circle.cols * 0.5;
        const double y = point.y + circle.rows * 0.5;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(point.edge_dx) ||
            !std::isfinite(point.edge_dy)) return 10;
        if (x < 8.0 || x > circle.cols - 9.0 || y < 8.0 || y > circle.rows - 9.0) continue;
        const double localized = std::abs(std::hypot(x - 63.35, y - 61.70) - 28.40);
        const double integerX = std::round(x);
        const double integerY = std::round(y);
        const double integer = std::abs(std::hypot(integerX - 63.35, integerY - 61.70) - 28.40);
        if (localized < 1.0)
        {
            circleLocalized.push_back(localized);
            circleInteger.push_back(integer);
        }
    }
    if (circleLocalized.size() < 30 || circleLocalized.size() != circleInteger.size()) return 11;
    std::sort(circleLocalized.begin(), circleLocalized.end());
    std::sort(circleInteger.begin(), circleInteger.end());
    if (!(circleLocalized[circleLocalized.size() / 2] <
          circleInteger[circleInteger.size() / 2] - 0.01)) return 12;

    // A reduced pyramid must retain disconnected small hole contours, not
    // only the strongest outer boundary.  This exercises the weak-component
    // hysteresis recovery used by the Devernay backend.
    cv::Mat fourHole(128, 128, CV_8UC1, cv::Scalar(20));
    cv::rectangle(fourHole, cv::Rect(16, 16, 96, 96), cv::Scalar(220), cv::FILLED);
    const std::vector<cv::Point> holeCenters{{32, 32}, {96, 32}, {96, 96}, {32, 96}};
    for (const cv::Point& center : holeCenters)
        cv::circle(fourHole, center, 8, cv::Scalar(20), cv::FILLED);
    const cv::Mat fourHoleMask(fourHole.size(), CV_8UC1, cv::Scalar(255));
    const T_T::EdgeMethod coarseMethods[] = {
        T_T::EDGE_CANNY_PIXEL, T_T::EDGE_CURRENT, T_T::EDGE_DEVERNAY};
    for (const T_T::EdgeMethod method : coarseMethods) {
        auto fourHoleModel = std::make_shared<T_T::Template>();
        if (!creator.createTemplate(fourHole, fourHoleMask, 2, 0, 0, 1.0, false, 5, 100,
                                    fourHoleModel, method) ||
            !fourHoleModel || fourHoleModel->templates.size() < 3 ||
            fourHoleModel->templates[2]->shape_angle.empty()) return 13;
        const auto& coarsePoints = fourHoleModel->templates[2]->shape_angle[0]->shape_point;
        int retainedHoles = 0;
        for (const cv::Point& center : holeCenters) {
            const cv::Point2d coarseCenter(center.x / 4.0, center.y / 4.0);
            bool found = false;
            for (const auto& point : coarsePoints) {
                const double x = point.x + 16.0;
                const double y = point.y + 16.0;
                if (std::hypot(x - coarseCenter.x, y - coarseCenter.y) <= 3.0) {
                    found = true;
                    break;
                }
            }
            if (found) ++retainedHoles;
        }
        if (retainedHoles < 4) return 14;
    }
    return 0;
}
