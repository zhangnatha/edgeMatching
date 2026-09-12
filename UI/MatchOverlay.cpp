#include "MatchOverlay.h"

#include "ContourBuilder.h"

#include <QLineF>

#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace MatchOverlay
{
namespace
{

T_T::Template::Ptr modelForResult(const T_T::MatchResult& result,
                                  const std::vector<T_T::Template::Ptr>& models)
{
    T_T::Template::Ptr fallback;
    for (const T_T::Template::Ptr& model : models) {
        if (!model) continue;
        if (!fallback) fallback = model;
        if (result.template_id == model->template_cfg.id) return model;
    }
    return result.template_id < 0 ? fallback : T_T::Template::Ptr();
}

}

QPointF transformCanonicalPoint(const QPointF& canonical,
                                const T_T::MatchResult& result)
{
    const double radians = result.pose.angle * CV_PI / 180.0;
    const double c = std::cos(radians);
    const double s = std::sin(radians);
    return QPointF(result.pose.x + result.scale * (canonical.x() * c + canonical.y() * s),
                   result.pose.y + result.scale * (-canonical.x() * s + canonical.y() * c));
}

QPolygonF rotatedFrame(const T_T::MatchResult& result,
                       const T_T::Template::Ptr& model)
{
    if (!model) return {};
    const double width = std::max(1, model->template_cfg.image_width);
    const double height = std::max(1, model->template_cfg.image_height);
    const double originX = model->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
        ? model->template_cfg.origin_x : width * 0.5;
    const double originY = model->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
        ? model->template_cfg.origin_y : height * 0.5;
    QPolygonF polygon;
    const QPointF corners[] = {
        QPointF(-originX, -originY), QPointF(width - originX, -originY),
        QPointF(width - originX, height - originY), QPointF(-originX, height - originY)
    };
    for (const QPointF& corner : corners)
        polygon << transformCanonicalPoint(corner, result);
    return polygon;
}

QVector<ImageView::OverlayPath> buildMatchOverlays(
    const std::vector<T_T::MatchResult>& results,
    const std::vector<T_T::Template::Ptr>& models)
{
    QVector<ImageView::OverlayPath> overlays;
    for (const T_T::MatchResult& result : results) {
        if (result.score <= 0.0) continue;
        const T_T::Template::Ptr model = modelForResult(result, models);
        if (!model || model->templates.empty() || !model->templates[0]) continue;

        const int width = std::max(1, model->template_cfg.image_width);
        const int height = std::max(1, model->template_cfg.image_height);
        // ContourBuilder reconstructs connected, de-duplicated paths from the
        // L0 canonical feature set.  Pixel values are irrelevant to it.
        const cv::Mat modelCanvas(height, width, CV_8UC1, cv::Scalar(0));
        const QPolygonF frame = rotatedFrame(result, model);
        if (frame.size() == 4) {
            ImageView::OverlayPath box;
            box.points = frame;
            box.closed = true;
            box.color = QColor(0, 255, 255);
            box.width = 2.0;
            box.cosmetic = true;
            overlays.push_back(box);

            const double direction = -result.pose.angle * CV_PI / 180.0;
            const double arrowLength = std::max(18.0, std::min(80.0,
                model->template_cfg.image_width * result.scale * 0.35));
            ImageView::OverlayPath arrow;
            arrow.points << QPointF(result.pose.x, result.pose.y)
                         << QPointF(result.pose.x + arrowLength * std::cos(direction),
                                    result.pose.y + arrowLength * std::sin(direction));
            arrow.color = QColor(0, 255, 255);
            arrow.width = 2.0;
            arrow.cosmetic = true;
            arrow.arrow = true;
            arrow.arrowSize = 8.0;
            overlays.push_back(arrow);
        }
    }
    return overlays;
}

QVector<ImageView::OverlayPath> buildMatchOverlays(
    const cv::Mat& image, const std::vector<T_T::MatchResult>& results,
    const std::vector<T_T::Template::Ptr>& models)
{
    return buildMatchOverlays(image, results, models, I_I::USE_POLARITY);
}

QVector<ImageView::OverlayPath> buildMatchOverlays(
    const cv::Mat& image, const std::vector<T_T::MatchResult>& results,
    const std::vector<T_T::Template::Ptr>& models, I_I::Metric metric)
{
    // Keep the established cyan frame/pose arrow, then classify the same
    // rotated ShapeAngle points and normalized gradients as the core drawer.
    QVector<ImageView::OverlayPath> overlays = buildMatchOverlays(results, models);
    cv::Mat gray, gx, gy;
    if (!image.empty()) {
        if (image.channels() == 1) gray = image;
        else if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        else if (image.channels() == 4) cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
        if (!gray.empty()) {
            cv::Sobel(gray, gx, CV_32F, 1, 0, 1);
            cv::Sobel(gray, gy, CV_32F, 0, 1, 1);
            cv::Mat magnitude;
            cv::magnitude(gx, gy, magnitude);
            cv::Mat valid = magnitude > 1e-6f;
            cv::divide(gx, magnitude, gx, 1.0, CV_32F);
            cv::divide(gy, magnitude, gy, 1.0, CV_32F);
            gx.setTo(0.0f, ~valid);
            gy.setTo(0.0f, ~valid);
        }
    }
    for (size_t ri = 0; ri < results.size(); ++ri) {
        const auto& result = results[ri];
        const T_T::Template::Ptr model = modelForResult(result, models);
        if (!model || model->templates.empty() || !model->templates[0] || model->templates[0]->shape_angle.empty()) continue;
        const auto& angles = model->templates[0]->shape_angle;
        auto selected = angles.front();
        double bestDiff = 1e9;
        for (const auto& candidate : angles) {
            const double d = std::abs(candidate->angle + result.pose.angle);
            if (d < bestDiff) { bestDiff = d; selected = candidate; }
        }
        const auto& shape = *selected;
        const int width = std::max(1, model->template_cfg.image_width);
        const int height = std::max(1, model->template_cfg.image_height);
        const cv::Mat canvas(height, width, CV_8UC1, cv::Scalar(0));
        const auto contours = ContourBuilder::buildTemplateContours(canvas, model);
        const QPointF contourOrigin(
            model->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                ? model->template_cfg.origin_x : width * 0.5,
            model->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID
                ? model->template_cfg.origin_y : height * 0.5);
        struct Sample {
            Sample(const QPointF& p, float s) : point(p), similarity(s), quality(0) {}
            QPointF point;
            float similarity;
            int quality;
        };
        std::vector<Sample> samples;
        samples.reserve(shape.shape_point.size());
        double globalSum = 0.0;
        for (const auto& point : shape.shape_point) {
            const QPointF imagePoint(result.pose.x + point.x * result.scale,
                                     result.pose.y + point.y * result.scale);
            const int x = cvRound(imagePoint.x()), y = cvRound(imagePoint.y());
            float similarity = 0.0f;
            if (!gx.empty() && x >= 0 && y >= 0 && x < gx.cols && y < gx.rows)
                similarity = gx.at<float>(y, x) * point.edge_dx +
                             gy.at<float>(y, x) * point.edge_dy;
            similarity = std::max(-1.0f, std::min(1.0f, similarity));
            globalSum += similarity;
            samples.push_back(Sample(imagePoint, similarity));
        }
        const float globalPolarity = globalSum < 0.0 ? -1.0f : 1.0f;
        for (auto& sample : samples) {
            float value = sample.similarity;
            if (metric == I_I::IGNORE_LOCAL_POLARITY) value = std::abs(value);
            else if (metric == I_I::IGNORE_GLOBAL_POLARITY) value *= globalPolarity;
            sample.quality = value >= 0.8f ? 2 : (value >= 0.4f ? 1 : 0);
        }
        for (const auto& contour : contours) {
            const int n = contour.points.size(); if (n < 2) continue;
            QVector<int> quality(n); QVector<QPointF> pts(n);
            for (int i=0;i<n;++i) {
                const QPointF canonical = contour.points[i] - contourOrigin;
                pts[i] = transformCanonicalPoint(canonical, result);
                int best = -1; double bestDistance = std::numeric_limits<double>::infinity();
                for (int k = 0; k < static_cast<int>(samples.size()); ++k) {
                    const double d = QLineF(pts[i], samples[k].point).length();
                    if (d < bestDistance) { bestDistance = d; best = k; }
                }
                quality[i] = best >= 0 ? samples[best].quality : 0;
            }
            const int lim = contour.closed ? n : n-1;
            for (int i=0;i<lim;++i) {
                const int j=(i+1)%n; ImageView::OverlayPath seg; seg.points << pts[i] << pts[j];
                const int q = std::min(quality[i], quality[j]);
                seg.color = q == 2 ? QColor(0,255,0) : (q == 1 ? QColor(255,255,0) : QColor(255,0,0));
                seg.width=1.0; seg.cosmetic=true; overlays.push_back(seg);
            }
        }
        ImageView::OverlayPath label;
        label.color = QColor(255,255,0); label.width = 1.0; label.cosmetic = true;
        label.label = QString("#%1").arg(static_cast<int>(ri)+1);
        label.labelPosition = QPointF(result.pose.x + 6.0, result.pose.y - 6.0);
        label.points << label.labelPosition << label.labelPosition + QPointF(0.01,0.01);
        overlays.push_back(label);
    }
    return overlays;
}

}
