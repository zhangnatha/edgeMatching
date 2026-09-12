#include "ContourBuilder.h"
#include "MatchOverlay.h"
#include "MakeTemplateV1.h"
#include "FindTemplateV1.h"

#include <QLineF>

#include <opencv2/imgproc.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <cstdlib>

namespace
{

T_T::Template::Ptr modelFromMask(const cv::Mat& mask)
{
    T_T::Template::Ptr model(new T_T::Template());
    model->templates.resize(1);
    model->templates[0].reset(new T_T::ShapeInfo());
    model->templates[0]->shape_angle.resize(1);
    model->templates[0]->shape_angle[0].reset(new T_T::ShapeAngle());
    const double cx = mask.cols * 0.5;
    const double cy = mask.rows * 0.5;
    for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
            if (mask.at<unsigned char>(y, x) == 0) continue;
            T_T::ShapePoint point;
            point.x = x - cx;
            point.y = y - cy;
            point.edge_dx = 1.0f;
            point.edge_dy = 0.0f;
            model->templates[0]->shape_angle[0]->shape_point.push_back(point);
        }
    }
    return model;
}

void assertBounded(const QVector<ImageView::ContourPath>& contours)
{
    for (const ImageView::ContourPath& contour : contours) {
        assert(contour.points.size() >= 2);
        for (int i = 1; i < contour.points.size(); ++i)
            assert(QLineF(contour.points.at(i - 1), contour.points.at(i)).length() <= 4.5 + 1e-9);
        if (contour.closed)
            assert(QLineF(contour.points.back(), contour.points.front()).length() <= 4.5 + 1e-9);
        for (int i = 0; i + 1 < contour.points.size(); ++i) {
            const QLineF first(contour.points.at(i), contour.points.at(i + 1));
            for (int j = i + 2; j + 1 < contour.points.size(); ++j) {
                if (contour.closed && i == 0 && j + 1 == contour.points.size() - 1)
                    continue;
                const QLineF second(contour.points.at(j), contour.points.at(j + 1));
                QPointF intersection;
                assert(first.intersects(second, &intersection) != QLineF::BoundedIntersection);
            }
        }
    }
}

}

int main()
{
    const cv::Mat image = cv::Mat::zeros(80, 80, CV_8UC1);

    cv::Mat circle = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::ellipse(circle, cv::Point(40, 40), cv::Size(14, 14), 0.0, 0.0, 360.0, 255, 1,
                cv::LINE_8);
    const QVector<ImageView::ContourPath> oneCircle =
        ContourBuilder::buildTemplateContours(image, modelFromMask(circle));
    assert(oneCircle.size() == 1);
    assertBounded(oneCircle);

    cv::Mat outerAndHole = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::rectangle(outerAndHole, cv::Rect(10, 10, 60, 60), 255, 1, cv::LINE_8);
    cv::rectangle(outerAndHole, cv::Rect(30, 30, 20, 20), 255, 1, cv::LINE_8);
    const QVector<ImageView::ContourPath> twoBoundaries =
        ContourBuilder::buildTemplateContours(image, modelFromMask(outerAndHole));
    assert(twoBoundaries.size() == 2);
    assertBounded(twoBoundaries);

    // Coarse-level style geometry: the external boundary and four disconnected
    // holes must each remain a closed chain, even when the holes are sparse.
    cv::Mat fourHoleBoundaries = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::rectangle(fourHoleBoundaries, cv::Rect(8, 8, 64, 64), 255, 1, cv::LINE_8);
    const std::vector<cv::Point> fourHoleCenters{{20, 20}, {60, 20}, {60, 60}, {20, 60}};
    for (const cv::Point& center : fourHoleCenters)
        cv::circle(fourHoleBoundaries, center, 4, 255, 1, cv::LINE_8);
    const QVector<ImageView::ContourPath> fiveClosed =
        ContourBuilder::buildTemplateContours(image, modelFromMask(fourHoleBoundaries));
    assert(fiveClosed.size() == 5);
    for (const ImageView::ContourPath& contour : fiveClosed) {
        assert(contour.closed);
        assertBounded(QVector<ImageView::ContourPath>{contour});
    }

    // Explicit pyramid level must use that level's ShapeInfo and its own
    // image centre; it must not silently reuse L0 coordinates.
    T_T::Template::Ptr twoLevel(new T_T::Template());
    twoLevel->templates.resize(2);
    cv::Mat l0Mask = cv::Mat::zeros(40, 40, CV_8UC1);
    cv::circle(l0Mask, cv::Point(30, 30), 6, 255, 1, cv::LINE_8);
    cv::Mat l1Mask = cv::Mat::zeros(20, 20, CV_8UC1);
    cv::circle(l1Mask, cv::Point(5, 5), 3, 255, 1, cv::LINE_8);
    twoLevel->templates[0] = modelFromMask(l0Mask)->templates[0];
    twoLevel->templates[1] = modelFromMask(l1Mask)->templates[0];
    const QVector<ImageView::ContourPath> levelOne =
        ContourBuilder::buildTemplateContours(cv::Mat::zeros(20, 20, CV_8UC1), twoLevel, 1);
    assert(levelOne.size() == 1);
    assertBounded(levelOne);
    QPointF mean;
    int count = 0;
    for (const QPointF& point : levelOne.front().points) { mean += point; ++count; }
    mean /= static_cast<qreal>(count);
    assert(std::abs(mean.x() - 5.0) < 1.0 && std::abs(mean.y() - 5.0) < 1.0);

    // A new model may use the valid mask/domain centroid as its reference
    // point.  This is opt-in; the legacy default remains image centre.
    cv::Mat asymmetricImage = cv::Mat::zeros(32, 32, CV_8UC1);
    cv::Mat asymmetricMask = cv::Mat::zeros(32, 32, CV_8UC1);
    cv::rectangle(asymmetricImage, cv::Rect(4, 6, 12, 18), 255, cv::FILLED);
    cv::rectangle(asymmetricMask, cv::Rect(4, 6, 12, 18), 255, cv::FILLED);
    T_T::Template::Ptr centroidModel(new T_T::Template());
    SM_V1::CreateTemplate centroidTrainer;
    assert(centroidTrainer.createTemplate(asymmetricImage, asymmetricMask, 0, -10, 10, 1.0,
                                          false, 10, 30, centroidModel, T_T::EDGE_CURRENT,
                                          T_T::ORIGIN_DOMAIN_CENTROID));
    assert(centroidModel->template_cfg.origin_mode == T_T::ORIGIN_DOMAIN_CENTROID);
    assert(std::abs(centroidModel->template_cfg.origin_x - 9.5) < 1e-6);
    assert(std::abs(centroidModel->template_cfg.origin_y - 14.5) < 1e-6);
    assert(std::abs(centroidModel->templates[0]->shape_angle[0]->angle) < 1e-12);

    const std::string angleModelPath = "/tmp/edge_matching_angle_phase3.json";
    assert(centroidTrainer.saveModelFile2Json(centroidModel, angleModelPath));
    SM_V1::SearchTemplate angleLoader;
    const T_T::Template::Ptr expanded = angleLoader.loadModelFileFromJson(angleModelPath);
    assert(expanded && !expanded->templates.empty());
    const auto& angles = expanded->templates[0]->shape_angle;
    bool hasZero = false, hasStart = false, hasEnd = false;
    for (size_t i = 0; i < angles.size(); ++i) {
        assert(angles[i]);
        const double angle = angles[i]->angle;
        assert(angle >= -10.0 - 1e-9 && angle <= 10.0 + 1e-9);
        if (std::abs(angle) < 1e-9) hasZero = true;
        if (std::abs(angle + 10.0) < 1e-9) hasStart = true;
        if (std::abs(angle - 10.0) < 1e-9) hasEnd = true;
        // The canonical zero entry is intentionally retained alongside the
        // historical sampled zero entry for matcher compatibility. Other
        // samples, especially the exact interval endpoints, are unique.
        for (size_t j = 0; j < i; ++j)
            if (std::abs(angle) > 1e-9) assert(std::abs(angle - angles[j]->angle) > 1e-9);
    }
    assert(hasZero && hasStart && hasEnd);

    // Automatic pyramid selection must back off on a small/sparse target and
    // never expose a selected top layer with fewer than four features.
    cv::Mat sparseImage = cv::Mat::zeros(64, 64, CV_8UC1);
    cv::Mat sparseMask = cv::Mat::zeros(64, 64, CV_8UC1);
    cv::rectangle(sparseImage, cv::Rect(27, 27, 10, 10), 255, cv::FILLED);
    cv::rectangle(sparseMask, cv::Rect(27, 27, 10, 10), 255, cv::FILLED);
    T_T::Template::Ptr autoModel(new T_T::Template());
    SM_V1::CreateTemplate autoTrainer;
    assert(autoTrainer.createTemplate(sparseImage, sparseMask, -1, -5, 5, 1.0,
                                      false, 10, 30, autoModel));
    for (int level = 0; level <= autoModel->template_cfg.num_levels; ++level) {
        assert(autoModel->templates[level] && !autoModel->templates[level]->shape_angle.empty());
        assert(autoModel->templates[level]->shape_angle[0]->shape_point.size() >= 4);
    }

    // Public pose angles use the same sign convention as the UI overlay:
    // x'=x cos(A)+y sin(A), y'=-x sin(A)+y cos(A).
    const T_T::MatchResult knownPose(T_T::Pose2d(100.0, 80.0, 90.0), 0.95, 2.0, 7);
    const QPointF transformed = MatchOverlay::transformCanonicalPoint(QPointF(3.0, 4.0), knownPose);
    assert(std::abs(transformed.x() - 108.0) < 1e-9);
    assert(std::abs(transformed.y() - 74.0) < 1e-9);

    // Two template IDs must select their own geometry, rather than applying
    // the first model to every match.
    T_T::Template::Ptr narrow = modelFromMask(circle);
    narrow->template_cfg.id = 7;
    narrow->template_cfg.image_width = 20;
    narrow->template_cfg.image_height = 20;
    T_T::Template::Ptr wide = modelFromMask(circle);
    wide->template_cfg.id = 8;
    wide->template_cfg.image_width = 40;
    wide->template_cfg.image_height = 40;
    const std::vector<T_T::Template::Ptr> models{narrow, wide};
    const std::vector<T_T::MatchResult> matches{
        T_T::MatchResult(T_T::Pose2d(30.0, 30.0, 0.0), 0.9, 1.0, 7),
        T_T::MatchResult(T_T::Pose2d(70.0, 70.0, 0.0), 0.9, 1.0, 8)};
    const QVector<ImageView::OverlayPath> overlays =
        MatchOverlay::buildMatchOverlays(matches, models);
    QVector<QPolygonF> frames;
    for (const ImageView::OverlayPath& overlay : overlays) {
        if (overlay.closed && overlay.points.size() == 4 &&
            overlay.color == QColor(0, 255, 255)) frames.push_back(overlay.points);
    }
    assert(frames.size() == 2);
    const auto frameArea = [](const QPolygonF& polygon) {
        return QLineF(polygon.at(0), polygon.at(1)).length();
    };
    assert(std::abs(frameArea(frames.at(0)) - 20.0) < 1e-4);
    assert(std::abs(frameArea(frames.at(1)) - 40.0) < 1e-4);

    // Optional diagnostic for real coarse models.  Kept environment-gated so
    // the unit test remains independent of repository assets.
    if (const char* modelPath = std::getenv("SHAPE_MATCH_DIAGNOSTIC_MODEL")) {
        SM_V1::SearchTemplate loader;
        const auto realModel = loader.loadModelFileFromJson(modelPath);
        const char* imagePath = std::getenv("SHAPE_MATCH_DIAGNOSTIC_IMAGE");
        if (realModel && imagePath) {
            const cv::Mat realImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
            const auto realContours = ContourBuilder::buildTemplateContours(realImage, realModel, 2);
            std::cout << "diagnostic L2 contours=" << realContours.size();
            for (const auto& contour : realContours)
                std::cout << " [" << contour.points.size() << "," << (contour.closed ? "closed" : "open")
                          << ",gap=" << QLineF(contour.points.front(), contour.points.back()).length()
                          << ",center=" << contour.points.boundingRect().center().x() << ","
                          << contour.points.boundingRect().center().y() << "]";
            std::cout << std::endl;
        }
    }

    std::cout << "ui contour and overlay geometry: 6/6" << std::endl;
    return 0;
}
