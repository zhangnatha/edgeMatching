#pragma once

#include "ImageView.h"
#include "Type.h"

#include <opencv2/core.hpp>

#include <vector>

namespace MatchOverlay
{

// Apply the same image-coordinate rotation as the core drawMatchResults path:
// a public pose angle A selects the internally generated shape angle -A.
QPointF transformCanonicalPoint(const QPointF& canonical,
                                const T_T::MatchResult& result);

// Build display-only vector paths.  The input image is never modified.
QVector<ImageView::OverlayPath> buildMatchOverlays(
    const std::vector<T_T::MatchResult>& results,
    const std::vector<T_T::Template::Ptr>& models);
QVector<ImageView::OverlayPath> buildMatchOverlays(
    const cv::Mat& image, const std::vector<T_T::MatchResult>& results,
    const std::vector<T_T::Template::Ptr>& models);
QVector<ImageView::OverlayPath> buildMatchOverlays(
    const cv::Mat& image, const std::vector<T_T::MatchResult>& results,
    const std::vector<T_T::Template::Ptr>& models, I_I::Metric metric);

// Exposed for data-only geometry tests and to keep frame rendering identical
// to the core cv::RotatedRect convention.
QPolygonF rotatedFrame(const T_T::MatchResult& result,
                       const T_T::Template::Ptr& model);

}
