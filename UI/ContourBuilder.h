#pragma once

#include "ImageView.h"
#include "MakeTemplateV1.h"

#include <opencv2/core.hpp>

namespace ContourBuilder
{

// Reconstruct display-only vector contours from the unordered canonical
// feature set.  The reconstruction keeps the original feature indices while
// tracing so nested raster boundaries can be de-duplicated without removing
// real holes.
QVector<ImageView::ContourPath> buildTemplateContours(
    const cv::Mat& image, const T_T::Template::Ptr& model);

// Build contours from the canonical shape information at an explicit
// pyramid level.  Coordinates are interpreted relative to that level's
// image centre, exactly as they are stored by CreateTemplate.
QVector<ImageView::ContourPath> buildTemplateContours(
    const cv::Mat& image, const T_T::Template::Ptr& model, int level);

}
