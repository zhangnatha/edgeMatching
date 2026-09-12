#pragma once

#include "ImageView.h"
#include "Type.h"

#include <opencv2/core.hpp>

#include <vector>

namespace MatchOverlay
{

// 使用与核心 drawMatchResults 相同的图像坐标旋转：公开位姿角 A 对应内部生成的
// 形状角 -A。
QPointF transformCanonicalPoint(const QPointF& canonical,
                                const T_T::MatchResult& result);

// 构建仅用于显示的矢量路径，不修改输入图像。
QVector<ImageView::OverlayPath> buildMatchOverlays(
    const std::vector<T_T::MatchResult>& results,
    const std::vector<T_T::Template::Ptr>& models);
QVector<ImageView::OverlayPath> buildMatchOverlays(
    const cv::Mat& image, const std::vector<T_T::MatchResult>& results,
    const std::vector<T_T::Template::Ptr>& models);
QVector<ImageView::OverlayPath> buildMatchOverlays(
    const cv::Mat& image, const std::vector<T_T::MatchResult>& results,
    const std::vector<T_T::Template::Ptr>& models, I_I::Metric metric);

// 对外提供数据几何测试接口，并保持外框绘制与核心 cv::RotatedRect 约定一致。
QPolygonF rotatedFrame(const T_T::MatchResult& result,
                       const T_T::Template::Ptr& model);

}
