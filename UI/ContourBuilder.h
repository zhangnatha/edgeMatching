#pragma once

#include "ImageView.h"
#include "MakeTemplateV1.h"

#include <opencv2/core.hpp>

namespace ContourBuilder
{

// 从无序 canonical 特征集合重建仅用于显示的矢量轮廓；追踪时保留原始特征索引，
// 以便去除嵌套栅格边界的重复线，同时不删除真实孔洞。
QVector<ImageView::ContourPath> buildTemplateContours(
    const cv::Mat& image, const T_T::Template::Ptr& model);

// 根据指定金字塔层的 canonical 形状信息构建轮廓；坐标相对于该层图像中心解释，
// 与 CreateTemplate 的存储方式完全一致。
QVector<ImageView::ContourPath> buildTemplateContours(
    const cv::Mat& image, const T_T::Template::Ptr& model, int level);

}
