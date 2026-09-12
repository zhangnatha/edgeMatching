#ifndef ROI_H
#define ROI_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

class ROI
{
public:
    enum ROIType { RECT, ROTATED_RECT, NONE};

    ROI() : type_(NONE) {}  // ✅ 默认构造函数，表示没有ROI

    ROI(const cv::Rect& rect)// 矩形 ROI : x, y, w, h(左上角坐标、宽高)
        : type_(RECT), rect_(rect) {}

    ROI(const cv::RotatedRect& rrect)// 旋转矩形 ROI : center, size, angle(中心点、尺寸、角度初始化)
        : type_(ROTATED_RECT), rrect_(rrect) {}

    ROIType type() const { return type_; }

    /**
     * @brief 绘制 ROI
     */
    void draw(const cv::Mat& src, cv::Mat& dst, const cv::Scalar& color = cv::Scalar(0, 255, 0), int thickness = 2) const
    {
        dst = src.clone();
        if (type_ == RECT)
        {
            cv::rectangle(dst, rect_, color, thickness);
        }
        else if (type_ == ROTATED_RECT)
        {
            cv::Point2f vertices[4];
            rrect_.points(vertices);
            for (int i = 0; i < 4; i++)
            {
                cv::line(dst, vertices[i], vertices[(i + 1) % 4], color, thickness);
            }
        }
    }

    /**
     * @brief ROI 坐标系 -> 图像坐标系
     */
    bool toImageCoord(double xinroi, double yinroi, double angleinroi,
                      double& xinimg, double& yinimg, double& angleinimg) const
    {
        if (type_ == RECT)
        {
            xinimg = xinroi + rect_.x;
            yinimg = yinroi + rect_.y;
            angleinimg = angleinroi;
        }
        else if (type_ == ROTATED_RECT)
        {
            cv::Point2f roiPt(xinroi - rrect_.size.width / 2.0,
                              yinroi - rrect_.size.height / 2.0);

            double rad = -rrect_.angle * CV_PI / 180.0;
            cv::Matx22f R(cos(rad), -sin(rad),
                          sin(rad),  cos(rad));

            cv::Point2f rotated = R * roiPt;
            cv::Point2f imgPt   = rotated + rrect_.center;

            xinimg = imgPt.x;
            yinimg = imgPt.y;

            angleinimg = angleinroi - rrect_.angle;
            while (angleinimg > 180)  angleinimg -= 360;
            while (angleinimg < -180) angleinimg += 360;
        }
        return true;
    }

    /**
     * @brief 从 ROI 裁剪图像
     * @param src 输入源图
     * @param dst 输出裁剪后的图像
     */
    bool crop(const cv::Mat& src, cv::Mat& dst) const
    {
        if (type_ == RECT)
        {
            cv::Rect validRect = rect_ & cv::Rect(0, 0, src.cols, src.rows); // 防止越界
            if (validRect.width <= 0 || validRect.height <= 0) return false;
            dst = src(validRect).clone();
            return true;
        }
        else if (type_ == ROTATED_RECT)
        {
            // 旋转 ROI 到水平
            cv::Mat M = cv::getRotationMatrix2D(rrect_.center, rrect_.angle, 1.0);

            cv::Mat rotated;
            cv::warpAffine(src, rotated, M, src.size(), cv::INTER_CUBIC);

            // 裁剪旋转后的 ROI
            cv::Size roi_size = rrect_.size;
            cv::getRectSubPix(rotated, roi_size, rrect_.center, dst);

            return true;
        }
        return false;
    }


    // 🔹 判断是否为空
    bool empty() const
    {
        if (type_ == RECT)
            return (rect_.width <= 0 || rect_.height <= 0);
        else if (type_ == ROTATED_RECT)
            return (rrect_.size.width <= 0 || rrect_.size.height <= 0);
        return true;
    }

    // 🔹 计算面积
    double area() const
    {
        if (empty()) return 0.0;
        if (type_ == RECT)
            return static_cast<double>(rect_.area());
        else if (type_ == ROTATED_RECT)
            return rrect_.size.area();
        return 0.0;
    }

    // 🔹 计算中心点
    cv::Point2f center() const
    {
        if (empty()) return cv::Point2f(-1, -1);
        if (type_ == RECT)
            return cv::Point2f(rect_.x + rect_.width / 2.0f,
                               rect_.y + rect_.height / 2.0f);
        else if (type_ == ROTATED_RECT)
            return rrect_.center;
        return cv::Point2f(-1, -1);
    }

private:
    ROIType type_;
    cv::Rect rect_;
    cv::RotatedRect rrect_;
};


#endif // ROI_H 宏定义结束

