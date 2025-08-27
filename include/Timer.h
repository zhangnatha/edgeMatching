#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

enum class TimerMethod
{
    HighResolutionClock,
    SteadyClock,
    OpenCVTick
};

class Timer
{
public:
    Timer(TimerMethod method = TimerMethod::HighResolutionClock)
        : method_(method)
    {
    }

    void start()
    {
        time_points_.clear(); // 清空之前的时间点
        function_names_.clear(); // 清空之前的函数名
        switch (method_)
        {
        case TimerMethod::HighResolutionClock:
        case TimerMethod::SteadyClock:
            time_points_.push_back(std::chrono::high_resolution_clock::now());
            break;
        case TimerMethod::OpenCVTick:
            time_points_cv_.push_back(cv::getTickCount());
            break;
        }
    }

    void record(const std::string& function_name)
    {
        function_names_.push_back(function_name); // 存储函数名
        switch (method_)
        {
        case TimerMethod::HighResolutionClock:
        case TimerMethod::SteadyClock:
            time_points_.push_back(std::chrono::high_resolution_clock::now());
            break;
        case TimerMethod::OpenCVTick:
            time_points_cv_.push_back(cv::getTickCount());
            break;
        }
    }

    void report() const
    {
        if (method_ == TimerMethod::OpenCVTick)
        {
            for (size_t i = 1; i < time_points_cv_.size(); ++i)
            {
                double duration = static_cast<double>(time_points_cv_[i] - time_points_cv_[i - 1]) /
                    cv::getTickFrequency() * 1000;
                std::cout << function_names_[i - 1] << " took: " << duration << " ms." << std::endl;
            }
        }
        else
        {
            for (size_t i = 1; i < time_points_.size(); ++i)
            {
                double duration = 0;
                switch (method_)
                {
                case TimerMethod::HighResolutionClock:
                    {
                        auto diff = std::chrono::duration<double, std::milli>(time_points_[i] - time_points_[i - 1]);
                        duration = diff.count();
                        break;
                    }
                case TimerMethod::SteadyClock:
                    {
                        duration = std::chrono::duration<double, std::milli>(time_points_[i] - time_points_[i - 1]).
                            count();
                        break;
                    }
                }
                std::cout << function_names_[i - 1] << " took: " << duration << " ms." << std::endl;
            }
        }
    }

    double get(const std::string& function_name) const
    {
        for (size_t i = 0; i < function_names_.size(); ++i)
        {
            if (function_names_[i] == function_name)
            {
                if (method_ == TimerMethod::OpenCVTick)
                {
                    if (i + 1 < time_points_cv_.size())
                    {
                        return static_cast<double>(time_points_cv_[i + 1] - time_points_cv_[i]) /
                            cv::getTickFrequency() * 1000;
                    }
                }
                else if (method_ == TimerMethod::HighResolutionClock)
                {
                    if (i + 1 < time_points_.size())
                    {
                        return std::chrono::duration<double, std::milli>(time_points_[i + 1] - time_points_[i]).count();
                    }
                }
                else if (method_ == TimerMethod::SteadyClock)
                {
                    if (i + 1 < time_points_s_.size())
                    {
                        return std::chrono::duration<double, std::milli>(time_points_s_[i + 1] - time_points_s_[i]).count();
                    }
                }
            }
        }
        return -1.0; // 返回-1表示未找到
    }

private:
    TimerMethod method_;
    std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> time_points_;
    std::vector<std::chrono::time_point<std::chrono::steady_clock>> time_points_s_;
    std::vector<int64_t> time_points_cv_; // 存储 OpenCV tick 时间戳
    std::vector<std::string> function_names_; // 存储函数名
};

