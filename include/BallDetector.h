#ifndef BALLDETECTOR_H
#define BALLDETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 表示检测到的小球位置及对应的时间戳
struct Position {
    double time;  // 时间 (秒)
    double x;     // X 坐标 (像素)
    double y;     // Y 坐标 (像素)
};

class BallDetector {
public:
    BallDetector();

    /**
     * @brief 从视频中检测小球的运动轨迹
     * @param videoPath 输入视频文件路径
     * @return 小球在各个时间点的位置序列 (time, x, y)
     */
    std::vector<Position> detectBallPositions(const std::string& videoPath);

    /**
     * @brief 通过交互方式校准小球颜色范围
     * @param videoPath 输入视频文件路径
     * @param sampleFrame 用于采样的帧编号 (默认第 0 帧)
     */
    void calibrateColorRange(const std::string& videoPath, int sampleFrame = 0);

    /**
     * @brief 绘制小球的运动轨迹
     * @param positions 小球的位置序列
     */
    void plotTrajectory(const std::vector<Position>& positions);

private:
    cv::Scalar lowerBlue;   ///< 小球颜色的下限 (HSV)
    cv::Scalar upperBlue;   ///< 小球颜色的上限 (HSV)

    // 颜色阈值参数 (供手动调节)
    int h_min, h_max;
    int s_min, s_max;
    int v_min, v_max;

    /**
     * @brief 在单帧图像中检测小球位置
     * @param frame 输入帧 (BGR 格式)
     * @param frameCount 当前帧编号
     * @return 小球质心坐标 (若未检测到则返回 (0,0))
     */
    cv::Point2f detectBallInFrame(const cv::Mat& frame, int frameCount);

    /**
     * @brief 显示检测效果并叠加轨迹信息
     * @param frame 输入帧 (BGR 格式)
     * @param position 小球位置
     * @param frameCount 当前帧编号
     * @param fps 视频帧率
     */
    void displayEnhancedDetection(const cv::Mat& frame, const cv::Point2f& position, 
                                 int frameCount, double fps);

    /**
     * @brief Trackbar 回调函数 (用于颜色阈值调整)
     * @param value 当前 Trackbar 值
     * @param userdata 传入的用户数据指针 (BallDetector 实例)
     */
    static void onTrackbarChange(int value, void* userdata);
};

/**
 * @brief 将检测结果保存到文件
 * @param positions 小球的位置序列
 * @param filename 输出文件路径 (CSV 或 TXT)
 */
void savePositionsToFile(const std::vector<Position>& positions, const std::string& filename);

#endif // BALLDETECTOR_H