#ifndef TRAJECTORYFITTER_H
#define TRAJECTORYFITTER_H

#include <vector>
#include <string>
#include "BallDetector.h"  // 包含 Position 结构体定义

/**
 * @brief 拟合结果结构体
 * 保存轨迹拟合后的关键物理参数和误差信息
 */
struct FitResult {
    double VX0;     ///< 初始水平速度 (px/s)
    double VY0;     ///< 初始垂直速度 (px/s)  
    double k;       ///< 阻尼系数 (s⁻¹)
    double g_px;    ///< 重力加速度 (px/s²)
    double rmse;    ///< 均方根误差 (px)
    int iterations; ///< 迭代次数
};

/**
 * @brief 抛体运动轨迹拟合器
 * 
 * 通过给定的点集数据，使用非线性优化方法估计初始速度、
 * 阻尼系数和重力加速度等物理参数，并提供预测与可视化功能。
 */
class TrajectoryFitter {
public:
    TrajectoryFitter();

    /**
     * @brief 根据观测点集拟合运动轨迹
     * @param positions 小球在各时刻的观测位置
     * @return 轨迹拟合结果（包括初速度、阻尼系数、重力加速度等）
     */
    FitResult fitTrajectory(const std::vector<Position>& positions);

    /**
     * @brief 使用拟合参数预测某时刻的位置
     * @param t 时间 (秒)
     * @param params 拟合参数
     * @param X0 初始位置的 X 坐标
     * @param Y0 初始位置的 Y 坐标
     * @return 预测位置 (time, x, y)
     */
    Position predictPosition(double t, const FitResult& params, double X0, double Y0);

    /**
     * @brief 评估拟合结果的准确性
     * @param positions 真实观测点集
     * @param result 拟合结果
     */
    void evaluateFit(const std::vector<Position>& positions, const FitResult& result);

    /**
     * @brief 将拟合结果保存到文件
     * @param result 拟合参数结果
     * @param filename 输出文件路径 (TXT/CSV)
     */
    void saveFitResult(const FitResult& result, const std::string& filename);

    /**
     * @brief 绘制原始观测点与拟合轨迹的对比图
     * @param positions 观测点集
     * @param result 拟合参数结果
     */
    void plotComparison(const std::vector<Position>& positions, const FitResult& result);

private:
    // Ceres Solver 相关实现细节在 .cpp 文件中
};

#endif // TRAJECTORYFITTER_H
