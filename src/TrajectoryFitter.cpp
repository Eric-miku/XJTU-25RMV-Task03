#include "TrajectoryFitter.h"
#include <ceres/ceres.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace ceres;
using namespace std;

// Ceres代价函数
struct TrajectoryCostFunction {
    TrajectoryCostFunction(double t, double x_obs, double y_obs, double x0, double y0)
        : t_(t), x_obs_(x_obs), y_obs_(y_obs), x0_(x0), y0_(y0) {}
    
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // params: [VX0, VY0, k, g_px]
        const T& VX0 = params[0];
        const T& VY0 = params[1];
        const T& k = params[2];
        const T& g_px = params[3];
        
        T x_pred, y_pred;
        
        // 处理k接近0的情况（数值稳定性）
        if (abs(k) < T(1e-8)) {
            // 无阻尼近似解
            x_pred = T(x0_) + VX0 * T(t_);
            y_pred = T(y0_) + VY0 * T(t_) - T(0.5) * g_px * T(t_) * T(t_);
        } else {
            T kt = k * T(t_);
            T exp_kt = exp(-kt);
            
            x_pred = T(x0_) + (VX0 / k) * (T(1.0) - exp_kt);
            y_pred = T(y0_) + (T(1.0) / k) * (VY0 + g_px / k) * (T(1.0) - exp_kt) - (g_px / k) * T(t_);
        }
        
        residual[0] = x_pred - T(x_obs_);
        residual[1] = y_pred - T(y_obs_);
        
        return true;
    }
    
private:
    double t_, x_obs_, y_obs_, x0_, y0_;
};

TrajectoryFitter::TrajectoryFitter() {}

FitResult TrajectoryFitter::fitTrajectory(const vector<Position>& positions) {
    if (positions.size() < 5) {
        cerr << "错误: 需要至少5个数据点进行拟合" << endl;
        return FitResult{0, 0, 0, 0, 0, 0};
    }
    
    // 获取初始位置（第一帧）
    double X0 = positions[0].x;
    double Y0 = positions[0].y;
    
    cout << "初始位置: X0=" << X0 << " px, Y0=" << Y0 << " px" << endl;
    
    // 调整时间基准，使第一帧时间为0
    vector<Position> adjusted_positions = positions;
    double t0 = positions[0].time;
    for (auto& pos : adjusted_positions) {
        pos.time -= t0;
    }
    
    // 估计初始速度（使用前5帧的数值微分）
    double VX0_est = 0.0, VY0_est = 0.0;
    int num_frames_vel = min(5, (int)adjusted_positions.size());
    for (int i = 1; i < num_frames_vel; i++) {
        double dt = adjusted_positions[i].time - adjusted_positions[i-1].time;
        if (dt > 0) {
            VX0_est += (adjusted_positions[i].x - adjusted_positions[i-1].x) / dt;
            VY0_est += (adjusted_positions[i].y - adjusted_positions[i-1].y) / dt;
        }
    }
    VX0_est /= (num_frames_vel - 1);
    VY0_est /= (num_frames_vel - 1);
    
    cout << "估计初始速度: VX0=" << VX0_est << " px/s, VY0=" << VY0_est << " px/s" << endl;
    
    // 设置参数初始值
    double params[4] = {VX0_est, VY0_est, 0.5, 550.0}; // [VX0, VY0, k, g_px]
    
    // 构建Ceres问题
    Problem problem;
    
    // 添加残差块
    for (const auto& pos : adjusted_positions) {
        CostFunction* cost_function = 
            new AutoDiffCostFunction<TrajectoryCostFunction, 2, 4>(
                new TrajectoryCostFunction(pos.time, pos.x, pos.y, X0, Y0));
        problem.AddResidualBlock(cost_function, nullptr, params);
    }
    
    // 设置参数边界
    problem.SetParameterLowerBound(params, 2, 0.01);  // k >= 0.01
    problem.SetParameterUpperBound(params, 2, 1.0);   // k <= 1.0
    problem.SetParameterLowerBound(params, 3, 100.0); // g_px >= 100
    problem.SetParameterUpperBound(params, 3, 1000.0); // g_px <= 1000
    
    // 配置求解器
    Solver::Options options;
    options.linear_solver_type = DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-6;
    options.parameter_tolerance = 1e-8;
    
    Solver::Summary summary;
    
    // 求解
    cout << "开始优化..." << endl;
    Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;
    
    // 计算RMSE
    double total_error = 0.0;
    int count = 0;
    for (const auto& pos : adjusted_positions) {
        Position pred = predictPosition(pos.time, 
            {params[0], params[1], params[2], params[3], 0, 0}, X0, Y0);
        double dx = pos.x - pred.x;
        double dy = pos.y - pred.y;
        total_error += dx*dx + dy*dy;
        count += 2;
    }
    double rmse = sqrt(total_error / count);
    
    FitResult result;
    result.VX0 = params[0];
    result.VY0 = params[1];
    result.k = params[2];
    result.g_px = params[3];
    result.rmse = rmse;
    result.iterations = summary.iterations.size();
    
    return result;
}

Position TrajectoryFitter::predictPosition(double t, const FitResult& params, double X0, double Y0) {
    Position pos;
    pos.time = t;
    
    double VX0 = params.VX0;
    double VY0 = params.VY0;
    double k = params.k;
    double g_px = params.g_px;
    
    if (abs(k) < 1e-8) {
        // 无阻尼情况
        pos.x = X0 + VX0 * t;
        pos.y = Y0 + VY0 * t - 0.5 * g_px * t * t;
    } else {
        double exp_kt = exp(-k * t);
        pos.x = X0 + (VX0 / k) * (1.0 - exp_kt);
        pos.y = Y0 + (1.0 / k) * (VY0 + g_px / k) * (1.0 - exp_kt) - (g_px / k) * t;
    }
    
    return pos;
}

void TrajectoryFitter::evaluateFit(const vector<Position>& positions, const FitResult& result) {
    double X0 = positions[0].x;
    double Y0 = positions[0].y;
    double t0 = positions[0].time;
    
    cout << "\n=== 拟合结果评估 ===" << endl;
    cout << "初始速度: VX0 = " << result.VX0 << " px/s, VY0 = " << result.VY0 << " px/s" << endl;
    cout << "阻尼系数: k = " << result.k << " s⁻¹" << endl;
    cout << "重力加速度: g_px = " << result.g_px << " px/s²" << endl;
    cout << "拟合RMSE: " << result.rmse << " px" << endl;
    cout << "迭代次数: " << result.iterations << endl;
    
    // 计算初始速度大小和角度
    double v0 = sqrt(result.VX0 * result.VX0 + result.VY0 * result.VY0);
    double angle = atan2(result.VY0, result.VX0) * 180.0 / M_PI;
    cout << "初始速度大小: " << v0 << " px/s" << endl;
    cout << "发射角度: " << angle << " 度" << endl;
    
    // 显示前几个点的预测误差
    cout << "\n前10个点的预测误差:" << endl;
    cout << "时间(s)\tX误差(px)\tY误差(px)\t总误差(px)" << endl;
    for (size_t i = 0; i < min((size_t)10, positions.size()); i++) {
        Position pred = predictPosition(positions[i].time - t0, result, X0, Y0);
        double dx = positions[i].x - pred.x;
        double dy = positions[i].y - pred.y;
        double total_error = sqrt(dx*dx + dy*dy);
        cout << fixed << setprecision(3) << positions[i].time - t0 << "\t" 
             << setprecision(2) << dx << "\t\t" << dy << "\t\t" << total_error << endl;
    }
}

void TrajectoryFitter::saveFitResult(const FitResult& result, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "错误: 无法创建文件 " << filename << endl;
        return;
    }
    
    file << "参数,值,单位" << endl;
    file << "VX0," << result.VX0 << ",px/s" << endl;
    file << "VY0," << result.VY0 << ",px/s" << endl;
    file << "k," << result.k << ",s⁻¹" << endl;
    file << "g_px," << result.g_px << ",px/s²" << endl;
    file << "RMSE," << result.rmse << ",px" << endl;
    file << "迭代次数," << result.iterations << "," << endl;
    
    file.close();
    cout << "拟合结果已保存到 " << filename << endl;
}


void TrajectoryFitter::plotComparison(const vector<Position>& positions, const FitResult& result) {
    if (positions.empty()) return;
    
    // 获取初始位置
    double X0 = positions[0].x;
    double Y0 = positions[0].y;
    double t0 = positions[0].time;
    
    // 找到数据范围
    double minX = positions[0].x, maxX = positions[0].x;
    double minY = positions[0].y, maxY = positions[0].y;
    double maxTime = 0;
    
    for (const auto& pos : positions) {
        minX = min(minX, pos.x);
        maxX = max(maxX, pos.x);
        minY = min(minY, pos.y);
        maxY = max(maxY, pos.y);
        maxTime = max(maxTime, pos.time - t0);
    }
    
    // 生成拟合轨迹的密集点
    vector<Position> fittedTrajectory;
    int numPoints = 200; // 生成200个点用于平滑显示
    for (int i = 0; i <= numPoints; i++) {
        double t = (maxTime * i) / numPoints;
        Position pred = predictPosition(t, result, X0, Y0);
        pred.time = t;
        fittedTrajectory.push_back(pred);
        
        // 更新范围以包含拟合轨迹
        minX = min(minX, pred.x);
        maxX = max(maxX, pred.x);
        minY = min(minY, pred.y);
        maxY = max(maxY, pred.y);
    }
    
    // 创建图像 (Y轴向上为正，所以我们需要翻转Y坐标来显示)
    int imgWidth = 1000;
    int imgHeight = 800;
    cv::Mat comparisonImg(imgHeight, imgWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // 计算缩放和偏移
    double marginX = (maxX - minX) * 0.1;
    double marginY = (maxY - minY) * 0.1;
    minX -= marginX; maxX += marginX;
    minY -= marginY; maxY += marginY;
    
    auto scaleX = [&](double x) { return static_cast<int>((x - minX) / (maxX - minX) * (imgWidth - 100) + 50); };
    auto scaleY = [&](double y) { return static_cast<int>(imgHeight - 50 - (y - minY) / (maxY - minY) * (imgHeight - 100)); };
    
    // 绘制坐标网格
    for (double x = ceil(minX/100)*100; x <= maxX; x += 100) {
        int x_img = scaleX(x);
        cv::line(comparisonImg, cv::Point(x_img, 50), cv::Point(x_img, imgHeight - 50), 
                cv::Scalar(200, 200, 200), 1);
        cv::putText(comparisonImg, to_string((int)x), cv::Point(x_img - 10, imgHeight - 20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }
    
    for (double y = ceil(minY/100)*100; y <= maxY; y += 100) {
        int y_img = scaleY(y);
        cv::line(comparisonImg, cv::Point(50, y_img), cv::Point(imgWidth - 50, y_img), 
                cv::Scalar(200, 200, 200), 1);
        cv::putText(comparisonImg, to_string((int)y), cv::Point(10, y_img + 5), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }
    
    // 绘制坐标轴
    cv::line(comparisonImg, cv::Point(50, 50), cv::Point(50, imgHeight - 50), cv::Scalar(0, 0, 0), 2);
    cv::line(comparisonImg, cv::Point(50, imgHeight - 50), cv::Point(imgWidth - 50, imgHeight - 50), cv::Scalar(0, 0, 0), 2);
    
    // 绘制拟合轨迹（蓝色曲线）
    for (size_t i = 1; i < fittedTrajectory.size(); i++) {
        int x1 = scaleX(fittedTrajectory[i-1].x);
        int y1 = scaleY(fittedTrajectory[i-1].y);
        int x2 = scaleX(fittedTrajectory[i].x);
        int y2 = scaleY(fittedTrajectory[i].y);
        cv::line(comparisonImg, cv::Point(x1, y1), cv::Point(x2, y2), 
                cv::Scalar(255, 0, 0), 3);
    }
    
    // 绘制原始检测点（红色圆点）
    for (const auto& pos : positions) {
        int x = scaleX(pos.x);
        int y = scaleY(pos.y);
        cv::circle(comparisonImg, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
        
        // 可选：显示时间标签
        // string timeLabel = to_string(pos.time - t0).substr(0, 4) + "s";
        // cv::putText(comparisonImg, timeLabel, cv::Point(x + 5, y - 5), 
        //            cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 100, 0), 1);
    }
    
    // 添加图例
    cv::putText(comparisonImg, "Fitted Trajectory", cv::Point(imgWidth - 200, 80), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
    cv::putText(comparisonImg, "Original Points", cv::Point(imgWidth - 200, 110), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    
    // 添加参数信息
    string paramText = "VX0: " + to_string(result.VX0).substr(0, 6) + " px/s";
    cv::putText(comparisonImg, paramText, cv::Point(imgWidth - 200, 150), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    paramText = "VY0: " + to_string(result.VY0).substr(0, 6) + " px/s";
    cv::putText(comparisonImg, paramText, cv::Point(imgWidth - 200, 170), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    paramText = "k: " + to_string(result.k).substr(0, 6) + " s^-1";
    cv::putText(comparisonImg, paramText, cv::Point(imgWidth - 200, 190), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    paramText = "g_px: " + to_string(result.g_px).substr(0, 6) + " px/s^2";
    cv::putText(comparisonImg, paramText, cv::Point(imgWidth - 200, 210), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    paramText = "RMSE: " + to_string(result.rmse).substr(0, 6) + " px";
    cv::putText(comparisonImg, paramText, cv::Point(imgWidth - 200, 230), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    // 添加标题
    cv::putText(comparisonImg, "Trajectory Fitting Comparison", cv::Point(imgWidth/2 - 150, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    
    // 坐标轴标签
    cv::putText(comparisonImg, "X (px)", cv::Point(imgWidth/2, imgHeight - 10), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(comparisonImg, "Y (px)", cv::Point(10, imgHeight/2), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    // 显示图像
    cv::namedWindow("Trajectory Comparison", cv::WINDOW_NORMAL);
    cv::resizeWindow("Trajectory Comparison", 1000, 800);
    cv::imshow("Trajectory Comparison", comparisonImg);
    
    // 保存图像
    cv::imwrite("trajectory_comparison.png", comparisonImg);
    cout << "对比图已保存为 trajectory_comparison.png" << endl;
    
    cv::waitKey(0);
}