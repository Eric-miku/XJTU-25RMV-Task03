#include "BallDetector.h"
#include "TrajectoryFitter.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main() {
    /*===========================第一阶段：小球检测=================================*/
    BallDetector detector;
    string videoPath = "../TASK03/video.mp4"; // 替换为你的视频路径
    // 检测小球位置
    vector<Position> positions = detector.detectBallPositions(videoPath);
    if (positions.empty()) {
        cout << "未检测到任何位置数据，请检查视频路径和颜色范围" << endl;
        return -1;
    }
    // 保存检测结果
    savePositionsToFile(positions, "ball_positions.csv");
    // 显示轨迹
    detector.plotTrajectory(positions);
    
    /*===========================第二阶段：轨迹拟合=================================*/
    TrajectoryFitter fitter;
    FitResult result = fitter.fitTrajectory(positions);
    // 评估和保存结果
    fitter.evaluateFit(positions, result);
    fitter.saveFitResult(result, "fit_result.csv");

    /*===========================第三阶段：轨迹对比可视化============================*/
    fitter.plotComparison(positions, result);

    cout << "\n轨迹拟合完成！" << endl;
    
    return 0;
}