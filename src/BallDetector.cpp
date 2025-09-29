#include "BallDetector.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <thread>
#include <chrono>

BallDetector::BallDetector() {
    // 默认天蓝色HSV范围
    lowerBlue = cv::Scalar(90, 50, 50);
    upperBlue = cv::Scalar(130, 255, 255);
    
    // 初始化颜色校准变量
    h_min = 90; h_max = 130;
    s_min = 50; s_max = 255;
    v_min = 50; v_max = 255;
}

// 静态回调函数
void BallDetector::onTrackbarChange(int, void* userdata) {
    cv::Mat* frame = static_cast<cv::Mat*>(userdata);
    cv::Mat hsv, mask, result;
    
    cv::cvtColor(*frame, hsv, cv::COLOR_BGR2HSV);
    
    // 注意：这里直接使用类的静态方法，需要通过其他方式获取当前实例
    // 在实际使用中，我们会通过其他方式传递实例指针
    cv::inRange(hsv, cv::Scalar(90, 50, 50), cv::Scalar(130, 255, 255), mask);
    cv::bitwise_and(*frame, *frame, result, mask);
    
    cv::imshow("Color Calibration", result);
}

std::vector<Position> BallDetector::detectBallPositions(const std::string& videoPath) {
    std::vector<Position> positions;
    
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "错误: 无法打开视频文件: " << videoPath << std::endl;
        return positions;
    }
    
    // 获取视频信息
    double fps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "视频信息: " << width << "x" << height 
              << ", FPS: " << fps << ", 总帧数: " << totalFrames << std::endl;
    
    // 创建显示窗口
    cv::namedWindow("Ball Detection - Original", cv::WINDOW_NORMAL);
    cv::namedWindow("Ball Detection - Processed", cv::WINDOW_NORMAL);
    cv::namedWindow("Ball Detection - Mask", cv::WINDOW_NORMAL);
    
    // 调整窗口大小
    cv::resizeWindow("Ball Detection - Original", 1200, 800);
    cv::resizeWindow("Ball Detection - Processed", 1200, 800);
    cv::resizeWindow("Ball Detection - Mask", 1200, 800);
    
    int frameCount = 0;
    cv::Mat frame;
    
    // 控制播放速度的参数
    int delayMs = 1; // 每帧之间的延迟（毫秒），可以调整这个值来控制速度
    bool paused = false;
    
    std::cout << "按以下键控制播放:" << std::endl;
    std::cout << "  SPACE: 暂停/继续" << std::endl;
    std::cout << "  ESC: 退出" << std::endl;
    std::cout << "  +: 加快速度" << std::endl;
    std::cout << "  -: 减慢速度" << std::endl;
    
    while (true) {
        if (!paused) {
            cap >> frame;
            if (frame.empty()) break;
            
            double currentTime = frameCount / fps;
            cv::Point2f ballPosition = detectBallInFrame(frame, frameCount);
            
            if (ballPosition.x >= 0 && ballPosition.y >= 0) {
                // 转换坐标系：Y向上为正
                double yTransformed = height - ballPosition.y;
                positions.push_back({currentTime, ballPosition.x, yTransformed});
                
                std::cout << "帧 " << frameCount << ": t=" << std::fixed << std::setprecision(3) 
                          << currentTime << "s, 位置=(" << ballPosition.x << ", " 
                          << yTransformed << ")" << std::endl;
            } else {
                std::cout << "警告: 第 " << frameCount << " 帧未检测到小球" << std::endl;
            }
            
            // 显示处理结果
            displayEnhancedDetection(frame, ballPosition, frameCount, fps);
            
            frameCount++;
        }
        
        // 键盘控制
        int key = cv::waitKey(delayMs) & 0xFF;
        if (key == 27) { // ESC键
            break;
        } else if (key == 32) { // 空格键
            paused = !paused;
            std::cout << (paused ? "已暂停" : "继续播放") << std::endl;
        } else if (key == 43) { // +键
            delayMs = std::max(10, delayMs - 10);
            std::cout << "加快速度，当前延迟: " << delayMs << "ms" << std::endl;
        } else if (key == 45) { // -键
            delayMs += 10;
            std::cout << "减慢速度，当前延迟: " << delayMs << "ms" << std::endl;
        }
        
        // 如果暂停，等待更短时间以保持响应性
        if (paused) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    std::cout << "检测完成: 共处理 " << frameCount << " 帧，成功检测 " 
              << positions.size() << " 个位置" << std::endl;
    
    return positions;
}

cv::Point2f BallDetector::detectBallInFrame(const cv::Mat& frame, int frameCount) {
    cv::Mat hsv, mask;
    
    // 转换为HSV颜色空间
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    
    // 创建颜色掩膜
    cv::inRange(hsv, lowerBlue, upperBlue, mask);
    
    // 形态学操作
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return cv::Point2f(-1, -1);
    }
    
    // 找到最大轮廓
    double maxArea = 0;
    int maxAreaIdx = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxAreaIdx = i;
        }
    }
    
    if (maxAreaIdx == -1) {
        return cv::Point2f(-1, -1);
    }
    
    // 计算质心
    cv::Moments m = cv::moments(contours[maxAreaIdx]);
    if (m.m00 == 0) {
        return cv::Point2f(-1, -1);
    }
    
    float cx = static_cast<float>(m.m10 / m.m00);
    float cy = static_cast<float>(m.m01 / m.m00);
    
    // 亚像素精度优化
    if (contours[maxAreaIdx].size() >= 5) {
        std::vector<cv::Point2f> contourPoints;
        for (const auto& pt : contours[maxAreaIdx]) {
            contourPoints.push_back(cv::Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)));
        }
        
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 0.01);
        cv::cornerSubPix(gray, contourPoints, cv::Size(3, 3), cv::Size(-1, -1), criteria);
        
        // 重新计算质心
        cv::Moments m_refined = cv::moments(contourPoints);
        if (m_refined.m00 != 0) {
            cx = static_cast<float>(m_refined.m10 / m_refined.m00);
            cy = static_cast<float>(m_refined.m01 / m_refined.m00);
        }
    }
    
    return cv::Point2f(cx, cy);
}

void BallDetector::displayEnhancedDetection(const cv::Mat& frame, const cv::Point2f& position, 
                                           int frameCount, double fps) {
    cv::Mat displayFrame = frame.clone();
    cv::Mat hsv, mask, processedFrame;
    
    // 处理帧用于显示
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, lowerBlue, upperBlue, mask);
    
    // 形态学操作
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    
    // 创建处理后的帧（高亮显示检测到的区域）
    cv::bitwise_and(frame, frame, processedFrame, mask);
    
    if (position.x >= 0 && position.y >= 0) {
        // 在原始帧上绘制检测到的小球位置
        cv::circle(displayFrame, position, 8, cv::Scalar(0, 255, 0), 2);
        cv::circle(displayFrame, position, 2, cv::Scalar(0, 255, 0), -1);
        
        // 在处理后的帧上绘制
        cv::circle(processedFrame, position, 8, cv::Scalar(0, 255, 0), 2);
        cv::circle(processedFrame, position, 2, cv::Scalar(0, 255, 0), -1);
        
        // 添加坐标信息
        std::string coordText = "(" + std::to_string((int)position.x) + ", " + 
                                std::to_string((int)position.y) + ")";
        cv::putText(displayFrame, coordText, 
                   cv::Point(position.x + 10, position.y - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    
    // 添加帧信息
    std::string frameText = "Frame: " + std::to_string(frameCount);
    std::string timeText = "Time: " + std::to_string(frameCount / fps).substr(0, 5) + "s";
    
    cv::putText(displayFrame, frameText, cv::Point(10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(displayFrame, timeText, cv::Point(10, 60), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    cv::putText(processedFrame, frameText, cv::Point(10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(processedFrame, timeText, cv::Point(10, 60), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    // 显示三个窗口
    cv::imshow("Ball Detection - Original", displayFrame);
    cv::imshow("Ball Detection - Processed", processedFrame);
    cv::imshow("Ball Detection - Mask", mask);
}

void BallDetector::calibrateColorRange(const std::string& videoPath, int sampleFrame) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "错误: 无法打开视频文件: " << videoPath << std::endl;
        return;
    }
    
    cap.set(cv::CAP_PROP_POS_FRAMES, sampleFrame);
    cv::Mat frame;
    cap >> frame;
    cap.release();
    
    if (frame.empty()) {
        std::cout << "无法读取样本帧" << std::endl;
        return;
    }
    
    // 创建校准窗口
    cv::namedWindow("Color Calibration");
    
    // 创建滑动条 - 使用lambda函数作为回调
    cv::createTrackbar("H_min", "Color Calibration", &h_min, 180, [](int, void* userdata) {
        cv::Mat* frame = static_cast<cv::Mat*>(userdata);
        cv::Mat hsv, mask, result;
        
        cv::cvtColor(*frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(90, 50, 50), cv::Scalar(130, 255, 255), mask);
        cv::bitwise_and(*frame, *frame, result, mask);
        
        cv::imshow("Color Calibration", result);
    }, &frame);
    
    cv::createTrackbar("H_max", "Color Calibration", &h_max, 180, [](int, void* userdata) {
        cv::Mat* frame = static_cast<cv::Mat*>(userdata);
        cv::Mat hsv, mask, result;
        
        cv::cvtColor(*frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(90, 50, 50), cv::Scalar(130, 255, 255), mask);
        cv::bitwise_and(*frame, *frame, result, mask);
        
        cv::imshow("Color Calibration", result);
    }, &frame);
    
    cv::createTrackbar("S_min", "Color Calibration", &s_min, 255, [](int, void* userdata) {
        cv::Mat* frame = static_cast<cv::Mat*>(userdata);
        cv::Mat hsv, mask, result;
        
        cv::cvtColor(*frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(90, 50, 50), cv::Scalar(130, 255, 255), mask);
        cv::bitwise_and(*frame, *frame, result, mask);
        
        cv::imshow("Color Calibration", result);
    }, &frame);
    
    cv::createTrackbar("S_max", "Color Calibration", &s_max, 255, [](int, void* userdata) {
        cv::Mat* frame = static_cast<cv::Mat*>(userdata);
        cv::Mat hsv, mask, result;
        
        cv::cvtColor(*frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(90, 50, 50), cv::Scalar(130, 255, 255), mask);
        cv::bitwise_and(*frame, *frame, result, mask);
        
        cv::imshow("Color Calibration", result);
    }, &frame);
    
    cv::createTrackbar("V_min", "Color Calibration", &v_min, 255, [](int, void* userdata) {
        cv::Mat* frame = static_cast<cv::Mat*>(userdata);
        cv::Mat hsv, mask, result;
        
        cv::cvtColor(*frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(90, 50, 50), cv::Scalar(130, 255, 255), mask);
        cv::bitwise_and(*frame, *frame, result, mask);
        
        cv::imshow("Color Calibration", result);
    }, &frame);
    
    cv::createTrackbar("V_max", "Color Calibration", &v_max, 255, [](int, void* userdata) {
        cv::Mat* frame = static_cast<cv::Mat*>(userdata);
        cv::Mat hsv, mask, result;
        
        cv::cvtColor(*frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(90, 50, 50), cv::Scalar(130, 255, 255), mask);
        cv::bitwise_and(*frame, *frame, result, mask);
        
        cv::imshow("Color Calibration", result);
    }, &frame);
    
    // 初始显示
    cv::Mat hsv, mask, result;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(90, 50, 50), cv::Scalar(130, 255, 255), mask);
    cv::bitwise_and(frame, frame, result, mask);
    cv::imshow("Color Calibration", result);
    
    std::cout << "调整滑动条校准颜色范围，按 's' 保存，按 'q' 退出" << std::endl;
    
    while (true) {
        int key = cv::waitKey(1) & 0xFF;
        if (key == 's') {
            lowerBlue = cv::Scalar(h_min, s_min, v_min);
            upperBlue = cv::Scalar(h_max, s_max, v_max);
            std::cout << "颜色范围已更新" << std::endl;
            break;
        } else if (key == 'q') {
            break;
        }
    }
    
    cv::destroyAllWindows();
}

// ... (其他函数保持不变 - plotTrajectory, savePositionsToFile)

void BallDetector::plotTrajectory(const std::vector<Position>& positions) {
    if (positions.empty()) return;
    
    // 创建轨迹图像
    int width = 800, height = 600;
    cv::Mat trajectoryImg(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // 找到坐标范围用于缩放
    double minX = positions[0].x, maxX = positions[0].x;
    double minY = positions[0].y, maxY = positions[0].y;
    
    for (const auto& pos : positions) {
        minX = std::min(minX, pos.x);
        maxX = std::max(maxX, pos.x);
        minY = std::min(minY, pos.y);
        maxY = std::max(maxY, pos.y);
    }
    
    // 添加边界
    double marginX = (maxX - minX) * 0.1;
    double marginY = (maxY - minY) * 0.1;
    minX -= marginX; maxX += marginX;
    minY -= marginY; maxY += marginY;
    
    // 坐标转换函数
    auto scaleX = [&](double x) { return static_cast<int>((x - minX) / (maxX - minX) * (width - 100) + 50); };
    auto scaleY = [&](double y) { return static_cast<int>(height - 50 - (y - minY) / (maxY - minY) * (height - 100)); };
    
    // 绘制轨迹
    for (size_t i = 0; i < positions.size(); i++) {
        int x = scaleX(positions[i].x);
        int y = scaleY(positions[i].y);
        cv::circle(trajectoryImg, cv::Point(x, y), 3, cv::Scalar(255, 0, 0), -1);
        
        if (i > 0) {
            int prevX = scaleX(positions[i-1].x);
            int prevY = scaleY(positions[i-1].y);
            cv::line(trajectoryImg, cv::Point(prevX, prevY), cv::Point(x, y), 
                    cv::Scalar(0, 0, 255), 1);
        }
    }
    
    // 添加坐标轴标签
    cv::putText(trajectoryImg, "X (px)", cv::Point(width/2, height-10), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(trajectoryImg, "Y (px)", cv::Point(10, height/2), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    cv::imshow("Trajectory", trajectoryImg);
    cv::waitKey(0);
}

void savePositionsToFile(const std::vector<Position>& positions, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "错误: 无法创建文件 " << filename << std::endl;
        return;
    }
    
    file << "time,x,y" << std::endl;
    for (const auto& pos : positions) {
        file << std::fixed << std::setprecision(6) << pos.time << "," 
             << pos.x << "," << pos.y << std::endl;
    }
    
    file.close();
    std::cout << "位置数据已保存到 " << filename << std::endl;
}