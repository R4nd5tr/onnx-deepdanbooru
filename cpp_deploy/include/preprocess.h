#pragma once
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>

constexpr int TARGET_HEIGHT = 512;
constexpr int TARGET_WIDTH = 512;
constexpr size_t TARGET_IMG_SIZE = TARGET_HEIGHT * TARGET_WIDTH * 3;

// area interpolation resize + border replicate padding to 512x512,
// normalize to [0,1], RGB NHWC Row Major (1, 512, 512, 3) float array
std::vector<float> preprocessImage(const std::filesystem::path& imagePath) {
    cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR_RGB);
    if (image.empty()) {
        throw std::runtime_error("Failed to read image: " + imagePath.string());
    };

    int imageWidth = image.cols;
    int imageHeight = image.rows;
    double scale = std::min(static_cast<double>(TARGET_WIDTH) / imageWidth, static_cast<double>(TARGET_HEIGHT) / imageHeight);
    double tx = (TARGET_WIDTH - imageWidth * scale) / 2.0;
    double ty = (TARGET_HEIGHT - imageHeight * scale) / 2.0;
    cv::Mat affine = (cv::Mat_<double>(2, 3) << scale, 0, tx, 0, scale, ty);
    cv::Mat transformed;
    cv::warpAffine(image, transformed, affine, cv::Size(TARGET_WIDTH, TARGET_HEIGHT), cv::INTER_AREA, cv::BORDER_REPLICATE);

    cv::Mat floatImage;
    transformed.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

    if (!floatImage.isContinuous()) floatImage = floatImage.clone();
    std::vector<float> result(floatImage.ptr<float>(), floatImage.ptr<float>() + TARGET_IMG_SIZE);

    return result;
}
