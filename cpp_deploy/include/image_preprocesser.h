#pragma once
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>

// linear interpolation resize + border replicate padding to 512x512, normalize to [0,1], RGB NHWC Row Major float array
std::vector<float> preprocessImage(const std::filesystem::path& imagePath) {
    cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR_RGB); // channel = 3
    if (image.empty()) {
        throw std::runtime_error("Failed to read image: " + imagePath.string());
    };

    int imageWidth = image.cols;
    int imageHeight = image.rows;
    int targetHeight = 512;
    int targetWidth = 512;
    double scale = std::min(static_cast<double>(targetWidth) / imageWidth, static_cast<double>(targetHeight) / imageHeight);
    double tx = (targetWidth - imageWidth * scale) / 2.0;
    double ty = (targetHeight - imageHeight * scale) / 2.0;
    cv::Mat affine = (cv::Mat_<double>(2, 3) << scale, 0, tx, 0, scale, ty);
    cv::Mat transformed;
    cv::warpAffine(image, transformed, affine, cv::Size(targetWidth, targetHeight), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::imshow("Transformed Image", transformed);
    cv::waitKey(0);

    cv::Mat floatImage;
    transformed.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

    if (!floatImage.isContinuous()) {
        floatImage = floatImage.clone();
    }
    size_t numElem = floatImage.total() * floatImage.channels();
    if (numElem != 512 * 512 * 3) {
        throw std::runtime_error("Unexpected number of elements after preprocessing.");
    }
    std::vector<float> result(floatImage.ptr<float>(), floatImage.ptr<float>() + numElem);

    return std::move(result);
}
