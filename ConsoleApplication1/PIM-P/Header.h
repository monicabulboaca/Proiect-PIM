#pragma once
#include <opencv2/core.hpp>
#include <vector>


void humandetect(const char* path, const char* outFileName);
void sift_img_test();
void bgSegm();
void camShift();
void histCalc();

cv::Mat drawTrajectory(cv::Mat frame, std::vector<cv::Rect> coords);
cv::Mat tracking_person(std::vector<cv::Rect> &cropped_coord, cv::Mat frame, bool init);
void nms(const std::vector<cv::Rect>& srcRects, std::vector<cv::Rect>& resRects, float thresh, int neighbors);
bool compareImagesVector(std::vector<cv::Mat> imagesToCompareWith, cv::Mat image);
bool compareImages(cv::Mat img1, cv::Mat img2);
bool personAlreadyTracking(std::vector<cv::Rect> crop_coord, cv::Rect cropped_image_coords);