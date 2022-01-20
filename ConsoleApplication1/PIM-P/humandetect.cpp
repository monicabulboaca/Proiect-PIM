#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <numeric>
#include <thread>
#include <direct.h>
#include <iostream>
#include <iomanip>
#include "Header.h"
using namespace cv;
using namespace std;


// bool sift_test(vector<Mat> frames, Mat frame_test)
// {
// 	// Read first frame 
// 	Mat output;
// 	Mat gray;
// 	vector<cv::KeyPoint> keypoints;

// 	// Create smart pointer for SIFT feature detector.
// 	Ptr<FastFeatureDetector> fp;
// 	fp = FastFeatureDetector::create();
// 	bool ok = true;
// 	for(vector<Mat>::iterator i = frames.begin(); i!= frames.end(); ++i)
// 	{
// 		Mat& frame = *i;
// 		if (frame.empty()) {
// 			cout << "FRAME EMPTY!\n";
// 			break;
// 		}
// 		cvtColor(frame, gray, COLOR_BGR2GRAY);

// 		// Detect the keypoints
// 		fp->detect(gray, keypoints);
// 		if(!keypoints.empty())
// 		{
// 			ok = false;
// 		}
// 		drawKeypoints(gray, keypoints, output);

// 		// imshow("Tracking", frame);
// 		imshow("Sift output", output);
// 		// Press  ESC on keyboard to exit
// 		char c = (char)waitKey(25);
// 		if (c == 27)
// 			break;
// 	}

// }

//https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/?_ga=2.230718431.1917472254.1638901971-814175245.1636552974


class Detector
{
	enum Mode { Default, Daimler } m;
	HOGDescriptor hog, hog_d;
public:

	Detector() : m(Default), hog(), hog_d(Size(48, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9)
	{
		hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
		hog_d.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());
	}

	void toggleMode() { m = (m == Default ? Daimler : Default); }
	string modeName() const { return (m == Default ? "Default" : "Daimler"); }

	vector<Rect> detect(InputArray img)
	{
		// Run the detector with default parameters. to get a higher hit-rate
		// (and more false alarms, respectively), decrease the hitThreshold and
		// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
		vector<Rect> found;
		if (m == Default)
			hog.detectMultiScale(img, found, 0, Size(4, 4), Size(4, 4), 1.4);
		else if (m == Daimler)
			hog_d.detectMultiScale(img, found, 0, Size(8, 8), Size(10, 10), 1.05, 2, true);
		return found;
	}

	void adjustRect(Rect& r) const
	{
		// The HOG detector returns slightly larger rectangles than the real objects,
		// so we slightly shrink the rectangles to get a nicer output.
		r.x += cvRound(r.width * 0.09);
		r.width = cvRound(r.width * 0.6);
		r.y += cvRound(r.height * 0.07);
		r.height = cvRound(r.height * 0.8);
	}
};


void humandetect(const char* path,const char* outFileName)
{
	Detector detector;
	Mat draw;

	int k = 0;
	// Read video
	//VideoCapture video("datasets/OneLeaveShopReenter1cor.mpg");
	//VideoCapture video("datasets/TwoLeaveShop2cor.mpg");
	//VideoCapture video("datasets/ThreePastShop2cor.mpg");
	//VideoCapture video("datasets/WalkByShop1cor.mpg");
	VideoCapture video(path);
	
	// Exit if video is not opened
	if (!video.isOpened())
	{
		cout << "Could not read video file" << endl;
		return;
	}
	// vector cu imaginile decupate
	vector<Mat> crop;
	Mat frame;
	vector<Rect> found2;
	vector<Rect> crop_coord;
	vector<Mat> imagesOfTracking;
	VideoWriter outputVideo;
	Mat frameOriginal;
	int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
	for (k = 0; k < 90; k++)
	{
		video.read(frame);
	}
	outputVideo.open("WalkByShop1corOutput.avi", codec, 25, frame.size(),frame.type());
	if (!outputVideo.isOpened()) {
		cerr << "Could not open the output video file for write\n";
		return;
	}
	for (;;)
	{
		k += 1;

		// frame = imread(c, IMREAD_GRAYSCALE);
		video.read(frame);
		//applyColorMap(frame, frame, COLORMAP_HSV);
		//threshold(frame, frame, 0, 255, THRESH_BINARY + THRESH_OTSU);
		//cvtColor(frame, frame, COLOR_RGB2GRAY);
		//equalizeHist(frame, frame);
		if (frame.empty())
		{
			cout << "Finished reading: empty frame" << endl;
			break;
		}
		frameOriginal = frame.clone();
		convertScaleAbs(frame, frame, 1.3, -50);
		Mat cropped_image(frame.size(), CV_8UC4, Scalar(0, 0, 0, 0));
		int64 t = getTickCount();
		Mat updatedFrame;
		{
			if (k % 5 == 0)
			{
				found2 = detector.detect(frame);	// vector cu bbox-ul persoanelor detectate
			}
			else
			{
				found2.clear();
			}
			if (!found2.empty())
			{
				vector<Rect> found(found2.capacity() + 1);
				nms(found2, found, (float)0.3,0);	// aplicare non maximum suppression
				for (vector<Rect>::iterator i = found.begin(); i != found.end(); ++i)
				{
					Rect& r = *i;
					detector.adjustRect(r);
					rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);	// VERDE -> detectare persoana
					//circle(draw, Point((r.tl().x + r.br().x) / 2, (r.tl().y + r.br().y) / 2), 3, cv::Scalar(255, 0, 255, 255), FILLED);

					// crop image
					cropped_image = frame(Range(r.tl().y, r.br().y), Range(r.tl().x, r.br().x));
					//if (compareImagesVector(imagesOfTracking, cropped_image) == false)
					//{
					if (personAlreadyTracking(crop_coord, r) == false)
					{
						crop.push_back(cropped_image);
						//imagesOfTracking.push_back(cropped_image);
						crop_coord.push_back(r);	// get cropped image's coordinates
						tracking_person(crop_coord, frame, true);
					}
					//}
						// add cropped image to vector
						//crop.push_back(cropped_image);
				}
			}

			updatedFrame = tracking_person(crop_coord, frame, false);
			outputVideo.write(drawTrajectory(frameOriginal, crop_coord));
			//updatedFrame = tracking_person(crop_coord, frame);
			// if (ok)	// s-a detectat persoana
			// {
			// 	// verific in vectorul cu sift
			// 	sift_test(crop);
			// }

			// if (ok)	// am gasit macar o persoana in frame ul curent
			// {
			// 	sprintf_s(img_detected, FILENAME_MAX - 1, "%s\\img_detected%d.jpg", path_out, k);
			// 	sprintf_s(img_circles, FILENAME_MAX - 1, "%s\\img_circles%d.png", path_out, k);
			// 	sprintf_s(img_cropped, FILENAME_MAX - 1, "%s\\img_cropped%d.png", path_crop_img, k);
			// 	bool result_detect = false, result_circles = false, result_crop = false;
			// 	try
			// 	{
			// 		result_detect = imwrite(img_detected, frame);
			// 		result_circles = imwrite(img_circles, draw);
			// 		result_crop = imwrite(img_cropped, cropped_image);
			// 	}
			// 	catch (const cv::Exception& ex)
			// 	{
			// 		fprintf_s(stderr, "Exception converting image to PNG format: %s\n", ex.what());
			// 	}
			// 	if (!result_detect || !result_circles || !result_crop)
			// 		exit(EXIT_FAILURE);
			// }
		}




		t = getTickCount() - t;
		// show the window
		{
			ostringstream buf;
			buf << "FPS: " << fixed << setprecision(1) << (getTickFrequency() / (double)t)
				<< "\nFRAME: " << k;
			putText(frame, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 2, LINE_AA);
		}
		imshow("People detector", updatedFrame);

		// interact with user
		const char key = (char)waitKey(1);
		if (key == 27 || key == 'q') // ESC
		{
			cout << "Exit requested" << endl;
			break;
		}
	}
}

