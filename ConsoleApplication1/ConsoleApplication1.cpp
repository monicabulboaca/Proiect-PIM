// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <numeric>
#include <thread>
#include <direct.h>
#include <iostream>
#include <iomanip>
using namespace cv;
using namespace std;
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
			hog.detectMultiScale(img, found, 0, Size(4, 4), Size(4,4),1.4);
		else if (m == Daimler)
			hog_d.detectMultiScale(img, found, 0, Size(8, 8), Size(10,10), 1.05, 2, true);
		return found;
	}
	void adjustRect(Rect & r) const
	{
		// The HOG detector returns slightly larger rectangles than the real objects,
		// so we slightly shrink the rectangles to get a nicer output.
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
	}
};



//https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/?_ga=2.230718431.1917472254.1638901971-814175245.1636552974
//Non-Maximum Suppresion - remove bounding boxes
//vector<Rect> nms(vector<Rect> v, float overlapThresh)
//{
//	if (v.empty)
//	{
//		return v;
//	}
//	vector<double> x1,x2,y1,y2,area,pick;
//	vector<int> indx;
//	for (vector<Rect>::iterator i = v.begin(); i != v.end(); ++i)
//	{
//		Rect r = *i;
//		x1.push_back(r.tl().x);
//		x2.push_back(r.br().x);
//		y1.push_back(r.tl().y);
//		y2.push_back(r.br().y);
//		area.push_back((x2.back() - x1.back() + 1)*(y2.back() - y1.back() + 1));
//	}
//	iota(indx.begin(), indx.end(), 0);
//	sort(indx.begin(), indx.end(), [&](int i, int j) {return y2.at(i) < y2.at(j); });
//	while (!indx.empty())
//	{
//		int last = indx.size();
//		int i = indx.at(last);
//		pick.push_back(i);
//		double xx1 = max_element(x1, x1.capacity());
//	}
//}

inline void nms(const vector<cv::Rect>& srcRects,vector<cv::Rect>& resRects,float thresh,int neighbors = 0)
{
	resRects.clear();

	const size_t size = srcRects.size();
	if (!size)
		return;

	// Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
	multimap<int, size_t> idxs;
	for (size_t i = 0; i < size; ++i)
	{
		idxs.emplace(srcRects[i].br().y, i);
	}

	// keep looping while some indexes still remain in the indexes list
	while (idxs.size() > 0)
	{
		// grab the last rectangle
		auto lastElem = --end(idxs);
		const cv::Rect& rect1 = srcRects[lastElem->second];

		int neigborsCount = 0;

		idxs.erase(lastElem);

		for (auto pos = std::begin(idxs); pos != end(idxs); )
		{
			// grab the current rectangle
			const cv::Rect& rect2 = srcRects[pos->second];

			float intArea = static_cast<float>((rect1 & rect2).area());
			float unionArea = rect1.area() + rect2.area() - intArea;
			float overlap = intArea / unionArea;

			// if there is sufficient overlap, suppress the current bounding box
			if (overlap > thresh)
			{
				pos = idxs.erase(pos);
				++neigborsCount;
			}
			else
			{
				++pos;
			}
		}
		if (neigborsCount >= neighbors)
			resRects.push_back(rect1);
	}
}

int main(int argc, char** argv)
{
	//string file2 = "D:\\#FACULTATE\\PIM-P\\datasets\\WalkByShop1cor.mpg";
	//VideoCapture cap;
	//if (file.empty())
	//	cap.open(camera);
	//else
	//	file = samples::findFileOrKeep(file);
	/*cap.open(file2);
	if (!cap.isOpened())
	{
		cout << "Can not open video stream: '" << (file2.empty() ? "<camera>" : file2) << "'" << endl;
		return 2;
	}
	cout << "Press 'q' or <ESC> to quit." << endl;
	cout << "Press <space> to toggle between Default and Daimler detector" << endl;*/
	Detector detector;
	Mat frame;
	Mat frame_orig;
	Mat draw;
	int k = 200;
	char c[FILENAME_MAX],path[FILENAME_MAX], img_detected[FILENAME_MAX],img_circles[FILENAME_MAX],path_out[FILENAME_MAX];
	bool gray_img = true;
	_getcwd(path, FILENAME_MAX-1);
	strcpy_s(path_out, path);
	strcat_s(path_out, "\\test2");
	cout << path;

	for (;;)
	{
		k+=5;
		sprintf_s(c, FILENAME_MAX-1, "%s\\datasets\\TwoLeaveShop2cor.tar\\TwoLeaveShop2cor%04d.jpg", path,k);
		//sprintf_s(c, FILENAME_MAX-1, "D:\\#FACULTATE\\PIM-P\\datasets\\WalkByShop1cor.tar\\WalkByShop1cor%04d.jpg", k);
		frame = imread(c, IMREAD_GRAYSCALE);
		if (frame.empty())
		{
			cout << "Finished reading: empty frame" << endl;
			break;
		}
		Mat draw(frame.size(), CV_8UC4, Scalar(0, 0, 0, 0));
		int64 t = getTickCount();
		
		{
			bool ok = false;
			vector<Rect> found2 = detector.detect(frame);
			vector<Rect> found(found2.capacity()+1);
			nms(found2, found, 0.3);
			for (vector<Rect>::iterator i = found.begin(); i != found.end(); ++i)
			{
				ok = true;
				Rect &r = *i;
				detector.adjustRect(r);
				rectangle(frame, r.tl(),r.br(), cv::Scalar(0, 255, 0), 2);
				circle(draw, Point((r.tl().x + r.br().x)/ 2, (r.tl().y + r.br().y) / 2),3, cv::Scalar(255,0, 255,255),FILLED);
			}
			if (ok)
			{
			sprintf_s(img_detected,FILENAME_MAX-1, "%s\\img_detected%d.jpg",path_out,k);
			sprintf_s(img_circles,FILENAME_MAX-1, "%s\\img_circles%d.png",path_out,k);
			bool result_detect = false,result_circles = false;
			try
			{
				result_detect = imwrite(img_detected, frame);
				result_circles = imwrite(img_circles, draw);
			}
			catch (const cv::Exception& ex)
			{
				fprintf_s(stderr, "Exception converting image to PNG format: %s\n", ex.what());
			}
			if(!result_detect || !result_circles)
				exit(EXIT_FAILURE);
			}
		}
		t = getTickCount() - t;
		// show the window
		{
			ostringstream buf;
			buf << "Mode: " << detector.modeName() << " ||| "
				<< "FPS: " << fixed << setprecision(1) << (getTickFrequency() / (double)t)
				<< "\nFRAME: " << k;
			putText(frame, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 2, LINE_AA);
		}
				imshow("People detector", frame);
		
		// interact with user
		const char key = (char)waitKey(1);
		if (key == 27 || key == 'q') // ESC
		{
			cout << "Exit requested" << endl;
			break;
		}
		else if (key == ' ')
		{
			detector.toggleMode();
		}
		else if (key == 't')
		{
			gray_img = !gray_img;
		}
	}
	return 0;
}