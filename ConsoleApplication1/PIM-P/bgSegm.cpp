#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/objdetect.hpp>
#include <stdio.h>
#include "Header.h"

using namespace cv;
using namespace std;

void bgSegm()
{
	Ptr<BackgroundSubtractor> pBackSub;
	//pBackSub = createBackgroundSubtractorMOG2();
	pBackSub = createBackgroundSubtractorKNN();
	VideoCapture capture("datasets/OneLeaveShopReenter1cor.mpg");
	if (!capture.isOpened()) {
		//error in opening the video input
		cerr << "Unable to open:" << endl;
		return;
	}
	VideoWriter output;
	int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
	Mat frame, fgMask;
	while (true) {
		capture >> frame;
		if (frame.empty())
			break;
		//update the background model
		pBackSub->apply(frame, fgMask);
		if (!output.isOpened())
			output.open("fgMask.avi", codec, 25, frame.size(), frame.type());
		//get the frame number and write it on the current frame
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		stringstream ss;
		ss << capture.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		//show the current frame and the fg masks
		Mat element = getStructuringElement(MorphShapes::MORPH_ELLIPSE,Size(5,5));
		//Mat neww;
		//medianBlur(fgMask, fgMask, 3);
		//erode(fgMask, neww, element);
		//erode(frame, frame, element);
		dilate(fgMask, fgMask, element);
		morphologyEx(fgMask, fgMask, MorphTypes::MORPH_OPEN, element);
		frame.setTo(0, fgMask == 0);
		imshow("Frame", frame);
		imshow("FG Mask", fgMask);
		//imshow("neww", neww);
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
			break;
		output.write(frame);
	}
}
