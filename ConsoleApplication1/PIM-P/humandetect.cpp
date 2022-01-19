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
#include "Header.h"
using namespace cv;
using namespace std;

vector<Ptr<Tracker>> trackers;
Mat tracking_person(vector<Rect> cropped_coord, Mat& frame,vector<Mat> &imagesToCompareWith, bool init)
{

	// imshow("Tracking", frame);
	Rect bbox;

	// bbox = selectROI(frame, false);


	// rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);	// Display bounding box. 
	if (init)
	{
		trackers.push_back(TrackerMIL::create());
		bbox = cropped_coord.back();
		trackers.back()->init(frame, bbox);
	}
	else
	{
		int n = trackers.size();
		for (int i = 0; i < n; i++)
		{
			bbox = cropped_coord[i];
			trackers[i]->update(frame, bbox);
			imagesToCompareWith.push_back(frame(Range(bbox.tl().y, bbox.br().y), Range(bbox.tl().x, bbox.br().x)));
			rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
		}
	}

	return frame;
}

bool compareImages(Mat img1, Mat img2)
{
	Mat out1, out2;
	vector<KeyPoint> keypoints1;
	vector<KeyPoint> keypoints2;
	vector< vector<DMatch> > nn_matches;

	Ptr<BRISK>msr = BRISK::create();
	//Ptr<AKAZE>akz = AKAZE::create();

	if (img1.cols < 100)
	{
		resize(img1, img1, Size(img1.cols * 2, img1.rows * 2));
	}

	if (img2.cols < 100)
	{
		resize(img2, img2, Size(img2.cols * 2, img2.rows * 2));
	}

	//Ptr<AgastFeatureDetector> afp;
	//Ptr<GFTTDetector> gfd;
	//gfd = GFTTDetector::create();
	//afp = AgastFeatureDetector::create();

	//fp = FastFeatureDetector::create();
	//afd = AffineFeatureDetector::create(fp);
	//ade = AffineDescriptorExtractor::create(fp);
	//afp->detectAndCompute(img1, noArray(), keypoints1, out1);
	//afp->detectAndCompute(img2, noArray(), keypoints2, out2);

	msr->detect(img1, keypoints1);
	msr->detect(img2, keypoints2);
	msr->compute(img1, keypoints1, out1);
	msr->compute(img2, keypoints2, out2);

	//afp->detect(img1,keypoints1);
	//afp->detect(img2,keypoints2);

	//afp->compute(img1, keypoints1, out1);
	//afp->compute(img2, keypoints2, out2);



	//fp->compute(img1, keypoints1, out1);
	//fp->compute(img2, keypoints2, out2);
	if (!out1.empty() && !out2.empty())
	{
		std::vector<DMatch> good_matches;
		//Ptr<BFMatcher> bf = BFMatcher::create(NORM_L2);
		BFMatcher bf;
		bf.knnMatch(out1, out2, nn_matches, 2);
		//bf.match(out1, out2, good_matches);

		//-- Filter matches using the Lowe's ratio test
		const float ratio_thresh = 0.9f;

		if (!nn_matches.empty())
		{
			for (size_t i = 0; i < nn_matches.size(); i++)
			{
				if (nn_matches[i].size() == 2)
				{
					if (nn_matches[i][0].distance < ratio_thresh * nn_matches[i][1].distance)
					{
						good_matches.push_back(nn_matches[i][0]);
					}
				}
			}
		}


		// Mat im3;
		// drawMatches(img1, keypoints1, img2, keypoints2, good_matches, im3, -1, -1, vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		// drawKeypoints(img1, keypoints1, img1);
		// drawKeypoints(img2, keypoints2, img2);
		// imshow("im1", img1);
		// imshow("im2", img2);
		// imshow("im3", im3);
		// imwrite("imaginematch.jpg", im3);
		// waitKey(0);
		if (!good_matches.empty())
		{
			return true;
		}
		else return false;
	}
	else
	{
		return false;
	}

}

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
		r.x += cvRound(r.width * 0.1);
		r.width = cvRound(r.width * 0.8);
		r.y += cvRound(r.height * 0.07);
		r.height = cvRound(r.height * 0.8);
	}
};


//https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/?_ga=2.230718431.1917472254.1638901971-814175245.1636552974

inline void nms(const vector<cv::Rect>& srcRects, vector<cv::Rect>& resRects, float thresh, int neighbors = 0)
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

bool compareImagesVector(vector<Mat> imagesToCompareWith, Mat image)
{
	int n = imagesToCompareWith.size();
	int i;
	if (imagesToCompareWith.empty())
	{
		return false;
	}
	else
	{
		for (i = 0; i < n; i++)
		{
			if (compareImages(imagesToCompareWith.at(i), image))
				break;
		}
		if (i == n)
			return false;
		return true;
	}
}

void humandetect()
{
	Detector detector;
	Mat draw;

	int k = 0;
	char c[FILENAME_MAX], path[FILENAME_MAX], path_out[FILENAME_MAX], path_crop_img[FILENAME_MAX];
	char img_detected[FILENAME_MAX], img_circles[FILENAME_MAX], img_cropped[FILENAME_MAX];
	bool gray_img = true;

	_getcwd(path, FILENAME_MAX - 1);

	strcpy_s(path_out, path);
	strcpy_s(path_crop_img, path);

	strcat_s(path_out, "\\test2");
	strcat_s(path_crop_img, "\\cropped_img");

	// Read video
	VideoCapture video("datasets/OneLeaveShopReenter1cor.mpg");

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
	for (;;)
	{
		k += 1;

		// frame = imread(c, IMREAD_GRAYSCALE);

		bool ok = video.read(frame);
		// cvtColor(frame, frame, COLOR_BGR2GRAY);
		if (frame.empty())
		{
			ok = false;
			cout << "Finished reading: empty frame" << endl;
			break;
		}
		Mat cropped_image(frame.size(), CV_8UC4, Scalar(0, 0, 0, 0));
		Mat draw(frame.size(), CV_8UC4, Scalar(0, 0, 0, 0));
		int64 t = getTickCount();
		Mat updatedFrame;
		{
			bool ok = false;	// flag pt persoana detectata in frame
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
				nms(found2, found, 0.3);	// aplicare non maximum suppression
				for (vector<Rect>::iterator i = found.begin(); i != found.end(); ++i)
				{
					ok = true;
					Rect& r = *i;
					detector.adjustRect(r);
					rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);	// VERDE -> detectare persoana
					circle(draw, Point((r.tl().x + r.br().x) / 2, (r.tl().y + r.br().y) / 2), 3, cv::Scalar(255, 0, 255, 255), FILLED);

					// crop image
					cropped_image = frame(Range(r.tl().y, r.br().y), Range(r.tl().x, r.br().x));
					if (compareImagesVector(imagesOfTracking, cropped_image) == false)
					{
						crop.push_back(cropped_image);
						//imagesOfTracking.push_back(cropped_image);
						crop_coord.push_back(r);	// get cropped image's coordinates
						tracking_person(crop_coord, frame,imagesOfTracking, true);
					}
					// add cropped image to vector
					//crop.push_back(cropped_image);
				}
			}

			updatedFrame = tracking_person(crop_coord, frame,imagesOfTracking, false);
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
		else if (key == 't')
		{
			gray_img = !gray_img;
		}
	}
}

