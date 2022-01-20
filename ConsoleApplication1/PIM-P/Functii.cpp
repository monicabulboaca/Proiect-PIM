#include "Header.h"
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

using namespace std;
using namespace cv;


vector<Ptr<Tracker>> trackers;
vector<Point> circlesToDraw;
vector<Scalar> coloursToDraw;
cv::Scalar COLORS[] = { cv::Scalar(60,60,180),cv::Scalar(0,128,128),cv::Scalar(145,25,180),cv::Scalar(19,239,200) , cv::Scalar(25,25,160), cv::Scalar(180,220,230) };



Mat drawTrajectory(Mat frame, vector<Rect> coords)
{
	size_t n = coords.size();
	int centerX, centerY;
	Mat newFrame = frame.clone();
	for (int i = 0; i < n; i++)
	{
		centerX = coords[i].tl().x + (coords[i].br().x - coords[i].tl().x) / 2;
		centerY = coords[i].tl().y + (coords[i].br().y - coords[i].tl().y) / 2;
		coloursToDraw.push_back(COLORS[i]);
		circlesToDraw.push_back(Point(centerX, centerY));
	}
	n = circlesToDraw.size();
	for (int i = 0; i < n; i++)
	{
		circle(newFrame, circlesToDraw[i], 1, coloursToDraw[i], FILLED);
	}
	return newFrame;
}


bool personAlreadyTracking(vector<Rect> crop_coord, Rect cropped_image_coords)
{
	size_t n = crop_coord.size();
	for (int i = 0; i < n; i++)
	{
		if (crop_coord[i].contains(cropped_image_coords.tl()) ||
			crop_coord[i].contains(cropped_image_coords.br()) ||
			cropped_image_coords.contains(crop_coord[i].tl()) ||
			cropped_image_coords.contains(crop_coord[i].br()) ||
			Rect(cropped_image_coords.tl().x - 15, cropped_image_coords.tl().y - 15, cropped_image_coords.br().x - cropped_image_coords.tl().x + 10, cropped_image_coords.br().y - cropped_image_coords.tl().y + 10).contains(crop_coord[i].tl()) ||
			Rect(cropped_image_coords.tl().x - 15, cropped_image_coords.tl().y - 15, cropped_image_coords.br().x - cropped_image_coords.tl().x + 10, cropped_image_coords.br().y - cropped_image_coords.tl().y + 10).contains(crop_coord[i].br()) ||
			Rect(crop_coord[i].tl().x - 15, crop_coord[i].tl().y - 15, crop_coord[i].br().x - crop_coord[i].tl().x + 10, crop_coord[i].br().y - crop_coord[i].tl().y + 10).contains(cropped_image_coords.tl()) ||
			Rect(crop_coord[i].tl().x - 15, crop_coord[i].tl().y - 15, crop_coord[i].br().x - crop_coord[i].tl().x + 10, crop_coord[i].br().y - crop_coord[i].tl().y + 10).contains(cropped_image_coords.br())
			)
			return true;
	}
	return false;
}


bool compareImages(Mat img1, Mat img2)
{
	Mat out1, out2;
	vector<KeyPoint> keypoints1;
	vector<KeyPoint> keypoints2;
	vector< vector<DMatch> > nn_matches;

	Ptr<SIFT>msr = SIFT::create();
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
		const float ratio_thresh = 0.7f;

		if (!nn_matches.empty())
		{
			for (size_t i = 0; i < nn_matches.size(); i++)
			{
				//if (nn_matches[i].size() == 2)
				//{
				if (nn_matches[i][0].distance < ratio_thresh * nn_matches[i][1].distance)
				{
					good_matches.push_back(nn_matches[i][0]);
				}
				//}
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


bool compareImagesVector(vector<Mat> imagesToCompareWith, Mat image)
{
	size_t n = imagesToCompareWith.size();
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

void nms(const std::vector<cv::Rect>& srcRects, std::vector<cv::Rect>& resRects, float thresh, int neighbors)
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
		{
			Rect xRect = rect1;
			xRect.x += cvRound(xRect.width * 0.1);
			xRect.width = cvRound(xRect.width * 0.9);
			xRect.y += cvRound(xRect.height * 0.07);
			xRect.height = cvRound(xRect.height * 0.8);
			resRects.push_back(xRect);
		}
	}
}

Mat tracking_person(vector<Rect> &cropped_coord, Mat frame, bool init)
{

	// imshow("Tracking", frame);
	Rect bbox;

	// bbox = selectROI(frame, false);

	bool ok;
	// rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);	// Display bounding box. 
	if (init)
	{
		trackers.push_back(TrackerCSRT::create());
		bbox = cropped_coord.back();
		trackers.back()->init(frame, bbox);
	}
	else
	{
		size_t n = trackers.size();
		for (int i = 0; i < n; i++)
		{
			ok = trackers[i]->update(frame, cropped_coord[i]);
			bbox = cropped_coord[i];
			if (ok)
			{
				//imagesToCompareWith.push_back(frame(Range(bbox.tl().y, bbox.br().y), Range(bbox.tl().x, bbox.br().x)));
				rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
			}
		}
	}

	return frame;
}

