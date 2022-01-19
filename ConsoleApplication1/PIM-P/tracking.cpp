
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/objdetect.hpp>
#include <stdio.h>
#include "Header.h"

using namespace cv;
using namespace std;


void tracking_person(vector<Rect> cropped_coord, Mat & new_frame)
{
	// Mat new_frame = frame;
	Ptr<Tracker> tracker = TrackerMIL::create();

	bool tracking = false;
	// imshow("Tracking", frame);
	Rect bbox;

	// bbox = selectROI(frame, false);

	bbox = cropped_coord[0];

	// rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);	// Display bounding box. 

	tracker->init(new_frame, bbox);
	tracking = true;

	if (tracking)
	{
		// Update the tracking result
		bool ok = tracker->update(new_frame, bbox);
	}
	else if (bbox.x >= 1 && bbox.y >= 1 && (bbox.x + bbox.width) < new_frame.cols && (bbox.y + bbox.height) < new_frame.rows)
	{
		tracking = false;
	}

	if (tracking)
	{
		// Tracking success : Draw the tracked object
		rectangle(new_frame, bbox, Scalar(255, 0, 0), 2, 1);
	}
	else
	{
		// Tracking failure detected.
		putText(new_frame, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
	}

	//return new_frame;
}
