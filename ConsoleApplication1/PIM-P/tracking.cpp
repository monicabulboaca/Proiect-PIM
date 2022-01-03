#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/objdetect.hpp>


using namespace cv;
using namespace std;


void tracking_test()
{
    Ptr<Tracker> tracker = TrackerMIL::create();
    // Read video
    VideoCapture video("datasets/OneLeaveShopReenter2cor.mpg");

    // Exit if video is not opened
    if (!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return;
    }

    // Read first frame 
    Mat frame;
    bool ok = video.read(frame);
    bool tracking = false;
    imshow("Tracking", frame);
    int key = waitKey(2500);
    Rect bbox;
    if (key == 'y')
    {
        bbox = Rect(287, 23, 86, 320);
        // Uncomment the line below to select a different bounding box 
        bbox = selectROI(frame, false);
        // Display bounding box. 
        rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);

        imshow("Tracking", frame);
        tracker->init(frame, bbox);
        tracking = true;
    }

    // Define initial bounding box 
    

    while (video.read(frame))
    {
        // Start timer
        //double timer = (double)getTickCount();
        resize(frame, frame, Size(800 / frame.cols * frame.cols, 600 / frame.rows * frame.rows));
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        if (tracking)
        {   
            // Update the tracking result
            bool ok = tracker->update(frame, bbox);
        }
        else if (bbox.x >= 1 && bbox.y >= 1 && (bbox.x + bbox.width) < frame.cols && (bbox.y + bbox.height) < frame.rows)
        {
            tracking = false;
        }

        // Calculate Frames per second (FPS)
        //float fps = getTickFrequency() / ((double)getTickCount() - timer);
        if (tracking)
        {
            if (ok)
            {
                // Tracking success : Draw the tracked object
                rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
            }
            else
            {
                // Tracking failure detected.
                putText(frame, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
            }
        }

        //ostringstream buf;
        //buf << "FPS:" << fps;
        // Display FPS on frame
        //putText(frame, buf.str(), Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

        // Display frame.
        imshow("Tracking", frame);

        // Exit if ESC pressed.
        int q = waitKey(1);
        if (q == 27)
        {
            break;
        }
        if (q == ' ' && tracking == false)
        {
            // Uncomment the line below to select a different bounding box 
            bbox = selectROI(frame, false);
            // Display bounding box. 
            rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);

            imshow("Tracking", frame);
            tracker->init(frame, bbox);
            tracking = true;
        }
    }
}