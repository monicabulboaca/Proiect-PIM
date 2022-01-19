
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <direct.h>
#include "Header.h"
using namespace cv;
using namespace std;

void sift_test(vector<Mat> frame)
{
    // Read first frame

    Mat output;
    Mat gray;
    vector<cv::KeyPoint> keypoints;

    // Create smart pointer for SIFT feature detector.
    Ptr<FastFeatureDetector> fp;
    fp = FastFeatureDetector::create();
    //Mat frame;
    while (!frame.empty())
    {
       cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect the keypoints
        fp->detect(gray, keypoints);

        drawKeypoints(gray, keypoints, output);

        // imshow("Tracking", frame);
        imshow("Sift output", output);
        // Press  ESC on keyboard to exit
           char c = (char)waitKey(25);
           if (c == 27)
                break;
    }
    if (frame.empty()) {
        cout << "FRAME EMPTY!\n";
    }

}

void sift_img_test()
{
    Mat img1 = imread("datasets/TwoLeaveShop2cor.tar/TwoLeaveShop2cor0257.jpg");//, IMREAD_GRAYSCALE);
    Mat img2 = imread("datasets/TwoLeaveShop2cor.tar/TwoLeaveShop2cor0337.jpg");//, IMREAD_GRAYSCALE);
    Mat out1, out2;
    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    vector< vector<DMatch> > nn_matches;

    Ptr<BRISK>msr = BRISK::create();
    Ptr<AKAZE>akz = AKAZE::create();
	Ptr<SIFT>sift = SIFT::create();
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
/*
    msr->detect(img1,keypoints1);
    msr->detect(img2,keypoints2);
    msr->compute(img1, keypoints1, out1);
    msr->compute(img2, keypoints2, out2);
*/
	sift->detect(img1,keypoints1);
    sift->detect(img2,keypoints2);
    sift->compute(img1, keypoints1, out1);
    sift->compute(img2, keypoints2, out2);

    //afp->detect(img1,keypoints1);
    //afp->detect(img2,keypoints2);

    //afp->compute(img1, keypoints1, out1);
    //afp->compute(img2, keypoints2, out2);



    //fp->compute(img1, keypoints1, out1);
    //fp->compute(img2, keypoints2, out2);
    if (!out1.empty() && !out2.empty())
    {
        std::vector<DMatch> good_matches;
        BFMatcher bf;
        bf.knnMatch(out1, out2, nn_matches, 2);
        //bf.match(out1, out2, good_matches);

        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.7f;

        if (nn_matches.size())
        {
            for (size_t i = 0; i < nn_matches.size(); i++)
            {
                if (nn_matches[i][0].distance < ratio_thresh * nn_matches[i][1].distance)
                {
                    good_matches.push_back(nn_matches[i][0]);
                }
            }
        }


        Mat im3;
        drawMatches(img1, keypoints1, img2, keypoints2, good_matches, im3, -1, -1, vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        drawKeypoints(img1, keypoints1, img1);
        drawKeypoints(img2, keypoints2, img2);
        imshow("im1", img1);
        imshow("im2", img2);
        imshow("im3", im3);
        imwrite("imaginematch.jpg", im3);
        waitKey(0);
    }
    else
    {
        cout << "\n\n\n\n\nnu\n";
        return;
    }

}
