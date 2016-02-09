/*
Name       : Ravi Kant
e-mail     : rkant@usc.edu	
Submission : Jan 29, 2016

 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <time.h>

using namespace cv;
using namespace std;


int main()
{


	Mat refImg1 = imread("refrence.jpg");
	Mat refImg;

	cvtColor(refImg1,refImg,CV_BGR2GRAY);

	SiftFeatureDetector detector;
	vector<KeyPoint> refKeyPoints,sceneKeyPoints;

	detector.detect(refImg,refKeyPoints);
	SiftDescriptorExtractor extractor;
	Mat refKeyPoint_descriptors, sceneKeyPointDescriptor;
	extractor.compute(refImg, refKeyPoints, refKeyPoint_descriptors);
	BFMatcher matcher(NORM_L2,true);

	VideoCapture cap("12");
	if(!cap.isOpened())  // check if we succeeded
		return -1;
	VideoWriter outcap;
	outcap.open("resultVideo.avi", CV_FOURCC('D','I','V','X'), cap.get(CV_CAP_PROP_FPS), Size(0.3*cap.get(CV_CAP_PROP_FRAME_WIDTH),0.3*cap.get(CV_CAP_PROP_FRAME_HEIGHT)));
	bool flag = false;
	double area;
	clock_t tStart = clock();
	int nFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
	Mat homography,oldHomography;
	vector<Point2f> oldCorners,destinationCorners(4);

	//ofstream fout;
	//fout.open("angles.txt");
	float data[9]= {1.0495656184032528e+03, 0.0, 6.3950000000000000e+02,
			0.0, 1.0495656184032528e+03, 3.5950000000000000e+02,
			0.0, 0.0, 1.0};
	Mat cameraMatrix(3,3,CV_32F,data);


	double dCoeff[5] = {0.058055443802597889, -0.75762481381245350, 0.0, 0.0,1.2767832543908491};
	vector<double> distCoeffs(&dCoeff[0], &dCoeff[0]+5);

	Mat rTemp,tTemp, rVec, tVec;
	Mat_<float> rotMat(3,3);
	vector<Point3f> objectPoints(4);
	objectPoints[0] = Point3f(-7.5,5.5,0);
	objectPoints[1] = Point3f(7.5,5.5,0);
	objectPoints[2] = Point3f(7.5,-5.5,0);
	objectPoints[3] = Point3f(-7.5,-5.5,0);

	vector<Point3f> objPts(3);
	objPts[0] = Point3f(0,0,0);
	objPts[1] = Point3f(0,5.5,0);
	objPts[2] = Point3f(5.5,0,0);
	Mat imagePointsRP;



	Mat nrml = Mat(objectPoints[0]).cross(Mat(objectPoints[1]));
	float daTA[4] = {nrml.at<float>(0,0), nrml.at<float>(1,0), nrml.at<float>(2,0), 1};
	Mat nrML(4,1,CV_32F,daTA);
	Mat RT;
	for(int h = 0; h < nFrames; h++)
	{
		Mat sceneImg1;
		cap>>sceneImg1;
		Mat sceneImg2,sceneImg;
		resize(sceneImg1, sceneImg2,Size(),0.3,0.3,CV_INTER_AREA);


		vector<Mat> channelImages;
		split(sceneImg2, channelImages);
		Mat empty_image = Mat::zeros(sceneImg2.rows,sceneImg2.cols,CV_8UC1);
		Mat red_image(sceneImg2.rows,sceneImg2.cols,CV_8UC3);


		Mat tempMat;
		//adaptiveThreshold(channelImages[0], tempMat, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
		imshow("orignal green",channelImages[1]);
		waitKey(0);
		/*	adaptiveThreshold(channelImages[1], tempMat, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 3);
		imshow("Adaptive",tempMat);
		waitKey(0);*/
		/*float da[9] = {-1,-1,-1,-1,9,-1,-1,-1,-1};
			Mat kerner(3,3,CV_32F,da);
			filter2D(channelImages[1],tempMat, CV_8U, kerner, Point(-1,-1), 0, BORDER_DEFAULT );
			imshow("filter",tempMat);
					waitKey(0);
		 */

		Canny( channelImages[1], tempMat, 100, 100*2, 3 );
		namedWindow("Canny",WINDOW_NORMAL);
		imshow("Canny",tempMat);
		waitKey(0);

		dilate(tempMat, tempMat, Mat(), Point(-1,-1), 1);
				imshow( "dilate", tempMat );
						waitKey(0);
		erode(tempMat, tempMat, Mat(), Point(-1,-1), 1);
		imshow( "Erode", tempMat );
								waitKey(0);
		/*		GaussianBlur(channelImages[1], tempMat, Size(3,3) , 0, 0,BORDER_DEFAULT );
		threshold( tempMat, tempMat, 0, 255,THRESH_BINARY|THRESH_OTSU );
		imshow("MAnual",tempMat);
				waitKey(0);*/
		//	adaptiveThreshold(channelImages[2], tempMat, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);

		//	imshow("red",tempMat);
		//	waitKey(0);
		//


		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours( tempMat, contours, hierarchy, RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

		double maxArea = contourArea(contours[0]);
		int index;
		vector<Point> tempCount;
		for( int i = 1; i< contours.size(); i++ )
		{
			approxPolyDP(contours[i], tempCount, 9, true);
			if(contourArea(tempCount)>maxArea){
				index = i;
				maxArea = contourArea(tempCount);
			}
		}
		Mat drawing = Mat::zeros( tempMat.size(), CV_8UC3 );
		drawContours( drawing, (contours), index, Scalar(0,0,255), 2 );

	/*	for( int i = 0; i< contours.size(); i++ )
		{
			drawContours( drawing, Mat(contours_poly[i]), i, Scalar(0,0,255), 2, 8,  vector<Vec4i>(), 0, Point() );
		}*/
		namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
		imshow( "Contours", drawing );
		waitKey(0);
		morphologyEx(drawing, drawing, MORPH_OPEN, Mat(), Point(-1,-1), 1 );
		imshow( "Morph", drawing );
				waitKey(0);
		cvtColor(sceneImg2,sceneImg,CV_BGR2GRAY);

		/*	namedWindow( "img", WINDOW_NORMAL );
	    		imshow("img",sceneImg);
	    		waitKey(0);*/

		detector.detect(sceneImg,sceneKeyPoints);

		extractor.compute(sceneImg, sceneKeyPoints, sceneKeyPointDescriptor);
		vector<DMatch> matchingPairs;

		matcher.match(refKeyPoint_descriptors, sceneKeyPointDescriptor, matchingPairs);

		float temp;
		float minDist = matchingPairs[0].distance, maxDist = matchingPairs[0].distance;
		for(int i = 1; i < matchingPairs.size(); i++) {
			temp = matchingPairs[i].distance;
			if(temp < minDist)
				minDist = temp;
			if(temp > maxDist)
				maxDist = temp;
		}

		vector<DMatch> goodMatches;
		vector<Point2f> matched_objectKeyPoint, matched_sceneKeyPoint;

		temp = minDist + 0.6 * (maxDist - minDist);
		for(int i = 0; i < matchingPairs.size(); i++){
			if(matchingPairs[i].distance < temp ) {
				goodMatches.push_back(matchingPairs[i]);
				matched_objectKeyPoint.push_back(refKeyPoints[matchingPairs[i].queryIdx].pt);
				matched_sceneKeyPoint.push_back(sceneKeyPoints[matchingPairs[i].trainIdx].pt);
			}
		}

		Mat matchedImage;
		drawMatches(refImg,refKeyPoints,sceneImg,sceneKeyPoints,goodMatches,matchedImage,Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
		/*		namedWindow("Result", WINDOW_NORMAL);
	    		imshow("Result",matchedImage);
	    		waitKey(0);*/

		if(matched_objectKeyPoint.size()>20){
			homography = findHomography( matched_objectKeyPoint, matched_sceneKeyPoint, CV_RANSAC);
		}
		vector<Point2f> refCorners(4);
		refCorners[0] = Point(0,0);
		refCorners[1] = Point(refImg.cols,0);
		refCorners[2] = Point(refImg.cols,refImg.rows);
		refCorners[3] = Point(0,refImg.rows);

		oldCorners = destinationCorners; // keeping the vales of corners from i-1th iteration
		perspectiveTransform( refCorners, destinationCorners, homography);

		/*Mat resultImg = matchedImage;//refImg1;
	    		line( resultImg, destinationCorners[0] + Point2f( refImg.cols, 0), destinationCorners[1] + Point2f( refImg.cols, 0), Scalar(0, 255, 0), 4 );
	    		line( resultImg, destinationCorners[1] + Point2f( refImg.cols, 0), destinationCorners[2] + Point2f( refImg.cols, 0), Scalar( 0, 255, 0), 4 );
	    		line( resultImg, destinationCorners[2] + Point2f( refImg.cols, 0), destinationCorners[3] + Point2f( refImg.cols, 0), Scalar( 0, 255, 0), 4 );
	    		line( resultImg, destinationCorners[3] + Point2f( refImg.cols, 0), destinationCorners[0] + Point2f( refImg.cols, 0), Scalar( 0, 255, 0), 4 );
		 */
		if(flag==false){
			area = norm(destinationCorners[0] - destinationCorners[1]) * norm(destinationCorners[1] - destinationCorners[2]);
		}
		flag = true;
		double tempArea = norm(destinationCorners[0] - destinationCorners[1]) * norm(destinationCorners[2] - destinationCorners[1]);

		double dotProd1 = Mat(destinationCorners[0] - destinationCorners[1]).dot(Mat(destinationCorners[2] - destinationCorners[1]))/tempArea;
		double dotProd2 = Mat(destinationCorners[0] - destinationCorners[3]).dot(Mat(destinationCorners[2] - destinationCorners[3]))/tempArea;
		double dotProd3 = Mat(destinationCorners[1] - destinationCorners[0]).dot(Mat(destinationCorners[3] - destinationCorners[0]))/tempArea;
		double dotProd4 = Mat(destinationCorners[1] - destinationCorners[2]).dot(Mat(destinationCorners[3] - destinationCorners[2]))/tempArea;


		Point3f vec1(destinationCorners[0].x - destinationCorners[1].x,destinationCorners[0].y - destinationCorners[1].y, 0);
		Point3f vec2(destinationCorners[2].x - destinationCorners[1].x,destinationCorners[2].y - destinationCorners[1].y, 0);

		Matx31d normal = Mat(vec1).cross(Mat(vec2));
		Point2f nrm(normal(0,0)/normal(2,0),normal(1,0)/normal(2,0));
		//	Mat normal = Mat(destinationCorners[0] - destinationCorners[1]).cross(Mat(destinationCorners[2] - destinationCorners[1]));
		//fout<<dotProd<<"\n";
		if((dotProd1>-0.7 && dotProd1<0.7) && (dotProd2>-0.7 && dotProd2<0.7) && (dotProd3>-0.7 && dotProd3<0.7) && (dotProd4>-0.7 && dotProd4<0.7) ) {// this program fails if 1st frame does not have the object
			//implement better check for valid
			//destinationCorners = oldCorners;





			solvePnP(objectPoints, destinationCorners, cameraMatrix, distCoeffs, rTemp, tTemp, false);

			projectPoints(Mat(objPts), rTemp, tTemp, cameraMatrix, distCoeffs, imagePointsRP);




			rTemp.convertTo(rVec,CV_32F);
			tTemp.convertTo(tVec ,CV_32F);
			Rodrigues(rVec, rotMat);

			hconcat( rotMat,tVec, RT);
			//	vector<double> toPlot;
			cout<<RT<<"\n\n\n";

			Mat toPlot = cameraMatrix * RT * nrML;

			// for x axis
			Point2f mid2((destinationCorners[1].x + destinationCorners[2].x)/2.0,(destinationCorners[1].y + destinationCorners[2].y)/2.0);

			Point2f mid3((destinationCorners[2].x + destinationCorners[3].x)/2.0,(destinationCorners[2].y + destinationCorners[3].y)/2.0);




			Point2f mid(((destinationCorners[0].x+destinationCorners[2].x)/2),((destinationCorners[0].y+destinationCorners[2].y)/2));
			Point2f end(toPlot.at<float>(0,0),toPlot.at<float>(1,0));

			Point2f vecZ((15 * ((end.x - mid.x)/norm(end - mid))) + mid.x,(15 * ((end.y - mid.y)/norm(end - mid))) + mid.y);

			Mat resultImg = sceneImg2;
			//	line( resultImg, destinationCorners[0], destinationCorners[1] , Scalar( 0, 255, 0), 2 );
			//	line( resultImg, destinationCorners[1], destinationCorners[2] , Scalar( 0, 255, 0), 2 );
			//	line( resultImg, destinationCorners[2], destinationCorners[3] , Scalar( 0, 255, 0), 2 );
			//	line( resultImg, destinationCorners[3], destinationCorners[0] , Scalar( 0, 255, 0), 2);
			line( resultImg, Point2f(imagePointsRP.at<float>(0,0),imagePointsRP.at<float>(0,1)), vecZ, Scalar( 255, 0, 0), 2);
			line( resultImg, Point2f(imagePointsRP.at<float>(0,0),imagePointsRP.at<float>(0,1)), Point2f(imagePointsRP.at<float>(1,0),imagePointsRP.at<float>(1,1)), Scalar( 0, 255, 0), 2);
			line( resultImg, Point2f(imagePointsRP.at<float>(0,0),imagePointsRP.at<float>(0,1)), Point2f(imagePointsRP.at<float>(2,0),imagePointsRP.at<float>(2,1)), Scalar( 0, 0, 255), 2);
			namedWindow("result", WINDOW_NORMAL);
			imshow("result",resultImg);
			waitKey(1);

			outcap<<resultImg;


		}
		else
		{
			outcap<<sceneImg2;
		}

	}
	printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
	cout << "Finished writing" << endl;

}// end of main
