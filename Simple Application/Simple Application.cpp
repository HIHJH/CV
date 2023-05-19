#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2\nonfree\features2d.hpp>

using namespace cv;


/**
 * @function main
 * @brief Main function
 */
int main()
{

	Mat img_object = imread("img1.png", CV_LOAD_IMAGE_GRAYSCALE); // �̹���1�� grayscale�� �޾ƿ�
	Mat img_scene = imread("img2.png", CV_LOAD_IMAGE_GRAYSCALE); // �̹���2�� grayscale�� �޾ƿ�

	if (!img_object.data || !img_scene.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return -1; // �̹��� ������ ���� �� ���ٸ� �����޼��� ���
	}

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400; // SURF feature detector�� �ּ� Hessian ���� 400���� ����

	SurfFeatureDetector detector(minHessian);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	detector.detect(img_object, keypoints_object); // img_object�� keypoints�� �����Ͽ� keypoints_object�� ����
	detector.detect(img_scene, keypoints_scene); // img_scene�� keypoints�� �����Ͽ� keypoints_scene�� ����

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	Mat descriptors_object, descriptors_scene;

	extractor.compute(img_object, keypoints_object, descriptors_object); // img_object�� descriptors�� ����Ͽ� descriptors_object�� ����
	extractor.compute(img_scene, keypoints_scene, descriptors_scene); // img_scene�� descriptors�� ����Ͽ� descriptors_scene�� ����

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_object, descriptors_scene, matches);
	// match �Լ��� ����Ͽ� img_object�� img_scene�� descriptors�� ��Ī�Ͽ� matches ���Ϳ� ����

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance; // matches ���Ϳ� ����� ��Ī ������� �Ÿ��� Ȯ��
		// �ִ� �Ÿ��� �ּ� �Ÿ� ������Ʈ
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist); // �ִ�Ÿ� ���
	printf("-- Min dist : %f \n", min_dist); // �ּҰŸ� ���

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist) // ��Ī ��� �Ÿ��� 3 * min_dist���� ���� ���
		{
			good_matches.push_back(matches[i]); // good_matches ���Ϳ� �߰�
		}
	}

	Mat img_matches; // ��Ī ����� ǥ���ϴ� Mat
	drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	// drawMatches �Լ��� ����Ͽ� Ű����Ʈ ��Ī ����� img_matches�� �׸�

	//-- Localize the object from img_1 in img_2 
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		// �� ��Ī���� Ű����Ʈ���� ������ obj, scene ���Ϳ� �߰�
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, CV_RANSAC);
	// RANSAC �˰����� ����� findHomography �Լ��� �̿��Ͽ� obj�� scene ������ ȣ��׷��� ��� H�� ����

	//-- Get the corners from the image_1 ( the object to be "detected" )
	// �̹���1�� �������� ��� �κ�
	std::vector<Point2f> obj_corners(4); // object image�� 4���� ������(�𼭸�) ����
	obj_corners[0] = cvPoint(0, 0); // ���� ��� �𼭸�
	obj_corners[1] = cvPoint(img_object.cols, 0); // ���� ��� �𼭸�
	obj_corners[2] = cvPoint(img_object.cols, img_object.rows); // ���� �ϴ� �𼭸�
	obj_corners[3] = cvPoint(0, img_object.rows); // ���� �ϴ� �𼭸�
	std::vector<Point2f> scene_corners(4); // scene image���� ��ȯ�� ��ü�� 4���� �������� ����

	perspectiveTransform(obj_corners, scene_corners, H);
	// ȣ��׷��� ��� H�� �̿��Ͽ� obj_coners�� scene_corners�� ��ȯ
	// object �̹����� �������� scene �̹��������� �ش� ��ġ�� ��ȯ


	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	// ��ȯ�� object�� 4���� �������� scene �̹������� �����ϴ� ���� �׸�
	line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);

	//-- Show detected matches
	imshow("Good Matches & Object detection", img_matches);
	// Good Matches & Object detection â�� img_matches�� ������

	waitKey(0);

	return 0;
}

