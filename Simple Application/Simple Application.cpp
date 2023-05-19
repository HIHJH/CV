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

	Mat img_object = imread("img1.png", CV_LOAD_IMAGE_GRAYSCALE); // 이미지1을 grayscale로 받아옴
	Mat img_scene = imread("img2.png", CV_LOAD_IMAGE_GRAYSCALE); // 이미지2를 grayscale로 받아옴

	if (!img_object.data || !img_scene.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return -1; // 이미지 파일을 읽을 수 없다면 오류메세지 출력
	}

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400; // SURF feature detector의 최소 Hessian 값을 400으로 설정

	SurfFeatureDetector detector(minHessian);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	detector.detect(img_object, keypoints_object); // img_object의 keypoints를 검출하여 keypoints_object에 저장
	detector.detect(img_scene, keypoints_scene); // img_scene의 keypoints를 검출하여 keypoints_scene에 저장

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	Mat descriptors_object, descriptors_scene;

	extractor.compute(img_object, keypoints_object, descriptors_object); // img_object의 descriptors를 계산하여 descriptors_object에 저장
	extractor.compute(img_scene, keypoints_scene, descriptors_scene); // img_scene의 descriptors를 계산하여 descriptors_scene에 저장

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_object, descriptors_scene, matches);
	// match 함수를 사용하여 img_object와 img_scene의 descriptors를 매칭하여 matches 백터에 저장

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance; // matches 벡터에 저장된 매칭 결과에서 거리를 확인
		// 최대 거리와 최소 거리 업데이트
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist); // 최대거리 출력
	printf("-- Min dist : %f \n", min_dist); // 최소거리 출력

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist) // 매칭 결과 거리가 3 * min_dist보다 작은 경우
		{
			good_matches.push_back(matches[i]); // good_matches 벡터에 추가
		}
	}

	Mat img_matches; // 매칭 결과를 표시하는 Mat
	drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	// drawMatches 함수를 사용하여 키포인트 매칭 결과를 img_matches에 그림

	//-- Localize the object from img_1 in img_2 
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		// 각 매칭에서 키포인트들을 가져와 obj, scene 벡터에 추가
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, CV_RANSAC);
	// RANSAC 알고리즘이 적용된 findHomography 함수를 이용하여 obj와 scene 사이의 호모그래피 행렬 H를 구함

	//-- Get the corners from the image_1 ( the object to be "detected" )
	// 이미지1의 꼭짓점을 얻는 부분
	std::vector<Point2f> obj_corners(4); // object image의 4개의 꼭짓점(모서리) 저장
	obj_corners[0] = cvPoint(0, 0); // 좌측 상단 모서리
	obj_corners[1] = cvPoint(img_object.cols, 0); // 우측 상단 모서리
	obj_corners[2] = cvPoint(img_object.cols, img_object.rows); // 우측 하단 모서리
	obj_corners[3] = cvPoint(0, img_object.rows); // 좌측 하단 모서리
	std::vector<Point2f> scene_corners(4); // scene image에서 변환된 객체의 4개의 꼭짓점을 저장

	perspectiveTransform(obj_corners, scene_corners, H);
	// 호모그래피 행렬 H를 이용하여 obj_coners를 scene_corners로 변환
	// object 이미지의 꼭짓점이 scene 이미지에서의 해당 위치로 변환


	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	// 변환된 object의 4개의 꼭짓점을 scene 이미지에서 연결하는 선을 그림
	line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);

	//-- Show detected matches
	imshow("Good Matches & Object detection", img_matches);
	// Good Matches & Object detection 창에 img_matches를 보여줌

	waitKey(0);

	return 0;
}

