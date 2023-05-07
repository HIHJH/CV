// opencv_test.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

vector<Point2f> MatToVec(const Mat input);
Mat NonMaximum_Suppression(const Mat input, Mat corner_mat, int radius);
Mat Mirroring(const Mat input, int n);
void type2str(int type);


int main() {

	//Use the following three images.
	//Mat input = imread("checkerboard.png", CV_LOAD_IMAGE_COLOR);
	Mat input = imread("checkerboard2.jpg", CV_LOAD_IMAGE_COLOR);
	//Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input 이미지 불러옴

		// check for validation
	if (!input.data) {
		printf("Could not open\n"); // 이미지 파일 데이터에 오류가 있으면 메세지 출력
		return -1;
	}

	int row = input.rows;
	int col = input.cols;

	Mat input_gray, input_visual;
	Mat output, output_norm, corner_mat;
	vector<Point2f> points;

	corner_mat = Mat::zeros(row, col, CV_8U); // 각 픽셀이 코너인지 아닌지를 나타내는 Mat (코너O:1, 코너X:0)

	bool NonMaxSupp = true; // non-maximum suppression 수행 여부 결정

	bool Subpixel = true; // subpixel refinement 수행 여부 결정


	cvtColor(input, input_gray, CV_RGB2GRAY);	// RGB 이미지를 흑백 이미지로 변환

	//Harris corner detection using 'cornerHarris'
	//Note that 'src' of 'cornerHarris' can be either 1) input single-channel 8-bit or 2) floating-point image.
	// 해리스 코너 검출
	// src = 입력 이미지 = input_gray
	// dst = 출력 이미지 = output
	// blockSize = 코너 감지를 위한 윈도우 크기 = 2
	// ksize = 커널 크기 = 3
	// k = 코너 감지에서 사용되는 Harris 감도 상수 = 0.04
	// borderType = 이미지 경계 처리 방법 = BORDER_DEFAULT
	cornerHarris(input_gray, output, 2, 3, 0.04, BORDER_DEFAULT);


	//Scale the Harris response map 'output' from 0 to 1.
	//This is for display purpose only.
	normalize(output, output_norm, 0, 1.0, NORM_MINMAX); // output을 0~1 사이 값으로 normalization
	namedWindow("Harris Response", WINDOW_AUTOSIZE); // Harris Response 창 띄움
	imshow("Harris Response", output_norm); // 창에 output_norm 이미지 보여줌


	//Threshold the Harris corner response.
	//corner_mat = 1 for corner, 0 otherwise.
	input_visual = input.clone(); // input_visual은 input을 복제한 것
	double minVal, maxVal; // 최소, 최대값을 저장할 변수
	Point minLoc, maxLoc; // 최소, 최대값의 좌표를 저장할 변수
	minMaxLoc(output, &minVal, &maxVal, &minLoc, &maxLoc); // output의 최소, 최대값과 그 위치를 찾아 저장

	//코너 시각화
	int corner_num = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (output.at<float>(i, j) > 0.01 * maxVal) // threshold =  0.01 * maxVal
			{
				//input_visual.at<Vec3b>(i, j)[0] = 0;		input_visual.at<Vec3b>(i, j)[1] = 0;	input_visual.at<Vec3b>(i, j)[2] = 255;
				circle(input_visual, Point(j, i), 2, Scalar(0, 0, 255), 1, 8, 0);	//You can also use this function of drawing a circle. For details, search 'circle' in OpenCV.
				// 해당 픽셀 위치에 빨간색 동그라미 표시함
				corner_mat.at<uchar>(i, j) = 1; // corner_mat 행렬에 1을 할당
				corner_num++;
			}

			else
				output.at<float>(i, j) = 0.0; // 코너가 아니면 0을 할당
		}
	}
	printf("After cornerHarris, corner number = %d\n\n", corner_num); // corner number를 출력
	namedWindow("Harris Corner", WINDOW_AUTOSIZE); // Harris Corner 창 띄움
	imshow("Harris Corner", input_visual); // 창에 input_visual 이미지 보여줌 

	//Non-maximum suppression
	if (NonMaxSupp) // NonMaxSupp가 true이면,
	{
		NonMaximum_Suppression(output, corner_mat, 2);
		// NonMaximum Suppression 수행
		// radius = 2

		//코너 시각화
		corner_num = 0;
		input_visual = input.clone(); // input_visual은 input을 복제한 것
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (corner_mat.at<uchar>(i, j) == 1) { // corner 이면,
					//input_visual.at<Vec3b>(i, j)[0] = 0;		input_visual.at<Vec3b>(i, j)[1] = 0;	input_visual.at<Vec3b>(i, j)[2] = 255;					
					circle(input_visual, Point(j, i), 2, Scalar(0, 0, 255), 1, 8, 0);	//You can also use this function of drawing a circle. For details, search 'circle' in OpenCV.
					// 해당 픽셀에 빨간색 동그라미 표시함
					corner_num++;
				}
			}
		}

		printf("After non-maximum suppression, corner number = %d\n\n", corner_num); // corner number 출력
		namedWindow("Harris Corner (Non-max)", WINDOW_AUTOSIZE); // Harris Corner (Non-max) 창 띄움
		imshow("Harris Corner (Non-max)", input_visual); // 창에 input_visual 이미지 보여줌
	}

	//Sub-pixel refinement for detected corners
	if (Subpixel) // Subpixel true이면,
	{
		Size subPixWinSize(3, 3); // subpixel 보정시 참고할 윈도우 크기 = 3*3
		TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
		// 최대 반복 횟수가 COUNT 또는 최대 반복 횟수 = 20, 변화량의 tolerance = 0.03 이면 refinement 종료

		points = MatToVec(corner_mat); // corner_mat을 벡터 형식으로 points에 저장

		// cornerSubPix
		// input_gray에서 감지된 코너 위치(points)를 subPixWinSize, termcrit을 고려하며 subpixel 레벨로 보정
		// 결과 코너 위치는 다시 points에 저장
		cornerSubPix(input_gray, points, subPixWinSize, Size(-1, -1), termcrit);

		//코너 시각화
		input_visual = input.clone(); // input_visual은 input을 복제한 것
		for (int k = 0; k < points.size(); k++) {

			int x = points[k].x; // points 벡터에 저장된 각 코너의 x, y 좌표값을 이용
			int y = points[k].y;

			if (x<0 || x>col - 1 || y<0 || y>row - 1)
			{
				points.pop_back(); // 각 코너의 좌표값이 이미지 크기를 벗어나면 해당 좌표값 제거
				continue;
			}

			//input_visual.at<Vec3b>(i, j)[0] = 0;		input_visual.at<Vec3b>(i, j)[1] = 0;	input_visual.at<Vec3b>(i, j)[2] = 255;
			circle(input_visual, Point(x, y), 2, Scalar(0, 0, 255), 1, 8, 0);	//You can also use this function of drawing a circle. For details, search 'circle' in OpenCV.
			// Point(x, y) 위치에 해당하는 픽셀에 빨간색 동그라미 표시함
		}

		printf("After subpixel-refinement, corner number = %d\n\n", points.size()); // corner number 출력
		namedWindow("Harris Corner (subpixel)", WINDOW_AUTOSIZE); // Harris Corner (subpixel) 창 띄움
		imshow("Harris Corner (subpixel)", input_visual); // 창에 input_visual 이미지 보여줌
	}

	waitKey(0);

	return 0;
}

// input 에서 값이 1인 픽셀의 위치를 Point2f 형태로 저장하여 vector에 반환하는 함수
vector<Point2f> MatToVec(const Mat input)
{
	vector<Point2f> points; // points vector 형성

	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.at<uchar>(i, j) == 1) { // input 값이 1이면
				points.push_back(Point2f((float)j, (float)i)); // 그 픽셀 위치를 Point2f 형태로 vector에 추가
			}
		}
	}

	return points; // points vector 반환
}

//corner_mat = 1 for corner, 0 otherwise.
// NonMaximum Suppression 수행
// 매개변수: input = NMS 수행할 이미지, corner_mat = 코너 위치를 나타내는 행렬
// radius = NMS 수행할 윈도우의 반지름 값
Mat NonMaximum_Suppression(const Mat input, Mat corner_mat, int radius)
{
	int row = input.rows;
	int col = input.cols;

	Mat input_mirror = Mirroring(input, radius); // boundary process로 mirroring 수행

	for (int i = radius; i < row + radius; i++) {
		for (int j = radius; j < col + radius; j++) {

			if (corner_mat.at<uchar>(i - radius, j - radius) != 1.0) { // 코너가 아니라면
				continue; // 다음 픽셀로 넘어감
			}

			//초기에 코너 픽셀 값을 최대로 생각 => is_max = true로 설정
			bool is_max = true;

			for (int a = i - radius; a <= i + radius; a++) { 
				for (int b = j - radius; b <= j + radius; b++) {
					if (a == i && b == j) {
						continue;
					}
					if (input_mirror.at<float>(a, b) > input_mirror.at<float>(i, j)) { // 해당 픽셀 값보다 윈도우 내의 주변 픽셀 값이 크다면
						is_max = false; // is_max 를 false로 변경
						break;
					}
				}
				if (!is_max) { // 가장 큰 값을 갖는 픽셀이 아니면,
					break;
				}
			}

			if (!is_max) { // 가장 큰 값을 갖는 픽셀이 아니면,
				corner_mat.at<uchar>(i - radius, j - radius) = 0.0; // corner_mat 값을 0으로 변경
			}
		}
	}

	return input;
}

// boundary process = mirroring
// 매개변수: input = mirroring할 이미지, n = neighbor의 범위
Mat Mirroring(const Mat input, int n)
{
	int row = input.rows;
	int col = input.cols;

	Mat input2 = Mat::zeros(row + 2 * n, col + 2 * n, input.type()); // input2에 mirroring 한 Mat 저장
	int row2 = input2.rows;
	int col2 = input2.cols;

	// input2의 중앙에 원본 이미지를 복사
	for (int i = n; i < row + n; i++) {
		for (int j = n; j < col + n; j++) {
			input2.at<float>(i, j) = input.at<float>(i - n, j - n);
		}
	}
	// 이미지가 좌우 대칭되도록 mirroing
	for (int i = n; i < row + n; i++) {
		for (int j = 0; j < n; j++) {
			input2.at<float>(i, j) = input2.at<float>(i, 2 * n - j);
		}
		for (int j = col + n; j < col2; j++) {
			input2.at<float>(i, j) = input2.at<float>(i, 2 * col - 2 + 2 * n - j);
		}
	}
	// 이미지가 상하 대칭되도록 mirroring
	for (int j = 0; j < col2; j++) {
		for (int i = 0; i < n; i++) {
			input2.at<float>(i, j) = input2.at<float>(2 * n - i, j);
		}
		for (int i = row + n; i < row2; i++) {
			input2.at<float>(i, j) = input2.at<float>(2 * row - 2 + 2 * n - i, j);
		}
	}

	return input2; // input2 Mat 반환
}


//If you want to know the type of 'Mat', use the following function
//For instance, for 'Mat input'
//type2str(input.type());
// Mat의 객체 타입을 문자열로반환
void type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	printf("Matrix: %s \n", r.c_str());
}