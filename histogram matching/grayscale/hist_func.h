#pragma once

#include <opencv2/opencv.hpp>
#include <stdio.h>

#define L 256		// # of intensity levels
#define IM_TYPE	CV_8UC3

using namespace cv;

// 흑백 이미지는 G, 컬러이미지는 C를 사용하도록 이미지 타입을 지정해준다.
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

// single channel 이미지의 PDF를 계산하는 함수
float* cal_PDF(Mat& input) {

	int count[L] = { 0 }; // intensity 레벨 크기의 int 배열(count)를 0으로 초기화하며 생성한다.
	float* PDF = (float*)calloc(L, sizeof(float)); // float 배열을 동적할당 해준다.

	// 픽셀을 Count 하는 부분이다. (히스토그램)
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			count[input.at<G>(i, j)]++;

	// count 값을 rows*cols로 나누어주어 PDF를 구한다.
	for (int i = 0; i < L; i++)
		PDF[i] = (float)count[i] / (float)(input.rows * input.cols);

	return PDF;
}

// color 이미지(RGB channel)의 PDF를 계산하는 함수
float** cal_PDF_RGB(Mat& input) {

	int count[L][3] = { 0 }; // channel이 R,G,B 3개이므로 [3]을 추가하여 2차원 배열을 만들어준다.
	float** PDF = (float**)malloc(sizeof(float*) * L);  // 포인터 배열을 동적할당 해준다.

	for (int i = 0; i < L; i++)
		PDF[i] = (float*)calloc(3, sizeof(float)); // float 배열을 동적할당해준다.

	// 픽셀을 Count 하는 부분이다. (히스토그램)
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			C pixel = input.at<C>(i, j);
			// 이미지에서 각 색상(해당 채널별)의 빈도수를 계산한다.
			count[pixel[0]][0]++;
			count[pixel[1]][1]++;
			count[pixel[2]][2]++;
		}
	}

	// 각 count 값을 rows*cols로 나누어주어 PDF를 구한다.
	for (int i = 0; i < L; i++) {
		PDF[i][0] = (float)count[i][0] / (float)(input.rows * input.cols); // R의 확률(PDF)을 계산
		PDF[i][1] = (float)count[i][1] / (float)(input.rows * input.cols); // G의 확률(PDF)을 계산
		PDF[i][2] = (float)count[i][2] / (float)(input.rows * input.cols); // B의 확률(PDF)을 계산
	}

	return PDF;
}

// single channel 이미지의 CDF를 계산하는 함수
float* cal_CDF(Mat& input) {

	int count[L] = { 0 }; // intensity 레벨 크기의 int 배열(count)를 0으로 초기화하며 생성한다.
	float* CDF = (float*)calloc(L, sizeof(float)); // float 배열을 동적할당 해준다.

	// 픽셀을 Count 한 것을 누적하는 부분이다.
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			count[input.at<G>(i, j)]++;

	// count 값을 rows*cols로 나누어주어 CDF를 구한다.
	for (int i = 0; i < L; i++) {
		CDF[i] = (float)count[i] / (float)(input.rows * input.cols);

		if (i != 0)
			CDF[i] += CDF[i - 1];
	}

	return CDF;
}

// color 이미지(RGB channel)의 CDF를 계산하는 함수
float** cal_CDF_RGB(Mat& input) {

	int count[L][3] = { 0 };
	float** CDF = (float**)malloc(sizeof(float*) * L);

	for (int i = 0; i < L; i++)
		CDF[i] = (float*)calloc(3, sizeof(float));

	// 픽셀을 Count 한 것을 누적하는 부분이다.
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			C pixel = input.at<C>(i, j);
			count[pixel[0]][0]++;
			count[pixel[1]][1]++;
			count[pixel[2]][2]++;
		}
	}

	// count 값을 rows*cols로 나누어주어 CDF를 구한다.
	for (int i = 0; i < L; i++) {
		CDF[i][0] = (float)count[i][0] / (float)(input.rows * input.cols);
		CDF[i][1] = (float)count[i][1] / (float)(input.rows * input.cols);
		CDF[i][2] = (float)count[i][2] / (float)(input.rows * input.cols);

		if (i != 0) {
			CDF[i][0] += CDF[i - 1][0]; // R의 누적 확률(CDF)을 계산
			CDF[i][1] += CDF[i - 1][1]; // G의 누적 확률(CDF)을 계산
			CDF[i][2] += CDF[i - 1][2]; // B의 누적 확률(CDF)을 계산
		}
	}

	return CDF;
}