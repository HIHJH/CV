#pragma once

#include <opencv2/opencv.hpp>
#include <stdio.h>

#define L 256		// # of intensity levels
#define IM_TYPE	CV_8UC3

using namespace cv;

// ��� �̹����� G, �÷��̹����� C�� ����ϵ��� �̹��� Ÿ���� �������ش�.
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

// single channel �̹����� PDF�� ����ϴ� �Լ�
float* cal_PDF(Mat& input) {

	int count[L] = { 0 }; // intensity ���� ũ���� int �迭(count)�� 0���� �ʱ�ȭ�ϸ� �����Ѵ�.
	float* PDF = (float*)calloc(L, sizeof(float)); // float �迭�� �����Ҵ� ���ش�.

	// �ȼ��� Count �ϴ� �κ��̴�. (������׷�)
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			count[input.at<G>(i, j)]++;

	// count ���� rows*cols�� �������־� PDF�� ���Ѵ�.
	for (int i = 0; i < L; i++)
		PDF[i] = (float)count[i] / (float)(input.rows * input.cols);

	return PDF;
}

// color �̹���(RGB channel)�� PDF�� ����ϴ� �Լ�
float** cal_PDF_RGB(Mat& input) {

	int count[L][3] = { 0 }; // channel�� R,G,B 3���̹Ƿ� [3]�� �߰��Ͽ� 2���� �迭�� ������ش�.
	float** PDF = (float**)malloc(sizeof(float*) * L);  // ������ �迭�� �����Ҵ� ���ش�.

	for (int i = 0; i < L; i++)
		PDF[i] = (float*)calloc(3, sizeof(float)); // float �迭�� �����Ҵ����ش�.

	// �ȼ��� Count �ϴ� �κ��̴�. (������׷�)
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			C pixel = input.at<C>(i, j);
			// �̹������� �� ����(�ش� ä�κ�)�� �󵵼��� ����Ѵ�.
			count[pixel[0]][0]++;
			count[pixel[1]][1]++;
			count[pixel[2]][2]++;
		}
	}

	// �� count ���� rows*cols�� �������־� PDF�� ���Ѵ�.
	for (int i = 0; i < L; i++) {
		PDF[i][0] = (float)count[i][0] / (float)(input.rows * input.cols); // R�� Ȯ��(PDF)�� ���
		PDF[i][1] = (float)count[i][1] / (float)(input.rows * input.cols); // G�� Ȯ��(PDF)�� ���
		PDF[i][2] = (float)count[i][2] / (float)(input.rows * input.cols); // B�� Ȯ��(PDF)�� ���
	}

	return PDF;
}

// single channel �̹����� CDF�� ����ϴ� �Լ�
float* cal_CDF(Mat& input) {

	int count[L] = { 0 }; // intensity ���� ũ���� int �迭(count)�� 0���� �ʱ�ȭ�ϸ� �����Ѵ�.
	float* CDF = (float*)calloc(L, sizeof(float)); // float �迭�� �����Ҵ� ���ش�.

	// �ȼ��� Count �� ���� �����ϴ� �κ��̴�.
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			count[input.at<G>(i, j)]++;

	// count ���� rows*cols�� �������־� CDF�� ���Ѵ�.
	for (int i = 0; i < L; i++) {
		CDF[i] = (float)count[i] / (float)(input.rows * input.cols);

		if (i != 0)
			CDF[i] += CDF[i - 1];
	}

	return CDF;
}

// color �̹���(RGB channel)�� CDF�� ����ϴ� �Լ�
float** cal_CDF_RGB(Mat& input) {

	int count[L][3] = { 0 };
	float** CDF = (float**)malloc(sizeof(float*) * L);

	for (int i = 0; i < L; i++)
		CDF[i] = (float*)calloc(3, sizeof(float));

	// �ȼ��� Count �� ���� �����ϴ� �κ��̴�.
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			C pixel = input.at<C>(i, j);
			count[pixel[0]][0]++;
			count[pixel[1]][1]++;
			count[pixel[2]][2]++;
		}
	}

	// count ���� rows*cols�� �������־� CDF�� ���Ѵ�.
	for (int i = 0; i < L; i++) {
		CDF[i][0] = (float)count[i][0] / (float)(input.rows * input.cols);
		CDF[i][1] = (float)count[i][1] / (float)(input.rows * input.cols);
		CDF[i][2] = (float)count[i][2] / (float)(input.rows * input.cols);

		if (i != 0) {
			CDF[i][0] += CDF[i - 1][0]; // R�� ���� Ȯ��(CDF)�� ���
			CDF[i][1] += CDF[i - 1][1]; // G�� ���� Ȯ��(CDF)�� ���
			CDF[i][2] += CDF[i - 1][2]; // B�� ���� Ȯ��(CDF)�� ���
		}
	}

	return CDF;
}