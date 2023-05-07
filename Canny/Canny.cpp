// opencv_test.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;


Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize);
Mat get_Laplacian_Kernel();
Mat Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s);
Mat Laplacianfilter(const Mat input);
Mat Mirroring(const Mat input, int n);


int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input �̹��� �ҷ���
	Mat input_gray, output;

	// check for validation
	if (!input.data) {
		printf("Could not open\n"); // �̹��� ���� �����Ϳ� ������ ������ �޼��� ���
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// RGB �̹����� ��� �̹����� ��ȯ
	//input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);	// 8-bit unsigned char -> 64-bit floating point

	// Canny Edge Detector
	// image = ���� ������ �̹��� = input_gray
	// edges = ��� �̹��� = output
	// threshold1 = ���� ������ ���� ���� �Ӱ谪 = 100
	// threshold2 = ���� ������ ���� ���� �Ӱ谪 = 160
	// apertureSize = Sobel ���꿡 ���Ǵ� Ŀ�� ũ�� = 3
	// L2gradient = �̺��� �� ���Ǵ� ��� (true = L2-norm, false = L1-norm)
	// = false (L1-norm)
	Canny(input_gray, output, 100, 160, 3, false);

	namedWindow("Grayscale", WINDOW_AUTOSIZE); // Grayscale â ���
	imshow("Grayscale", input_gray); // â�� input_gray �̹��� ������

	namedWindow("Canny", WINDOW_AUTOSIZE); // Canny â ���
	imshow("Canny", output); // â�� output �̹��� ������

	waitKey(0);

	return 0;
}



