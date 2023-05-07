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

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input 이미지 불러옴
	Mat input_gray, output;

	// check for validation
	if (!input.data) {
		printf("Could not open\n"); // 이미지 파일 데이터에 오류가 있으면 메세지 출력
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// RGB 이미지를 흑백 이미지로 변환
	//input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);	// 8-bit unsigned char -> 64-bit floating point

	// Canny Edge Detector
	// image = 엣지 검출할 이미지 = input_gray
	// edges = 출력 이미지 = output
	// threshold1 = 엣지 검출을 위한 하위 임계값 = 100
	// threshold2 = 엣지 검출을 위한 상위 임계값 = 160
	// apertureSize = Sobel 연산에 사용되는 커널 크기 = 3
	// L2gradient = 미분할 때 사용되는 방법 (true = L2-norm, false = L1-norm)
	// = false (L1-norm)
	Canny(input_gray, output, 100, 160, 3, false);

	namedWindow("Grayscale", WINDOW_AUTOSIZE); // Grayscale 창 띄움
	imshow("Grayscale", input_gray); // 창에 input_gray 이미지 보여줌

	namedWindow("Canny", WINDOW_AUTOSIZE); // Canny 창 띄움
	imshow("Canny", output); // 창에 output 이미지 보여줌

	waitKey(0);

	return 0;
}



