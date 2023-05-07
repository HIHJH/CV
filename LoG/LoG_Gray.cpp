// opencv_test.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;


Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize);
Mat get_Laplacian_Kernel();
Mat Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s);
Mat Laplacianfilter(const Mat input);
Mat Mirroring(const Mat input, int n);


int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input 이미지 불러옴
	Mat input_gray;

	// check for validation
	if (!input.data) {
		printf("Could not open\n"); // 이미지 파일 데이터에 오류가 있으면 메세지 출력
		return -1;
	}

	//Gaussian smoothing parameters
	int window_radius = 2;
	double sigma_t = 2.0;
	double sigma_s = 2.0;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// RGB 이미지를 흑백 이미지로 변환
	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);	// 8-bit unsigned char -> 64-bit floating point

	// 1. gaussain filtering 적용
	Mat h_f = Gaussianfilter(input_gray, window_radius, sigma_t, sigma_s);	// h(x,y) * f(x,y)

	// 2. Laplacian filtering 적용
	Mat Laplacian = Laplacianfilter(h_f); // Gaussain filtering 수행 후 Laplacian filtering 수행

	normalize(Laplacian, Laplacian, 0, 1, CV_MINMAX); // 0-1 사이 값으로 정규화

	namedWindow("Grayscale", WINDOW_AUTOSIZE); // GrayScale 창 띄움
	imshow("Grayscale", input_gray); // 창에 input_gray 이미지 보여줌

	namedWindow("Gaussian blur", WINDOW_AUTOSIZE); // Gaussian blur 창 띄움
	imshow("Gaussian blur", h_f); // 창에 h_f 이미지 보여줌

	namedWindow("Laplacian filter", WINDOW_AUTOSIZE); // Laplacian filter 창 띄움
	imshow("Laplacian filter", Laplacian); // 창에 Laplacian 이미지 보여줌

	waitKey(0);

	return 0;
}


// Gaussian filtering
// 매개변수: input = 필터를 적용할 이미지, n = neighbor의 범위
// sigma_t = t 관련 gaussian 분포의 표준편차(x-coordinate), sigma_s = s 관련 gaussian 분포의 표준편차(y-coordinate)
Mat Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s) {

	int row = input.rows;
	int col = input.cols;

	// generate gaussian kernel
	Mat kernel = get_Gaussian_Kernel(n, sigma_t, sigma_s, true);
	Mat output = Mat::zeros(row, col, input.type()); // output은 input과 크기, 타입이 같은 matrix이고 값은 0으로 초기화

	//Intermediate data generation for mirroring
	Mat input_mirror = Mirroring(input, n);

	//convolution
	for (int i = n; i < row + n; i++) {
		for (int j = n; j < col + n; j++) {		
			
			float sum = 0.0;
			for (int a = -n; a <= n; a++) { // for each kernel window
				for (int b = -n; b <= n; b++) {
					sum += kernel.at<double>(a + n, b + n) * input_mirror.at<double>(i + a, j + b);
					// filtering할 neighbor 각각에 weight를 곱하고 이를 더해줌
				}
			}
			output.at<double>(i - n, j - n) = (double)sum; // 더한 값을 output의 픽셀에 반영

		}
	}

	return output; // filtering 적용한 output matrix를 return
}

// Laplacian filtering
// 매개변수: input = 필터를 적용할 이미지
Mat Laplacianfilter(const Mat input) {

	int row = input.rows;
	int col = input.cols;

	// generate laplacian kernel
	Mat kernel = get_Laplacian_Kernel();
	Mat output = Mat::zeros(row, col, input.type()); // output은 input과 크기, 타입이 같은 matrix이고 값은 0으로 초기화

	int n = 1; // laplacian filter kernel N
	//Intermediate data generation for mirroring
	Mat input_mirror = Mirroring(input, n);

	for (int i = n; i < row + n; i++) {
		for (int j = n; j < col + n; j++) {

			float sum = 0.0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					sum += kernel.at<double>(a + n, b + n) * input_mirror.at<double>(i + a, j + b);
					// filtering할 neighbor 각각에 weight를 곱하고 이를 더해줌
				}
			}
			float out = abs(sum);  // 더한 값의 절댓값을 output의 픽셀에 반영
			output.at<double>(i - n, j - n) = saturate_cast<double>(out); // 값을 0~255 범위 내로 하여 output 각 픽셀에 반영

		}
	}

	return output; // filtering 적용한 output matrix를 return
}

// boundary process = mirroring
// 매개변수: input = mirroring할 이미지, n = neighbor의 범위
Mat Mirroring(const Mat input, int n)
{
	int row = input.rows;
	int col = input.cols;

	Mat input2 = Mat::zeros(row + 2 * n, col + 2 * n, input.type());
	int row2 = input2.rows;
	int col2 = input2.cols;

	// input2의 중앙에 원본 이미지를 복사
	for (int i = n; i < row + n; i++) {
		for (int j = n; j < col + n; j++) {
			input2.at<double>(i, j) = input.at<double>(i - n, j - n);
		}
	}
	// 이미지가 좌우 대칭되도록 mirroing
	for (int i = n; i < row + n; i++) {
		for (int j = 0; j < n; j++) {
			input2.at<double>(i, j) = input2.at<double>(i, 2 * n - j);
		}
		for (int j = col + n; j < col2; j++) {
			input2.at<double>(i, j) = input2.at<double>(i, 2 * col - 2 + 2 * n - j);
		}
	}
	// 이미지가 상하 대칭되도록 mirroring
	for (int j = 0; j < col2; j++) {
		for (int i = 0; i < n; i++) {
			input2.at<double>(i, j) = input2.at<double>(2 * n - i, j);
		}
		for (int i = row + n; i < row2; i++) {
			input2.at<double>(i, j) = input2.at<double>(2 * row - 2 + 2 * n - i, j);
		}
	}

	return input2; // input2 Mat 반환
}

// Gaussian kernel 값 지정
// 매개변수: n = neighbor의 범위, sigma_t =  t 관련 gaussian 분포의 표준편차(x-coordinate),
// sigma_s = s 관련 gaussian 분포의 표준편차(y-coordinate), normalize = 정규화 여부
Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize) {

	int kernel_size = (2 * n + 1); // (2N+1)*(2N+1)
	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_64F);
	// kernel matrix를 kernel size에 맞게 64bit float형태로 설정하고, 값은 0으로 초기화
	double kernel_sum = 0.0;

	for (int i = -n; i <= n; i++) {
		for (int j = -n; j <= n; j++) {
			// gaussian 함수를 이용해 kernel 값 업데이트
			kernel.at<double>(i + n, j + n) = exp(-((i * i) / (2.0 * sigma_t * sigma_t) + (j * j) / (2.0 * sigma_s * sigma_s)));
			kernel_sum += kernel.at<double>(i + n, j + n); // 정규화를 하기 위해 kernel_sum에 각 weight를 계산한 값의 합을 저장
		}
	}

	if (normalize) { // 정규화한다면,
		for (int i = 0; i < kernel_size; i++)
			for (int j = 0; j < kernel_size; j++)
				kernel.at<double>(i, j) /= kernel_sum;		// kernel_sum 으로 나누어 정규화
	}

	return kernel; // gaussian kernel을 return
}

// Laplacian kernel 값 지정
Mat get_Laplacian_Kernel() {

	Mat kernel = Mat::zeros(3, 3, CV_64F);
	// kernel을 64bit float형태의 3*3 matirx로 생성하고 0으로 초기화

	// weight 값 넣어줌
	kernel.at<double>(0, 1) = 1.0;
	kernel.at<double>(2, 1) = 1.0;
	kernel.at<double>(1, 0) = 1.0;
	kernel.at<double>(1, 2) = 1.0;
	kernel.at<double>(1, 1) = -4.0;

	return kernel; // laplacian kernel을 return
}