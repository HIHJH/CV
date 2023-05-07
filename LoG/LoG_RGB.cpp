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

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input �̹��� �ҷ���

	// check for validation
	if (!input.data) {
		printf("Could not open\n"); // �̹��� ���� �����Ϳ� ������ ������ �޼��� ���
		return -1;
	}

	//Gaussian smoothing parameters
	int window_radius = 2;
	double sigma_t = 2.0;
	double sigma_s = 2.0;

	// 1. gaussain filtering ����
	Mat h_f = Gaussianfilter(input, window_radius, sigma_t, sigma_s);	// h(x,y) * f(x,y)

	// 2. Laplacian filtering ����
	Mat Laplacian = Laplacianfilter(h_f); // Gaussain filtering ���� �� Laplacian filtering ����

	normalize(Laplacian, Laplacian, 0, 1, CV_MINMAX); // 0-1 ���� ������ ����ȭ

	namedWindow("Original", WINDOW_AUTOSIZE); // Original â ���
	imshow("Original", input); // â�� input �̹��� ������

	namedWindow("Gaussian blur", WINDOW_AUTOSIZE); // Gaussian blur â ���
	imshow("Gaussian blur", h_f); // â�� h_f �̹��� ������

	namedWindow("Laplacian filter", WINDOW_AUTOSIZE); // Laplacian filter â ���
	imshow("Laplacian filter", Laplacian); // â�� Laplacian �̹��� ������

	waitKey(0);

	return 0;
}


// Gaussian filtering
// �Ű�����: input = ���͸� ������ �̹���, n = neighbor�� ����
// sigma_t = t ���� gaussian ������ ǥ������(x-coordinate), sigma_s = s ���� gaussian ������ ǥ������(y-coordinate)
Mat Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s) {

	int row = input.rows;
	int col = input.cols;

	// generate gaussian kernel
	Mat kernel = get_Gaussian_Kernel(n, sigma_t, sigma_s, true);
	Mat output = Mat::zeros(row, col, input.type()); // output�� input�� ũ��, Ÿ���� ���� matrix�̰� ���� 0���� �ʱ�ȭ

	//Intermediate data generation for mirroring
	Mat input_mirror = Mirroring(input, n);

	//convolution
	for (int i = n; i < row + n; i++) {
		for (int j = n; j < col + n; j++) {

			double sum_r = 0.0;
			double sum_g = 0.0;
			double sum_b = 0.0;
			for (int a = -n; a <= n; a++) { // for each kernel window
				for (int b = -n; b <= n; b++) {
					sum_r += kernel.at<double>(a + n, b + n) * (double)(input_mirror.at<Vec3b>(i + a, j + b)[0]);
					sum_g += kernel.at<double>(a + n, b + n) * (double)(input_mirror.at<Vec3b>(i + a, j + b)[1]);
					sum_b += kernel.at<double>(a + n, b + n) * (double)(input_mirror.at<Vec3b>(i + a, j + b)[2]);
					// �� ä�κ��� filtering�� neighbor ������ weight�� ���ϰ� �̸� ������
				}
			}
			output.at<Vec3b>(i - n, j - n)[0] = (double)sum_r; // ���� ���� output�� �� ä�κ� �ȼ��� �ݿ�
			output.at<Vec3b>(i - n, j - n)[1] = (double)sum_g;
			output.at<Vec3b>(i - n, j - n)[2] = (double)sum_b;

		}
	}

	return output; // filtering ������ output matrix�� return
}

// Laplacian filtering
// �Ű�����: input = ���͸� ������ �̹���
Mat Laplacianfilter(const Mat input) {

	int row = input.rows;
	int col = input.cols;

	// generate laplacian kernel
	Mat kernel = get_Laplacian_Kernel();
	Mat output = Mat::zeros(row, col, CV_64F);
	// laplacian filter�� output�� grayscale�̹Ƿ� output�� 64bit float, ä�� 1�� Ÿ���� matrix

	int n = 1; // laplacian filter kernel N
	//Intermediate data generation for mirroring
	Mat input_mirror = Mirroring(input, n);

	for (int i = n; i < row + n; i++) {
		for (int j = n; j < col + n; j++) {
			
			double sum_r = 0.0;
			double sum_g = 0.0;
			double sum_b = 0.0;

			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					sum_r += kernel.at<double>(a + n, b + n) * (double)(input_mirror.at<Vec3b>(i + a, j + b)[0]);
					sum_g += kernel.at<double>(a + n, b + n) * (double)(input_mirror.at<Vec3b>(i + a, j + b)[1]);
					sum_b += kernel.at<double>(a + n, b + n) * (double)(input_mirror.at<Vec3b>(i + a, j + b)[2]);
					// �� ä�κ��� filtering�� neighbor ������ weight�� ���ϰ� �̸� ������
				}
			}
			double out_r = abs(sum_r);  // �� ä�κ��� ���� ���� ������ output�� �ȼ��� �ݿ�
			double out_g = abs(sum_g);
			double out_b = abs(sum_b);
			double out1 = saturate_cast<uchar>(out_r); // ���� 0~255 ���� ���� �Ͽ� out ������ ����
			double out2 = saturate_cast<uchar>(out_g);
			double out3 = saturate_cast<uchar>(out_b);
			//max �Լ��� ����Ͽ� ���� ū ���� output�� �ݿ�
			output.at<double>(i - n, j - n) = max({ out1, out2, out3 });
		}
	}

	return output; // filtering ������ output matrix�� return
}

// boundary process = mirroring
// �Ű�����: input = mirroring�� �̹���, n = neighbor�� ����
Mat Mirroring(const Mat input, int n)
{
	int row = input.rows;
	int col = input.cols;

	Mat input2 = Mat::zeros(row + 2 * n, col + 2 * n, input.type());
	int row2 = input2.rows;
	int col2 = input2.cols;

	// input2�� �߾ӿ� ���� �̹����� ����
	for (int i = n; i < row + n; i++) {
		for (int j = n; j < col + n; j++) {
			input2.at<Vec3b>(i, j)[0] = input.at<Vec3b>(i - n, j - n)[0];
			input2.at<Vec3b>(i, j)[1] = input.at<Vec3b>(i - n, j - n)[1];
			input2.at<Vec3b>(i, j)[2] = input.at<Vec3b>(i - n, j - n)[2];
		}
	}
	// �̹����� �¿� ��Ī�ǵ��� mirroing
	for (int i = n; i < row + n; i++) {
		for (int j = 0; j < n; j++) {
			input2.at<Vec3b>(i, j)[0] = input2.at<Vec3b>(i, 2 * n - j)[0];
			input2.at<Vec3b>(i, j)[1] = input2.at<Vec3b>(i, 2 * n - j)[1];
			input2.at<Vec3b>(i, j)[2] = input2.at<Vec3b>(i, 2 * n - j)[2];
		}
		for (int j = col + n; j < col2; j++) {
			input2.at<Vec3b>(i, j)[0] = input2.at<Vec3b>(i, 2 * col - 2 + 2 * n - j)[0];
			input2.at<Vec3b>(i, j)[1] = input2.at<Vec3b>(i, 2 * col - 2 + 2 * n - j)[1];
			input2.at<Vec3b>(i, j)[2] = input2.at<Vec3b>(i, 2 * col - 2 + 2 * n - j)[2];
		}
	}
	// �̹����� ���� ��Ī�ǵ��� mirroring
	for (int j = 0; j < col2; j++) {
		for (int i = 0; i < n; i++) {
			input2.at<Vec3b>(i, j)[0] = input2.at<Vec3b>(2 * n - i, j)[0];
			input2.at<Vec3b>(i, j)[1] = input2.at<Vec3b>(2 * n - i, j)[1];
			input2.at<Vec3b>(i, j)[2] = input2.at<Vec3b>(2 * n - i, j)[2];
		}
		for (int i = row + n; i < row2; i++) {
			input2.at<Vec3b>(i, j)[0] = input2.at<Vec3b>(2 * row - 2 + 2 * n - i, j)[0];
			input2.at<Vec3b>(i, j)[1] = input2.at<Vec3b>(2 * row - 2 + 2 * n - i, j)[1];
			input2.at<Vec3b>(i, j)[2] = input2.at<Vec3b>(2 * row - 2 + 2 * n - i, j)[2];
		}
	}

	return input2; // input2 Mat ��ȯ
}

// Gaussian kernel �� ����
// �Ű�����: n = neighbor�� ����, sigma_t =  t ���� gaussian ������ ǥ������(x-coordinate),
// sigma_s = s ���� gaussian ������ ǥ������(y-coordinate), normalize = ����ȭ ����
Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize) {

	int kernel_size = (2 * n + 1); // (2N+1)*(2N+1)
	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_64F);
	// kernel matrix�� kernel size�� �°� 64bit float���·� �����ϰ�, ���� 0���� �ʱ�ȭ
	double kernel_sum = 0.0;

	for (int i = -n; i <= n; i++) {
		for (int j = -n; j <= n; j++) {
			// gaussian �Լ��� �̿��� kernel �� ������Ʈ
			kernel.at<double>(i + n, j + n) = exp(-((i * i) / (2.0 * sigma_t * sigma_t) + (j * j) / (2.0 * sigma_s * sigma_s)));
			kernel_sum += kernel.at<double>(i + n, j + n); // ����ȭ�� �ϱ� ���� kernel_sum�� �� weight�� ����� ���� ���� ����
		}
	}

	if (normalize) { // ����ȭ�Ѵٸ�,
		for (int i = 0; i < kernel_size; i++)
			for (int j = 0; j < kernel_size; j++)
				kernel.at<double>(i, j) /= kernel_sum;		// kernel_sum ���� ������ ����ȭ
	}

	return kernel; // gaussian kernel�� return
}

// Laplacian kernel �� ����
Mat get_Laplacian_Kernel() {

	Mat kernel = Mat::zeros(3, 3, CV_64F);
	// kernel�� 64bit float������ 3*3 matirx�� �����ϰ� 0���� �ʱ�ȭ

	// weight �� �־���
	kernel.at<double>(0, 1) = 1.0;
	kernel.at<double>(2, 1) = 1.0;
	kernel.at<double>(1, 0) = 1.0;
	kernel.at<double>(1, 2) = 1.0;
	kernel.at<double>(1, 1) = -4.0;

	return kernel; // laplacian kernel�� return
}