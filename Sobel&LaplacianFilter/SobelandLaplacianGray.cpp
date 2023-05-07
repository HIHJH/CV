#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

using namespace cv;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
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

Mat sobelfilter(const Mat input);
Mat laplacianfilter(const Mat input);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input �̹��� �ҷ���
	Mat input_gray; // ��� �̹����� ������ Mat
	Mat output, output2; // ���� �̹����� ������ Mat


	cvtColor(input, input_gray, CV_RGB2GRAY); // RGB �̹����� ��� �̹����� ��ȯ



	if (!input.data)
	{
		std::cout << "Could not open" << std::endl; // �̹��� ���� �����Ϳ� ������ ������ �޼��� ���
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE); // Grayscale â ���
	imshow("Grayscale", input_gray); // â�� input_gray �̹��� ������
	output = sobelfilter(input_gray);
	// output�� input_gray �̹����� sobelfilter�� ������ ��
	namedWindow("Sobel Filter", WINDOW_AUTOSIZE); // Sobel Filter â ���
	imshow("Sobel Filter", output); // â�� output �̹��� ������
	output2 = laplacianfilter(input_gray);
	// output2�� input_gray �̹����� laplacianfilter�� ������ ��
	namedWindow("Laplacian Filter", WINDOW_AUTOSIZE); // Laplacian Filter â ���
	imshow("Laplacian Filter", output2); // â�� output2 �̹��� ������


	waitKey(0);

	return 0;
}

// sobel filter
// �Ű�����: input = ���͸� ������ �̹���
Mat sobelfilter(const Mat input) {

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N

	// Initialiazing 2 Kernel Matrix with 3x3 size for Sx and Sy
	//Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)
	int data_x[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 }; // Sx filter�� kernel weight ����
	Mat Sx(3, 3, CV_32S, data_x); // Sx matrix�� 3*3 ũ���� 32 bit int ���·� �����ϰ�, ���� data_x�� ����.
	int data_y[] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 }; // Sy filter�� kernel weight ����
	Mat Sy(3, 3, CV_32S, data_y); // Sy matrix�� 3*3 ũ���� 32 bit int ���·� �����ϰ�, ���� data_y�� ����.

	Mat output = Mat::zeros(row, col, input.type()); // output�� input ũ��, Ÿ���� matrix�̰�, ���� 0���� �ʱ�ȭ

	int tempa, tempb;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) { // output�� �� �ȼ����� �ݺ�
			float sum_x = 0.0;
			float sum_y = 0.0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					if (i + a > row - 1) {  // filter�� �Ʒ��� boundary�� �ʰ����� ��,
						tempa = i - a; // boundary �� ���� mirroring
					}
					else if (i + a < 0) { // filter�� ���� boundary�� �ʰ����� ��,
						tempa = -(i + a); // boundary �� ���� mirroring
					}
					else {
						tempa = i + a;
					}
					if (j + b > col - 1) { // filter�� ������ boundary�� �ʰ����� ��,
						tempb = j - b; // boundary �� ���� mirroring
					}
					else if (j + b < 0) { // filter�� ���� boundary�� �ʰ����� ��,
						tempb = -(j + b); // boundary �� ���� mirroring
					}
					else {
						tempb = j + b;
					}
					sum_x += Sx.at<int>( a + n, b + n ) * (float)(input.at<G>(tempa, tempb)); // filtering �� neighbor ������ Sx weight�� ���ϰ� �̸� ������
					sum_y += Sy.at<int>( a + n, b + n ) * (float)(input.at<G>(tempa, tempb)); // filtering �� neighbor ������ Sy weight�� ���ϰ� �̸� ������
				}
			}
			float out = sqrt(sum_x * sum_x + sum_y * sum_y); // sum_x,y�� ���� ������ ���� ���ϰ� ��Ʈ ���� output�� �ȼ��� �ݿ�
			output.at<G>(i, j) = saturate_cast<G>(out); // ���� 0-255 ���� ���� �Ͽ� output �� ä�κ� �ȼ��� �ݿ�
		}
	}
	return output; // filtering ������ output matrix �� return
}

// laplacian filter
// �Ű�����: input = ���͸� ������ �̹���
Mat laplacianfilter(const Mat input) {

	int row = input.rows;
	int col = input.cols;
	int n = 1; // laplacian Filter Kernel N

	int data[] = {0, 1, 0, 1, -4, 1, 0, 1, 0}; // Laplacian filter�� kernel weight ����
	Mat kernel(3, 3, CV_32S, data); // kernel matrix�� 3*3 ũ���� 32 bit int ���·� �����ϰ�, ���� data�� ����.


	Mat output = Mat::zeros(row, col, input.type()); // output�� input ũ��, Ÿ���� matrix�̰�, ���� 0���� �ʱ�ȭ

	int tempa, tempb;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) { // output�� �� �ȼ����� �ݺ�
			float sum = 0.0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					if (i + a > row - 1) {  // filter�� �Ʒ��� boundary�� �ʰ����� ��,
						tempa = i - a; // boundary �� ���� mirroring
					}
					else if (i + a < 0) { // filter�� ���� boundary�� �ʰ����� ��,
						tempa = -(i + a); // boundary �� ���� mirroring
					}
					else {
						tempa = i + a;
					}
					if (j + b > col - 1) { // filter�� ������ boundary�� �ʰ����� ��,
						tempb = j - b; // boundary �� ���� mirroring
					}
					else if (j + b < 0) { // filter�� ���� boundary�� �ʰ����� ��,
						tempb = -(j + b); // boundary �� ���� mirroring
					}
					else {
						tempb = j + b;
					}
					sum += kernel.at<int>( a + n, b + n ) * (float)(input.at<G>(tempa, tempb));
					// filtering �� neighbor ������ weight�� ���ϰ� �̸� ������
				}
			}
			float out = abs(sum); // ���� ���� ������ output�� �ȼ��� �ݿ�
			output.at<G>(i, j) = saturate_cast<G>(out); // ���� 0-255 ���� ���� �Ͽ� output �� ä�κ� �ȼ��� �ݿ�
		}
	}
	return output; // filtering ������ output matrix �� return
}