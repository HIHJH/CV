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

Mat UnsharpMask(const Mat input, int n, float sigmaT, float sigmaS, const char* opt, float k);
Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input �̹��� �ҷ���
	Mat input_gray; // ��� �̹����� ������ Mat
	Mat output; // ���� �̹����� ������ Mat


	cvtColor(input, input_gray, CV_RGB2GRAY); // RGB �̹����� ��� �̹����� ��ȯ

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl; // �̹��� ���� �����Ϳ� ������ ������ �޼��� ���
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE); // Grayscale â ���
	imshow("Grayscale", input_gray); // â�� input_gray �̹��� ������
	output = UnsharpMask(input_gray, 1, 1, 1, "zero-paddle", 0.5);
	// output�� input_gray �̹����� UnsharpMasking�� ������ ��
	// filter�� n = 1, sigmaT = 1, sigmaS = 1 �� �⺻ ����
	// Boundary process�� zero-paddle, mirroring, adjustkernel �� �ϳ��� �����ϸ� ��.
	// k�� 0.5�� �⺻ ����
	namedWindow("UnsharpMasking", WINDOW_AUTOSIZE); // UnsharpMasking â ���
	imshow("UnsharpMasking", output); // â�� output �̹��� ������


	waitKey(0);

	return 0;
}


// gaussian filter
// �Ű�����: input = ���͸� ������ �̹���, n = neighbor�� ����, sigmaT =  t ���� gaussian ������ ǥ������,
// sigmaS = s ���� gaussian ������ ǥ������, opt = boundary process ���
Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1); // (2N+1)*(2N+1) 
	int tempa;
	int tempb;
	float denom;
	float kernelvalue;

	// Initialiazing Kernel Matrix 
	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);
	// kernel matrix�� kernel size�� �°� 32bit float���·� �����ϰ�, ���� 0���� �ʱ�ȭ


	denom = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))) - (pow(b, 2) / (2 * pow(sigmaT, 2)))); // gaussian filter weight�� ���ڰ� ���
			kernel.at<float>(a + n, b + n) = value1; // �� Ŀ�ο� ���ڰ�(value1) ������Ʈ
			denom += value1; // ����þ� filter weight�� �и� ����ϱ� ���� denom�� �� ���ڸ� ����� ���� ���� ����
		}
	}

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom; // gaussian filter weight�� �и�(denom)���� �ݿ��Ͽ� kernel ������Ʈ
		}
	}

	Mat output = Mat::zeros(row, col, input.type()); // output�� input ũ��, Ÿ���� matrix�̰�, ���� 0���� �ʱ�ȭ


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) { // output�� �� �ȼ����� �ݺ�

			if (!strcmp(opt, "zero-paddle")) { // boundary process�� zero-paddle �� ��
				float sum = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							// �ȼ��� border �ۿ� ������ �ȼ� ���� 0���� ����Ƿ�, �ȿ� �ִ� ��츸 �������
							sum += kernel.at<float>(a + n, b + n) * (float)(input.at<G>(i + a, j + b)); // filtering �� neighbor ������ weight�� ���ϰ� �̸� ������ 
						}
					}
				}
				output.at<G>(i, j) = (G)sum; // ���� ���� output�� �ȼ��� �ݿ�
			}

			else if (!strcmp(opt, "mirroring")) { // boundary process�� mirroring �� ��
				float sum = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {
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
						sum += kernel.at<float>(a + n, b + n) * (float)(input.at<G>(tempa, tempb)); // filtering �� neighbor ������ weight�� ���ϰ� �̸� ������
					}
				}
				output.at<G>(i, j) = (G)sum; // ���� ���� output�� �ȼ��� �ݿ�
			}


			else if (!strcmp(opt, "adjustkernel")) { // boundary process�� adjustkernel �� ��,
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							// �ȼ��� border �ۿ� ������ �ȼ� ���� 0���� ����Ƿ�, �ȿ� �ִ� ��츸 �������
							sum1 += kernel.at<float>(a + n, b + n) * (float)(input.at<G>(i + a, j + b)); // filtering �� neighbor ������ weight�� ���ϰ� �̸� �� ���Ͽ� sum1�� ����
							sum2 += kernel.at<float>(a + n, b + n); // �� weight ���鸸 ���ؼ� sum2�� ����
						}
					}
				}
				output.at<G>(i, j) = (G)(sum1 / sum2); // sum1�� sum2�� ���� ���� output �ȼ��� �ݿ�
			}
		}
	}
	return output; // filtering ������ output matrix �� return
}

//UnsharpMasking
// �Ű�����: input = ���͸� ������ �̹���, n = neighbor�� ����, sigmaT =  t ���� gaussian ������ ǥ������,
// sigmaS = s ���� gaussian ������ ǥ������, opt = boundary process ���
// k = low frequency�� ���̴� ���� �����ϴ� scaling�� �̿�Ǵ� �� (0<=k<=1)
// k�� Ŭ���� output�� sharper ������, k�� 0.5 ���Ϸ� �����
Mat UnsharpMask(const Mat input, int n, float sigmaT, float sigmaS, const char* opt, float k) {

	int row = input.rows;
	int col = input.cols;

	Mat output = Mat::zeros(row, col, input.type()); // output�� input ũ��, Ÿ���� matrix�̰�, ���� 0���� �ʱ�ȭ

	Mat L = gaussianfilter(input, n, sigmaT, sigmaS, opt); // input�� gaussian filter�� ������ matrix�� L�� ����

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float out = (input.at<G>(i, j) - k * L.at<G>(i, j)) / (1 - k); // output = (I - kL) / (1 - k)
			output.at<G>(i, j) = saturate_cast<G>(out); // ���� 0-255 ���� ���� �Ͽ� output �ȼ��� �ݿ�
		}
	}

	return output; // unsharp masking ������ output matrix �� return
}