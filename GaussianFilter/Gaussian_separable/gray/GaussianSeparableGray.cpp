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

Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);

int main() {
	clock_t start, end;
	double result;

	start = clock(); // ���� �ð� ���� ����

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
	output = gaussianfilter(input_gray, 1, 1, 1, "zero-paddle");
	// output�� input_gray �̹����� gaussianfilter�� ������ ��
	// filter�� n = 1, sigmaT = 1, sigmaS = 1�� �⺻ ����
	// Boundary process�� zero-paddle, mirroring, adjustkernel �� �ϳ��� �����ϸ� ��.
	namedWindow("Gaussian Filter", WINDOW_AUTOSIZE); // Gaussian Filter â ���
	imshow("Gaussian Filter", output); // â�� output �̹��� ������

	end = clock(); // ���� �ð� ���� ��
	result = (double)(end - start); // �ɸ� �ð� ���

	printf("%f", result); // ���� �ð� ���

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

	// Initialiazing Kernel Matrix 
	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);
	// kernel matrix�� kernel size�� �°� 32bit float���·� �����ϰ�, ���� 0���� �ʱ�ȭ

	// Ws(s)�� Wt(t)�� ���� ����ϰ� �����־� kernel�� �ݿ�
	// Ws(s) ���
	float* ws = new float[kernel_size]; // kernel_size ũ�� ��ŭ �迭�� �����Ҵ�
	float denom_s = 0.0;
	for (int a = -n; a <= n; a++) {
		ws[a+n] = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2)))); // Ws(s)�� ���ڰ� ����Ͽ� �迭�� ����
		denom_s += ws[a+n]; // Ws(s)�� �и� ����ϱ� ���� denom_s �� �� ���ڸ� ����� ���� ���� ����
	}
	for (int a = -n; a <= n; a++) {
		ws[a + n] /= denom_s; // Ws(s)�� �и𰪱��� �ݿ��Ͽ� �迭 �� ������Ʈ
	}

	// Wt(t) ��� 
	float* wt = new float[kernel_size]; // kernel_size ũ�� ��ŭ �迭�� �����Ҵ�
	float denom_t = 0.0;
	for (int b = -n; b <= n; b++) {
		wt[b + n] = exp(-(pow(b, 2) / (2 * pow(sigmaT, 2)))); // Wt(t)�� ���ڰ� ����Ͽ� �迭�� ����
		denom_t += wt[b + n]; // Wt(t)�� �и� ����ϱ� ���� denom_t �� �� ���ڸ� ����� ���� ���� ����
	}
	for (int b = -n; b <= n; b++) {
		wt[b + n] /= denom_t; // Wt(t)�� �и𰪱��� �ݿ��Ͽ� �迭 �� ������Ʈ
	}

	//�� weight ���
	for (int a = -n; a <= n; a++) {
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) = ws[a + n] * wt[b + n]; // Ws(s) ���� Wt(t) ���� �����־�, ���� gaussian filter�� weight ���� kernel matrix�� ������Ʈ
		}
	}

	delete[] ws; // ���� �Ҵ� �ߴ� �޸� ����
	delete[] wt;

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