#include <iostream>
#include <opencv2/opencv.hpp>

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

Mat adaptive_thres(const Mat input, int n, float b);

int main() {

	Mat input = imread("writing.jpg", CV_LOAD_IMAGE_COLOR); // input �̹��� �ҷ���
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

	output = adaptive_thres(input_gray, 2, 0.9);
	// output�� input_gray �̹����� adaptive thresholding�� ������ ��
	// filtering ��, neighbor�� ���� = 2, weight = 0.9�� �⺻ ����
	// filtering�� boundary process�� zero padding�� ����� uniform mean filtering�� �����ϵ��� ������

	namedWindow("Adaptive_threshold", WINDOW_AUTOSIZE); // Adaptive â ���
	imshow("Adaptive_threshold", output); // â�� output �̹��� ������


	waitKey(0);

	return 0;
}

// Adaptive thresholding
// filtering�� boundary process�� zero padding�� ����� uniform mean filtering�� �����ϵ��� ������
// �Ű�����: input = thresholding�� �̹���, n = neighbor�� ����, bnumber = weight
Mat adaptive_thres(const Mat input, int n, float bnumber) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1); // (2N+1)*(2N*1)

	// Initialiazing Kernel Matrix 
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
	// kernel matrix�� kernel size�� �°� 32bit float���·� ����
	// kernel matrix�� �� ���� 1 / (kernel_size * kernel_size) �� �ʱ�ȭ
	float kernelvalue = kernel.at<float>(0, 0);
	// filter�� uniform �ϱ� ������, ��� kernel�� ��ҵ��� ���� ���� ����. �� ���� kernelvalue�� ����

	Mat output = Mat::zeros(row, col, input.type()); // output�� input ũ��, Ÿ���� matrix�̰�, ���� 0���� �ʱ�ȭ
	

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) { // output�� �� �ȼ����� �ݺ�
			// uniform mean filtering with zero paddle border process.
			float sum1 = 0.0;
			for (int a = -n; a <= n; a++) { // for each kernel window
				for (int b = -n; b <= n; b++) {

					if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
						// �ȼ��� border �ۿ� ������ �ȼ� ���� 0���� ����Ƿ�, �ȿ� �ִ� ��츸 �������
						sum1 += kernelvalue * (float)(input.at<G>(i + a, j + b));
						// filtering �� neighbor ������ weight(kernelvalue)�� ���ϰ� �̸� ������ 
					}
				}
			}
			float temp = bnumber * (G)sum1; // threshold = weight * mean intensity
			//thresholding
			if (input.at<G>(i, j) > temp)
				output.at<G>(i, j) = 255; // input �̹����� intensity�� temp ������ ũ�� output image�� intensity = 255
			else
				output.at<G>(i, j) = 0; // input �̹����� intensity�� temp ������ ������ output image�� intensity = 0
		}
	}
	return output; // output matrix�� return
}