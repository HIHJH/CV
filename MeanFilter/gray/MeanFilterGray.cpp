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

Mat meanfilter(const Mat input, int n, const char* opt);

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
	output = meanfilter(input_gray, 3, "mirroring");
	// output�� input_gray �̹����� meanfilter�� ������ ��
	// filter�� n = 3���� �⺻ ����
	// Boundary process�� zero-paddle, mirroring, adjustkernel �� �ϳ��� �����ϸ� ��.
	namedWindow("Mean Filter", WINDOW_AUTOSIZE); // Mean Filter â ���
	imshow("Mean Filter", output); //â�� output �̹��� ������


	waitKey(0);

	return 0;
}

// mean filter
// �Ű�����: input = ���͸� ������ �̹���, n = neighbor�� ����, opt = boundary process ���
Mat meanfilter(const Mat input, int n, const char* opt) {

	Mat kernel;
	
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1); // (2N+1)*(2N+1) 
	int tempa; 
	int tempb;

 // Initialiazing Kernel Matrix 
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
	// kernel matrix�� kernel size�� �°� 32bit float���·� ����
	// kernel matrix�� �� ���� 1 / (kernel_size * kernel_size) �� �ʱ�ȭ
	float kernelvalue=kernel.at<float>(0, 0);
	// filter�� uniform �ϱ� ������, ��� kernel�� ��ҵ��� ���� ���� ����. �� ���� kernelvalue�� ����
	
	Mat output = Mat::zeros(row, col, input.type()); // output�� input ũ��, Ÿ���� matrix�̰�, ���� 0���� �ʱ�ȭ
	

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {  // output�� �� �ȼ����� �ݺ�
			
			if (!strcmp(opt, "zero-paddle")) { // boundary process�� zero-paddle �� ��
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							// �ȼ��� border �ۿ� ������ �ȼ� ���� 0���� ����Ƿ�, �ȿ� �ִ� ��츸 �������
							sum1 += kernelvalue*(float)(input.at<G>(i + a, j + b));
							// filtering �� neighbor ������ weight(kernelvalue)�� ���ϰ� �̸� ������ 
						}
					}
				}
				output.at<G>(i, j) = (G)sum1; // ���� ���� output�� �ȼ��� �ݿ�
			}
			
			else if (!strcmp(opt, "mirroring")) { // boundary process�� mirroring �� ��
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
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
						sum1 += kernelvalue*(float)(input.at<G>(tempa, tempb)); // filtering �� neighbor ������ weight�� ���ϰ� �̸� ������
					}
				}
				output.at<G>(i, j) = (G)sum1; // ���� ���� output�� �ȼ��� �ݿ�
			}
			
			else if (!strcmp(opt, "adjustkernel")) { // boundary process�� adjustkernel �� ��,
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							// �ȼ��� border �ۿ� ������ �ȼ� ���� 0���� ����Ƿ�, �ȿ� �ִ� ��츸 �������
							sum1 += kernelvalue*(float)(input.at<G>(i + a, j + b)); // filtering �� neighbor ������ weight�� ���ϰ� �̸� �� ���Ͽ� sum1�� ����
							sum2 += kernelvalue; // �� weight ���鸸 ���ؼ� sum2�� ����
						}
					}
				}
				output.at<G>(i, j) = (G)(sum1/sum2); // sum1�� sum2�� ���� ���� output �ȼ��� �ݿ�
			}
		}
	}
	return output; // filtering ������ output matrix �� return
}