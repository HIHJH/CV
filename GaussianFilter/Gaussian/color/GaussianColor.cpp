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

	start = clock(); // 수행 시간 측정 시작

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input 이미지 불러옴
	Mat output; // 최종 이미지를 저장할 Mat


	if (!input.data)
	{
		std::cout << "Could not open" << std::endl; // 이미지 파일 데이터에 오류가 있으면 메세지 출력
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE); // Original 창 띄움
	imshow("Original", input); // 창에 input 이미지 보여줌
	output = gaussianfilter(input, 1, 1, 1, "zero-paddle");
	// output은 input 이미지에 gaussianfilter를 적용한 것
	// filter는 n = 1, sigmaT = 1, sigmaS = 1로 기본 설정
	// Boundary process는 zero-paddle, mirroring, adjustkernel 중 하나를 선택하면 됨.
	namedWindow("Gaussian Filter", WINDOW_AUTOSIZE); // Gaussian Filter 창 띄움
	imshow("Gaussian Filter", output); // 창에 output 이미지 보여줌

	end = clock(); // 수행 시간 측정 끝
	result = (double)(end - start); // 걸린 시간 계산

	printf("%f", result); // 수행 시간 출력

	waitKey(0);

	return 0;
}

// gaussian filter
// 매개변수: input = 필터를 적용할 이미지, n = neighbor의 범위, sigmaT =  t 관련 gaussian 분포의 표준편차,
// sigmaS = s 관련 gaussian 분포의 표준편차, opt = boundary process 방법
Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1); // (2N+1)*(2N+1) 
	int tempa;
	int tempb;
	float denom;

	// Initialiazing Kernel Matrix 
	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);
	// kernel matrix를 kernel size에 맞게 32bit float형태로 설정하고, 값은 0으로 초기화

	denom = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))) - (pow(b, 2) / (2 * pow(sigmaT, 2)))); // gaussian filter weight의 분자값 계산
			kernel.at<float>(a + n, b + n) = value1; // 각 커널에 분자값(value1) 업데이트
			denom += value1; // 가우시안 filter weight의 분모를 계산하기 위해 denom에 각 분자를 계산한 값의 합을 저장
		}
	}

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom; // gaussian filter weight의 분모값(denom)까지 반영하여 kernel 업데이트
		}
	}

	Mat output = Mat::zeros(row, col, input.type());  // output은 input 크기, 타입인 matrix이고, 값은 0으로 초기화


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) { // output의 각 픽셀마다 반복


			if (!strcmp(opt, "zero-paddle")) { // boundary process가 zero-paddle 일 때
				float sum_r = 0.0;
				float sum_g = 0.0;
				float sum_b = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							// 픽셀이 border 밖에 있으면 픽셀 값을 0으로 여기므로, 안에 있는 경우만 계산해줌
							sum_r += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[0]); // 각 채널별로 filtering 할 neighbor 각각에 weight를 곱하고 이를 더해줌 
							sum_g += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[1]); 
							sum_b += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[2]);
						}
					}
				}
				output.at<C>(i, j)[0] = (G)sum_r; // 더한 값을 output의 각 채널별 픽셀에 반영
				output.at<C>(i, j)[1] = (G)sum_g;
				output.at<C>(i, j)[2] = (G)sum_b;
			}

			else if (!strcmp(opt, "mirroring")) { // boundary process가 mirroring 일 때
				float sum_r = 0.0;
				float sum_g = 0.0;
				float sum_b = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {
						if (i + a > row - 1) {  // filter가 아래쪽 boundary를 초과했을 때,
							tempa = i - a; // boundary 쪽 값을 mirroring
						}
						else if (i + a < 0) { // filter가 위쪽 boundary를 초과했을 때,
							tempa = -(i + a); // boundary 쪽 값을 mirroring
						}
						else {
							tempa = i + a;
						}
						if (j + b > col - 1) { // filter가 오른쪽 boundary를 초과했을 때,
							tempb = j - b; // boundary 쪽 값을 mirroring
						}
						else if (j + b < 0) { // filter가 왼쪽 boundary를 초과했을 때,
							tempb = -(j + b); // boundary 쪽 값을 mirroring
						}
						else {
							tempb = j + b;
						}
						// 각 채널별로 filtering 할 neighbor 각각에 weight를 곱하고 이를 더해줌
						sum_r += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[0]);
						sum_g += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[1]);
						sum_b += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[2]);
					}
				}
				output.at<C>(i, j)[0] = (G)sum_r; // 더한 값을 output의 각 채널별 픽셀에 반영
				output.at<C>(i, j)[1] = (G)sum_g;
				output.at<C>(i, j)[2] = (G)sum_b;
			}


			else if (!strcmp(opt, "adjustkernel")) { // boundary process가 adjustkernel 일 때,
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							// 픽셀이 border 밖에 있으면 픽셀 값을 0으로 여기므로, 안에 있는 경우만 계산해줌
							// 각 채널별로 filtering 할 neighbor 각각에 weight를 곱하고 이를 다 더해서 sum1에 저장
							sum1_r += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[0]);
							sum1_g += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[1]);
							sum1_b += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[2]);
							sum2 += kernel.at<float>(a + n, b + n); // 각 weight 값들만 더해서 sum2에 저장
						}
					}
				}
				output.at<C>(i, j)[0] = (G)(sum1_r / sum2); // sum1을 sum2로 나눈 값을 output 각 채널별 픽셀에 반영
				output.at<C>(i, j)[1] = (G)(sum1_g / sum2);
				output.at<C>(i, j)[2] = (G)(sum1_b / sum2);
			}
		}
	}
	return output; // filtering 적용한 output matrix 를 return
}