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

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input 이미지 불러옴
	Mat input_gray; // 흑백 이미지를 저장할 Mat
	Mat output, output2; // 최종 이미지를 저장할 Mat


	cvtColor(input, input_gray, CV_RGB2GRAY); // RGB 이미지를 흑백 이미지로 변환



	if (!input.data)
	{
		std::cout << "Could not open" << std::endl; // 이미지 파일 데이터에 오류가 있으면 메세지 출력
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE); // Grayscale 창 띄움
	imshow("Grayscale", input_gray); // 창에 input_gray 이미지 보여줌
	output = sobelfilter(input_gray);
	// output은 input_gray 이미지에 sobelfilter를 적용한 것
	namedWindow("Sobel Filter", WINDOW_AUTOSIZE); // Sobel Filter 창 띄움
	imshow("Sobel Filter", output); // 창에 output 이미지 보여줌
	output2 = laplacianfilter(input_gray);
	// output2는 input_gray 이미지에 laplacianfilter를 적용한 것
	namedWindow("Laplacian Filter", WINDOW_AUTOSIZE); // Laplacian Filter 창 띄움
	imshow("Laplacian Filter", output2); // 창에 output2 이미지 보여줌


	waitKey(0);

	return 0;
}

// sobel filter
// 매개변수: input = 필터를 적용할 이미지
Mat sobelfilter(const Mat input) {

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N

	// Initialiazing 2 Kernel Matrix with 3x3 size for Sx and Sy
	//Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)
	int data_x[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 }; // Sx filter의 kernel weight 값들
	Mat Sx(3, 3, CV_32S, data_x); // Sx matrix를 3*3 크기의 32 bit int 형태로 설정하고, 값은 data_x를 넣음.
	int data_y[] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 }; // Sy filter의 kernel weight 값들
	Mat Sy(3, 3, CV_32S, data_y); // Sy matrix를 3*3 크기의 32 bit int 형태로 설정하고, 값은 data_y를 넣음.

	Mat output = Mat::zeros(row, col, input.type()); // output은 input 크기, 타입인 matrix이고, 값은 0으로 초기화

	int tempa, tempb;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) { // output의 각 픽셀마다 반복
			float sum_x = 0.0;
			float sum_y = 0.0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
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
					sum_x += Sx.at<int>( a + n, b + n ) * (float)(input.at<G>(tempa, tempb)); // filtering 할 neighbor 각각에 Sx weight를 곱하고 이를 더해줌
					sum_y += Sy.at<int>( a + n, b + n ) * (float)(input.at<G>(tempa, tempb)); // filtering 할 neighbor 각각에 Sy weight를 곱하고 이를 더해줌
				}
			}
			float out = sqrt(sum_x * sum_x + sum_y * sum_y); // sum_x,y를 각각 제곱한 것을 더하고 루트 씌워 output의 픽셀에 반영
			output.at<G>(i, j) = saturate_cast<G>(out); // 값을 0-255 범위 내로 하여 output 각 채널별 픽셀에 반영
		}
	}
	return output; // filtering 적용한 output matrix 를 return
}

// laplacian filter
// 매개변수: input = 필터를 적용할 이미지
Mat laplacianfilter(const Mat input) {

	int row = input.rows;
	int col = input.cols;
	int n = 1; // laplacian Filter Kernel N

	int data[] = {0, 1, 0, 1, -4, 1, 0, 1, 0}; // Laplacian filter의 kernel weight 값들
	Mat kernel(3, 3, CV_32S, data); // kernel matrix를 3*3 크기의 32 bit int 형태로 설정하고, 값은 data를 넣음.


	Mat output = Mat::zeros(row, col, input.type()); // output은 input 크기, 타입인 matrix이고, 값은 0으로 초기화

	int tempa, tempb;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) { // output의 각 픽셀마다 반복
			float sum = 0.0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
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
					sum += kernel.at<int>( a + n, b + n ) * (float)(input.at<G>(tempa, tempb));
					// filtering 할 neighbor 각각에 weight를 곱하고 이를 더해줌
				}
			}
			float out = abs(sum); // 더한 값의 절댓값을 output의 픽셀에 반영
			output.at<G>(i, j) = saturate_cast<G>(out); // 값을 0-255 범위 내로 하여 output 각 채널별 픽셀에 반영
		}
	}
	return output; // filtering 적용한 output matrix 를 return
}