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
	
	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input 이미지 불러옴
	Mat input_gray; // 흑백 이미지를 저장할 Mat
	Mat output; // 최종 이미지를 저장할 Mat

	cvtColor(input, input_gray, CV_RGB2GRAY); // RGB 이미지를 흑백 이미지로 변환
	

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl; // 이미지 파일 데이터에 오류가 있으면 메세지 출력
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE); // Grayscale 창 띄움
	imshow("Grayscale", input_gray); // 창에 input_gray 이미지 보여줌
	output = meanfilter(input_gray, 3, "mirroring");
	// output은 input_gray 이미지에 meanfilter를 적용한 것
	// filter는 n = 3으로 기본 설정
	// Boundary process는 zero-paddle, mirroring, adjustkernel 중 하나를 선택하면 됨.
	namedWindow("Mean Filter", WINDOW_AUTOSIZE); // Mean Filter 창 띄움
	imshow("Mean Filter", output); //창에 output 이미지 보여줌


	waitKey(0);

	return 0;
}

// mean filter
// 매개변수: input = 필터를 적용할 이미지, n = neighbor의 범위, opt = boundary process 방법
Mat meanfilter(const Mat input, int n, const char* opt) {

	Mat kernel;
	
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1); // (2N+1)*(2N+1) 
	int tempa; 
	int tempb;

 // Initialiazing Kernel Matrix 
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
	// kernel matrix를 kernel size에 맞게 32bit float형태로 설정
	// kernel matrix의 각 값은 1 / (kernel_size * kernel_size) 로 초기화
	float kernelvalue=kernel.at<float>(0, 0);
	// filter가 uniform 하기 때문에, 모든 kernel의 요소들은 같은 값을 가짐. 그 값을 kernelvalue에 저장
	
	Mat output = Mat::zeros(row, col, input.type()); // output은 input 크기, 타입인 matrix이고, 값은 0으로 초기화
	

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {  // output의 각 픽셀마다 반복
			
			if (!strcmp(opt, "zero-paddle")) { // boundary process가 zero-paddle 일 때
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							// 픽셀이 border 밖에 있으면 픽셀 값을 0으로 여기므로, 안에 있는 경우만 계산해줌
							sum1 += kernelvalue*(float)(input.at<G>(i + a, j + b));
							// filtering 할 neighbor 각각에 weight(kernelvalue)를 곱하고 이를 더해줌 
						}
					}
				}
				output.at<G>(i, j) = (G)sum1; // 더한 값을 output의 픽셀에 반영
			}
			
			else if (!strcmp(opt, "mirroring")) { // boundary process가 mirroring 일 때
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
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
						sum1 += kernelvalue*(float)(input.at<G>(tempa, tempb)); // filtering 할 neighbor 각각에 weight를 곱하고 이를 더해줌
					}
				}
				output.at<G>(i, j) = (G)sum1; // 더한 값을 output의 픽셀에 반영
			}
			
			else if (!strcmp(opt, "adjustkernel")) { // boundary process가 adjustkernel 일 때,
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							// 픽셀이 border 밖에 있으면 픽셀 값을 0으로 여기므로, 안에 있는 경우만 계산해줌
							sum1 += kernelvalue*(float)(input.at<G>(i + a, j + b)); // filtering 할 neighbor 각각에 weight를 곱하고 이를 다 더하여 sum1에 저장
							sum2 += kernelvalue; // 각 weight 값들만 더해서 sum2에 저장
						}
					}
				}
				output.at<G>(i, j) = (G)(sum1/sum2); // sum1을 sum2로 나눈 값을 output 픽셀에 반영
			}
		}
	}
	return output; // filtering 적용한 output matrix 를 return
}