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

	Mat input = imread("writing.jpg", CV_LOAD_IMAGE_COLOR); // input 이미지 불러옴
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

	output = adaptive_thres(input_gray, 2, 0.9);
	// output은 input_gray 이미지에 adaptive thresholding을 수행한 것
	// filtering 시, neighbor의 범위 = 2, weight = 0.9로 기본 설정
	// filtering은 boundary process는 zero padding을 사용한 uniform mean filtering을 수행하도록 고정됨

	namedWindow("Adaptive_threshold", WINDOW_AUTOSIZE); // Adaptive 창 띄움
	imshow("Adaptive_threshold", output); // 창에 output 이미지 보여줌


	waitKey(0);

	return 0;
}

// Adaptive thresholding
// filtering은 boundary process는 zero padding을 사용한 uniform mean filtering을 수행하도록 고정됨
// 매개변수: input = thresholding할 이미지, n = neighbor의 범위, bnumber = weight
Mat adaptive_thres(const Mat input, int n, float bnumber) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1); // (2N+1)*(2N*1)

	// Initialiazing Kernel Matrix 
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
	// kernel matrix를 kernel size에 맞게 32bit float형태로 설정
	// kernel matrix의 각 값은 1 / (kernel_size * kernel_size) 로 초기화
	float kernelvalue = kernel.at<float>(0, 0);
	// filter가 uniform 하기 때문에, 모든 kernel의 요소들은 같은 값을 가짐. 그 값을 kernelvalue에 저장

	Mat output = Mat::zeros(row, col, input.type()); // output은 input 크기, 타입인 matrix이고, 값은 0으로 초기화
	

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) { // output의 각 픽셀마다 반복
			// uniform mean filtering with zero paddle border process.
			float sum1 = 0.0;
			for (int a = -n; a <= n; a++) { // for each kernel window
				for (int b = -n; b <= n; b++) {

					if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
						// 픽셀이 border 밖에 있으면 픽셀 값을 0으로 여기므로, 안에 있는 경우만 계산해줌
						sum1 += kernelvalue * (float)(input.at<G>(i + a, j + b));
						// filtering 할 neighbor 각각에 weight(kernelvalue)를 곱하고 이를 더해줌 
					}
				}
			}
			float temp = bnumber * (G)sum1; // threshold = weight * mean intensity
			//thresholding
			if (input.at<G>(i, j) > temp)
				output.at<G>(i, j) = 255; // input 이미지의 intensity가 temp 값보다 크면 output image의 intensity = 255
			else
				output.at<G>(i, j) = 0; // input 이미지의 intensity가 temp 값보다 작으면 output image의 intensity = 0
		}
	}
	return output; // output matrix를 return
}