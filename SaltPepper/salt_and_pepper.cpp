#include <opencv2/opencv.hpp>
#include <stdio.h>

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

Mat Add_salt_pepper_Noise(const Mat input, float ps, float pp);
Mat Salt_pepper_noise_removal_Gray(const Mat input, int n, const char *opt);
Mat Salt_pepper_noise_removal_RGB(const Mat input, int n, const char *opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input 이미지 불러옴
	Mat input_gray; // 흑백 이미지를 저장할 Mat

	// check for validation
	if (!input.data) {
		printf("Could not open\n"); // 이미지 파일에 데이터 오류가 있으면 메세지 출력
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// RGB 이미지를 흑백 이미지로 변환하여 input_gray에 저장 

	// 원본 이미지에 noise 추가
	Mat noise_Gray = Add_salt_pepper_Noise(input_gray, 0.1f, 0.1f);
	// 원본 흑백 이미지에 salt, pepper 각각 density가 0.1f인 salt and pepper noise를 추가하여 noise_Gray에 저장
	Mat noise_RGB = Add_salt_pepper_Noise(input, 0.1f, 0.1f);
	// 원본 이미지에 salt, pepper 각각 density가 0.1f인 salt and pepper noise를 추가하여 noise_RGB에 저장

	// Denoise, using median filter
	int window_radius = 2; // window radius를 2로 초기화
	Mat Denoised_Gray = Salt_pepper_noise_removal_Gray(noise_Gray, window_radius, "zero-padding");
	// noise_Gray에 window radius = 2인 median filtering를 적용하여 Salt and pepper noise를 제거
	// boundary process는 zero-padding 방법을 사용
	Mat Denoised_RGB = Salt_pepper_noise_removal_RGB(noise_RGB, window_radius, "adjustkernel");
	// noise_RGB에 window radius = 2인 median filtering를 적용하여 Salt and pepper noise를 제거
	// boundary process는 adjustkernel 방법을 사용

	namedWindow("Grayscale", WINDOW_AUTOSIZE); // Grayscale 창 띄움
	imshow("Grayscale", input_gray); // 창에 input_gray 이미지 보여줌
	
	namedWindow("RGB", WINDOW_AUTOSIZE); // RGB 창 띄움
	imshow("RGB", input); // 창에 input 이미지 보여줌

	namedWindow("Impulse Noise (Grayscale)", WINDOW_AUTOSIZE); // Impulse Noise (Grayscale) 창 띄움
	imshow("Impulse Noise (Grayscale)", noise_Gray); // 창에 noise_Gray 이미지 보여줌

	namedWindow("Impulse Noise (RGB)", WINDOW_AUTOSIZE); // Impulse Noise (RGB) 창 띄움
	imshow("Impulse Noise (RGB)", noise_RGB); // 창에 noise_RGB 이미지 보여줌

	namedWindow("Denoised (Grayscale)", WINDOW_AUTOSIZE); // Denoised (Grayscale) 창 띄움
	imshow("Denoised (Grayscale)", Denoised_Gray); // 창에 Denoised_Gray 이미지 보여줌

	namedWindow("Denoised (RGB)", WINDOW_AUTOSIZE); // Denoised (RGB) 창 띄움
	imshow("Denoised (RGB)", Denoised_RGB); // 창에 Denoised_RGB 이미지 보여줌

	waitKey(0);

	return 0;
}


// Adding salt and pepper noise
// 매개변수: input = noise를 추가할 이미지, ps = salt의 density, pp = pepper의 density
Mat Add_salt_pepper_Noise(const Mat input, float ps, float pp)
{
	Mat output = input.clone(); // input matrix를 복제하여 output matrix에 저장
	RNG rng; // Random Number Generator

	int amount1 = (int)(output.rows * output.cols * pp); // output 이미지에 추가될 pepper noise의 양
	int amount2 = (int)(output.rows * output.cols * ps); // output 이미지에 추가될 salt noise의 양

	int x, y;

	// Grayscale image
	if (output.channels() == 1) {
		for (int counter = 0; counter < amount1; ++counter)
			output.at<G>(rng.uniform(0, output.rows), rng.uniform(0, output.cols)) = 0;
			// output의 랜덤한 위치에 pepper noise (값=0) 를 추가
			// rng.uniform(a,b)는 (a, b] 범위에서 균일하게 분포된 정수 난수를 반환

		for (int counter = 0; counter < amount2; ++counter)
			output.at<G>(rng.uniform(0, output.rows), rng.uniform(0, output.cols)) = 255;
			// output의 랜덤한 위치에 salt noise (값=255) 를 추가
	}
	// Color image
	else if (output.channels() == 3) {
		// pepper noise (값=0) 추가
		for (int counter = 0; counter < amount1; ++counter) {
			x = rng.uniform(0, output.rows); // 랜덤으로 행 선택
			y = rng.uniform(0, output.cols); // 랜덤으로 열 선택
			output.at<C>(x, y)[0] = 0; // R channel에서 랜덤으로 선택된 픽셀에 pepper noise 추가

			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[1] = 0; // G channel에서 랜덤으로 선택된 픽셀에 pepper noise 추가

			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[2] = 0; // B channel에서 랜덤으로 선택된 픽셀에 pepper noise 추가
		}
		// salt noise (값=255) 추가
		for (int counter = 0; counter < amount2; ++counter) {
			x = rng.uniform(0, output.rows); // 랜덤으로 행 선택
			y = rng.uniform(0, output.cols); // 랜덤으로 열 선택
			output.at<C>(x, y)[0] = 255; // R channel에서 랜덤으로 선택된 픽셀에 salt noise 추가

			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[1] = 255; // G channel에서 랜덤으로 선택된 픽셀에 salt noise 추가

			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[2] = 255; // B channel에서 랜덤으로 선택된 픽셀에 salt noise 추가
		}
	}

	return output; // output 이미지를 반환
}

// removing salt and pepper noise with Median filter (Grayscale image)
// 매개변수: input = noise를 제거할 이미지, n = neighbor의 범위, opt = boundary process 방법
Mat Salt_pepper_noise_removal_Gray(const Mat input, int n, const char *opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1); // (2N+1)*(2N+1)
	int median;		// index of median value
	int tempa, tempb;

	// initialize median filter kernel
	Mat kernel = Mat::zeros(kernel_size * kernel_size, 1, input.type());
	// kernel은 kernel_size*kernel_size 개의 행, 1개의 열로 이루어진 matrix
	// matrix 값들은 input과 같은 type이고 0으로 초기화
	
	Mat output = Mat::zeros(row, col, input.type());
	// output은 input의 크기, 타입인 matrix이고, 값은 0으로 초기화

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) { // output의 각 픽셀마다 반복
					
			if (!strcmp(opt, "zero-padding")) { // boundary process가 zero-padding 일 때
				int count = 0; // count는 kernel의 행 index를 의미
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							// 픽셀이 border 밖에 있으면 픽셀 값을 0으로 여기므로, 안에 있는 경우만 값을 업데이트해줌
							kernel.at<G>(count, 0) = input.at<G>(i + x, j + y);
							// kernel에 해당 위치의 input 픽셀 값 저장
						}
						count++; // kernel 행 index ++
					}
				}
				median = kernel_size * kernel_size / 2;
				// median 은 행 크기 / 2 에 저장된 값
				// 행을 정렬한 후에 적용해주어야 올바른 median(중앙값)의 index가 될 것!
			}

			else if (!strcmp(opt, "mirroring")) { // boundary process가 mirroring 일 때
				int count = 0; // count는 kernel의 행 index를 의미
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
		
						if (i + x > row - 1) {  // filter가 아래쪽 boundary를 초과했을 때,
							tempa = i - x; // boundary 쪽 값을 mirroring
						}
						else if (i + x < 0) { // filter가 위쪽 boundary를 초과했을 때,
							tempa = -(i + x); // boundary 쪽 값을 mirroring
						}
						else {
							tempa = i + x;
						}
						if (j + y > col - 1) { // filter가 오른쪽 boundary를 초과했을 때,
							tempb = j - y; // boundary 쪽 값을 mirroring
						}
						else if (j + y < 0) { // filter가 왼쪽 boundary를 초과했을 때,
							tempb = -(j + y); // boundary 쪽 값을 mirroring
						}
						else {
							tempb = j + y;
						}
						kernel.at<G>(count, 0) = input.at<G>(tempa, tempb);
						// kernel에 해당 위치의 input 픽셀 값 저장
						count++; // kernel 행 index ++
					}
				}
				median = kernel_size * kernel_size / 2;
				// median index = 행 크기 / 2
				// 행을 정렬한 후에 적용해주어야 올바른 median(중앙값)의 index가 될 것!
			}

			else if (!strcmp(opt, "adjustkernel")) { // boundary process가 adjustkernel 일 때
				int count = 0; // count는 kernel의 행 index를 의미
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {		
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							// 픽셀이 border 밖에 있는 경우가 존재하면 이를 제외하고 kernel 사이즈를 작게 생각
							kernel.at<G>(count, 0) = input.at<G>(i + x, j + y);
							// kernel에 해당 위치의 input 픽셀 값 저장
							count++; // kernel 행 index ++
						}
					}
				}
				median = count / 2;
				// median index = count / 2
				// border 밖에 있는 픽셀을 제외하고 kernel 사이즈를 아예 작게 취급하므로,
				// count / 2 를 이용하여 median index를 결정
			}

			// Sort the kernels in ascending order
			sort(kernel, kernel, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			// kernel을 열끼리 독립적으로 오름차순 정렬

			output.at<G>(i, j) = kernel.at<G>(median, 0);
			// kernel의 median 값을 output 픽셀에 update
		}
	}

	return output; // output 이미지를 반환
}

// removing salt and pepper noise with Median filter (RGB image)
// 매개변수: input = noise를 제거할 이미지, n = neighbor의 범위, opt = boundary process 방법
Mat Salt_pepper_noise_removal_RGB(const Mat input, int n, const char *opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1); // (2N + 1) * (2N + 1)
	int median;		// index of median value
	int channel = input.channels();
	int tempa, tempb;

	// initialize median filter kernel
	Mat kernel = Mat::zeros(kernel_size * kernel_size, channel, input.type() - 16);
	// kernel은 kernel_size*kernel_size 개의 행, channel 개의 열로 이루어진 matrix
	// initialize ( (TypeX with 3 channel) - (TypeX with 1 channel) = 16 )
	// ex) CV_8UC3 - CV_8U = 16
	// matrix 값들은 0으로 초기화

	Mat output = Mat::zeros(row, col, input.type());
	// output은 input의 크기, 타입인 matrix이고, 값은 0으로 초기화

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) { // output의 각 픽셀마다 반복
					
			if (!strcmp(opt, "zero-padding")) { // boundary process가 zero-padding 일 때
				int count = 0; // count는 kernel의 행 index를 의미
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							// 픽셀이 border 밖에 있으면 픽셀 값을 0으로 여기므로, 안에 있는 경우만 값을 업데이트해줌
							kernel.at<G>(count, 0) = input.at<C>(i + x, j + y)[0]; // kernel의 첫번째 열에는 해당 위치의 input의 R channel pixel 값을 저장
							kernel.at<G>(count, 1) = input.at<C>(i + x, j + y)[1]; // kernel의 두번째 열에는 해당 위치의 input의 G channel pixel 값을 저장
							kernel.at<G>(count, 2) = input.at<C>(i + x, j + y)[2]; // kernel의 세번째 열에는 해당 위치의 input의 B channel pixel 값을 저장
						}
						count++; // kernel 행 index ++
					}
				}
				median = kernel_size * kernel_size / 2;
				// median 은 행 크기 / 2 에 저장된 값
				// 행을 정렬한 후에 적용해주어야 올바른 median(중앙값)의 index가 될 것!
			}

			else if (!strcmp(opt, "mirroring")) { // boundary process가 mirroring 일 때
				int count = 0; // count는 kernel의 행 index를 의미
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
	
						if (i + x > row - 1) {  // filter가 아래쪽 boundary를 초과했을 때,
							tempa = i - x; // boundary 쪽 값을 mirroring
						}
						else if (i + x < 0) { // filter가 위쪽 boundary를 초과했을 때,
							tempa = -(i + x); // boundary 쪽 값을 mirroring
						}
						else {
							tempa = i + x;
						}
						if (j + y > col - 1) { // filter가 오른쪽 boundary를 초과했을 때,
							tempb = j - y; // boundary 쪽 값을 mirroring
						}
						else if (j + y < 0) { // filter가 왼쪽 boundary를 초과했을 때,
							tempb = -(j + y); // boundary 쪽 값을 mirroring
						}
						else {
							tempb = j + y;
						}
						kernel.at<G>(count, 0) = input.at<C>(tempa, tempb)[0];  // kernel의 첫번째 열에는 해당 위치의 input의 R channel pixel 값을 저장
						kernel.at<G>(count, 1) = input.at<C>(tempa, tempb)[1];  // kernel의 두번째 열에는 해당 위치의 input의 G channel pixel 값을 저장
						kernel.at<G>(count, 2) = input.at<C>(tempa, tempb)[2];  // kernel의 세번째 열에는 해당 위치의 input의 B channel pixel 값을 저장
						count++; // kernel 행 index ++
					}
				}
				median = kernel_size * kernel_size / 2;
				// median index = 행 크기 / 2
				// 행을 정렬한 후에 적용해주어야 올바른 median(중앙값)의 index가 될 것!
			}

			else if (!strcmp(opt, "adjustkernel")) { // boundary process가 adjustkernel 일 때
				int count = 0; // count는 kernel의 행 index를 의미
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {	
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							// 픽셀이 border 밖에 있는 경우가 존재하면 이를 제외하고 kernel 사이즈를 작게 생각
							kernel.at<G>(count, 0) = input.at<C>(i + x, j + y)[0]; // kernel의 첫번째 열에는 해당 위치의 input의 R channel pixel 값을 저장
							kernel.at<G>(count, 1) = input.at<C>(i + x, j + y)[1]; // kernel의 두번째 열에는 해당 위치의 input의 G channel pixel 값을 저장
							kernel.at<G>(count, 2) = input.at<C>(i + x, j + y)[2]; // kernel의 세번째 열에는 해당 위치의 input의 B channel pixel 값을 저장
							count++; // kernel 행 index ++
						}
					}
				}
				median = count / 2;
				// median index = count / 2
				// border 밖에 있는 픽셀을 제외하고 kernel 사이즈를 아예 작게 취급하므로,
				// count / 2 를 이용하여 median index를 결정
			}

			// Sort the kernels in ascending order
			sort(kernel, kernel, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			// kernel을 열끼리 독립적으로 오름차순 정렬
			
			// 각 채널별로 kernel의 median 값을 output 픽셀에 update
			output.at<C>(i, j)[0] = kernel.at<G>(median, 0);
			output.at<C>(i, j)[1] = kernel.at<G>(median, 1);
			output.at<C>(i, j)[2] = kernel.at<G>(median, 2);
		}
	}

	return output; // output 이미지를 반환

}