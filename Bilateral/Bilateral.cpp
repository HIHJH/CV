#include <opencv2/opencv.hpp>
#include <stdio.h>

#define IM_TYPE	CV_64FC3

using namespace cv;

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt);
Mat Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input 이미지 불러옴
	Mat input_gray; // 흑백 이미지를 저장할 Mat

	// check for validation
	if (!input.data) {
		printf("Could not open\n"); // 이미지 파일에 데이터 오류가 있으면 메세지 출력
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// RGB 이미지를 흑백 이미지로 변환하여 input_gray에 저장

	// 8-bit unsigned char -> 64-bit floating point
	input.convertTo(input, CV_64FC3, 1.0 / 255);
	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);

	// 원본 이미지에 noise 추가
	Mat noise_Gray = Add_Gaussian_noise(input_gray, 0, 0.1);
	// 원본 흑백 이미지에 gaussian noise를 추가하여 noise_Gray에 저장
	// 평균 = 0, 표준편차 = 0.1 인 gaussian 설정
	Mat noise_RGB = Add_Gaussian_noise(input, 0, 0.1);
	// 원본 이미지에 gaussian noise를 추가하여 noise_RGB에 저장
	// 평균 = 0, 표준편차 = 0.1 인 gaussian 설정

	// Denoise, using gaussian filter
	Mat Denoised_Gray = Bilateralfilter_Gray(noise_Gray, 3, 10, 10, 0.3, "zero-padding");
	// Denoised_Gray는 noise_Gray에 bilateral filtering을 적용한 것
	// filter는 n = 3, sigma_t = 10, sigma_s = 10, sigma_r = 0.3 으로 기본 설정
	// boundary process는 zero-padding을 기본 적용
	Mat Denoised_RGB = Bilateralfilter_RGB(noise_RGB, 3, 10, 10, 0.3, "adjustkernel");
	// Denoised_RGB는 noise_RGB에 gaussian filtering을 적용한 것
	// filter는 n = 3, sigma_t = 10, sigma_s = 10, sigma_r = 0.3 으로 기본 설정
	// boundary process는 adjustkernel을 기본 적용

	namedWindow("Grayscale", WINDOW_AUTOSIZE); // Grayscale 창 띄움
	imshow("Grayscale", input_gray); // 창에 input_gray 이미지 보여줌

	namedWindow("RGB", WINDOW_AUTOSIZE); // RGB 창 띄움
	imshow("RGB", input); // 창에 input 이미지 보여줌

	namedWindow("Gaussian Noise (Grayscale)", WINDOW_AUTOSIZE); // Gaussian Noise (Grayscale) 창 띄움
	imshow("Gaussian Noise (Grayscale)", noise_Gray); // 창에 noise_Gray 이미지 보여줌

	namedWindow("Gaussian Noise (RGB)", WINDOW_AUTOSIZE); // Gaussian Noise (RGB) 창 띄움
	imshow("Gaussian Noise (RGB)", noise_RGB); // 창에 noise_RGB 이미지 보여줌

	namedWindow("Denoised (Grayscale)", WINDOW_AUTOSIZE); // Denoised (Grayscale) 창 띄움
	imshow("Denoised (Grayscale)", Denoised_Gray); // 창에 Denoised_Gray 이미지 보여줌

	namedWindow("Denoised (RGB)", WINDOW_AUTOSIZE); // Denoised (RGB) 창 띄움
	imshow("Denoised (RGB)", Denoised_RGB); // 창에 Denoised_RGB 이미지 보여줌

	waitKey(0);

	return 0;
}

// Adding gaussian noise
// 매개변수: input = noise를 추가할 이미지, mean = noise가 따르는 정규 분포의 평균,
// sigma = noise가 따르는 정규 분포의 표준편차
Mat Add_Gaussian_noise(const Mat input, double mean, double sigma) {

	Mat NoiseArr = Mat::zeros(input.rows, input.cols, input.type());
	// input과 같은 크기, type을 가지는 NoiseArr matrix, 값은 0으로 초기화
	RNG rng; // Random Number Generator
	rng.fill(NoiseArr, RNG::NORMAL, mean, sigma);
	// 평균 = mean, 표준편차 = sigma인 정규(gaussian) 분포를 따르는 난수를 생성하여 NoiseArr에 저장

	add(input, NoiseArr, NoiseArr); // NoiseArr <- input + NoiseArr

	return NoiseArr; // NoiseArr을 반환
}

// bilateral filtering (Grayscale image)
// 매개변수: input = 필터를 적용할 이미지, n = neighbor의 범위, sigma_t =  t 관련 gaussian 분포의 표준편차(x-coordinate),
// sigma_s = s 관련 gaussian 분포의 표준편차(y-coordinate), sigma_r = intensity 관련 gaussian 분포의 표준편차(color),
// opt = boundary process 방법
Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1); // (2N+1)*(2N+1) 
	int tempa;
	int tempb;
	float denom_g;

	// Initialiazing Kernel Matrix 
	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);
	// kernel matrix를 kernel size에 맞게 32bit float형태로 설정하고, 값은 0으로 초기화

	// 초기설정: 거리만 고려하는 kernel (gaussian filtering의 kernel과 같음)
	denom_g = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2)))); // gaussian filter weight의 분자값 계산
			kernel.at<float>(a + n, b + n) = value1; // 각 커널에 분자값(value1) 업데이트
			denom_g += value1; // 가우시안 filter weight의 분모를 계산하기 위해 denom에 각 분자를 계산한 값의 합을 저장
		}
	}

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom_g; // gaussian filter weight의 분모값(denom)까지 반영하여 kernel 업데이트
		}
	}

	Mat output = Mat::zeros(row, col, input.type());
	// output은 input의 크기, 타입인 matrix이고, 값은 0으로 초기화

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-padding")) { // boundary process가 zerop-padding 일 때
				float sum = 0.0;
				float denom = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							// 픽셀이 border 밖에 있으면 픽셀 값을 0으로 여기므로, 안에 있는 경우만 계산해줌
							float weight = kernel.at<float>(x + n, y + n) * exp(-pow(input.at<double>(i, j) - input.at<double>(i + x, j + y), 2) / (2 * pow(sigma_r, 2)));
							// 거리만 고려하던 kernel weight에 intensity를 고려한 weight를 곱해주어 weight 값 업데이트
							sum += input.at<double>(i + x, j + y) * weight; // filtering 할 neighbor 각각에 weight를 곱하고 이를 더해줌
							denom += weight; // weight의 합은 denom에 저장
						}

					}
				}
				output.at<double>(i, j) = (double)(sum / denom);
				// sum을 denom로 나누어 줌으로써 bilateral filtering의 전체 weight가 모두 반영됨
				// 그리고 반영된 값을 output 픽셀에 넣어줌
			}

			else if (!strcmp(opt, "mirroring")) { // boundary process가 mirroring 일 때
				float sum = 0.0;
				float denom = 0.0;
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
						float weight = kernel.at<float>(x + n, y + n) * exp(-pow(input.at<double>(i, j) - input.at<double>(tempa, tempb), 2) / (2 * pow(sigma_r, 2)));
						// 거리만 고려하던 kernel weight에 intensity를 고려한 weight를 곱해주어 weight 값 업데이트
						sum += input.at<double>(tempa, tempb) * weight; // filtering 할 neighbor 각각에 weight를 곱하고 이를 더해줌
						denom += weight; // weight의 합은 denom에 저장
					}
				}
				output.at<double>(i, j) = (double)(sum / denom);
				// sum을 denom로 나누어 줌으로써 bilateral filtering의 전체 weight가 모두 반영됨
				// 그리고 반영된 값을 output 픽셀에 넣어줌
			}

			else if (!strcmp(opt, "adjustkernel")) { // boundary process가 adjustkernel 일 때
				float sum = 0.0;
				float denom = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							// 픽셀이 border 밖에 있는 경우가 존재하면 이를 제외하고 kernel 사이즈를 작게 생각
							float weight = kernel.at<float>(x + n, y + n) * exp(-pow(input.at<double>(i, j) - input.at<double>(i + x, j + y), 2) / (2 * pow(sigma_r, 2)));
							// 거리만 고려하던 kernel weight에 intensity를 고려한 weight를 곱해주어 weight 값 업데이트
							sum += input.at<double>(i + x, j + y) * weight; // filtering 할 neighbor 각각에 weight를 곱하고 이를 더해줌
							denom += weight; // weight의 합은 denom에 저장
						}
					}
				}
				
				//adjust kernel 방법을 사용하기 위해 위의 코드를 일부 반복하여 bilateral filtering의 전체 weight 합을 구함 => weight_sum 에 저장
				float weight_sum = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							float weight = kernel.at<float>(x + n, y + n) * exp(-pow(input.at<double>(i, j) - input.at<double>(i + x, j + y), 2) / (2 * pow(sigma_r, 2)));
							weight_sum += (weight / denom); // weight_sum 에 bilateral filtering의 최종 weight 합을 구함
						}
					}
				}
				output.at<double>(i, j) = (double)((sum / denom) / weight_sum);
				// sum을 denom로 나누어 줌으로써 bilateral filtering의 전체 weight가 모두 반영됨
				// 이 값을 weight_sum 으로 나누어 줌으로써 kernel의 사이즈를 줄인 adjust kernel 방법을 적용
				// 최종값을 output 픽셀에 넣어줌

			}
		}
	}

	return output; // filtering 적용한 output matrix를 반환
}


// bilateral filtering (RGB image)
// 매개변수: input = 필터를 적용할 이미지, n = neighbor의 범위, sigma_t =  t 관련 gaussian 분포의 표준편차(x-coordinate),
// sigma_s = s 관련 gaussian 분포의 표준편차(y-coordinate), sigma_r = intensity 관련 gaussian 분포의 표준편차(color),
// opt = boundary process 방법
Mat Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom_g;

	// Initialiazing Kernel Matrix
	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);
	// kernel matrix를 kernel size에 맞게 32bit float형태로 설정하고, 값은 0으로 초기화

	// 초기설정: 거리만 고려하는 kernel (gaussian filtering의 kernel과 같음)
	denom_g = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2)))); // gaussian filter weight의 분자값 계산
			kernel.at<float>(a + n, b + n) = value1; // 각 커널에 분자값(value1) 업데이트
			denom_g += value1; // 가우시안 filter weight의 분모를 계산하기 위해 denom에 각 분자를 계산한 값의 합을 저장
		}
	}

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom_g; // gaussian filter weight의 분모값(denom)까지 반영하여 kernel 업데이트
		}
	}

	Mat output = Mat::zeros(row, col, input.type());
	// output은 input의 크기, 타입인 matrix이고, 값은 0으로 초기화

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-padding")) { // boundary process가 zerop-padding 일 때
				float sum_r = 0.0;
				float sum_g = 0.0;
				float sum_b = 0.0;
				float denom = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							// 픽셀이 border 밖에 있으면 픽셀 값을 0으로 여기므로, 안에 있는 경우만 계산해줌
							float value_r = exp(-pow((float)input.at<Vec3d>(i, j)[0] - (float)input.at<Vec3d>(i + x, j + y)[0], 2) / (2 * pow(sigma_r, 2)));
							float value_g = exp(-pow((float)input.at<Vec3d>(i, j)[1] - (float)input.at<Vec3d>(i + x, j + y)[1], 2) / (2 * pow(sigma_r, 2)));
							float value_b = exp(-pow((float)input.at<Vec3d>(i, j)[2] - (float)input.at<Vec3d>(i + x, j + y)[2], 2) / (2 * pow(sigma_r, 2)));
							// RGB 채널 각각 intensity 차이를 고려한 weight를 계산
							float weight = kernel.at<float>(x + n, y + n) * value_r * value_g * value_b;
							// 거리만 고려하던 kernel weight에 intensity를 고려한 weight를 곱해주어 weight 값 업데이트
							sum_r += input.at<Vec3d>(i + x, j + y)[0] * weight;
							sum_g += input.at<Vec3d>(i + x, j + y)[1] * weight;
							sum_b += input.at<Vec3d>(i + x, j + y)[2] * weight;
							// 각 채널별로 filtering 할 neighbor 각각에 weight를 곱하고 이를 더해줌
							denom += weight; // weight의 합은 denom에 저장
						}
					}
				}
				output.at<Vec3d>(i, j)[0] = (double)(sum_r / denom);
				output.at<Vec3d>(i, j)[1] = (double)(sum_g / denom);
				output.at<Vec3d>(i, j)[2] = (double)(sum_b / denom);
				// 각 채널별로 sum을 denom로 나누어 줌으로써 bilateral filtering의 전체 weight가 모두 반영됨
				// 그리고 반영된 값을 output 픽셀에 넣어줌
			}

			else if (!strcmp(opt, "mirroring")) { // boundary process가 mirroring 일 때
				float sum_r = 0.0;
				float sum_g = 0.0;
				float sum_b = 0.0;
				float denom = 0.0;
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
						float value_r = exp(-pow((float)input.at<Vec3d>(i, j)[0] - (float)input.at<Vec3d>(tempa, tempb)[0], 2) / (2 * pow(sigma_r, 2)));
						float value_g = exp(-pow((float)input.at<Vec3d>(i, j)[1] - (float)input.at<Vec3d>(tempa, tempb)[1], 2) / (2 * pow(sigma_r, 2)));
						float value_b = exp(-pow((float)input.at<Vec3d>(i, j)[2] - (float)input.at<Vec3d>(tempa, tempb)[2], 2) / (2 * pow(sigma_r, 2)));
						// RGB 채널 각각 intensity 차이를 고려한 weight를 계산
						float weight = kernel.at<float>(x + n, y + n) * value_r * value_g * value_b;
						// 거리만 고려하던 kernel weight에 intensity를 고려한 weight를 곱해주어 weight 값 업데이트
						sum_r += input.at<Vec3d>(tempa, tempb)[0] * weight;
						sum_g += input.at<Vec3d>(tempa, tempb)[1] * weight;
						sum_b += input.at<Vec3d>(tempa, tempb)[2] * weight;
						// 각 채널별로 filtering 할 neighbor 각각에 weight를 곱하고 이를 더해줌
						denom += weight; // weight의 합은 denom에 저장
					}
				}
				output.at<Vec3d>(i, j)[0] = (double)(sum_r / denom);
				output.at<Vec3d>(i, j)[1] = (double)(sum_g / denom);
				output.at<Vec3d>(i, j)[2] = (double)(sum_b / denom);
				// 각 채널별로 sum을 denom로 나누어 줌으로써 bilateral filtering의 전체 weight가 모두 반영됨
				// 그리고 반영된 값을 output 픽셀에 넣어줌
			}

			else if (!strcmp(opt, "adjustkernel")) {
				float sum_r = 0.0;
				float sum_g = 0.0;
				float sum_b = 0.0;
				float denom = 0.0;
				//float sum2 = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							// 픽셀이 border 밖에 있으면 픽셀 값을 0으로 여기므로, 안에 있는 경우만 계산해줌
							// 각 채널별로 filtering 할 neighbor 각각에 weight를 곱하고 이를 다 더해서 sum1에 저장
							float value_r = exp(-pow((float)input.at<Vec3d>(i, j)[0] - (float)input.at<Vec3d>(i + x, j + y)[0], 2) / (2 * pow(sigma_r, 2)));
							float value_g = exp(-pow((float)input.at<Vec3d>(i, j)[1] - (float)input.at<Vec3d>(i + x, j + y)[1], 2) / (2 * pow(sigma_r, 2)));
							float value_b = exp(-pow((float)input.at<Vec3d>(i, j)[2] - (float)input.at<Vec3d>(i + x, j + y)[2], 2) / (2 * pow(sigma_r, 2)));
							// RGB 채널 각각 intensity 차이를 고려한 weight를 계산
							float weight = kernel.at<float>(x + n, y + n) * value_r * value_g * value_b;
							// 거리만 고려하던 kernel weight에 intensity를 고려한 weight를 곱해주어 weight 값 업데이트
							sum_r += (float)input.at<Vec3d>(i + x, j + y)[0] * weight;
							sum_g += (float)input.at<Vec3d>(i + x, j + y)[1] * weight;
							sum_b += (float)input.at<Vec3d>(i + x, j + y)[2] * weight;
							// 각 채널별로 filtering 할 neighbor 각각에 weight를 곱하고 이를 더해줌
							denom += weight; // weight의 합은 denom에 저장
						}
					}
				}

				//adjust kernel 방법을 사용하기 위해 위의 코드를 일부 반복하여 bilateral filtering의 전체 weight 합을 구함 => weight_sum 에 저장
				float weight_sum = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							float value_r = exp(-pow((float)input.at<Vec3d>(i, j)[0] - (float)input.at<Vec3d>(i + x, j + y)[0], 2) / (2 * pow(sigma_r, 2)));
							float value_g = exp(-pow((float)input.at<Vec3d>(i, j)[1] - (float)input.at<Vec3d>(i + x, j + y)[1], 2) / (2 * pow(sigma_r, 2)));
							float value_b = exp(-pow((float)input.at<Vec3d>(i, j)[2] - (float)input.at<Vec3d>(i + x, j + y)[2], 2) / (2 * pow(sigma_r, 2)));
							float weight = kernel.at<float>(x + n, y + n) * value_r * value_g * value_b;
							weight_sum += (weight / denom); // weight_sum 에 bilateral filtering의 최종 weight 합을 구함
						}
					}
				}

				output.at<Vec3d>(i, j)[0] = (double)((sum_r / denom) / weight_sum);
				output.at<Vec3d>(i, j)[1] = (double)((sum_g / denom) / weight_sum);
				output.at<Vec3d>(i, j)[2] = (double)((sum_b / denom) / weight_sum);
				// 각 채널별로 sum을 denom로 나누어 줌으로써 bilateral filtering의 전체 weight가 모두 반영됨
				// 이 값을 weight_sum 으로 나누어 줌으로써 kernel의 사이즈를 줄인 adjust kernel 방법을 적용
				// 최종값을 output 픽셀에 넣어줌
			}
		}
	}

	return output; // filtering 적용한 output matrix를 반환
}