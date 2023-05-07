#include <opencv2/opencv.hpp>
#include <stdio.h>

#define IM_TYPE	CV_64FC3

using namespace cv;

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt);
Mat Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input �̹��� �ҷ���
	Mat input_gray; // ��� �̹����� ������ Mat

	// check for validation
	if (!input.data) {
		printf("Could not open\n"); // �̹��� ���Ͽ� ������ ������ ������ �޼��� ���
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// RGB �̹����� ��� �̹����� ��ȯ�Ͽ� input_gray�� ����

	// 8-bit unsigned char -> 64-bit floating point
	input.convertTo(input, CV_64FC3, 1.0 / 255);
	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);

	// ���� �̹����� noise �߰�
	Mat noise_Gray = Add_Gaussian_noise(input_gray, 0, 0.1);
	// ���� ��� �̹����� gaussian noise�� �߰��Ͽ� noise_Gray�� ����
	// ��� = 0, ǥ������ = 0.1 �� gaussian ����
	Mat noise_RGB = Add_Gaussian_noise(input, 0, 0.1);
	// ���� �̹����� gaussian noise�� �߰��Ͽ� noise_RGB�� ����
	// ��� = 0, ǥ������ = 0.1 �� gaussian ����

	// Denoise, using gaussian filter
	Mat Denoised_Gray = Bilateralfilter_Gray(noise_Gray, 3, 10, 10, 0.3, "zero-padding");
	// Denoised_Gray�� noise_Gray�� bilateral filtering�� ������ ��
	// filter�� n = 3, sigma_t = 10, sigma_s = 10, sigma_r = 0.3 ���� �⺻ ����
	// boundary process�� zero-padding�� �⺻ ����
	Mat Denoised_RGB = Bilateralfilter_RGB(noise_RGB, 3, 10, 10, 0.3, "adjustkernel");
	// Denoised_RGB�� noise_RGB�� gaussian filtering�� ������ ��
	// filter�� n = 3, sigma_t = 10, sigma_s = 10, sigma_r = 0.3 ���� �⺻ ����
	// boundary process�� adjustkernel�� �⺻ ����

	namedWindow("Grayscale", WINDOW_AUTOSIZE); // Grayscale â ���
	imshow("Grayscale", input_gray); // â�� input_gray �̹��� ������

	namedWindow("RGB", WINDOW_AUTOSIZE); // RGB â ���
	imshow("RGB", input); // â�� input �̹��� ������

	namedWindow("Gaussian Noise (Grayscale)", WINDOW_AUTOSIZE); // Gaussian Noise (Grayscale) â ���
	imshow("Gaussian Noise (Grayscale)", noise_Gray); // â�� noise_Gray �̹��� ������

	namedWindow("Gaussian Noise (RGB)", WINDOW_AUTOSIZE); // Gaussian Noise (RGB) â ���
	imshow("Gaussian Noise (RGB)", noise_RGB); // â�� noise_RGB �̹��� ������

	namedWindow("Denoised (Grayscale)", WINDOW_AUTOSIZE); // Denoised (Grayscale) â ���
	imshow("Denoised (Grayscale)", Denoised_Gray); // â�� Denoised_Gray �̹��� ������

	namedWindow("Denoised (RGB)", WINDOW_AUTOSIZE); // Denoised (RGB) â ���
	imshow("Denoised (RGB)", Denoised_RGB); // â�� Denoised_RGB �̹��� ������

	waitKey(0);

	return 0;
}

// Adding gaussian noise
// �Ű�����: input = noise�� �߰��� �̹���, mean = noise�� ������ ���� ������ ���,
// sigma = noise�� ������ ���� ������ ǥ������
Mat Add_Gaussian_noise(const Mat input, double mean, double sigma) {

	Mat NoiseArr = Mat::zeros(input.rows, input.cols, input.type());
	// input�� ���� ũ��, type�� ������ NoiseArr matrix, ���� 0���� �ʱ�ȭ
	RNG rng; // Random Number Generator
	rng.fill(NoiseArr, RNG::NORMAL, mean, sigma);
	// ��� = mean, ǥ������ = sigma�� ����(gaussian) ������ ������ ������ �����Ͽ� NoiseArr�� ����

	add(input, NoiseArr, NoiseArr); // NoiseArr <- input + NoiseArr

	return NoiseArr; // NoiseArr�� ��ȯ
}

// bilateral filtering (Grayscale image)
// �Ű�����: input = ���͸� ������ �̹���, n = neighbor�� ����, sigma_t =  t ���� gaussian ������ ǥ������(x-coordinate),
// sigma_s = s ���� gaussian ������ ǥ������(y-coordinate), sigma_r = intensity ���� gaussian ������ ǥ������(color),
// opt = boundary process ���
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
	// kernel matrix�� kernel size�� �°� 32bit float���·� �����ϰ�, ���� 0���� �ʱ�ȭ

	// �ʱ⼳��: �Ÿ��� ����ϴ� kernel (gaussian filtering�� kernel�� ����)
	denom_g = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2)))); // gaussian filter weight�� ���ڰ� ���
			kernel.at<float>(a + n, b + n) = value1; // �� Ŀ�ο� ���ڰ�(value1) ������Ʈ
			denom_g += value1; // ����þ� filter weight�� �и� ����ϱ� ���� denom�� �� ���ڸ� ����� ���� ���� ����
		}
	}

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom_g; // gaussian filter weight�� �и�(denom)���� �ݿ��Ͽ� kernel ������Ʈ
		}
	}

	Mat output = Mat::zeros(row, col, input.type());
	// output�� input�� ũ��, Ÿ���� matrix�̰�, ���� 0���� �ʱ�ȭ

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-padding")) { // boundary process�� zerop-padding �� ��
				float sum = 0.0;
				float denom = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							// �ȼ��� border �ۿ� ������ �ȼ� ���� 0���� ����Ƿ�, �ȿ� �ִ� ��츸 �������
							float weight = kernel.at<float>(x + n, y + n) * exp(-pow(input.at<double>(i, j) - input.at<double>(i + x, j + y), 2) / (2 * pow(sigma_r, 2)));
							// �Ÿ��� ����ϴ� kernel weight�� intensity�� ����� weight�� �����־� weight �� ������Ʈ
							sum += input.at<double>(i + x, j + y) * weight; // filtering �� neighbor ������ weight�� ���ϰ� �̸� ������
							denom += weight; // weight�� ���� denom�� ����
						}

					}
				}
				output.at<double>(i, j) = (double)(sum / denom);
				// sum�� denom�� ������ �����ν� bilateral filtering�� ��ü weight�� ��� �ݿ���
				// �׸��� �ݿ��� ���� output �ȼ��� �־���
			}

			else if (!strcmp(opt, "mirroring")) { // boundary process�� mirroring �� ��
				float sum = 0.0;
				float denom = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if (i + x > row - 1) {  // filter�� �Ʒ��� boundary�� �ʰ����� ��,
							tempa = i - x; // boundary �� ���� mirroring
						}
						else if (i + x < 0) { // filter�� ���� boundary�� �ʰ����� ��,
							tempa = -(i + x); // boundary �� ���� mirroring
						}
						else {
							tempa = i + x;
						}
						if (j + y > col - 1) { // filter�� ������ boundary�� �ʰ����� ��,
							tempb = j - y; // boundary �� ���� mirroring
						}
						else if (j + y < 0) { // filter�� ���� boundary�� �ʰ����� ��,
							tempb = -(j + y); // boundary �� ���� mirroring
						}
						else {
							tempb = j + y;
						}
						float weight = kernel.at<float>(x + n, y + n) * exp(-pow(input.at<double>(i, j) - input.at<double>(tempa, tempb), 2) / (2 * pow(sigma_r, 2)));
						// �Ÿ��� ����ϴ� kernel weight�� intensity�� ����� weight�� �����־� weight �� ������Ʈ
						sum += input.at<double>(tempa, tempb) * weight; // filtering �� neighbor ������ weight�� ���ϰ� �̸� ������
						denom += weight; // weight�� ���� denom�� ����
					}
				}
				output.at<double>(i, j) = (double)(sum / denom);
				// sum�� denom�� ������ �����ν� bilateral filtering�� ��ü weight�� ��� �ݿ���
				// �׸��� �ݿ��� ���� output �ȼ��� �־���
			}

			else if (!strcmp(opt, "adjustkernel")) { // boundary process�� adjustkernel �� ��
				float sum = 0.0;
				float denom = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							// �ȼ��� border �ۿ� �ִ� ��찡 �����ϸ� �̸� �����ϰ� kernel ����� �۰� ����
							float weight = kernel.at<float>(x + n, y + n) * exp(-pow(input.at<double>(i, j) - input.at<double>(i + x, j + y), 2) / (2 * pow(sigma_r, 2)));
							// �Ÿ��� ����ϴ� kernel weight�� intensity�� ����� weight�� �����־� weight �� ������Ʈ
							sum += input.at<double>(i + x, j + y) * weight; // filtering �� neighbor ������ weight�� ���ϰ� �̸� ������
							denom += weight; // weight�� ���� denom�� ����
						}
					}
				}
				
				//adjust kernel ����� ����ϱ� ���� ���� �ڵ带 �Ϻ� �ݺ��Ͽ� bilateral filtering�� ��ü weight ���� ���� => weight_sum �� ����
				float weight_sum = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							float weight = kernel.at<float>(x + n, y + n) * exp(-pow(input.at<double>(i, j) - input.at<double>(i + x, j + y), 2) / (2 * pow(sigma_r, 2)));
							weight_sum += (weight / denom); // weight_sum �� bilateral filtering�� ���� weight ���� ����
						}
					}
				}
				output.at<double>(i, j) = (double)((sum / denom) / weight_sum);
				// sum�� denom�� ������ �����ν� bilateral filtering�� ��ü weight�� ��� �ݿ���
				// �� ���� weight_sum ���� ������ �����ν� kernel�� ����� ���� adjust kernel ����� ����
				// �������� output �ȼ��� �־���

			}
		}
	}

	return output; // filtering ������ output matrix�� ��ȯ
}


// bilateral filtering (RGB image)
// �Ű�����: input = ���͸� ������ �̹���, n = neighbor�� ����, sigma_t =  t ���� gaussian ������ ǥ������(x-coordinate),
// sigma_s = s ���� gaussian ������ ǥ������(y-coordinate), sigma_r = intensity ���� gaussian ������ ǥ������(color),
// opt = boundary process ���
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
	// kernel matrix�� kernel size�� �°� 32bit float���·� �����ϰ�, ���� 0���� �ʱ�ȭ

	// �ʱ⼳��: �Ÿ��� ����ϴ� kernel (gaussian filtering�� kernel�� ����)
	denom_g = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2)))); // gaussian filter weight�� ���ڰ� ���
			kernel.at<float>(a + n, b + n) = value1; // �� Ŀ�ο� ���ڰ�(value1) ������Ʈ
			denom_g += value1; // ����þ� filter weight�� �и� ����ϱ� ���� denom�� �� ���ڸ� ����� ���� ���� ����
		}
	}

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom_g; // gaussian filter weight�� �и�(denom)���� �ݿ��Ͽ� kernel ������Ʈ
		}
	}

	Mat output = Mat::zeros(row, col, input.type());
	// output�� input�� ũ��, Ÿ���� matrix�̰�, ���� 0���� �ʱ�ȭ

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-padding")) { // boundary process�� zerop-padding �� ��
				float sum_r = 0.0;
				float sum_g = 0.0;
				float sum_b = 0.0;
				float denom = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							// �ȼ��� border �ۿ� ������ �ȼ� ���� 0���� ����Ƿ�, �ȿ� �ִ� ��츸 �������
							float value_r = exp(-pow((float)input.at<Vec3d>(i, j)[0] - (float)input.at<Vec3d>(i + x, j + y)[0], 2) / (2 * pow(sigma_r, 2)));
							float value_g = exp(-pow((float)input.at<Vec3d>(i, j)[1] - (float)input.at<Vec3d>(i + x, j + y)[1], 2) / (2 * pow(sigma_r, 2)));
							float value_b = exp(-pow((float)input.at<Vec3d>(i, j)[2] - (float)input.at<Vec3d>(i + x, j + y)[2], 2) / (2 * pow(sigma_r, 2)));
							// RGB ä�� ���� intensity ���̸� ����� weight�� ���
							float weight = kernel.at<float>(x + n, y + n) * value_r * value_g * value_b;
							// �Ÿ��� ����ϴ� kernel weight�� intensity�� ����� weight�� �����־� weight �� ������Ʈ
							sum_r += input.at<Vec3d>(i + x, j + y)[0] * weight;
							sum_g += input.at<Vec3d>(i + x, j + y)[1] * weight;
							sum_b += input.at<Vec3d>(i + x, j + y)[2] * weight;
							// �� ä�κ��� filtering �� neighbor ������ weight�� ���ϰ� �̸� ������
							denom += weight; // weight�� ���� denom�� ����
						}
					}
				}
				output.at<Vec3d>(i, j)[0] = (double)(sum_r / denom);
				output.at<Vec3d>(i, j)[1] = (double)(sum_g / denom);
				output.at<Vec3d>(i, j)[2] = (double)(sum_b / denom);
				// �� ä�κ��� sum�� denom�� ������ �����ν� bilateral filtering�� ��ü weight�� ��� �ݿ���
				// �׸��� �ݿ��� ���� output �ȼ��� �־���
			}

			else if (!strcmp(opt, "mirroring")) { // boundary process�� mirroring �� ��
				float sum_r = 0.0;
				float sum_g = 0.0;
				float sum_b = 0.0;
				float denom = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						if (i + x > row - 1) {  // filter�� �Ʒ��� boundary�� �ʰ����� ��,
							tempa = i - x; // boundary �� ���� mirroring
						}
						else if (i + x < 0) { // filter�� ���� boundary�� �ʰ����� ��,
							tempa = -(i + x); // boundary �� ���� mirroring
						}
						else {
							tempa = i + x;
						}
						if (j + y > col - 1) { // filter�� ������ boundary�� �ʰ����� ��,
							tempb = j - y; // boundary �� ���� mirroring
						}
						else if (j + y < 0) { // filter�� ���� boundary�� �ʰ����� ��,
							tempb = -(j + y); // boundary �� ���� mirroring
						}
						else {
							tempb = j + y;
						}
						float value_r = exp(-pow((float)input.at<Vec3d>(i, j)[0] - (float)input.at<Vec3d>(tempa, tempb)[0], 2) / (2 * pow(sigma_r, 2)));
						float value_g = exp(-pow((float)input.at<Vec3d>(i, j)[1] - (float)input.at<Vec3d>(tempa, tempb)[1], 2) / (2 * pow(sigma_r, 2)));
						float value_b = exp(-pow((float)input.at<Vec3d>(i, j)[2] - (float)input.at<Vec3d>(tempa, tempb)[2], 2) / (2 * pow(sigma_r, 2)));
						// RGB ä�� ���� intensity ���̸� ����� weight�� ���
						float weight = kernel.at<float>(x + n, y + n) * value_r * value_g * value_b;
						// �Ÿ��� ����ϴ� kernel weight�� intensity�� ����� weight�� �����־� weight �� ������Ʈ
						sum_r += input.at<Vec3d>(tempa, tempb)[0] * weight;
						sum_g += input.at<Vec3d>(tempa, tempb)[1] * weight;
						sum_b += input.at<Vec3d>(tempa, tempb)[2] * weight;
						// �� ä�κ��� filtering �� neighbor ������ weight�� ���ϰ� �̸� ������
						denom += weight; // weight�� ���� denom�� ����
					}
				}
				output.at<Vec3d>(i, j)[0] = (double)(sum_r / denom);
				output.at<Vec3d>(i, j)[1] = (double)(sum_g / denom);
				output.at<Vec3d>(i, j)[2] = (double)(sum_b / denom);
				// �� ä�κ��� sum�� denom�� ������ �����ν� bilateral filtering�� ��ü weight�� ��� �ݿ���
				// �׸��� �ݿ��� ���� output �ȼ��� �־���
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
							// �ȼ��� border �ۿ� ������ �ȼ� ���� 0���� ����Ƿ�, �ȿ� �ִ� ��츸 �������
							// �� ä�κ��� filtering �� neighbor ������ weight�� ���ϰ� �̸� �� ���ؼ� sum1�� ����
							float value_r = exp(-pow((float)input.at<Vec3d>(i, j)[0] - (float)input.at<Vec3d>(i + x, j + y)[0], 2) / (2 * pow(sigma_r, 2)));
							float value_g = exp(-pow((float)input.at<Vec3d>(i, j)[1] - (float)input.at<Vec3d>(i + x, j + y)[1], 2) / (2 * pow(sigma_r, 2)));
							float value_b = exp(-pow((float)input.at<Vec3d>(i, j)[2] - (float)input.at<Vec3d>(i + x, j + y)[2], 2) / (2 * pow(sigma_r, 2)));
							// RGB ä�� ���� intensity ���̸� ����� weight�� ���
							float weight = kernel.at<float>(x + n, y + n) * value_r * value_g * value_b;
							// �Ÿ��� ����ϴ� kernel weight�� intensity�� ����� weight�� �����־� weight �� ������Ʈ
							sum_r += (float)input.at<Vec3d>(i + x, j + y)[0] * weight;
							sum_g += (float)input.at<Vec3d>(i + x, j + y)[1] * weight;
							sum_b += (float)input.at<Vec3d>(i + x, j + y)[2] * weight;
							// �� ä�κ��� filtering �� neighbor ������ weight�� ���ϰ� �̸� ������
							denom += weight; // weight�� ���� denom�� ����
						}
					}
				}

				//adjust kernel ����� ����ϱ� ���� ���� �ڵ带 �Ϻ� �ݺ��Ͽ� bilateral filtering�� ��ü weight ���� ���� => weight_sum �� ����
				float weight_sum = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							float value_r = exp(-pow((float)input.at<Vec3d>(i, j)[0] - (float)input.at<Vec3d>(i + x, j + y)[0], 2) / (2 * pow(sigma_r, 2)));
							float value_g = exp(-pow((float)input.at<Vec3d>(i, j)[1] - (float)input.at<Vec3d>(i + x, j + y)[1], 2) / (2 * pow(sigma_r, 2)));
							float value_b = exp(-pow((float)input.at<Vec3d>(i, j)[2] - (float)input.at<Vec3d>(i + x, j + y)[2], 2) / (2 * pow(sigma_r, 2)));
							float weight = kernel.at<float>(x + n, y + n) * value_r * value_g * value_b;
							weight_sum += (weight / denom); // weight_sum �� bilateral filtering�� ���� weight ���� ����
						}
					}
				}

				output.at<Vec3d>(i, j)[0] = (double)((sum_r / denom) / weight_sum);
				output.at<Vec3d>(i, j)[1] = (double)((sum_g / denom) / weight_sum);
				output.at<Vec3d>(i, j)[2] = (double)((sum_b / denom) / weight_sum);
				// �� ä�κ��� sum�� denom�� ������ �����ν� bilateral filtering�� ��ü weight�� ��� �ݿ���
				// �� ���� weight_sum ���� ������ �����ν� kernel�� ����� ���� adjust kernel ����� ����
				// �������� output �ȼ��� �־���
			}
		}
	}

	return output; // filtering ������ output matrix�� ��ȯ
}