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

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input �̹��� �ҷ���
	Mat input_gray; // ��� �̹����� ������ Mat

	// check for validation
	if (!input.data) {
		printf("Could not open\n"); // �̹��� ���Ͽ� ������ ������ ������ �޼��� ���
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// RGB �̹����� ��� �̹����� ��ȯ�Ͽ� input_gray�� ���� 

	// ���� �̹����� noise �߰�
	Mat noise_Gray = Add_salt_pepper_Noise(input_gray, 0.1f, 0.1f);
	// ���� ��� �̹����� salt, pepper ���� density�� 0.1f�� salt and pepper noise�� �߰��Ͽ� noise_Gray�� ����
	Mat noise_RGB = Add_salt_pepper_Noise(input, 0.1f, 0.1f);
	// ���� �̹����� salt, pepper ���� density�� 0.1f�� salt and pepper noise�� �߰��Ͽ� noise_RGB�� ����

	// Denoise, using median filter
	int window_radius = 2; // window radius�� 2�� �ʱ�ȭ
	Mat Denoised_Gray = Salt_pepper_noise_removal_Gray(noise_Gray, window_radius, "zero-padding");
	// noise_Gray�� window radius = 2�� median filtering�� �����Ͽ� Salt and pepper noise�� ����
	// boundary process�� zero-padding ����� ���
	Mat Denoised_RGB = Salt_pepper_noise_removal_RGB(noise_RGB, window_radius, "adjustkernel");
	// noise_RGB�� window radius = 2�� median filtering�� �����Ͽ� Salt and pepper noise�� ����
	// boundary process�� adjustkernel ����� ���

	namedWindow("Grayscale", WINDOW_AUTOSIZE); // Grayscale â ���
	imshow("Grayscale", input_gray); // â�� input_gray �̹��� ������
	
	namedWindow("RGB", WINDOW_AUTOSIZE); // RGB â ���
	imshow("RGB", input); // â�� input �̹��� ������

	namedWindow("Impulse Noise (Grayscale)", WINDOW_AUTOSIZE); // Impulse Noise (Grayscale) â ���
	imshow("Impulse Noise (Grayscale)", noise_Gray); // â�� noise_Gray �̹��� ������

	namedWindow("Impulse Noise (RGB)", WINDOW_AUTOSIZE); // Impulse Noise (RGB) â ���
	imshow("Impulse Noise (RGB)", noise_RGB); // â�� noise_RGB �̹��� ������

	namedWindow("Denoised (Grayscale)", WINDOW_AUTOSIZE); // Denoised (Grayscale) â ���
	imshow("Denoised (Grayscale)", Denoised_Gray); // â�� Denoised_Gray �̹��� ������

	namedWindow("Denoised (RGB)", WINDOW_AUTOSIZE); // Denoised (RGB) â ���
	imshow("Denoised (RGB)", Denoised_RGB); // â�� Denoised_RGB �̹��� ������

	waitKey(0);

	return 0;
}


// Adding salt and pepper noise
// �Ű�����: input = noise�� �߰��� �̹���, ps = salt�� density, pp = pepper�� density
Mat Add_salt_pepper_Noise(const Mat input, float ps, float pp)
{
	Mat output = input.clone(); // input matrix�� �����Ͽ� output matrix�� ����
	RNG rng; // Random Number Generator

	int amount1 = (int)(output.rows * output.cols * pp); // output �̹����� �߰��� pepper noise�� ��
	int amount2 = (int)(output.rows * output.cols * ps); // output �̹����� �߰��� salt noise�� ��

	int x, y;

	// Grayscale image
	if (output.channels() == 1) {
		for (int counter = 0; counter < amount1; ++counter)
			output.at<G>(rng.uniform(0, output.rows), rng.uniform(0, output.cols)) = 0;
			// output�� ������ ��ġ�� pepper noise (��=0) �� �߰�
			// rng.uniform(a,b)�� (a, b] �������� �����ϰ� ������ ���� ������ ��ȯ

		for (int counter = 0; counter < amount2; ++counter)
			output.at<G>(rng.uniform(0, output.rows), rng.uniform(0, output.cols)) = 255;
			// output�� ������ ��ġ�� salt noise (��=255) �� �߰�
	}
	// Color image
	else if (output.channels() == 3) {
		// pepper noise (��=0) �߰�
		for (int counter = 0; counter < amount1; ++counter) {
			x = rng.uniform(0, output.rows); // �������� �� ����
			y = rng.uniform(0, output.cols); // �������� �� ����
			output.at<C>(x, y)[0] = 0; // R channel���� �������� ���õ� �ȼ��� pepper noise �߰�

			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[1] = 0; // G channel���� �������� ���õ� �ȼ��� pepper noise �߰�

			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[2] = 0; // B channel���� �������� ���õ� �ȼ��� pepper noise �߰�
		}
		// salt noise (��=255) �߰�
		for (int counter = 0; counter < amount2; ++counter) {
			x = rng.uniform(0, output.rows); // �������� �� ����
			y = rng.uniform(0, output.cols); // �������� �� ����
			output.at<C>(x, y)[0] = 255; // R channel���� �������� ���õ� �ȼ��� salt noise �߰�

			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[1] = 255; // G channel���� �������� ���õ� �ȼ��� salt noise �߰�

			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[2] = 255; // B channel���� �������� ���õ� �ȼ��� salt noise �߰�
		}
	}

	return output; // output �̹����� ��ȯ
}

// removing salt and pepper noise with Median filter (Grayscale image)
// �Ű�����: input = noise�� ������ �̹���, n = neighbor�� ����, opt = boundary process ���
Mat Salt_pepper_noise_removal_Gray(const Mat input, int n, const char *opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1); // (2N+1)*(2N+1)
	int median;		// index of median value
	int tempa, tempb;

	// initialize median filter kernel
	Mat kernel = Mat::zeros(kernel_size * kernel_size, 1, input.type());
	// kernel�� kernel_size*kernel_size ���� ��, 1���� ���� �̷���� matrix
	// matrix ������ input�� ���� type�̰� 0���� �ʱ�ȭ
	
	Mat output = Mat::zeros(row, col, input.type());
	// output�� input�� ũ��, Ÿ���� matrix�̰�, ���� 0���� �ʱ�ȭ

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) { // output�� �� �ȼ����� �ݺ�
					
			if (!strcmp(opt, "zero-padding")) { // boundary process�� zero-padding �� ��
				int count = 0; // count�� kernel�� �� index�� �ǹ�
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							// �ȼ��� border �ۿ� ������ �ȼ� ���� 0���� ����Ƿ�, �ȿ� �ִ� ��츸 ���� ������Ʈ����
							kernel.at<G>(count, 0) = input.at<G>(i + x, j + y);
							// kernel�� �ش� ��ġ�� input �ȼ� �� ����
						}
						count++; // kernel �� index ++
					}
				}
				median = kernel_size * kernel_size / 2;
				// median �� �� ũ�� / 2 �� ����� ��
				// ���� ������ �Ŀ� �������־�� �ùٸ� median(�߾Ӱ�)�� index�� �� ��!
			}

			else if (!strcmp(opt, "mirroring")) { // boundary process�� mirroring �� ��
				int count = 0; // count�� kernel�� �� index�� �ǹ�
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
						kernel.at<G>(count, 0) = input.at<G>(tempa, tempb);
						// kernel�� �ش� ��ġ�� input �ȼ� �� ����
						count++; // kernel �� index ++
					}
				}
				median = kernel_size * kernel_size / 2;
				// median index = �� ũ�� / 2
				// ���� ������ �Ŀ� �������־�� �ùٸ� median(�߾Ӱ�)�� index�� �� ��!
			}

			else if (!strcmp(opt, "adjustkernel")) { // boundary process�� adjustkernel �� ��
				int count = 0; // count�� kernel�� �� index�� �ǹ�
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {		
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							// �ȼ��� border �ۿ� �ִ� ��찡 �����ϸ� �̸� �����ϰ� kernel ����� �۰� ����
							kernel.at<G>(count, 0) = input.at<G>(i + x, j + y);
							// kernel�� �ش� ��ġ�� input �ȼ� �� ����
							count++; // kernel �� index ++
						}
					}
				}
				median = count / 2;
				// median index = count / 2
				// border �ۿ� �ִ� �ȼ��� �����ϰ� kernel ����� �ƿ� �۰� ����ϹǷ�,
				// count / 2 �� �̿��Ͽ� median index�� ����
			}

			// Sort the kernels in ascending order
			sort(kernel, kernel, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			// kernel�� ������ ���������� �������� ����

			output.at<G>(i, j) = kernel.at<G>(median, 0);
			// kernel�� median ���� output �ȼ��� update
		}
	}

	return output; // output �̹����� ��ȯ
}

// removing salt and pepper noise with Median filter (RGB image)
// �Ű�����: input = noise�� ������ �̹���, n = neighbor�� ����, opt = boundary process ���
Mat Salt_pepper_noise_removal_RGB(const Mat input, int n, const char *opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1); // (2N + 1) * (2N + 1)
	int median;		// index of median value
	int channel = input.channels();
	int tempa, tempb;

	// initialize median filter kernel
	Mat kernel = Mat::zeros(kernel_size * kernel_size, channel, input.type() - 16);
	// kernel�� kernel_size*kernel_size ���� ��, channel ���� ���� �̷���� matrix
	// initialize ( (TypeX with 3 channel) - (TypeX with 1 channel) = 16 )
	// ex) CV_8UC3 - CV_8U = 16
	// matrix ������ 0���� �ʱ�ȭ

	Mat output = Mat::zeros(row, col, input.type());
	// output�� input�� ũ��, Ÿ���� matrix�̰�, ���� 0���� �ʱ�ȭ

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) { // output�� �� �ȼ����� �ݺ�
					
			if (!strcmp(opt, "zero-padding")) { // boundary process�� zero-padding �� ��
				int count = 0; // count�� kernel�� �� index�� �ǹ�
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							// �ȼ��� border �ۿ� ������ �ȼ� ���� 0���� ����Ƿ�, �ȿ� �ִ� ��츸 ���� ������Ʈ����
							kernel.at<G>(count, 0) = input.at<C>(i + x, j + y)[0]; // kernel�� ù��° ������ �ش� ��ġ�� input�� R channel pixel ���� ����
							kernel.at<G>(count, 1) = input.at<C>(i + x, j + y)[1]; // kernel�� �ι�° ������ �ش� ��ġ�� input�� G channel pixel ���� ����
							kernel.at<G>(count, 2) = input.at<C>(i + x, j + y)[2]; // kernel�� ����° ������ �ش� ��ġ�� input�� B channel pixel ���� ����
						}
						count++; // kernel �� index ++
					}
				}
				median = kernel_size * kernel_size / 2;
				// median �� �� ũ�� / 2 �� ����� ��
				// ���� ������ �Ŀ� �������־�� �ùٸ� median(�߾Ӱ�)�� index�� �� ��!
			}

			else if (!strcmp(opt, "mirroring")) { // boundary process�� mirroring �� ��
				int count = 0; // count�� kernel�� �� index�� �ǹ�
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
						kernel.at<G>(count, 0) = input.at<C>(tempa, tempb)[0];  // kernel�� ù��° ������ �ش� ��ġ�� input�� R channel pixel ���� ����
						kernel.at<G>(count, 1) = input.at<C>(tempa, tempb)[1];  // kernel�� �ι�° ������ �ش� ��ġ�� input�� G channel pixel ���� ����
						kernel.at<G>(count, 2) = input.at<C>(tempa, tempb)[2];  // kernel�� ����° ������ �ش� ��ġ�� input�� B channel pixel ���� ����
						count++; // kernel �� index ++
					}
				}
				median = kernel_size * kernel_size / 2;
				// median index = �� ũ�� / 2
				// ���� ������ �Ŀ� �������־�� �ùٸ� median(�߾Ӱ�)�� index�� �� ��!
			}

			else if (!strcmp(opt, "adjustkernel")) { // boundary process�� adjustkernel �� ��
				int count = 0; // count�� kernel�� �� index�� �ǹ�
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {	
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							// �ȼ��� border �ۿ� �ִ� ��찡 �����ϸ� �̸� �����ϰ� kernel ����� �۰� ����
							kernel.at<G>(count, 0) = input.at<C>(i + x, j + y)[0]; // kernel�� ù��° ������ �ش� ��ġ�� input�� R channel pixel ���� ����
							kernel.at<G>(count, 1) = input.at<C>(i + x, j + y)[1]; // kernel�� �ι�° ������ �ش� ��ġ�� input�� G channel pixel ���� ����
							kernel.at<G>(count, 2) = input.at<C>(i + x, j + y)[2]; // kernel�� ����° ������ �ش� ��ġ�� input�� B channel pixel ���� ����
							count++; // kernel �� index ++
						}
					}
				}
				median = count / 2;
				// median index = count / 2
				// border �ۿ� �ִ� �ȼ��� �����ϰ� kernel ����� �ƿ� �۰� ����ϹǷ�,
				// count / 2 �� �̿��Ͽ� median index�� ����
			}

			// Sort the kernels in ascending order
			sort(kernel, kernel, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
			// kernel�� ������ ���������� �������� ����
			
			// �� ä�κ��� kernel�� median ���� output �ȼ��� update
			output.at<C>(i, j)[0] = kernel.at<G>(median, 0);
			output.at<C>(i, j)[1] = kernel.at<G>(median, 1);
			output.at<C>(i, j)[2] = kernel.at<G>(median, 2);
		}
	}

	return output; // output �̹����� ��ȯ

}