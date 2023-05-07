#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt);

int main()
{
	Mat input, rotated;
	
	// Read each image
	input = imread("lena.jpg");

	// Check for invalid input
	if (!input.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	
	// original image
	namedWindow("image");
	imshow("image", input);

	rotated = myrotate<Vec3b>(input, 45, "bilinear");

	// rotated image
	namedWindow("rotated");
	imshow("rotated", rotated);

	waitKey(0);

	return 0;
}

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt) {
	int row = input.rows;
	int col = input.cols;

	float radian = angle * CV_PI / 180;

	float sq_row = ceil(row * sin(radian) + col * cos(radian));
	float sq_col = ceil(col * sin(radian) + row * cos(radian));

	Mat output = Mat::zeros(sq_row, sq_col, input.type());

	for (int i = 0; i < sq_row; i++) {
		for (int j = 0; j < sq_col; j++) {
			float x = (j - sq_col / 2) * cos(radian) - (i - sq_row / 2) * sin(radian) + col / 2;
			float y = (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;

			if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) {
				if (!strcmp(opt, "nearest")) {
					int nearest_x = round(x);
					int nearest_y = round(y);
					output.at<Vec3b>(i, j) = input.at<Vec3b>(nearest_y, nearest_x);
				}
				else if (!strcmp(opt, "bilinear")) {
					int x0 = floor(x);
					int x1 = x0 + 1;
					int y0 = floor(y);
					int y1 = y0 + 1;
					float x_ratio = x - x0;
					float y_ratio = y - y0;
					output.at<Vec3b>(i, j) = (1 - x_ratio) * (1 - y_ratio) * input.at<Vec3b>(y0, x0) +
						x_ratio * (1 - y_ratio) * input.at<Vec3b>(y0, x1) +
						(1 - x_ratio) * y_ratio * input.at<Vec3b>(y1, x0) +
						x_ratio * y_ratio * input.at<Vec3b>(y1, x1);
				}
			}
		}
	}

	return output;
}