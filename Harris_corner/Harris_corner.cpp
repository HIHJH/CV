// opencv_test.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

vector<Point2f> MatToVec(const Mat input);
Mat NonMaximum_Suppression(const Mat input, Mat corner_mat, int radius);
Mat Mirroring(const Mat input, int n);
void type2str(int type);


int main() {

	//Use the following three images.
	//Mat input = imread("checkerboard.png", CV_LOAD_IMAGE_COLOR);
	Mat input = imread("checkerboard2.jpg", CV_LOAD_IMAGE_COLOR);
	//Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input �̹��� �ҷ���

		// check for validation
	if (!input.data) {
		printf("Could not open\n"); // �̹��� ���� �����Ϳ� ������ ������ �޼��� ���
		return -1;
	}

	int row = input.rows;
	int col = input.cols;

	Mat input_gray, input_visual;
	Mat output, output_norm, corner_mat;
	vector<Point2f> points;

	corner_mat = Mat::zeros(row, col, CV_8U); // �� �ȼ��� �ڳ����� �ƴ����� ��Ÿ���� Mat (�ڳ�O:1, �ڳ�X:0)

	bool NonMaxSupp = true; // non-maximum suppression ���� ���� ����

	bool Subpixel = true; // subpixel refinement ���� ���� ����


	cvtColor(input, input_gray, CV_RGB2GRAY);	// RGB �̹����� ��� �̹����� ��ȯ

	//Harris corner detection using 'cornerHarris'
	//Note that 'src' of 'cornerHarris' can be either 1) input single-channel 8-bit or 2) floating-point image.
	// �ظ��� �ڳ� ����
	// src = �Է� �̹��� = input_gray
	// dst = ��� �̹��� = output
	// blockSize = �ڳ� ������ ���� ������ ũ�� = 2
	// ksize = Ŀ�� ũ�� = 3
	// k = �ڳ� �������� ���Ǵ� Harris ���� ��� = 0.04
	// borderType = �̹��� ��� ó�� ��� = BORDER_DEFAULT
	cornerHarris(input_gray, output, 2, 3, 0.04, BORDER_DEFAULT);


	//Scale the Harris response map 'output' from 0 to 1.
	//This is for display purpose only.
	normalize(output, output_norm, 0, 1.0, NORM_MINMAX); // output�� 0~1 ���� ������ normalization
	namedWindow("Harris Response", WINDOW_AUTOSIZE); // Harris Response â ���
	imshow("Harris Response", output_norm); // â�� output_norm �̹��� ������


	//Threshold the Harris corner response.
	//corner_mat = 1 for corner, 0 otherwise.
	input_visual = input.clone(); // input_visual�� input�� ������ ��
	double minVal, maxVal; // �ּ�, �ִ밪�� ������ ����
	Point minLoc, maxLoc; // �ּ�, �ִ밪�� ��ǥ�� ������ ����
	minMaxLoc(output, &minVal, &maxVal, &minLoc, &maxLoc); // output�� �ּ�, �ִ밪�� �� ��ġ�� ã�� ����

	//�ڳ� �ð�ȭ
	int corner_num = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (output.at<float>(i, j) > 0.01 * maxVal) // threshold =  0.01 * maxVal
			{
				//input_visual.at<Vec3b>(i, j)[0] = 0;		input_visual.at<Vec3b>(i, j)[1] = 0;	input_visual.at<Vec3b>(i, j)[2] = 255;
				circle(input_visual, Point(j, i), 2, Scalar(0, 0, 255), 1, 8, 0);	//You can also use this function of drawing a circle. For details, search 'circle' in OpenCV.
				// �ش� �ȼ� ��ġ�� ������ ���׶�� ǥ����
				corner_mat.at<uchar>(i, j) = 1; // corner_mat ��Ŀ� 1�� �Ҵ�
				corner_num++;
			}

			else
				output.at<float>(i, j) = 0.0; // �ڳʰ� �ƴϸ� 0�� �Ҵ�
		}
	}
	printf("After cornerHarris, corner number = %d\n\n", corner_num); // corner number�� ���
	namedWindow("Harris Corner", WINDOW_AUTOSIZE); // Harris Corner â ���
	imshow("Harris Corner", input_visual); // â�� input_visual �̹��� ������ 

	//Non-maximum suppression
	if (NonMaxSupp) // NonMaxSupp�� true�̸�,
	{
		NonMaximum_Suppression(output, corner_mat, 2);
		// NonMaximum Suppression ����
		// radius = 2

		//�ڳ� �ð�ȭ
		corner_num = 0;
		input_visual = input.clone(); // input_visual�� input�� ������ ��
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (corner_mat.at<uchar>(i, j) == 1) { // corner �̸�,
					//input_visual.at<Vec3b>(i, j)[0] = 0;		input_visual.at<Vec3b>(i, j)[1] = 0;	input_visual.at<Vec3b>(i, j)[2] = 255;					
					circle(input_visual, Point(j, i), 2, Scalar(0, 0, 255), 1, 8, 0);	//You can also use this function of drawing a circle. For details, search 'circle' in OpenCV.
					// �ش� �ȼ��� ������ ���׶�� ǥ����
					corner_num++;
				}
			}
		}

		printf("After non-maximum suppression, corner number = %d\n\n", corner_num); // corner number ���
		namedWindow("Harris Corner (Non-max)", WINDOW_AUTOSIZE); // Harris Corner (Non-max) â ���
		imshow("Harris Corner (Non-max)", input_visual); // â�� input_visual �̹��� ������
	}

	//Sub-pixel refinement for detected corners
	if (Subpixel) // Subpixel true�̸�,
	{
		Size subPixWinSize(3, 3); // subpixel ������ ������ ������ ũ�� = 3*3
		TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
		// �ִ� �ݺ� Ƚ���� COUNT �Ǵ� �ִ� �ݺ� Ƚ�� = 20, ��ȭ���� tolerance = 0.03 �̸� refinement ����

		points = MatToVec(corner_mat); // corner_mat�� ���� �������� points�� ����

		// cornerSubPix
		// input_gray���� ������ �ڳ� ��ġ(points)�� subPixWinSize, termcrit�� ����ϸ� subpixel ������ ����
		// ��� �ڳ� ��ġ�� �ٽ� points�� ����
		cornerSubPix(input_gray, points, subPixWinSize, Size(-1, -1), termcrit);

		//�ڳ� �ð�ȭ
		input_visual = input.clone(); // input_visual�� input�� ������ ��
		for (int k = 0; k < points.size(); k++) {

			int x = points[k].x; // points ���Ϳ� ����� �� �ڳ��� x, y ��ǥ���� �̿�
			int y = points[k].y;

			if (x<0 || x>col - 1 || y<0 || y>row - 1)
			{
				points.pop_back(); // �� �ڳ��� ��ǥ���� �̹��� ũ�⸦ ����� �ش� ��ǥ�� ����
				continue;
			}

			//input_visual.at<Vec3b>(i, j)[0] = 0;		input_visual.at<Vec3b>(i, j)[1] = 0;	input_visual.at<Vec3b>(i, j)[2] = 255;
			circle(input_visual, Point(x, y), 2, Scalar(0, 0, 255), 1, 8, 0);	//You can also use this function of drawing a circle. For details, search 'circle' in OpenCV.
			// Point(x, y) ��ġ�� �ش��ϴ� �ȼ��� ������ ���׶�� ǥ����
		}

		printf("After subpixel-refinement, corner number = %d\n\n", points.size()); // corner number ���
		namedWindow("Harris Corner (subpixel)", WINDOW_AUTOSIZE); // Harris Corner (subpixel) â ���
		imshow("Harris Corner (subpixel)", input_visual); // â�� input_visual �̹��� ������
	}

	waitKey(0);

	return 0;
}

// input ���� ���� 1�� �ȼ��� ��ġ�� Point2f ���·� �����Ͽ� vector�� ��ȯ�ϴ� �Լ�
vector<Point2f> MatToVec(const Mat input)
{
	vector<Point2f> points; // points vector ����

	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.at<uchar>(i, j) == 1) { // input ���� 1�̸�
				points.push_back(Point2f((float)j, (float)i)); // �� �ȼ� ��ġ�� Point2f ���·� vector�� �߰�
			}
		}
	}

	return points; // points vector ��ȯ
}

//corner_mat = 1 for corner, 0 otherwise.
// NonMaximum Suppression ����
// �Ű�����: input = NMS ������ �̹���, corner_mat = �ڳ� ��ġ�� ��Ÿ���� ���
// radius = NMS ������ �������� ������ ��
Mat NonMaximum_Suppression(const Mat input, Mat corner_mat, int radius)
{
	int row = input.rows;
	int col = input.cols;

	Mat input_mirror = Mirroring(input, radius); // boundary process�� mirroring ����

	for (int i = radius; i < row + radius; i++) {
		for (int j = radius; j < col + radius; j++) {

			if (corner_mat.at<uchar>(i - radius, j - radius) != 1.0) { // �ڳʰ� �ƴ϶��
				continue; // ���� �ȼ��� �Ѿ
			}

			//�ʱ⿡ �ڳ� �ȼ� ���� �ִ�� ���� => is_max = true�� ����
			bool is_max = true;

			for (int a = i - radius; a <= i + radius; a++) { 
				for (int b = j - radius; b <= j + radius; b++) {
					if (a == i && b == j) {
						continue;
					}
					if (input_mirror.at<float>(a, b) > input_mirror.at<float>(i, j)) { // �ش� �ȼ� ������ ������ ���� �ֺ� �ȼ� ���� ũ�ٸ�
						is_max = false; // is_max �� false�� ����
						break;
					}
				}
				if (!is_max) { // ���� ū ���� ���� �ȼ��� �ƴϸ�,
					break;
				}
			}

			if (!is_max) { // ���� ū ���� ���� �ȼ��� �ƴϸ�,
				corner_mat.at<uchar>(i - radius, j - radius) = 0.0; // corner_mat ���� 0���� ����
			}
		}
	}

	return input;
}

// boundary process = mirroring
// �Ű�����: input = mirroring�� �̹���, n = neighbor�� ����
Mat Mirroring(const Mat input, int n)
{
	int row = input.rows;
	int col = input.cols;

	Mat input2 = Mat::zeros(row + 2 * n, col + 2 * n, input.type()); // input2�� mirroring �� Mat ����
	int row2 = input2.rows;
	int col2 = input2.cols;

	// input2�� �߾ӿ� ���� �̹����� ����
	for (int i = n; i < row + n; i++) {
		for (int j = n; j < col + n; j++) {
			input2.at<float>(i, j) = input.at<float>(i - n, j - n);
		}
	}
	// �̹����� �¿� ��Ī�ǵ��� mirroing
	for (int i = n; i < row + n; i++) {
		for (int j = 0; j < n; j++) {
			input2.at<float>(i, j) = input2.at<float>(i, 2 * n - j);
		}
		for (int j = col + n; j < col2; j++) {
			input2.at<float>(i, j) = input2.at<float>(i, 2 * col - 2 + 2 * n - j);
		}
	}
	// �̹����� ���� ��Ī�ǵ��� mirroring
	for (int j = 0; j < col2; j++) {
		for (int i = 0; i < n; i++) {
			input2.at<float>(i, j) = input2.at<float>(2 * n - i, j);
		}
		for (int i = row + n; i < row2; i++) {
			input2.at<float>(i, j) = input2.at<float>(2 * row - 2 + 2 * n - i, j);
		}
	}

	return input2; // input2 Mat ��ȯ
}


//If you want to know the type of 'Mat', use the following function
//For instance, for 'Mat input'
//type2str(input.type());
// Mat�� ��ü Ÿ���� ���ڿ��ι�ȯ
void type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	printf("Matrix: %s \n", r.c_str());
}