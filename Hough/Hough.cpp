// opencv_test.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main() {

	Mat src = imread("building.jpg", CV_LOAD_IMAGE_COLOR); // �̹��� �ҷ���
	Mat dst, color_dst;

	// check for validation
	if (!src.data) {
		printf("Could not open\n"); // �̹��� ���� �ȿ����� ���� �޼��� ���
		return -1;
	}

	Canny(src, dst, 50, 200, 3);
	// �Ӱ谪�� 50, 200�̰� sobel�� aperture size�� 3���� �Ͽ� src�� canny edge ����
	// ����� dst�� ����
	cvtColor(dst, color_dst, COLOR_GRAY2BGR); // grayscale -> RGB

	//Standard Hough transform (using 'HoughLines')
#if 1
	vector<Vec2f> lines;

	// hough ��ȯ �����Ͽ� �� ����
	// image: hough ��ȯ ������ �̹���(canny edge ������ ���� ���� edge �̹���) = dst
	// lines: ����� �� ������ ���� = lines
	// rho: �Ÿ� �ػ� = 1
	// theta: ���� �ػ� = CV_PI / 180
	// threshold: ���� �����ϱ� ���� �ּ� ��ǥ �� = 150
	// srn, stn: ���� hough ��ȯ�� ���� �ɼ� ���� = 0, 0
	HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);

	// ����� �� �ð�ȭ
	for (size_t i = 0; i < lines.size(); i++)
	{
		// ����ǥ������ r, theta
		float rho = lines[i][0];
		float theta = lines[i][1];
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho; // ����ǥ���� x0, y0 ������ǥ�� ��ȯ
		Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a))); // ���� ������: �� ���� ���� 1000 ���� �̵�
		Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a))); // ���� ����: �� ���� ���� 1000 ���� �̵�
		line(color_dst, pt1, pt2, Scalar(0, 0, 255), 1, 8);
	}

	//Probabilistic Hough transform (using 'HoughLinesP')

#else
	vector<Vec4i> lines;

	// Ȯ�������� hough ��ȯ �����Ͽ� �� ����
	// image: hough ��ȯ ������ �̹���(canny edge ������ ���� ���� edge �̹���) = dst
	// lines: ����� �� ������ ���� = lines
	// rho: �Ÿ� �ػ� = 1
	// theta: ���� �ػ� = CV_PI / 180
	// threshold: ���� �����ϱ� ���� �ּ� ��ǥ �� = 80
	// minLineLength: ������ ���� �ּ� ���� = 30
	// maxLineGap: ���� ���� �ִ� ��� ���� = 10
	HoughLinesP(dst, lines, 1, CV_PI / 180, 80, 30, 10);

	// ����� �� �ð�ȭ
	for (size_t i = 0; i < lines.size(); i++)
	{
		line(color_dst, Point(lines[i][0], lines[i][1]),
			Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 1, 8);
	}

#starting ending point �ݿ��ϴ� ��� - pseudo code

		CascadeHough(image, initialScale, minScale, maxScale, scaleStep, houghThreshold) {
			currentScale = initialScale
				while currentScale <= maxScale:
			currentImage = image�� currentScale�� ����

				if currentScale == initialScale :
					edgeMap = currentImage�� ���� edge ���� ����
				else :
					edgeMap = ���� scale���� ����� ������ currentScale�� ���߾� edgeMap ����

					lineCandidates = HoughTransform(edgeMap, houghThreshold)

					if currentScale < minScale :
						topKLines = lineCandidates �߿��� Ȯ���� ���� ���� k���� �� ����
					else:
			topKLines = ���� scale���� ����� ���� �߿��� currentScale�� �´� ���� ����

				topKLines�� lineResults�� �߰�

				currentScale += scaleStep

				return lineResults
		}
	
		HoughTransform(edgeMap, houghThreshold) {
			lineCandidates = �� �迭

			for ��� pixels(x, y) in edgeMap :
				if (x, y)�� Hough Transform�� �����ϴ� ������ ����:
					 ��� ������ ���� ���� ��ǥ ����

			for ��ǥ ����� ���� ��� �� :
				if ���� threshold �Ѿ����� Ȯ��:
					���� lineCandidates�� �߰�

				return lineCandidates
		}
	

#endif
	namedWindow("Source", 1); // Source â ���
	imshow("Source", src); // â�� src ������
	namedWindow("Detected Lines", 1); // Detected Lines â ���
	imshow("Detected Lines", color_dst); // â�� color_dst ������
	waitKey(0);

	return 0;
}