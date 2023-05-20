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

	Mat src = imread("building.jpg", CV_LOAD_IMAGE_COLOR); // 이미지 불러옴
	Mat dst, color_dst;

	// check for validation
	if (!src.data) {
		printf("Could not open\n"); // 이미지 파일 안열리면 오류 메세지 출력
		return -1;
	}

	Canny(src, dst, 50, 200, 3);
	// 임계값이 50, 200이고 sobel의 aperture size을 3으로 하여 src의 canny edge 검출
	// 결과는 dst에 저장
	cvtColor(dst, color_dst, COLOR_GRAY2BGR); // grayscale -> RGB

	//Standard Hough transform (using 'HoughLines')
#if 1
	vector<Vec2f> lines;

	// hough 변환 수행하여 선 검출
	// image: hough 변환 수행할 이미지(canny edge 검출을 통해 얻은 edge 이미지) = dst
	// lines: 검출된 선 저장할 벡터 = lines
	// rho: 거리 해상도 = 1
	// theta: 각도 해상도 = CV_PI / 180
	// threshold: 선을 검출하기 위한 최소 투표 수 = 150
	// srn, stn: 향상된 hough 변환을 위한 옵션 여부 = 0, 0
	HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);

	// 검출된 선 시각화
	for (size_t i = 0; i < lines.size(); i++)
	{
		// 극좌표에서의 r, theta
		float rho = lines[i][0];
		float theta = lines[i][1];
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho; // 극좌표값을 x0, y0 직교좌표로 변환
		Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a))); // 선의 시작점: 선 방향 따라 1000 정도 이동
		Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a))); // 선의 끝점: 선 방향 따라 1000 정도 이동
		line(color_dst, pt1, pt2, Scalar(0, 0, 255), 1, 8);
	}

	//Probabilistic Hough transform (using 'HoughLinesP')

#else
	vector<Vec4i> lines;

	// 확률적으로 hough 변환 수행하여 선 검출
	// image: hough 변환 수행할 이미지(canny edge 검출을 통해 얻은 edge 이미지) = dst
	// lines: 검출된 선 저장할 벡터 = lines
	// rho: 거리 해상도 = 1
	// theta: 각도 해상도 = CV_PI / 180
	// threshold: 선을 검출하기 위한 최소 투표 수 = 80
	// minLineLength: 검출할 선의 최소 길이 = 30
	// maxLineGap: 선들 간의 최대 허용 간격 = 10
	HoughLinesP(dst, lines, 1, CV_PI / 180, 80, 30, 10);

	// 검출된 선 시각화
	for (size_t i = 0; i < lines.size(); i++)
	{
		line(color_dst, Point(lines[i][0], lines[i][1]),
			Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 1, 8);
	}

#starting ending point 반영하는 방법 - pseudo code

		CascadeHough(image, initialScale, minScale, maxScale, scaleStep, houghThreshold) {
			currentScale = initialScale
				while currentScale <= maxScale:
			currentImage = image를 currentScale로 조정

				if currentScale == initialScale :
					edgeMap = currentImage에 대해 edge 검출 수행
				else :
					edgeMap = 이전 scale에서 검출된 선들을 currentScale에 맞추어 edgeMap 생성

					lineCandidates = HoughTransform(edgeMap, houghThreshold)

					if currentScale < minScale :
						topKLines = lineCandidates 중에서 확률이 높은 상위 k개의 선 선택
					else:
			topKLines = 이전 scale에서 검출된 선들 중에서 currentScale에 맞는 선들 선택

				topKLines를 lineResults에 추가

				currentScale += scaleStep

				return lineResults
		}
	
		HoughTransform(edgeMap, houghThreshold) {
			lineCandidates = 빈 배열

			for 모든 pixels(x, y) in edgeMap :
				if (x, y)가 Hough Transform에 참여하는 조건을 만족:
					 모든 가능한 선에 대해 투표 진행

			for 투표 결과로 얻은 모든 선 :
				if 선이 threshold 넘었는지 확인:
					선을 lineCandidates에 추가

				return lineCandidates
		}
	

#endif
	namedWindow("Source", 1); // Source 창 띄움
	imshow("Source", src); // 창에 src 보여줌
	namedWindow("Detected Lines", 1); // Detected Lines 창 띄움
	imshow("Detected Lines", color_dst); // 창에 color_dst 보여줌
	waitKey(0);

	return 0;
}