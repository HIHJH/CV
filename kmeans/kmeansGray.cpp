#include <iostream>
#include <opencv2/opencv.hpp>

#define IM_TYPE	CV_8U;

using namespace cv;


int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_GRAYSCALE); // input 이미지 불러옴

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl; // 이미지 파일에 데이터 오류가 있으면 메세지 출력
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE); // Original 창 띄움
	imshow("Original", input); // 창에 input 이미지 보여줌


	// sample1
	// input: intensity
	Mat samples1(input.rows * input.cols, 1, CV_32F);
	// (input의 전체 픽셀 수 * 고려해야할 요소 수(intensity => 1개)) 크기의 sample matrix 생성
	// samples에는 clustering할 데이터가 저장
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
			samples1.at<float>(y + x * input.rows, 0) = input.at<uchar>(y, x); // 2차원 샘플 matrix에 input의 픽셀값 반영

	int clusterCount = 10; // cluster의 개수 = 10개
	Mat labels1; // 각 데이터 포인트가 속한 cluster의 index를 저장
	int attempts = 5; //초기 center를 선택하는 횟수 = 5번
	Mat centers1; // 각 cluster의 center 저장

	// kmeans clustering 수행
	// samples: clustering할 데이터
	// clusterCount: cluster 개수
	// labels: 각 데이터 포인트가 속한 cluster의 index를 저장
	// criteria: 중단 기준 => 최대 10000번 반복하거나, center 변화량이 0.0001보다 작아지면 알고리즘 중단
	// attempts: 초기 center를 선택하는 횟수
	// KMEANS_PP_CENTERS: k-means++ 알고리즘을 사용하여 초기 center 선택
	// centers: 각 cluster의 center 저장
	kmeans(samples1, clusterCount, labels1, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers1);

	Mat new_image1(input.size(), input.type()); // cluster 결과 이미지를 저장할 Mat
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
		{
			int cluster_idx = labels1.at<int>(y + x * input.rows, 0); // 1차원 행렬인 labels에 저장된 cluster의 index 값을 저장
			// 각 채널의 픽셀마다 cluster center 값을 new_image Mat에 받아옴
			new_image1.at<uchar>(y, x) = centers1.at<float>(cluster_idx, 0);
		}


	// sample2
	// input: intensity + position (intensity,x,y)
	Mat samples2(input.rows * input.cols, 3, CV_32F);
	// (input의 전체 픽셀 수 * 고려해야할 요소 수(intensity, x, y => 3개)) 크기의 sample matrix 생성
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
		{
			samples2.at<float>(y + x * input.rows, 0) = input.at<uchar>(y, x) / 255.0; // position 까지 고려해야하므로 intensity 값을 normalize 해줌
			samples2.at<float>(y + x * input.rows, 1) = x / (float)input.cols; // intensity도 고려하므로 x position 값을 nomalization
			samples2.at<float>(y + x * input.rows, 2) = y / (float)input.rows; // intensity도 고려하므로 y positon 값을 normalization
		}

	Mat labels2;
	Mat centers2;

	// kmeans clustering 수행
	kmeans(samples2, clusterCount, labels2, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers2);

	Mat new_image2(input.size(), input.type());
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
		{
			int cluster_idx = labels2.at<int>(y + x * input.rows, 0);
			// intensity 값을 normalize 해주었으므로, 다시 255.0을 곱하여 range를 복원시킴
			new_image2.at<uchar>(y, x) = centers2.at<float>(cluster_idx, 0) * 255.0;
		}

	namedWindow("clustered image1", WINDOW_AUTOSIZE); // clustered image1 창 띄움
	imshow("clustered image1", new_image1); // 창에 new_image1 이미지 보여줌
	namedWindow("clustered image2", WINDOW_AUTOSIZE); // clustered image2 창 띄움
	imshow("clustered image2", new_image2); // 창에 new_image2 이미지 보여줌

	waitKey(0);

	return 0;
}
