#include <iostream>
#include <opencv2/opencv.hpp>

#define IM_TYPE	CV_8UC3

using namespace cv;


int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR); // input �̹��� �ҷ���

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl; // �̹��� ���Ͽ� ������ ������ ������ �޼��� ���
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE); // Original â ���
	imshow("Original", input); // â�� input �̹��� ������


	// sample1
	// input: color(r,g,b)
	Mat samples1(input.rows * input.cols, 3, CV_32F);
	// (input�� ��ü �ȼ� �� * ����ؾ��� ��� ��(r, g, b => 3��)) ũ���� sample matrix ����
	// samples���� clustering�� �����Ͱ� ����
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
		{
			for (int z = 0; z < 3; z++)
				samples1.at<float>(y + x * input.rows, z) = input.at<Vec3b>(y, x)[z]; // 2���� ���� matrix�� input�� �ȼ��� �ݿ�
		}
	
	int clusterCount = 10; // cluster�� ���� = 10��
	Mat labels1; // �� ������ ����Ʈ�� ���� cluster�� index�� ����
	int attempts = 5; //�ʱ� center�� �����ϴ� Ƚ�� = 5��
	Mat centers1; // �� cluster�� center ����

	// kmeans clustering ����
	// samples: clustering�� ������
	// clusterCount: cluster ����
	// labels: �� ������ ����Ʈ�� ���� cluster�� index�� ����
	// criteria: �ߴ� ���� => �ִ� 10000�� �ݺ��ϰų�, center ��ȭ���� 0.0001���� �۾����� �˰��� �ߴ�
	// attempts: �ʱ� center�� �����ϴ� Ƚ��
	// KMEANS_PP_CENTERS: k-means++ �˰����� ����Ͽ� �ʱ� center ����
	// centers: �� cluster�� center ����
	kmeans(samples1, clusterCount, labels1, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers1);

	Mat new_image1(input.size(), input.type()); // cluster ��� �̹����� ������ Mat
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
		{
			int cluster_idx = labels1.at<int>(y + x*input.rows, 0); // 1���� ����� labels�� ����� cluster�� index ���� ����
			// �� ä���� �ȼ����� cluster center ���� new_image Mat�� �޾ƿ�
			new_image1.at<Vec3b>(y, x)[0] = centers1.at<float>(cluster_idx, 0);
			new_image1.at<Vec3b>(y, x)[1] = centers1.at<float>(cluster_idx, 1);
			new_image1.at<Vec3b>(y, x)[2] = centers1.at<float>(cluster_idx, 2);
		}


	// sample2
	// input: color + position (r,g,b,x,y)
	Mat samples2(input.rows * input.cols, 5, CV_32F);
	// (input�� ��ü �ȼ� �� * ����ؾ��� ��� ��(r, g, b, x, y => 5��)) ũ���� sample matrix ����
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
		{
			for (int z = 0; z < 3; z++)
				samples2.at<float>(y + x * input.rows, z) = input.at<Vec3b>(y, x)[z] / 255.0; // position ���� ����ؾ��ϹǷ� intensity ���� normalize ����

			samples2.at<float>(y + x * input.rows, 3) = x / (float)input.cols; // intensity�� ����ϹǷ� x position ���� nomalization
			samples2.at<float>(y + x * input.rows, 4) = y / (float)input.rows; // intensity�� ����ϹǷ� y positon ���� normalization
		}

	Mat labels2;
	Mat centers2;

	// kmeans clustering ����
	kmeans(samples2, clusterCount, labels2, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers2);

	Mat new_image2(input.size(), input.type());
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
		{
			int cluster_idx = labels2.at<int>(y + x * input.rows, 0);
			// intensity ���� normalize ���־����Ƿ�, �ٽ� 255.0�� ���Ͽ� range�� ������Ŵ
			new_image2.at<Vec3b>(y, x)[0] = centers2.at<float>(cluster_idx, 0) * 255.0;
			new_image2.at<Vec3b>(y, x)[1] = centers2.at<float>(cluster_idx, 1) * 255.0;
			new_image2.at<Vec3b>(y, x)[2] = centers2.at<float>(cluster_idx, 2) * 255.0;
		}

	namedWindow("clustered image1", WINDOW_AUTOSIZE); // clustered image1 â ���
	imshow("clustered image1", new_image1); // â�� new_image1 �̹��� ������
	namedWindow("clustered image2", WINDOW_AUTOSIZE); // clustered image2 â ���
	imshow("clustered image2", new_image2); // â�� new_image2 �̹��� ������

	waitKey(0);

	return 0;
}

