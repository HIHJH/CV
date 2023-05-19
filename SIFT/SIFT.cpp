#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#define RATIO_THR 0.4 // threshold ratio = 0.4

using namespace std;
using namespace cv;

double euclidDistance(Mat& vec1, Mat& vec2);
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold);

int main() {

	Mat input1 = imread("input1.jpg", CV_LOAD_IMAGE_COLOR); // input1 �÷� �̹��� �ε�
	Mat input2 = imread("input2.jpg", CV_LOAD_IMAGE_COLOR); // input2 �÷� �̹��� �ε�
	Mat input1_gray, input2_gray;

	if (!input1.data || !input2.data)
	{
		std::cout << "Could not open" << std::endl; // �̹����� �ҷ����� ���ϸ� �����޼��� ���
		return -1;
	}

	//resize(input1, input1, Size(input1.cols / 2, input1.rows / 2));
	//resize(input2, input2, Size(input2.cols / 2, input2.rows / 2));

	//input�� RGB���� Grayscale�� ��ȯ
	cvtColor(input1, input1_gray, CV_RGB2GRAY);
	cvtColor(input2, input2_gray, CV_RGB2GRAY);

	//feature ����� ��ü ���� (detector)
	FeatureDetector* detector = new SiftFeatureDetector(
		0,		// nFeatures: ������ Ư¡���� �ִ� ����, 0�̸� ���Ѿ���
		4,		// nOctaveLayers: ocatave ���̾��� ��
		0.04,	// contrastThreshold: Ư¡�� ���� ���� �Ӱ谪
		10,		// edgeThreshold: ������ �ִ� Ư���� �ɷ����� ���� �Ӱ谪
		1.6		// sigma: Gaussian smoothing�� �̿�Ǵ� �ñ׸���
	);

	// descriptor ����� ��ü�� ���� (extractor)
	DescriptorExtractor* extractor = new SiftDescriptorExtractor();

	// Create a image for displaying mathing keypoints
	Size size = input2.size();
	Size sz = Size(size.width + input1_gray.size().width, max(size.height, input1_gray.size().height));
	// �ΰ��� �̹����� ���� �� �ֵ��� ũ�� ����
	Mat matchingImage = Mat::zeros(sz, CV_8UC3);

	// input1�� matchingImage�� �����ʿ� ����
	input1.copyTo(matchingImage(Rect(size.width, 0, input1_gray.size().width, input1_gray.size().height)));
	// input2�� matchingImage�� ���ʿ� ����
	input2.copyTo(matchingImage(Rect(0, 0, size.width, size.height)));

	// Compute keypoints and descriptor from the source image in advance
	vector<KeyPoint> keypoints1; //keypoints1 ���� ����
	Mat descriptors1; // descriptors1 matrix ����

	detector->detect(input1_gray, keypoints1); // �̹���1���� keypoints�� �����Ͽ� ����
	extractor->compute(input1_gray, keypoints1, descriptors1);
	// �̹���1, keypoints1�� ���� descriptor�� ����Ͽ� descriptors1�� ����
	printf("input1 : %d keypoints are found.\n", (int)keypoints1.size()); // ����� keypoint�� ���� ���

	//�̹���2�� ���ؼ��� ���� ���� �ݺ�
	vector<KeyPoint> keypoints2;
	Mat descriptors2;

	detector->detect(input2_gray, keypoints2);
	extractor->compute(input2_gray, keypoints2, descriptors2);

	printf("input2 : %zd keypoints are found.\n", keypoints2.size());
	
	//matchingImage�� keypoint1,2 ǥ��
	for (int i = 0; i < keypoints1.size(); i++) {
		KeyPoint kp = keypoints1[i]; //i��° keypoint1�� kp�� �Ҵ�
		kp.pt.x += size.width; //�̹���1, 2�� �̾���̱� ���� kp�� x��ǥ�� width�� ������
		circle(matchingImage, kp.pt, cvRound(kp.size * 0.25), Scalar(255, 255, 0), 1, 8, 0);
		// matchingImage�� keypoint1�� ������ ǥ��
	}
	for (int i = 0; i < keypoints2.size(); i++) {
		KeyPoint kp = keypoints2[i]; //i��° keypoint2�� kp�� �Ҵ�
		circle(matchingImage, kp.pt, cvRound(kp.size * 0.25), Scalar(255, 255, 0), 1, 8, 0);
		// matchingImage�� keypoint2�� ������ ǥ��
	}

	// nearest neighbor �� ã��
	vector<Point2f> srcPoints; // ã�� neighbor ���� ������ �Ǵ� Ű����Ʈ���� ��ǥ ����
	vector<Point2f> dstPoints; // srcPoints�� �����ϴ� neighbor ���� ��ǥ�� ����
	bool crossCheck = true; //crossCheck ����
	bool ratio_threshold = true; //ratio_threshold ����
	findPairs(keypoints2, descriptors2, keypoints1, descriptors1, srcPoints, dstPoints, crossCheck, ratio_threshold);
	// keypoints1�� keypoints2 ���̿��� ���� ����� �̿����� ã�� srcPoints, dstPoints ���Ϳ� ����
	// crossCheck ������ ratio threshold ������ ���������� ����
	printf("%zd keypoints are matched.\n", srcPoints.size()); // ã�� neighbor ���� ���� ���

	// nearest neighbor ���� ������ ����
	for (int i = 0; i < (int)srcPoints.size(); ++i) {
		Point2f pt1 = srcPoints[i];
		Point2f pt2 = dstPoints[i];
		Point2f from = pt1;
		Point2f to = Point(size.width + pt2.x, pt2.y);
		line(matchingImage, from, to, Scalar(0, 0, 255)); // matchingImage�� �̿��� ���ἱ ǥ��
	}

	namedWindow("Matching"); // Matching â ���
	imshow("Matching", matchingImage); // â�� matchingImage ������

	waitKey(0);

	return 0;
}

// vec1�� vec2 ������ ��Ŭ����Ÿ��� ����ϴ� �Լ�
double euclidDistance(Mat& vec1, Mat& vec2) {
	double sum = 0.0;
	int dim = vec1.cols; // ������ ������ dim�� ����
	for (int i = 0; i < dim; i++) {
		// �� ���� ������ ������ sum�� ����
		sum += (vec1.at<uchar>(0, i) - vec2.at<uchar>(0, i)) * (vec1.at<uchar>(0, i) - vec2.at<uchar>(0, i));
	}
	return sqrt(sum); // �������� ����ؼ� ��Ŭ���� �Ÿ� ��ȯ
}

// keypoint�κ��� ���� ����� �̿� ����Ʈ�� index�� ã�� �Լ�
// vec: ã�����ϴ� Ư¡ ����
// keypoints: Ű����Ʈ�� ��ġ ������ ���Ե� ����
// descriptors: �� Ű����Ʈ�� ���� descriptor�� ������ ����
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1; // neighbor index�� �����ϴ� neighbor, -1�� �ʱ�ȭ
	double minDist = 1e6; // �ּҰŸ��� �����ϴ� minDist, ū ��(1e6)���� �ʱ�ȭ

	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);		// each row of descriptor

		double dist = euclidDistance(vec, v); // vec�� v ������ ��Ŭ���� �Ÿ� ���

		if (dist < minDist) { // dist�� �ּҰŸ��̸� �� ������Ʈ
			minDist = dist;
			neighbor = i;
		}
	}

	return neighbor; // ���� ����� �̿��� �ε��� ��ȯ
}

// Ű�����Ϳ� ��ũ���Ϳ� ���� �ּ� �Ÿ��� ���� �� ���� ã�� �Լ�
// keypoints1: ù��° �̹����� Ű����Ʈ�� ��� �ִ� ����
// descriptors1: ù��° �̹����� ��ũ���͸� ��� �ִ� ���
// keypoints2: �ι�° �̹����� Ű����Ʈ�� ��� �ִ� ����
// descriptors2: �ι�° �̹����� ��ũ���͸� ��� �ִ� ���
// srcPoints: ã�� ��Ī �� �ֿ��� ù��° �̹����� ������ �����ϴ� ����
// dstPoints: ã�� ��Ī �� �ֿ��� �ι�° �̹����� ������ �����ϴ� ����
// crossCheck: crossCheck ����
// ratio_threshold: ratio thresholding ����
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold) {
	for (int i = 0; i < descriptors1.rows; i++) {
		KeyPoint pt1 = keypoints1[i]; // i��° Ű����Ʈ pt1
		Mat desc1 = descriptors1.row(i); // i��° �ش� ��ũ���� desc1

		int nn = nearestNeighbor(desc1, keypoints2, descriptors2);
		// desc1, keypoints2, descriptors2�� �̿��Ͽ� ���� ����� �̿��� �ε��� nn�� ã��
		
		// Refine matching points using ratio-based thresholding
		if (ratio_threshold) {
			Mat desc2 = descriptors2.row(nn);// nn �ε����� �ش��ϴ� �ι�° ��ũ���͸� ������
			int nn2 = nearestNeighbor(desc2, keypoints1, descriptors1);
			// desc2�� ù��° �̹����� ��� Ű����Ʈ, ��ũ���͸� ���ؼ� ���� ����� �̿��� �ε��� nn2�� ã��

			if (nn2 == i) { // nn2�� i�� ��ġ�ϸ�
				double dist1 = euclidDistance(desc1, desc2); // �� ��ũ���� ������ ��Ŭ���� �Ÿ� ���
				Mat desc3 = descriptors1.row(nn2); // nn2 �ε����� �ش��ϴ� ù��° �̹����� ��ũ����
				double dist2 = euclidDistance(desc1, desc3);

				if (dist1 < RATIO_THR * dist2) { // dist1/dist2 < Ratio_thr �� ���, ��Ī ����Ʈ�� ����
					srcPoints.push_back(pt1.pt); // pt1 ��ǥ�� srcPoints�� �߰�
					dstPoints.push_back(keypoints2[nn].pt); // nn �ε��� �ι�° �̹����� Ű����Ʈ ��ǥ�� dstPoints�� �߰�
				}
			}
		}

		// Refine matching points using cross-checking
		if (crossCheck) {
			Mat desc4 = descriptors2.row(nn); // nn�ε����� �ش��ϴ� �ι�° �̹����� ��ũ���� ������
			int nn2 = nearestNeighbor(desc4, keypoints1, descriptors1);
			// desc2�� ù��° �̹����� ��� Ű����Ʈ, ��ũ���͸� ���ؼ� ���� ����� �̿��� �ε��� nn2�� ã��

			if (nn2 == i) { // nn2�� ���� �ε��� i�� ��ġ�ϸ� (reliable)
				srcPoints.push_back(pt1.pt); // pt1 ��ǥ srcPoints�� �߰�
				dstPoints.push_back(keypoints2[nn].pt); // nn �ε��� �ι�° �̹����� Ű����Ʈ ��ǥ�� dstPoints�� �߰�
			}
		}
		
		if (ratio_threshold == 0 && crossCheck == 0) {
			// ��Ī�� ����Ʈ�� �����̹���(srcPoints)�� ����̹���(dstPoints)�� ��ǥ�� ���� ����
			KeyPoint pt2 = keypoints2[nn];
			srcPoints.push_back(pt1.pt); // pt1�� ��ġ�� srcPoints�� �߰�
			dstPoints.push_back(pt2.pt); // pt2�� ��ġ�� dstPoints�� �߰�
		}
	}
}