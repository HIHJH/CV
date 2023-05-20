#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <unordered_set>

#define RATIO_THR 0.4 // threshold ratio = 0.4

using namespace std;
using namespace cv;


double euclidDistance(Mat& vec1, Mat& vec2);
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints);
template <typename T>
Mat cal_affine_RANSAC(const vector<Point2f>& srcPoints, const vector<Point2f>& dstPoints, int number_of_points);
void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int diff_x, int diff_y, float alpha);

int main() {

	Mat input1 = imread("input1.jpg", CV_LOAD_IMAGE_COLOR); // input1 컬러 이미지 로드
	Mat input2 = imread("input2.jpg", CV_LOAD_IMAGE_COLOR); // input2 컬러 이미지 로드
	Mat input1_gray, input2_gray;

	if (!input1.data || !input2.data)
	{
		std::cout << "Could not open" << std::endl; // 이미지를 불러오지 못하면 에러메세지 출력
		return -1;
	}

	//input을 RGB에서 Grayscale로 변환
	cvtColor(input1, input1_gray, CV_RGB2GRAY);
	cvtColor(input2, input2_gray, CV_RGB2GRAY);

// 1) Feature matching

	//feature 검출기 객체 생성 (detector)
	FeatureDetector* detector = new SiftFeatureDetector(
		0,		// nFeatures: 검출할 특징점의 최대 개수, 0이면 제한없음
		4,		// nOctaveLayers: ocatave 레이어의 수
		0.04,	// contrastThreshold: 특징점 검출 위한 임계값
		10,		// edgeThreshold: 엣지에 있는 특정점 걸러내기 위한 임계값
		1.6		// sigma: Gaussian smoothing에 이용되는 시그마값
	);

	// descriptor 추출기 객체를 생성 (extractor)
	DescriptorExtractor* extractor = new SiftDescriptorExtractor();

	// Create a image for displaying mathing keypoints
	Size size = input2.size();
	Size sz = Size(size.width + input1_gray.size().width, max(size.height, input1_gray.size().height));
	// 두개의 이미지를 붙일 수 있도록 크기 조절
	Mat matchingImage = Mat::zeros(sz, CV_8UC3);

	// input1을 matchingImage의 오른쪽에 복사
	input1.copyTo(matchingImage(Rect(size.width, 0, input1_gray.size().width, input1_gray.size().height)));
	// input2를 matchingImage의 왼쪽에 복사
	input2.copyTo(matchingImage(Rect(0, 0, size.width, size.height)));

	// Compute keypoints and descriptor from the source image in advance
	vector<KeyPoint> keypoints1; //keypoints1 벡터 생성
	Mat descriptors1; // descriptors1 matrix 생성

	detector->detect(input1_gray, keypoints1); // 이미지1에서 keypoints를 검출하여 저장
	extractor->compute(input1_gray, keypoints1, descriptors1);
	// 이미지1, keypoints1에 대해 descriptor를 계산하여 descriptors1에 저장
	printf("input1 : %d keypoints are found.\n", (int)keypoints1.size()); // 검출된 keypoint의 개수 출력

	//이미지2에 대해서도 같은 과정 반복
	vector<KeyPoint> keypoints2;
	Mat descriptors2;

	detector->detect(input2_gray, keypoints2);
	extractor->compute(input2_gray, keypoints2, descriptors2);

	printf("input2 : %zd keypoints are found.\n", keypoints2.size());

	//matchingImage에 keypoint1,2 표시
	for (int i = 0; i < keypoints1.size(); i++) {
		KeyPoint kp = keypoints1[i]; //i번째 keypoint1을 kp에 할당
		kp.pt.x += size.width; //이미지1, 2를 이어붙이기 위해 kp의 x좌표에 width를 더해줌
		circle(matchingImage, kp.pt, cvRound(kp.size * 0.25), Scalar(255, 255, 0), 1, 8, 0);
		// matchingImage에 keypoint1을 원으로 표시
	}
	for (int i = 0; i < keypoints2.size(); i++) {
		KeyPoint kp = keypoints2[i]; //i번째 keypoint2를 kp에 할당
		circle(matchingImage, kp.pt, cvRound(kp.size * 0.25), Scalar(255, 255, 0), 1, 8, 0);
		// matchingImage에 keypoint2를 원으로 표시
	}

	// nearest neighbor 쌍 찾기
	vector<Point2f> srcPoints; // 찾은 neighbor 쌍의 기준이 되는 키포인트들의 좌표 저장
	vector<Point2f> dstPoints; // srcPoints에 대응하는 neighbor 쌍의 좌표가 저장
	findPairs(keypoints2, descriptors2, keypoints1, descriptors1, srcPoints, dstPoints);
	// keypoints1과 keypoints2 사이에서 가장 가까운 이웃쌍을 찾아 srcPoints, dstPoints 벡터에 저장
	// crossCheck 과정과 ratio threshold 과정은 선택적으로 진행
	printf("%zd keypoints are matched.\n", srcPoints.size()); // 찾은 neighbor 쌍의 개수 출력

	// nearest neighbor 쌍을 선으로 연결
	for (int i = 0; i < (int)srcPoints.size(); ++i) {
		Point2f pt1 = srcPoints[i];
		Point2f pt2 = dstPoints[i];
		Point2f from = pt1;
		Point2f to = Point2f(size.width + pt2.x, pt2.y);
		line(matchingImage, from, to, Scalar(0, 0, 255)); // matchingImage에 이웃쌍 연결선 표시
	}

	namedWindow("Matching"); // Matching 창 띄움
	imshow("Matching", matchingImage); // 창에 matchingImage 보여줌

// 2. Affine transform estimation & 3) Perform image stitching

	// 이미지 픽셀값 normalization
	input1.convertTo(input1, CV_32FC3, 1.0 / 255);
	input2.convertTo(input2, CV_32FC3, 1.0 / 255);

	// height(row), width(col) of each image
	const float I1_row = input1.rows;
	const float I1_col = input1.cols;
	const float I2_row = input2.rows;
	const float I2_col = input2.cols;

	// srcPoints의 사이즈로 corresponding points 쌍의 개수 받아옴
	int number_of_points = srcPoints.size();

	// cal_affine_RANSAC을 이용해 A12, A21 matrix 계산
	// A12: srcPoints -> dstPoints
	// A21: dstPoints -> srcPoints
	Mat A12 = cal_affine_RANSAC<float>(srcPoints, dstPoints, number_of_points);
	Mat A21 = cal_affine_RANSAC<float>(dstPoints, srcPoints, number_of_points);

	// compute corners (p1, p2, p3, p4)
	// p1: (0,0)
	// p2: (row, 0)
	// p3: (row, col)
	// p4: (0, col)
	Point2f p1(A21.at<float>(0) * 0 + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * 0 + A21.at<float>(5));
	Point2f p2(A21.at<float>(0) * 0 + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * I2_row + A21.at<float>(5));
	Point2f p3(A21.at<float>(0) * I2_col + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * I2_row + A21.at<float>(5));
	Point2f p4(A21.at<float>(0) * I2_col + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * 0 + A21.at<float>(5));

	// compute boundary for merged image(I_f)
	// bound_u <= 0
	// bound_b >= I1_row-1
	// bound_l <= 0
	// bound_b >= I1_col-1
	int bound_u = (int)round(min(0.0f, min(p1.y, p4.y)));
	int bound_b = (int)round(max(I1_row - 1, max(p2.y, p3.y)));
	int bound_l = (int)round(min(0.0f, min(p1.x, p2.x)));
	int bound_r = (int)round(max(I1_col - 1, max(p3.x, p4.x)));

	// initialize merged image
	Mat I_f(bound_b - bound_u + 1, bound_r - bound_l + 1, CV_32FC3, Scalar(0));

	// inverse warping with bilinear interplolation
	for (int i = bound_u; i <= bound_b; i++) {
		for (int j = bound_l; j <= bound_r; j++) {
			float x = A12.at<float>(0) * j + A12.at<float>(1) * i + A12.at<float>(2) - bound_l;
			float y = A12.at<float>(3) * j + A12.at<float>(4) * i + A12.at<float>(5) - bound_u;

			float y1 = floor(y);
			float y2 = ceil(y);
			float x1 = floor(x);
			float x2 = ceil(x);

			// 픽셀과의 거리도 고려
			float mu = y - y1;
			float lambda = x - x1;

			if (x1 >= 0 && x2 < I2_col && y1 >= 0 && y2 < I2_row)
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = lambda * (mu * input2.at<Vec3f>(y2, x2) + (1 - mu) * input2.at<Vec3f>(y1, x2)) +
				(1 - lambda) * (mu * input2.at<Vec3f>(y2, x1) + (1 - mu) * input2.at<Vec3f>(y1, x1));
		}
	}

	// image stitching with blend
	// blend_stitching 함수를 이용하여 두 이미지를 연결한 경계부분 blending
	blend_stitching(input1, input2, I_f, bound_l, bound_u, 0.5);

	namedWindow("stitching"); // stitching 창 띄움
	imshow("stitching", I_f);  // 창에 l_f 보여줌

	waitKey(0);

	return 0;
}

// vec1과 vec2 사이의 유클리드거리를 계산하는 함수
double euclidDistance(Mat& vec1, Mat& vec2) {
	double sum = 0.0;
	int dim = vec1.cols; // 벡터의 차원을 dim에 저장
	for (int i = 0; i < dim; i++) {
		// 두 원소 차이의 제곱이 sum에 누적
		sum += (vec1.at<uchar>(0, i) - vec2.at<uchar>(0, i)) * (vec1.at<uchar>(0, i) - vec2.at<uchar>(0, i));
	}
	return sqrt(sum); // 제곱근을 계산해서 유클리드 거리 반환
}

// keypoint로부터 가장 가까운 이웃 포인트의 index를 찾는 함수
// vec: 찾고자하는 특징 벡터
// keypoints: 키포인트의 위치 정보가 포함된 벡터
// descriptors: 각 키포인트에 대한 descriptor로 구성된 벡터
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1; // neighbor index를 저장하는 neighbor, -1로 초기화
	double minDist = 1e6; // 최소거리를 저장하는 minDist, 큰 값(1e6)으로 초기화

	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);		// each row of descriptor

		double dist = euclidDistance(vec, v); // vec와 v 사이의 유클리드 거리 계산

		if (dist < minDist) { // dist가 최소거리이면 값 업데이트
			minDist = dist;
			neighbor = i;
		}
	}

	return neighbor; // 가장 가까운 이웃의 인덱스 반환
}

// 키포인터와 디스크립터에 대해 최소 거리를 갖는 점 쌍을 찾는 함수
// keypoints1: 첫번째 이미지의 키포인트를 담고 있는 벡터
// descriptors1: 첫번째 이미지의 디스크립터를 담고 있는 행렬
// keypoints2: 두번째 이미지의 키포인트를 담고 있는 벡터
// descriptors2: 두번째 이미지의 디스크립터를 담고 있는 행렬
// srcPoints: 찾은 매칭 점 쌍에서 첫번째 이미지의 점들을 저장하는 벡터
// dstPoints: 찾은 매칭 점 쌍에서 두번째 이미지의 점들을 저장하는 벡터
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints) {
	for (int i = 0; i < descriptors1.rows; i++) {
		KeyPoint pt1 = keypoints1[i]; // i번째 키포인트 pt1
		Mat desc1 = descriptors1.row(i); // i번째 해당 디스크립터 desc1

		int nn = nearestNeighbor(desc1, keypoints2, descriptors2);
		// desc1, keypoints2, descriptors2를 이용하여 가장 가까운 이웃의 인덱스 nn을 찾음

		// Refine matching points using ratio-based thresholding
			Mat desc2 = descriptors2.row(nn);// nn 인덱스에 해당하는 두번째 디스크립터를 가져옴
			int nn2 = nearestNeighbor(desc2, keypoints1, descriptors1);
			// desc2와 첫번째 이미지의 모든 키포인트, 디스크립터를 비교해서 가장 가까운 이웃의 인덱스 nn2를 찾음

			if (nn2 == i) { // nn2가 i와 일치하면
				double dist1 = euclidDistance(desc1, desc2); // 두 디스크립터 사이의 유클리드 거리 계산
				Mat desc3 = descriptors1.row(nn2); // nn2 인덱스에 해당하는 첫번째 이미지의 디스크립터
				double dist2 = euclidDistance(desc1, desc3);

				if (dist1 < RATIO_THR * dist2) { // dist1/dist2 < Ratio_thr 인 경우, 매칭 포인트로 선별
					srcPoints.push_back(pt1.pt); // pt1 좌표를 srcPoints에 추가
					dstPoints.push_back(keypoints2[nn].pt); // nn 인덱스 두번째 이미지의 키포인트 좌표를 dstPoints에 추가
				}
			}

		// Refine matching points using cross-checking
			Mat desc4 = descriptors2.row(nn); // nn인덱스에 해당하는 두번째 이미지의 디스크립터 가져옴
			int nn3 = nearestNeighbor(desc4, keypoints1, descriptors1);
			// desc2와 첫번째 이미지의 모든 키포인트, 디스크립터를 비교해서 가장 가까운 이웃의 인덱스 nn3를 찾음

			if (nn3 == i) { // nn3가 현재 인덱스 i와 일치하면 (reliable)
				srcPoints.push_back(pt1.pt); // pt1 좌표 srcPoints에 추가
				dstPoints.push_back(keypoints2[nn].pt); // nn 인덱스 두번째 이미지의 키포인트 좌표를 dstPoints에 추가
			}

	}
}

// RANSAC 알고리즘을 이용해 affine matrix를 계산하는 함수
// srcPoints: 기준이 되는 원본 corresponding points (x, y)
// dstPoints: 변환되는 corresponding points (x', y')
// number of points: corresponding points의 개수
template <typename T>
Mat cal_affine_RANSAC(const vector<Point2f>& srcPoints, const vector<Point2f>& dstPoints, int number_of_points) {

	// 입력 데이터가 충분한지 확인
	if (srcPoints.size() < 3 || dstPoints.size() < 3) {
		std::cout << "Not enough points" << std::endl;
		return Mat();
	}

	const double inlierThreshold = 0.5; // inlier를 구할 때 사용되는 distance threshold = 0.5
	int k = 3; // sample의 개수 = 3
	int max_iterations; // RANSAC 과정 반복 횟수 S
	
	double P = 0.99; // 적어도 한 subset이 outlier에 free할 확률
	double e = 0.5; // outlier ratio

	double numerator = log(1 - P);
	double denominator = log(1 - pow(1 - e, k));
	// 1 - P = ( 1 - ( 1 - e ) ^ k ) ^ S (= 모든 subset에 outlier가 존재할 확률)
	// S = log( 1 - P ) / log ( 1 - ( 1 - e ) ^ k )
	max_iterations = (int)(numerator / denominator);

	int bestInlierCount = 0;
	Mat bestModel;

	for (int iteration = 0; iteration < max_iterations; iteration++) {
		// 1. randomly sample k data (k=3)
		
		// 겹치지 않는 난수 생성
		random_device rd; // 시드 생성
		mt19937 gen(rd()); // 난수 생성과 관련된 메르센 트위스터 엔진을 초기화
		uniform_int_distribution<int> dist(0, number_of_points - 1); // 0부터 number_of_points - 1 까지의 난수 생성
		unordered_set<int> selectedIndices; // unordered_set인 selectedIndices에 선택된 인덱스를 저장하여 중복 허용하지 않음

		while (selectedIndices.size() < k) {
			int randomIdx = dist(gen); // 난수 k개 생성
			selectedIndices.insert(randomIdx); // 생성된 난수는 selectedIndices에 저장
		}

		vector<Point2f> sampledSrcPoints, sampledDstPoints;
		// 랜덤 인덱스에 따라 샘플이 될 포인트 추출
		for (int idx : selectedIndices) {
			sampledSrcPoints.push_back(srcPoints[idx]);
			sampledDstPoints.push_back(dstPoints[idx]);
		}

		// 2. estimate the affine transform

		// Mx = b 형태
		Mat M(2 * k, 6, CV_32F, Scalar(0));
		Mat b(2 * k, 1, CV_32F, Scalar(0));

		Mat M_trans, temp, affineM;

		// Mx = b 형태에서 matrix M과 b의 요소들에 0인 부분을 제외하고
		// sample corresponding points의 값 넣음
		for (int i = 0; i < k; i++) {
			M.at<T>(2 * i, 0) = sampledSrcPoints[i].x;		M.at<T>(2 * i, 1) = sampledSrcPoints[i].y;		M.at<T>(2 * i, 2) = 1;
			M.at<T>(2 * i + 1, 3) = sampledSrcPoints[i].x;		M.at<T>(2 * i + 1, 4) = sampledSrcPoints[i].y;		M.at<T>(2 * i + 1, 5) = 1;
			b.at<T>(2 * i) = sampledDstPoints[i].x;		b.at<T>(2 * i + 1) = sampledDstPoints[i].y;
		}

		// (M^T * M)^(−1) * M^T * b ( * : Matrix multiplication)
		transpose(M, M_trans); // M_trans = M^T
		invert(M_trans * M, temp); // temp = (M^T * M)^(-1)
		affineM = temp * M_trans * b; // affineM = (M^T * M)^(−1) * M^T * b

		// 3. score by computing the number of inliers
		 // affineM을 이용하여 모든 데이터 포인트의 예측 위치 계산
		vector<Point2f> transformedPoints;
		for (int i = 0; i < number_of_points; i++) {
			Point2f transformedPoint = Point2f(
				affineM.at<T>(0) * srcPoints[i].x + affineM.at<T>(1) * srcPoints[i].y + affineM.at<T>(2),
				affineM.at<T>(3) * srcPoints[i].x + affineM.at<T>(4) * srcPoints[i].y + affineM.at<T>(5)
			);
			// 계산한 위치는 transformedPoints에 저장
			transformedPoints.push_back(transformedPoint);
		}

		//인라이어 개수 계산
		int inlierCount = 0;
		for (int i = 0; i < number_of_points; i++) {
			double distance = norm(transformedPoints[i] - dstPoints[i]); // 거리 계산
			if (distance < inlierThreshold) { // inlierThreshold보다 작으면 inlier
				inlierCount++;
			}
		}

		// 현재 모델의 인라이어 개수가 최고 인라이어 개수보다 크다면 업데이트
		if (inlierCount > bestInlierCount) {
			bestInlierCount = inlierCount;
			// 4. select the best affine transformation
			bestModel = affineM;
		}
	}

	// 5. inlier들로 Mx=b 계산하여 추가적인 모델 추정
	if (bestInlierCount >= 3) {

		// inlier 구하기
		vector<Point2f> inlierSrcPoints, inlierDstPoints;
		for (int i = 0; i < number_of_points; i++) {
			Point2f transformedPoint_re = Point2f(
				bestModel.at<T>(0) * srcPoints[i].x + bestModel.at<T>(1) * srcPoints[i].y + bestModel.at<T>(2),
				bestModel.at<T>(3) * srcPoints[i].x + bestModel.at<T>(4) * srcPoints[i].y + bestModel.at<T>(5)
			);
			double distance = cv::norm(transformedPoint_re - dstPoints[i]);
			if (distance < inlierThreshold) {
				inlierSrcPoints.push_back(srcPoints[i]);
				inlierDstPoints.push_back(dstPoints[i]);
			}
		}

		// 모델 재추정
		int inliersize = inlierSrcPoints.size();

		if (inlierSrcPoints.size() >= 3) {

			Mat M_re(2 * inliersize, 6, CV_32F, Scalar(0));
			Mat b_re(2 * inliersize, 1, CV_32F, Scalar(0));

			Mat M_trans_re, temp_re;

			// initialize matrix
			for (int i = 0; i < inliersize; i++) {
				M_re.at<T>(2 * i, 0) = inlierSrcPoints[i].x;		M_re.at<T>(2 * i, 1) = inlierSrcPoints[i].y;		M_re.at<T>(2 * i, 2) = 1;
				M_re.at<T>(2 * i + 1, 3) = inlierSrcPoints[i].x;		M_re.at<T>(2 * i + 1, 4) = inlierSrcPoints[i].y;		M_re.at<T>(2 * i + 1, 5) = 1;
				b_re.at<T>(2 * i) = inlierDstPoints[i].x;		b_re.at<T>(2 * i + 1) = inlierDstPoints[i].y;
			}

			// (M^T * M)^(−1) * M^T * b ( * : Matrix multiplication)
			transpose(M_re, M_trans_re);
			invert(M_trans_re * M_re, temp_re);
			bestModel = temp_re * M_trans_re * b_re;
		}
	}
	return bestModel; // best affine transformation 반환
}

// 이미지를 연결한 경계 부분을 blending
// I1: 왼쪽 이미지
// I2: 오른쪽 이미지
// I_f: 합친 이미지
// bound_l: 왼쪽 경계
// bound_r: 오른쪽 경계
// alpha: 이미지 blending에 사용될 weight
void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int bound_l, int bound_u, float alpha) {

	int col = I_f.cols;
	int row = I_f.rows;

	// I2 is already in I_f by inverse warping
	for (int i = 0; i < I1.rows; i++) {
		for (int j = 0; j < I1.cols; j++) {
			// 해당 픽셀이 이미 I2 이미지에서 blending 되어 I_f에 들어왔는지 확인
			// = 0이 아닌 값을 가지고 있는지 확인
			bool cond_I2 = I_f.at<Vec3f>(i - bound_u, j - bound_l) != Vec3f(0, 0, 0) ? true : false;

			if (cond_I2) // 이미 블렌딩
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = alpha * I1.at<Vec3f>(i, j) + (1 - alpha) * I_f.at<Vec3f>(i - bound_u, j - bound_l);
			// weight에 따라 블렌딩하며 픽셀값 지정
			else
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = I1.at<Vec3f>(i, j); // I1 픽셀값 바로 가져옴

		}
	}
}