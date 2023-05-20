Read me!
<matching_stitching.cpp, stitching with RANSAC.cpp>

-목적
: <matching_stitching.cpp>은 SIFT descripter를 사용하여 두 이미지의 feature matching을 수행하고, matching 된 corresponding points로 affine transform을 수행하여 두 이미지를 연결하는 코드입니다. <stitching with RANSAC.cpp>은 affine transform을 수행하는 과정에서 RANSAC 알고리즘이 적용된 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) stitching을 수행할 이미지 파일을 프로젝트로 불러온 후, cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명을 넣어줍니다.
RATIO_THR에 matching threshold ratio 값을 넣어줍니다. RANSAC이 포함된 경우, sample의 수 k, distance threshold인 inlierThreshold, 적어도 한 subset이 outlier에 free할 확률인 P, outlier ratio인 e를 설정해줍니다.

-코드의 기본 설정
: 기본 코드는 "input1.jpg", "input2.jpg" 파일 이미지를 흑백으로 변환하고, 두 개의 이미지에 대해 SIFT를 사용하여 키포인트를 검출하고 디스크립터를 추출하여 두 이미지간의 매칭된 키포인트를 찾습니다. RATIO_THR은 0.4로 설정되어있고, crossCheck, ratio_threshold을 수행합니다.
input1과 input2의 키포인트 개수와 매칭된 키포인트 개수가 출력되며, 매칭 이미지에는 키포인트, 매칭된 키포인트가 시각화되어 "Matching" 창에 띄워지도록 설정되어 있습니다.
이후 affine transform을 수행하는데, RANSAC이 포함된 경우 [sample의 수는 3개, distance threshold = 0.5, 적어도 한 subset이 outlier에 free할 확률 = 0.99, outlier ratio=0.5]로 설정하여 최대 반복 횟수를 구하고 있습니다.
affine transform도 수행된 후, stitching 된 결과가 "stitching" 창에 띄워지도록 설정되어 있습니다.

-실행 함수 및 변수 설명
1. euclidDistance(Mat& vec1, Mat& vec2)
: 이 함수는 vec1과 vec2 사이의 유클리드거리를 계산하는 함수입니다.

2. nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors)
: 이 함수는 keypoint로부터 가장 가까운 이웃 포인트의 index를 찾는 함수입니다.
매개변수는 다음과 같습니다.
- vec: 찾고자하는 특징 벡터
- keypoints: 키포인트의 위치 정보가 포함된 벡터
- descriptors: 각 키포인트에 대한 descriptor로 구성된 벡터

3. findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints)
: 이 함수는 키포인터와 디스크립터에 대해 최소 거리를 갖는 점 쌍을 찾는 함수입니다. 매개변수는 다음과 같습니다.
- keypoints1: 첫번째 이미지의 키포인트를 담고 있는 벡터
- descriptors1: 첫번째 이미지의 디스크립터를 담고 있는 행렬
- keypoints2: 두번째 이미지의 키포인트를 담고 있는 벡터
- descriptors2: 두번째 이미지의 디스크립터를 담고 있는 행렬
- srcPoints: 찾은 매칭 점 쌍에서 첫번째 이미지의 점들을 저장하는 벡터
- dstPoints: 찾은 매칭 점 쌍에서 두번째 이미지의 점들을 저장하는 벡터

4. template <typename T>
Mat cal_affine(const vector<Point2f>& srcPoints, const vector<Point2f>& dstPoints, int number_of_points)
: 이 함수는 affine matrix를 계산하는 함수입니다.
- srcPoints: 기준이 되는 원본 corresponding points (x, y)
- dstPoints: 변환되는 corresponding points (x', y')
- number of points: corresponding points의 개수

5. template <typename T>
Mat cal_affine_RANSAC(const vector<Point2f>& srcPoints, const vector<Point2f>& dstPoints, int number_of_points)
: 이 함수는 affine matrix를 RANSAC 알고리즘을 이용하여 계산하는 함수입니다.
- srcPoints: 기준이 되는 원본 corresponding points (x, y)
- dstPoints: 변환되는 corresponding points (x', y')
- number of points: corresponding points의 개수

6. void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int bound_l, int bound_u, float alpha)
: 이 함수는 이미지를 연결한 경계 부분을 blending하는 함수입니다.
- I1: 왼쪽 이미지
- I2: 오른쪽 이미지
- I_f: 합친 이미지
- bound_l: 왼쪽 경계
- bound_r: 오른쪽 경계
- alpha: 이미지 blending에 사용될 weight