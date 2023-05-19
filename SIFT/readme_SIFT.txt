Read me!
<SIFT.cpp>

-목적
: 이 코드는 두 개의 이미지에서 SIFT를 사용하여 키포인트를 검출하고, 디스크립터를 추출한 다음, 두 이미지간의 매칭된 키포인트를 찾는 과정을 수행하는 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) SIFT를 수행할 이미지 파일을 프로젝트로 불러온 후, cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명을 넣어줍니다.
RATIO_THR에 threshold ratio 값을 넣어줍니다.
crossCheck의 수행 여부와 ratio_threshold의 수행 여부를 bool 형태로 넣어줍니다.

-코드의 기본 설정
: 기본 코드는 "input1.jpg", "input2.jpg" 파일 이미지를 흑백으로 변환하고, 두 개의 이미지에 대해 SIFT를 사용하여 키포인트를 검출하고 디스크립터를 추출하여 두 이미지간의 매칭된 키포인트를 찾습니다.
RATIO_THR은 0.4로 설정되어있고, crossCheck, ratio_threshold 모두 true로 설정되어 있습니다.
input1과 input2의 키포인트 개수와 매칭된 키포인트 개수가 출력되며, 매칭 이미지에는 키포인트, 매칭된 키포인트가 시각화되어 "Matching" 창에 띄워지도록 설정되어 있습니다.

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
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold)
: 이 함수는 키포인터와 디스크립터에 대해 최소 거리를 갖는 점 쌍을 찾는 함수입니다. 매개변수는 다음과 같습니다.
- keypoints1: 첫번째 이미지의 키포인트를 담고 있는 벡터
- descriptors1: 첫번째 이미지의 디스크립터를 담고 있는 행렬
- keypoints2: 두번째 이미지의 키포인트를 담고 있는 벡터
- descriptors2: 두번째 이미지의 디스크립터를 담고 있는 행렬
- srcPoints: 찾은 매칭 점 쌍에서 첫번째 이미지의 점들을 저장하는 벡터
- dstPoints: 찾은 매칭 점 쌍에서 두번째 이미지의 점들을 저장하는 벡터
- crossCheck: crossCheck 여부
- ratio_threshold: ratio thresholding 여부