Read me!
<kmeansGray.cpp>

-목적
: 이 코드는 opencv의 kmeans 함수를 이용한 clustering을 수행하여 흑백 이미지 내에서 비슷한 픽셀 값을 가지는 것들을 분류하는 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) filtering을 수행할 이미지 파일을 프로젝트로 불러온 후, cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명을 넣어줍니다.

-코드의 기본 설정
: 기본 코드는 "lena.jpg" 파일의 이미지를 k-means clustering 합니다. sample1은 intensity 값을 기준으로 clustering을 수행하고, sample2는 intensity,x,y intensity+position 값을 기준으로 clustering을 수행합니다.
둘 다 cluster의 개수는 10개, 초기 center의 선택하는 수는 5번으로 설정됩니다.
그리고 원본 이미지는 "Original", sample1을 clustering 한 이미지는 "clustered image1", sample2를 clustering 한 이미지는 "clustered image2" 창에 띄워지도록 설정되어 있습니다.

-실행 함수 및 변수 설명
kmeans(samples1, clusterCount, labels1, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers1);
: 이 함수는 이 코드에서 사용된 opencv에 내장된 kmeans clustering을 수행하는 함수입니다.
코드 내에서 사용중인 함수의 매개 변수들은 아래의 것들을 의미합니다.
- samples: clustering할 데이터
- clusterCount: cluster 개수
- labels: 각 데이터 포인트가 속한 cluster의 index를 저장
- criteria: 중단 기준 => 최대 10000번 반복하거나, center 변화량이 0.0001보다 작아지면 알고리즘 중단
- attempts: 초기 center를 선택하는 횟수
- KMEANS_PP_CENTERS: k-means++ 알고리즘을 사용하여 초기 center 선택
- centers: 각 cluster의 center 저장