Read me!
<adaptivethreshold.cpp>

-목적
: 이 코드는 이미지를 여러 영역으로 나눈 후, 주변 픽셀 값을 기준으로 계산하여 영역마다의 threshold를 지정하는 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) filtering을 수행할 이미지 파일을 프로젝트로 불러온 후, cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명을 넣어줍니다.
adaptive_thres() 함수의 매개변수로 threshold를 수행할 이미지, neighbor의 범위, weight 값을 넣어줍니다.

-코드의 기본 설정
: 기본 코드는 "writing.jpg" 파일의 이미지를 filter neighbor의 범위 = 2, weight = 0.9로 갖는 adaptive thresholding을 수행합니다. 값을 결정하는 과정에서 zero paddle boundary process를 거치는 uniform mean filtering을 수행합니다. 그리고 원본 이미지는 "Grayscale", thresholding한 이미지는 "Adaptive_threshold" 창에 띄워지도록 설정되어 있습니다.

-실행 함수 및 변수 설명
adaptive_thres(const Mat input, int n, float bnumber)
: 이 함수는 uniform mean filtering(with zero paddle boundary process)을 이용하여 thresholding을 수행하는 함수입니다.
함수의 매개 변수들은 아래의 것들을 의미합니다.
- input = thresholding 할 이미지
- n = neighbor의 범위
- bnumber = weight