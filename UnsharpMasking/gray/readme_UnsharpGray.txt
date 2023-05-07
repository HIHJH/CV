Read me!
<UnsharpGray.cpp>

-목적
: 이 코드는 흑백 이미지의 에지 부분(high frequency)을 강조하여 이미지가 더 날카로워지도록 하기 위해 Unsharp Masking을 수행하는 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) filtering을 수행할 이미지 파일을 프로젝트로 불러온 후, cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명을 넣어줍니다.
UnsharpMask() 함수의 매개변수로 필터를 적용할 이미지, neighbor의 범위, t 관련 gaussian 분포의 표준편차, s 관련 gaussian 분포의 표준편차, boundary process 방법, low frequency를 줄이는 양을 결정할 scaling 값을 차례대로 넣어줍니다.

-코드의 기본 설정
: 기본 코드는 "lena.jpg" 파일의 흑백 이미지를 unsharp masking 합니다. filter는 n = 1, sigmaT = 1, sigmaS = 1로 기본 설정되고 zero-paddle로 boundary process를 수행합니다. 그리고 low frequency component를 구한 후 0.5만큼 scaling을 수행하며 masking 합니다. 그리고 원본 흑백 이미지는 "Grayscale", filtering한 이미지는 "UnsharpMasking" 창에 띄워지도록 설정되어 있습니다.

-실행 함수 및 변수 설명
 gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) : 이 함수는 Gaussian filtering을 수행하는 함수입니다.
함수의 매개 변수들은 아래의 것들을 의미합니다.
- input = 필터를 적용할 이미지
- n = neighbor의 범위
- sigmaT =  t 관련 gaussian 분포의 표준편차
- sigmaS = s 관련 gaussian 분포의 표준편차
- opt = boundary process 방법

UnsharpMask(const Mat input, int n, float sigmaT, float sigmaS, const char* opt, float k) : 이 함수는 Unsharp masking을 수행하는 함수입니다.
함수의 매개 변수들은 아래의 것들을 의미합니다.
- input = 필터를 적용할 이미지
- n = neighbor의 범위
- sigmaT =  t 관련 gaussian 분포의 표준편차
- sigmaS = s 관련 gaussian 분포의 표준편차
- opt = boundary process 방법
- k = low frequency 를 줄이는 양을 결정하는 scaling에 이용되는 값 (0<=k<=1)