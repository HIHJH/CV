Read me!
<Bilateral.cpp>

-목적
: 이 코드는 흑백, 컬러 이미지에 gaussian noise를 주고, 이를 bilateral filtering을 통해 제거하는 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) add noise, remove noise를 수행할 이미지 파일을 프로젝트로 불러온 후, cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명을 넣어줍니다.
Add_Gaussian_noise() 함수의 매개변수로 noise를 추가할 이미지, 노이즈가 생성되는 확률 분포의 평균, 표준편차를 차례대로 넣어줍니다.
Bilateralfilter_Gray() 함수와 Bilateralfilter_RGB() 함수의 매개변수로 필터를 적용할 이미지, neighbor의 범위, t 관련 gaussian 분포의 표준편차, s 관련 gaussian 분포의 표준편차, intensity 관련 gaussian 분포의 표준편차, boundary process 방법을 차례대로 넣어줍니다.

-코드의 기본 설정
: 기본 코드는 "lena.jpg" 파일에 흑백, 컬러 이미지 각각 평균 = 0, 표준편차 = 0.1 인 정규 분포를 따르는 gaussian noise를 추가합니다. 그리고 noise image에 bilateral filtering을 수행합니다. filter는 n = 3, sigma_t = 10, sigma_s = 10, sigma_r=0.3 으로 기본 설정되고, 흑백이미지는 zero-padding으로, 컬러이미지는 adjustkerenl 방식으로 boundary process를 수행합니다. 그리고 원본 흑백 이미지는 "Grayscale", 컬러 이미지는 "RGB", noise를 추가한 흑백이미지는 "Gaussian Noise (Gray)", 컬러 이미지는 "Gaussian Noise (RGB)", filtering한 흑백 이미지는 "Denoised (Gray)", 컬러 이미지는 "Denoised (RGB)" 창에 띄워지도록 설정되어 있습니다.

-실행 함수 및 변수 설명
1. Add_Gaussian_noise(const Mat input, double mean, double sigma) : 이 함수는 이미지에 Gaussian 확률 분포를 따르는 noise를 추가합니다. 함수의 매개변수들은 아래의 것들을 의미합니다.
- input = noise를 추가할 이미지
- mean = noise가 따르는 정규 분포의 평균,
- sigma = noise가 따르는 정규 분포의 표준편차

2. Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt) : 이 함수는 흑백 noise 이미지에 noise를 제거하기 위해 Bilateral filtering을 수행하는 함수입니다. 함수의 매개변수들은 아래의 것들을 의미합니다.
- input = 필터를 적용할 이미지
- n = neighbor의 범위
- sigma_t =  t 관련 gaussian 분포의 표준편차(x-coordinate)
- sigma_s = s 관련 gaussian 분포의 표준편차(y-coordinate)
- sigma_r = intensity 관련 gaussian 분포의 표준편차(color)
- opt = boundary process 방법

3. Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt) : 이 함수는 컬러 noise 이미지에 noise를 제거하기 위해 Bilateral filtering을 수행하는 함수입니다. 함수의 매개변수들은 아래의 것들을 의미합니다.
- input = 필터를 적용할 이미지
- n = neighbor의 범위
- sigma_t =  t 관련 gaussian 분포의 표준편차(x-coordinate)
- sigma_s = s 관련 gaussian 분포의 표준편차(y-coordinate)
- sigma_r = intensity 관련 gaussian 분포의 표준편차(color)
- opt = boundary process 방법