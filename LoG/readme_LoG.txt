Read me!
<LoG_Gray.cpp>, <LoG_RGB.cpp>

-목적
: 이 코드는 이미지의 noise를 제거한 후 edge를 검출하기위해 gaussian filtering을 적용한 후 laplacian filtering을 적용하는 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) LoG를 수행할 이미지 파일을 프로젝트로 불러온 후, cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명을 넣어줍니다.
gaussian filltering의 parameter인 window_radius, sigma_t, sigma_s 값을 넣어줍니다.

-코드의 기본 설정
: 기본 코드는 "lena.jpg" 파일 이미지에 Laplacian filtering of Gaussian filtering을 수행합니다. 기본 gaussian filtering의 매개변수는 window_radius = 2, sigma_t = 2.0, sigma_s = 2.0 입니다. boundary process는 mirroring을 사용합니다. 그리고 laplacian filter의 weight는 {0, 1, 0, 1, -4, 1, 0, 1, 0}으로 기본 설정되어 있습니다. 컬러 이미지의 경우 output이 grayscale 이므로 R,G,B 채널별 output 픽셀 값 중 최댓값을 output에 반영합니다.
원본 이미지는 "Grayscale" 또는 "Original" 창에, gaussian filtering 적용 이미지는 "Gaussian blue" 창에, laplacian filtering 적용 이미지는 "Laplacian filter" 창에 각각 띄워지도록 설정되어 있습니다.

-실행 함수 및 변수 설명
1. Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s)
: 이 함수는 gaussian filtering을 수행하는 함수입니다. 매개변수는 다음과 같습니다- input = 필터를 적용할 이미지
- n = neighbor의 범위
- sigma_t = t 관련 gaussian 분포의 표준편차(x-coordinate)
- sigma_s = s 관련 gaussian 분포의 표준편차(y-coordinate)

2. Laplacianfilter(const Mat input)
: 이 함수는 laplacianfilter를 수행하는 함수입니다.
함수의 매개 변수인 input은 필터를 적용할 이미지를 의미합니다.

3. get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize)
: 이 함수는 gaussian kernel의 weight 값을 지정하는 함수입니다. 매개변수는 다음과 같습니다.
- n = neighbor의 범위
- sigma_t =  t 관련 gaussian 분포의 표준편차(x-coordinate)
- sigma_s = s 관련 gaussian 분포의 표준편차(y-coordinate)
- normalize = 정규화 여부

4. Mirroring(const Mat input, int n)
: 이 함수는 boundary process를 mirroring 방법으로 처리하는 함수입니다. 매개변수는 다음과 같습니다.
- input = mirroring할 이미지
- n = neighbor의 범위
