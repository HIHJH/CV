Read me!
<salt_and_pepper.cpp>

-목적
: 이 코드는 흑백, 컬러 이미지에 salt and pepper noise를 주고, 이를 median filtering을 통해 제거하는 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) add noise, remove noise를 수행할 이미지 파일을 프로젝트로 불러온 후, cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명을 넣어줍니다.
Add_salt_pepper_Noise() 함수의 매개변수로 noise를 추가할 이미지, pepper의 density, salt의 density를 차례대로 넣어줍니다.
Salt_pepper_noise_removal_Gray() 함수와 Salt_pepper_noise_removal_RGB() 함수의 매개변수로 noise를 제거할 이미지, neighbor의 범위, boundary process 방법을 차례대로 넣어줍니다.

-코드의 기본 설정
: 기본 코드는 "lena.jpg" 파일 흑백, 컬러 이미지에 salt, pepper 각각 density가 0.1f인 salt and pepper noise를 추가합니다. 그리고 noise image에 n=2인 median filtering을 수행합니다. 흑백이미지는 zero-padding으로, 컬러이미지는 adjustkerenl 방식으로 boundary process를 수행합니다. 그리고 원본 흑백 이미지는 "Grayscale", 컬러 이미지는 "RGB", noise를 추가한 흑백이미지는 "Impulse Noise (Gray)", 컬러 이미지는 "Impulse Noise (RGB)", filtering한 흑백 이미지는 "Denoised (Gray)", 컬러 이미지는 "Denoised (RGB)" 창에 띄워지도록 설정되어 있습니다.

-실행 함수 및 변수 설명
1. Add_salt_pepper_Noise(const Mat input, float ps, float pp)
: 이 함수는 이미지에 salt and pepper noise를 추가합니다. 함수의 매개변수들은 아래의 것들을 의미합니다.
- input = noise를 추가할 이미지
- ps = salt의 density
- pp = pepper의 density

2. Salt_pepper_noise_removal_Gray(const Mat input, int n, const char *opt)
: 이 함수는 salt and pepper noise 흑백 이미지에 noise를 제거하기 위해 median filtering을 수행하는 함수입니다. 함수의 매개변수들은 아래의 것들을 의미합니다.
- input = noise를 제거할 이미지
- n = neighbor의 범위
- opt = boundary process 방법

3. Salt_pepper_noise_removal_RGB(const Mat input, int n, const char *opt)
: 이 함수는 salt and pepper noise 컬러 이미지에 noise를 제거하기 위해 median filtering을 수행하는 함수입니다. 함수의 매개변수들은 아래의 것들을 의미합니다.
- input = noise를 제거할 이미지
- n = neighbor의 범위
- opt = boundary process 방법