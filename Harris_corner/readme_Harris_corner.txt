Read me!
<Harris_corner.cpp>

-목적
: 이 코드는 이미지에서 Harris corner detector을 이용해 코너를 검출하는 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) harris corner detect를 수행할 이미지 파일을 프로젝트로 불러온 후, cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명을 넣어줍니다.
그리고 NonMaxSupp, Subpixel 에 boolean 값을 넣어 NMS와 subpixel 함수의 수행 여부를 입력합니다.
cornerHarris() 함수의 매개변수로 입력 이미지, 출력 이미지, 코너 감지를 위한 윈도우 크기, 커널 크기, 코너 감지에서 사용되는 Harris 감도 상수, 이미지 경계 처리 방법을 차례대로 넣어줍니다.
circle() 함수의 매개변수로 원을 그릴 이미지, 원의 중심점, 원의 반지름, 색상, 두께, 선 타입, shift operation을 차례대로 넣어줍니다.
NonMaximum_Suppression() 함수의 매개변수로 NMS 수행할 이미지, 코너 위치를 나타내는 행렬, NMS 수행할 윈도우의 반지름 값도 차례대로 넣어줍니다.
cornerSubPix() 함수의 매개변수로 input 이미지, 코너 위치 mat, 윈도우 사이즈, 중단 기준을 각각 넣어줍니다.

-코드의 기본 설정
: 기본 코드는 "checkerboard.png", "checkerboard2.jpg", "lena.jpg" 파일 이미지에 Harris corner detection을 수행합니다. cornerHarris 기본 설정은 코너 감지를 위한 윈도우의 크기 = 2, 커널 크기 = 3, 감도 상수 = 0.04 이고, NonMaxSupp, Suppixel을 모두 수행합니다. NMS의 radius = 2, subpixel의 윈도우 사이즈는 3*3 입니다. filtering 시에는 boundary process로 mirroring을 사용합니다. 그리고 cornerHarris 함수 수행 후 정규화한 이미지는 "Harris Response"에, 코너를 thresholding 하고 시각화한 이미지는 "Harris Corner"에, NMS 수행 후 이미지는 "Harris Corner (Non-max)"에, cornerSubPix 수행 이미지는 "Harris Corner (subpixel)" 창에 각각 띄워지도록 설정되어 있습니다.

-실행 함수 및 변수 설명
1. cornerHarris(src, dst, blockSize, ksize, k, borderType)
: 이 함수는 해리스 코너 검출을 수행하기 위한 함수입니다. 매개변수는 다음과 같습니다.
- src = 입력 이미지 
- dst = 출력 이미지 
- blockSize = 코너 감지를 위한 윈도우 크기
- ksize = 커널 크기
- k = 코너 감지에서 사용되는 Harris 감도 상수
- borderType = 이미지 경계 처리 방법

2. NonMaximum_Suppression(input, corner_mat, radius)
: 이 함수는 NMS를 수행하기 위한 함수입니다. 매개변수는 다음과 같습니다.
- input = NMS 수행할 이미지
- corner_mat = 코너 위치를 나타내는 행렬
- radius = NMS 수행할 윈도우의 반지름 값

3. cornerSubPix(image, corners, winSize, zeroZone, criteria)
: 이 함수는 corner의 위치를 정밀한 단위로 찾아주기 위한 함수입니다. 매개변수는 다음과 같습니다.
- image = 입력 이미지
- corners = 초기 코너 포인트 좌표 담은 배열
- winSize = 코너 포인트 윈도우 크기
- zeroZone = 반지름이 'zeroZone'인 원 안에서는 검색x (가장자리문제때문)
- criteria = 알고리즘 멈추는 기준