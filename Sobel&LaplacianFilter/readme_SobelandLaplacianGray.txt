Read me!
<SobelandLaplacianGray.cpp>

-목적
: 이 코드는 흑백 이미지의 에지(high frequency component)를 보여주기 위해 sobel filtering과 laplacian filtering을 수행하는 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) filtering을 수행할 이미지 파일을 프로젝트로 불러온 후, cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명을 넣어줍니다.
sobelfilter() 함수의 매개변수로 필터를 적용할 이미지를 넣어줍니다.
laplacianfilter() 함수의 매개변수로 필터를 적용할 이미지를 넣어줍니다.

-코드의 기본 설정
: 기본 코드는 "lena.jpg" 파일의 흑백 이미지를 sobel filtering, laplacian filtering 합니다. sobel filter는 Sx={ -1, 0, 1, -2, 0, 2, -1, 0, 1 }, Sy={ -1, -2, -1, 0, 0, 0, 1, 2, 1 }으로, laplacian filter는 {0, 1, 0, 1, -4, 1, 0, 1, 0}으로 기본 설정되어있습니다. 그리고 원본 흑백 이미지는 "Grayscale", sobel filtering한 이미지는 "Sobel Filter", laplacian filtering한 이미지는 "Laplacian Filter"창에 띄워지도록 설정되어 있습니다.

-실행 함수 및 변수 설명
 sobelfilter(const Mat input)
: 이 함수는 sobel filtering을 수행하는 함수입니다.
함수의 매개 변수인 input은 필터를 적용할 이미지를 의미합니다.

laplacianfilter(const Mat input)
: 이 함수는 laplacianfilter를 수행하는 함수입니다.
함수의 매개 변수인 input은 필터를 적용할 이미지를 의미합니다.