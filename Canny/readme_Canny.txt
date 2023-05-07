Read me!
<Canny.cpp>

-목적
: 이 코드는 이미지에서 canny edge detector을 이용해 엣지를 검출하는 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) canny edge detect를 수행할 이미지 파일을 프로젝트로 불러온 후, cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명을 넣어줍니다.
Canny() 함수의 매개변수로 엣지 검출할 이미지, 출력 이미지, 엣지 검출을 위한 하위 임계값, 상위 임계값, Sobel 연산에 사용되는 커널 크기, 미분할 때 사용되는 방법 (true = L2-norm, false = L1-norm)을 차례대로 넣어줍니다.

-코드의 기본 설정
: 기본 코드는 "lena.jpg" 파일의 흑백 이미지에 Canny edge detection을 수행합니다. 기본 설정은 임계값이 100, 160이고, sobel 연산의 커널 크기는 3, 미분할 때는 L1-norm이 사용됩니다. 그리고 원본 흑백 이미지는 "Grayscale", Canny edge detect가 적용된 엣지 이미지는 "Canny" 창에 띄워지도록 설정되어 있습니다.

-실행 함수 및 변수 설명
1. Canny(image, edges, threshold1, threshold2, apertureSize, L2gradient)
: 이 함수는 Canny edge detection을 수행하는 함수입니다. 함수의 매개변수는 아래의 것들을 의미합니다.
- image = 엣지 검출할 이미지
- edges = 출력 이미지
- threshold1 = 엣지 검출을 위한 하위 임계값
- threshold2 = 엣지 검출을 위한 상위 임계값
- apertureSize = Sobel 연산에 사용되는 커널 크기
- L2gradient = 미분할 때 사용되는 방법 (true = L2-norm, false = L1-norm)