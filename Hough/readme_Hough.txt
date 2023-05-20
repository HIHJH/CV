Read me!
<Hough.cpp>

-목적
: <Hough.cpp>은 이미지에 canny edge detector를 적용해 edge를 검출하고 그 edge들에 hough transform을 적용해 fitting line을 검출하는 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) line fitting을 수행할 이미지 파일을 프로젝트로 불러온 후, cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명을 넣어줍니다.

-코드의 기본 설정
: 기본 코드는 "building.jpg" 에 Canny 엣지 검출을 수행한 후 hough transform을 수행합니다. 또는 probabilistic hough transform을 수행합니다. 실행 함수 및 변수 설명에 기본 설정 값도 포함되어 있습니다. 원본 이미지는 "Source" 창에, canny edge에 fitting line 된 결과는 "Detected Lines" 창에 띄워지도록 설정되어 있습니다.

-실행 함수 및 변수 설명
1. HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0)
: 이 함수는 hough 변환 수행하여 선 검출하는 함수입니다.
- image: hough 변환 수행할 이미지(canny edge 검출을 통해 얻은 edge 이미지) = dst
- lines: 검출된 선 저장할 벡터 = lines
- rho: 거리 해상도 = 1
- theta: 각도 해상도 = CV_PI / 180
- threshold: 선을 검출하기 위한 최소 투표 수 = 150
- srn, stn: 향상된 hough 변환을 위한 옵션 여부 = 0, 0

2. HoughLinesP(dst, lines, 1, CV_PI / 180, 80, 30, 10)
: 이 함수는 확률적으로 hough 변환 수행하여 선 검출하는 함수 입니다.
- image: hough 변환 수행할 이미지(canny edge 검출을 통해 얻은 edge 이미지) = dst
- lines: 검출된 선 저장할 벡터 = lines
- rho: 거리 해상도 = 1
- theta: 각도 해상도 = CV_PI / 180
- threshold: 선을 검출하기 위한 최소 투표 수 = 80
- minLineLength: 검출할 선의 최소 길이 = 30
- maxLineGap: 선들 간의 최대 허용 간격 = 10

3. Canny(image, edges, threshold1, threshold2, apertureSize)
: 이 함수는 Canny edge detection을 수행하는 함수입니다.
- image = 엣지 검출할 이미지 = src
- edges = 출력 이미지 = dst
- threshold1 = 엣지 검출을 위한 하위 임계값 = 50
- threshold2 = 엣지 검출을 위한 상위 임계값 = 200
- apertureSize = Sobel 연산에 사용되는 커널 크기 = 3