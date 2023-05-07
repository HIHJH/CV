Read me!
<hist_stretching.cpp>

-목적
: 이 코드는 흑백 이미지 파일의 intensity range를 늘려주어 contrast를 높여주기 위해 histogram stretching을 수행하는 코드입니다. 이 코드는 stretching 방법 중 linear stretching 방법을 사용하고 있습니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) hist_func.h 헤더 파일과 함께 stretching 시킬 이미지 파일도 프로젝트로 불러온 후, hist_stretching.cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명을 넣어줍니다.

-코드의 기본 설정
: 기본 파일은 "input.jpg" 파일의 PDF, stretching 시킨 후의 PDF, transfer function을 "PDF.txt", "stretched_PDF.txt", "trans_func_stretch.txt"에 저장합니다. 또한 원본 흑백 이미지는 "Grayscale", stretching 된 이미지는 "Stretched" 창에 띄워지도록 설정되어 있습니다.

-실행 함수 및 변수 설명
linear_stretching(Mat &input, Mat &stretched, G *trans_func, G x1, G x2, G y1, G y2)
: 이 함수는 histogram linear stretching을 수행하는 함수입니다.
함수의 매개 변수들은 아래의 것들을 의미합니다.
  - input: stretching 수행하고자 하는 이미지
  - stretched: stretching 수행 후의 결과가 저장될 곳
  - trans_func: stretching 과정을 담은 함수
  - x1, x2, y1, y2: (x1,x2) -> (y1, y2)로 stretch 하고자 할 때의 좌표들