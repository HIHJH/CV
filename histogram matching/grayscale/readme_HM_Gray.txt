Read me!
<HM_Gray.cpp>

-목적
: 이 코드는 흑백 이미지 파일(input)을 레퍼런스 파일(reference)의 intensity distribution과 비슷하게 바꾸도록 histogram matching을 수행하는 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) hist_func.h 헤더 파일과 함께 input, reference 이미지 파일도 프로젝트로 불러온 후, HM_Gray.cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명들을 넣어줍니다.

-코드의 기본 설정
: 기본 설정은 "input.jpg" 파일을 "reference.jpg" 파일과 histogram matching하고, input PDF, 결과로 나온 output PDF, transfer function을 "input_PDF_RGB.txt", "output_PDF.txt", "trans_func.txt" 파일에 저장합니다. 그리고 원본 이미지는 "input", 레퍼런스 이미지는 "Reference", 매칭 이후 이미지는 "Output" 창에 띄워지도록 설정되어 있습니다.

-실행 함수 및 변수 설명
hist_matching(Mat& input, Mat& reference, Mat& output, G* trans_func, float* input_CDF, float* ref_CDF)
: 이 함수는 histogram matching을 수행하는 함수입니다.
함수의 매개 변수들은 아래의 것들을 의미합니다.
  - input: matching: 수행하고자 하는 이미지
  - reference: matching의 목표가 될 이미지
  - output: matching 수행 후의 결과가 저장될 곳
  - trans_func: matching 과정을 담은 함수
  - input_CDF, ref_CDF: 중간에 CDF 계산이 이용되므로 불러올 포인터