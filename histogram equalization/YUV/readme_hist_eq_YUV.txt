Read me!
<hist_eq_YUV.cpp>

-목적
: 이 코드는 컬러 이미지의 contrast를 높이기 위해 uniform한 probability distribution을 갖도록 YUV 채널 중 V 채널에서만 histogram equalization을 수행하는 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) hist_func.h 헤더 파일과 함께 equalization을 수행할 이미지 파일도 프로젝트로 불러온 후, hist_eq_YUV.cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명을 넣어줍니다.

-코드의 기본 설정
: 기본 코드는 "input.jpg" 파일의 PDF, equalization을 수행한 뒤의 PDF, transfer function을 "PDF_RGB.txt", "equalized_PDF_YUV.txt", "trans_func_eq_YUV.txt"에 저장합니다. 그리고 원본 이미지는 "RGB", Equalization한 이미지는 "Equalized_YUV" 창에 띄워지도록 설정되어 있습니다.

-실행 함수 및 변수 설명
hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF)
: 이 함수는 histogram equalization을 수행하는 함수입니다.
 함수의 매개 변수들은 아래의 것들을 의미합니다.
  - input: eq 수행하고자 하는 이미지
  - equalized: eq 수행 후의 결과를 저장할 곳
  - trans_func: equalization 과정을 담은 함수
  - CDF: 중간에 CDF 계산이 이용되므로 불러올 포인터