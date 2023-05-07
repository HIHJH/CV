Read me!
<PDF_CDF.cpp>

-목적
: 이 코드는 이미지 파일의 PDF(Probability Density Funciton), CDF(Cumulative Distribution Function)를 담은 text file을 생성하고, 히스토그램 그래프를 그려주기 위해 사용하는 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) hist_func.h 헤더 파일과 함께 PDF, CDF를 구할 이미지 파일도 프로젝트로 불러온 후, PDF_CDF.cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 이미지 파일명을 넣어줍니다.

-코드의 기본 설정
: 기본 파일은 "input.jpg" 파일의 PDF, CDF를 "PDF.txt", "CDF.txt"에 저장합니다. 그리고 해당 데이터를 plot한 히스토그램 그래프도 "PDF Histogram", "CDF Histogram" 창으로 띄워줍니다. 또한 원본 이미지는 "RGB", 흑백 이미지는 "Grayscale" 창에 띄워지도록 설정되어 있습니다.

-실행 함수 및 변수 설명
: 이 코드는 hist_func.h 헤더 파일 속에 포함된 PDF, CDF 계산 함수를 사용합니다.
cal_PDF(Mat &input)은 input의 PDF를 계산해줍니다.
cal_CDF(Mat &input)은 input의 CDF를 계산해줍니다.
여기서 input은 이미지 파일을 읽어온 matrix 입니다.