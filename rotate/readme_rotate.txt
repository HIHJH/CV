Read me!
<rotate_skeleton_v2.cpp>

-목적
: 이 코드는 이미지 파일을 원하는 각도로 회전하기 위해 사용하는 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) 회전시킬 이미지 파일도 프로젝트로 불러온 후, rotate_skeleton_v2.cpp 파일을 실행시킵니다.

-입력할 매개변수
: imread() 함수의 매개변수로 회전시킬 이미지 파일명을 넣어줍니다. *line14
myrotate<Vec3b>(,,) 함수의 매개변수로 이미지 파일을 읽은 변수명, 각도, interpolation 방법을 각각 입력합니다.(예시: input, 45, "bilinear") *line26

-코드의 기본 설정
: 기본 파일은 "lena.jpg" 파일을 45도 회전시키고 bilinear interpolation 한 것을 "rotated" 창에 띄워지도록 설정되어 있습니다. 또한, 기존 "lena.jpg" 파일도 "image" 창에 띄워지도록 되어 있습니다.

-실행 함수 및 변수 설명
: Mat myrotate(const Mat input, float angle, const char* opt) 함수를 사용합니다. 이 함수는 input 픽셀들을 원점을 기준으로 angle만큼 회전하도록 이동시키고 nearest 또는 bilinear 방법으로 interpolation 시켜줍니다.
여기서 input은 회전할 이미지 파일을 읽어온 변수명, angle은 회전할 각도, opt는 interpolation 방법을 의미합니다.