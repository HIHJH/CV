Read me!
<stitching.cpp>

-목적
: 이 코드는 같은 object가 포함된 2개의 이미지 파일(다른 시야)이 존재할 때, 두 이미지 파일의 object를 겹쳐 하나의 이미지로 합치기 위한 코드입니다.

-사용방법
: opencv를 다운 받고, visual studio에서 새 프로젝트를 만들어 해당 파일을 불러옵니다. opencv를 사용하기 위해 프로젝트 속성들을 변경해줍니다.(변경 방법은 인터넷 참고) stitching 할 이미지 파일들도 프로젝트로 불러온 후, stitching.cpp 파일을 실행시킵니다.

-코드의 기본 설정
: "stitchingL.jpg", "stitchingR.jpg" 파일의 겹치는 픽셀들을 affine transform을 이용하여 연결한 뒤에, bilinear interpolation을 적용하고 blending 해줍니다. 이렇게 stitching 된 이미지를 "result.png" 파일로 저장하고, "result" 창에 띄워지도록 설정되어 있습니다. 또한, 기존 "stitchingL.jpg", "stitchingR.jpg" 파일도 각각 "Left Image", "Right Image" 창에 띄워지도록 되어 있습니다.

-실행 함수 및 변수 설명
1. cal_affine(int ptl_x[], int ptl_y[], int ptr_x[], int ptr_y[], int number_of_points) 함수를 사용합니다. 이 함수는 두 이미지 간의 corresponding pixels를 바탕으로, 두 점 사이의 이동을 나타내는 2*3 크기의 affine matrix를 계산하는 함수입니다. 여기서 ptl_x[], ptl_y[], ptr_x[], prt_y[]는 왼쪽 이미지와 오른쪽 이미지의 겹치는 픽셀들의 x, y 값 각각을 의미합니다. number_of_points는 이 픽셀들의 개수를 의미합니다. 
2. blend_stitching(const Mat I1, const Mat I2, Mat &I_f, int bound_l, int bound_u, float alpha) 함수를 사용합니다. 이 함수는 합쳐진 이미지의 픽셀값을 입력된 alpha 비율에 맞게 I1, I_f의 픽셀값에서 받아오는 역할을 합니다.
그리고 blend_stitching 함수 사용 전에 I2는 inverse warping을 수행하고, bilinear interpolation을 이용하여 두 이미지의 경계 부분이 자연스럽게 이어지도록 합니다.
I1은 왼쪽 이미지 행렬, I2는 오른쪽 이미지 행렬을 의미하고, bound_l는 왼쪽 경계의 위치, bound_u는 위쪽 경계의 위치를 나타냅니다. alpha는 이미지 blending에 사용될 가중치(weight) 값을 의미합니다.