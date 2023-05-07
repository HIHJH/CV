#include "hist_func.h" // hist_func.h 헤더파일 내용 포함

void hist_eq_Color(Mat& input, Mat& equalized, G(*trans_func)[3], float** CDF);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR); // 컬러 이미지 받아옴
	Mat equalized_RGB = input.clone(); // input matrix를 equalized_RGB matrix에 저장해둠

	// PDF, transfer function 텍스트 파일
	FILE* f_equalized_PDF_RGB, * f_PDF_RGB; // 원본의 PDF와 equalization을 수행한 후의 PDF를 저장할 파일들
	FILE* f_trans_func_eq_RGB; // transfer function을 저장할 파일

	fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+"); // 읽고 쓰기 위해 "PDF_RGB.txt"라는 빈 파일을 연다. (내용이 있을 시에는 비운다.)
	fopen_s(&f_equalized_PDF_RGB, "equalized_PDF_RGB.txt", "w+");
	fopen_s(&f_trans_func_eq_RGB, "trans_func_eq_RGB.txt", "w+");

	float** PDF_RGB = cal_PDF_RGB(input);	// PDF of Input image(RGB) : [L][3]
	float** CDF_RGB = cal_CDF_RGB(input);	// CDF of Input image(RGB) : [L][3]

	G trans_func_eq_RGB[L][3] = { 0 };		// transfer function 배열

	// 컬러 이미지의 histogram equalization 수행
	hist_eq_Color(input, equalized_RGB, trans_func_eq_RGB, CDF_RGB);

	// HE 수행 후의 PDF를 계산하여 equalized_PDF_RGB에 저장
	float** equalized_PDF_RGB = cal_PDF_RGB(equalized_RGB);

	for (int i = 0; i < L; i++) {
		// 출력 스트림(text 파일)에 RGB 각각의 히스토그램을 작성
		fprintf(f_PDF_RGB, "%d\t%f\t%f\t%f\n", i, PDF_RGB[i][0], PDF_RGB[i][1], PDF_RGB[i][2]);
		fprintf(f_equalized_PDF_RGB, "%d\t%f\t%f\t%f\n", i, equalized_PDF_RGB[i][0], equalized_PDF_RGB[i][1], equalized_PDF_RGB[i][2]);

		// 출력 스트림(text 파일)에 transfer function을 작성
		fprintf(f_trans_func_eq_RGB, "%d\t%f\t%f\t%f\n", i, trans_func_eq_RGB[i][0], trans_func_eq_RGB[i][1], trans_func_eq_RGB[i][2]);
	}

	// 메모리를 반납하고 수행한 작업 정리
	free(PDF_RGB);
	free(CDF_RGB);
	fclose(f_PDF_RGB);
	fclose(f_equalized_PDF_RGB);
	fclose(f_trans_func_eq_RGB);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE); // 창 이름 "RGB"
	imshow("RGB", input); // 원본 이미지 띄우기

	namedWindow("Equalized_RGB", WINDOW_AUTOSIZE);
	imshow("Equalized_RGB", equalized_RGB);  // HE 수행한 이미지 띄우기

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram equalization on 3 channel image
// input: eq 수행하고자 하는 이미지, equalized: eq 수행 후의 결과를 여기에 저장
// trans_func: equalization 과정을 담은 함수, CDF: 중간에 CDF 계산이 이용되어 포인터 불러옴
void hist_eq_Color(Mat& input, Mat& equalized, G(*trans_func)[3], float** CDF) {

	// compute transfer function
	for (int i = 0; i < L; i++) {
		for (int j = 0; j < 3; j++) { // 채널마다 계산
			trans_func[i][j] = (G)((L - 1) * CDF[i][j]); // output = (L-1) * CDF(input)
		}
	}

	// perform the transfer function
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			//input RGB 각각에 transfer function을 적용시켜 equalized RGB 각각에 저장
			equalized.at<C>(i, j)[0] = trans_func[input.at<C>(i, j)[0]][0];
			equalized.at<C>(i, j)[1] = trans_func[input.at<C>(i, j)[1]][1];
			equalized.at<C>(i, j)[2] = trans_func[input.at<C>(i, j)[2]][2];
		}
	}

}