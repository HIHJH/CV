#include "hist_func.h" // hist_func.h 헤더파일 내용 포함

void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR); // 컬러 이미지 받아옴
	Mat input_gray;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// RGB를 GrayScale으로 변환

	Mat equalized = input_gray.clone(); // equalized matrix는 input_gray를 복제한 것으로 초기화

	// PDF, transfer function 텍스트 파일
	FILE* f_PDF; // 원본의 PDF를 저장할 파일
	FILE* f_equalized_PDF_gray; //equalization을 수행한 후의 PDF를 저장할 파일
	FILE* f_trans_func_eq; // transfer function을 저장할 파일

	fopen_s(&f_PDF, "PDF.txt", "w+"); // 읽고 쓰기 위해 "PDF.txt"라는 빈 파일을 연다. (내용이 있을 시에는 비운다.)
	fopen_s(&f_equalized_PDF_gray, "equalized_PDF_gray.txt", "w+");
	fopen_s(&f_trans_func_eq, "trans_func_eq.txt", "w+");

	float* PDF = cal_PDF(input_gray);	// PDF of Input image(Grayscale) : [L]
	float* CDF = cal_CDF(input_gray);	// CDF of Input image(Grayscale) : [L]

	G trans_func_eq[L] = { 0 };			// transfer function 배열

	// 흑백 이미지의 histogram equalization 수행
	hist_eq(input_gray, equalized, trans_func_eq, CDF);
	// HE 수행 후의 PDF를 계산하여 equalized_PDF_gray에 저장
	float* equalized_PDF_gray = cal_PDF(equalized);

	for (int i = 0; i < L; i++) {
		// 출력 스트림(text 파일)에 히스토그램을 작성
		fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);
		fprintf(f_equalized_PDF_gray, "%d\t%f\n", i, equalized_PDF_gray[i]);

		// 출력 스트림(text 파일)에 transfer function을 작성
		fprintf(f_trans_func_eq, "%d\t%d\n", i, trans_func_eq[i]);
	}

	// 메모리를 반납하고 수행한 작업 정리
	free(PDF);
	free(CDF);
	fclose(f_PDF);
	fclose(f_equalized_PDF_gray);
	fclose(f_trans_func_eq);

	////////////////////// Show each image ///////////////////////

	namedWindow("Grayscale", WINDOW_AUTOSIZE); // 창 이름 "Grayscale"
	imshow("Grayscale", input_gray); // 흑백 이미지 띄우기

	namedWindow("Equalized", WINDOW_AUTOSIZE);
	imshow("Equalized", equalized); // HE 수행한 이미지 띄우기

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram equalization
// input: eq 수행하고자 하는 이미지, equalized: eq 수행 후의 결과를 여기에 저장
// trans_func: equalization 과정을 담은 함수, CDF: 중간에 CDF 계산이 이용되어 포인터 불러옴
void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF) {

	// compute transfer function
	for (int i = 0; i < L; i++)
		trans_func[i] = (G)((L - 1) * CDF[i]);  // output = (L-1) * CDF(input)

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			// input에 transfer function 적용시켜 equalized에 저장
			equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}
