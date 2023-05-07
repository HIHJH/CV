#include "hist_func.h" // hist_func.h 헤더파일 내용 포함

void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR); // 컬러 이미지 받아옴
	Mat equalized_YUV;

	cvtColor(input, equalized_YUV, CV_RGB2YUV);	// RGB를 YUV로 변환하고, equalized_YUV에 저장

	// Y, U, V 채널 각각을 나누기
	Mat channels[3];
	split(equalized_YUV, channels);
	Mat Y = channels[0];						// U = channels[1], V = channels[2]

	// PDF, transfer function 텍스트 파일
	FILE* f_equalized_PDF_YUV, * f_PDF_RGB; // 원본의 PDF와 equalization을 수행한 후의 PDF를 저장할 파일들
	FILE* f_trans_func_eq_YUV; //  transfer function을 저장할 파일

	float** PDF_RGB = cal_PDF_RGB(input);		// PDF of Input image(RGB) : [L][3]
	float* CDF_YUV = cal_CDF(Y);				// CDF of Y channel image

	fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+"); // 읽고 쓰기 위해 "PDF_RGB.txt"라는 빈 파일을 연다. (내용이 있을 시에는 비운다.)
	fopen_s(&f_equalized_PDF_YUV, "equalized_PDF_YUV.txt", "w+");
	fopen_s(&f_trans_func_eq_YUV, "trans_func_eq_YUV.txt", "w+");

	G trans_func_eq_YUV[L] = { 0 };			// transfer function 배열

	//  Y channel 에서만 histogram equalization 수행
	hist_eq(Y, Y, trans_func_eq_YUV, CDF_YUV);

	// Y에서 HE 수행 후 다시 합치기(Y, U, V channels)
	merge(channels, 3, equalized_YUV);

	// 합친 YUV를 다시 RGB(use "CV_YUV2RGB" flag)
	cvtColor(equalized_YUV, equalized_YUV, CV_YUV2RGB);

	// 다시 변환한 RGB의 PDF를 계산 (RGB)하여 equalized_PDF_YUV에 저장
	float** equalized_PDF_YUV = cal_PDF_RGB(equalized_YUV);

	for (int i = 0; i < L; i++) {
		// 출력 스트림(text 파일)에 히스토그램을 작성
		fprintf(f_PDF_RGB, "%d\t%f\t%f\t%f\n", i, PDF_RGB[i][0], PDF_RGB[i][1], PDF_RGB[i][2]);
		fprintf(f_equalized_PDF_YUV, "%d\t%f\t%f\t%f\n", i, equalized_PDF_YUV[i][0], equalized_PDF_YUV[i][1], equalized_PDF_YUV[i][2]);
		// 출력 스트림(text 파일)에 transfer function을 작성
		fprintf(f_trans_func_eq_YUV, "%d\t%d\n", i, trans_func_eq_YUV[i]);
	}

	// 메모리를 반납하고 수행한 작업 정리
	free(PDF_RGB);
	free(CDF_YUV);
	fclose(f_PDF_RGB);
	fclose(f_equalized_PDF_YUV);
	fclose(f_trans_func_eq_YUV);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE); // 창 이름 "RGB"
	imshow("RGB", input); // 원본 이미지 띄우기

	namedWindow("Equalized_YUV", WINDOW_AUTOSIZE);
	imshow("Equalized_YUV", equalized_YUV);  // HE 수행한 이미지 띄우기

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
		trans_func[i] = (G)((L - 1) * CDF[i]); // output = (L-1) * CDF(input)

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			// input에 transfer function 적용시켜 equalized에 저장
			equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}