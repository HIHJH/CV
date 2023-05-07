#include "hist_func.h" // hist_func.h 헤더파일 내용 포함


int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR); // 컬러 이미지 받아옴
	Mat input_gray;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// RGB를 GrayScale으로 변환

	FILE* f_PDF, * f_CDF; // PDF, CDF 텍스트 파일

	fopen_s(&f_PDF, "PDF.txt", "w+"); // 읽고 쓰기 위해 "PDF_RGB.txt"라는 빈 파일을 연다. (내용이 있을 시에는 비운다.)
	fopen_s(&f_CDF, "CDF.txt", "w+");

	// 흑백 input의 PDF, CDF를 계산한다.
	// hist_func.h 헤더파일의 계산 함수를 사용한다.
	float* PDF = cal_PDF(input_gray);		// PDF of Input image(Grayscale) : [L]
	float* CDF = cal_CDF(input_gray);		// CDF of Input image(Grayscale) : [L]

	for (int i = 0; i < L; i++) {
		// 출력 스트림(text 파일)에 히스토그램을 작성
		fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);
		fprintf(f_CDF, "%d\t%f\n", i, CDF[i]);
	}

	// 입력 이미지에 대한 히스토그램을 계산
	int hist[256] = { 0 }; // 히스토그램 배열 생성, 0으로 초기화
	for (int i = 0; i < input_gray.rows; i++) {
		for (int j = 0; j < input_gray.cols; j++) {
			int intensity = (int)input_gray.at<G>(i, j); // 모든 픽셀을 순회하며, 해당 픽셀의 intensity를 저장
			hist[intensity]++; // 해당 픽셀의 intensity에 따라 히스토그램 값을 증가시킴
		}
	}

	// 히스토그램 그래프 그리기 (plot it)
	// Draw histogram for PDF
	Mat hist_pdf = Mat::zeros(L, L, IM_TYPE); // 이미지 크기의 hist_pdf 배열 생성
	for (int i = 0; i < L; i++) {
		float pdf_val = PDF[i] * (L - 1); // PDF 값에 (높이-1)을 곱하여 pdf_val에 저장
		line(hist_pdf, Point(i, (L - 1)), Point(i, (L - 1) - pdf_val), Scalar(255, 255, 255), 1); // 이미지의 높이(256)를 기준으로 선을 그림.
		//Scalar(255, 255, 255)는 흰색 선을 의미
	}
	namedWindow("PDF Histogram", WINDOW_AUTOSIZE); // 창 이름 "PDF Histogram"
	imshow("PDF Histogram", hist_pdf); // 창에 히스토그램 띄우기

	// Draw histogram for CDF
	// PDF 히스토그램 출력 방식과 동일
	Mat hist_cdf = Mat::zeros(256, 256, IM_TYPE);
	for (int i = 0; i < L; i++) {
		float cdf_val = CDF[i] * (L - 1);
		line(hist_cdf, Point(i, (L - 1)), Point(i, (L - 1) - cdf_val), Scalar(255, 255, 255), 1);
	}
	namedWindow("CDF Histogram", WINDOW_AUTOSIZE);
	imshow("CDF Histogram", hist_cdf);

	// 메모리를 반납하고 수행한 작업 정리
	free(PDF);
	free(CDF);
	fclose(f_PDF);
	fclose(f_CDF);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE); // 창 이름 "RGB"
	imshow("RGB", input); // 원본 이미지 띄우기

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray); // 흑백 이미지 띄우기

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}