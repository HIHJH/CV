#include "hist_func.h" // hist_func.h 헤더파일 내용 포함

void linear_stretching(Mat& input, Mat& stretched, G* trans_func, G x1, G x2, G y1, G y2);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR); // 컬러 이미지 받아옴
	Mat input_gray;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// RGB를 GrayScale으로 변환

	Mat stretched = input_gray.clone(); // stretched matrix는 input_gray를 복제한 것으로 초기화

	// PDF, transfer function 텍스트 파일
	FILE* f_PDF; // 원본의 PDF를 저장할 파일
	FILE* f_stretched_PDF; //stretching을 수행한 후의 PDF를 저장할 파일
	FILE* f_trans_func_stretch; // transfer function을 저장할 파일

	fopen_s(&f_PDF, "PDF.txt", "w+"); // 읽고 쓰기 위해 "PDF.txt"라는 빈 파일을 연다. (내용이 있을 시에는 비운다.)
	fopen_s(&f_stretched_PDF, "stretched_PDF.txt", "w+");
	fopen_s(&f_trans_func_stretch, "trans_func_stretch.txt", "w+");

	G trans_func_stretch[L] = { 0 }; // transfer function 배열

	float* PDF = cal_PDF(input_gray); // PDF of Input image(Grayscale) : [L]

	//  histogram stretching (50 ~ 110 -> 10 ~ 110)을 수행
	linear_stretching(input_gray, stretched, trans_func_stretch, 50, 110, 10, 110);
	// stretching 수행 후의 PDF를 계산하여 stretched_PDF에 저장
	float* stretched_PDF = cal_PDF(stretched);

	for (int i = 0; i < L; i++) {
		// 출력 스트림(text 파일)에 히스토그램을 작성
		fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);
		fprintf(f_stretched_PDF, "%d\t%f\n", i, stretched_PDF[i]);

		// 출력 스트림(text 파일)에 transfer function을 작성
		fprintf(f_trans_func_stretch, "%d\t%d\n", i, trans_func_stretch[i]);
	}

	// 메모리를 반납하고 수행한 작업 정리
	free(PDF);
	free(stretched_PDF);
	fclose(f_PDF);
	fclose(f_stretched_PDF);
	fclose(f_trans_func_stretch);

	////////////////////// Show each image ///////////////////////

	namedWindow("Grayscale", WINDOW_AUTOSIZE); // 창 이름 "Grayscale"
	imshow("Grayscale", input_gray); // 흑백 이미지 띄우기

	namedWindow("Stretched", WINDOW_AUTOSIZE);
	imshow("Stretched", stretched); // stretching 수행한 이미지 띄우기

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram stretching (linear method)
// input: stretching 수행하고자 하는 이미지, stretched: stretching 수행 후의 결과를 여기에 저장
// trans_func: stretching 과정을 담은 함수, (x1,x2) -> (y1, y2)로 stretch 하고자 함
void linear_stretching(Mat& input, Mat& stretched, G* trans_func, G x1, G x2, G y1, G y2) {

	float constant = (y2 - y1) / (float)(x2 - x1); // linear stretching function의 기울기를 constant에 저장 

	// compute transfer function
	// linear stretching function의 형태 -> (a(i), b(i), a(i+1), b(i+2)) 인 경우
	// y = {( b(i+1) - b(i) ) / ( a(i+1) - a(i) )} ( x - a(i) ) + b(i)
	for (int i = 0; i < L; i++) {
		if (i >= 0 && i <= x1) // x1 보다 작은 경우
			trans_func[i] = (G)(y1 / x1 * i); // 
		else if (i > x1 && i <= x2) // x1 ~ x2 사이인 경우
			trans_func[i] = (G)(constant * (i - x1) + y1);
		else // x2 보다 큰 경우
			trans_func[i] = (G)((L - 1 - x2) / (L - 1 - y2) * (i - x2) + y2);
	}

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			// input에 transfer function 적용시켜 stretched에 저장
			stretched.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}