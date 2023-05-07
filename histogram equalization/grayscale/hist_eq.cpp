#include "hist_func.h" // hist_func.h ������� ���� ����

void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR); // �÷� �̹��� �޾ƿ�
	Mat input_gray;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// RGB�� GrayScale���� ��ȯ

	Mat equalized = input_gray.clone(); // equalized matrix�� input_gray�� ������ ������ �ʱ�ȭ

	// PDF, transfer function �ؽ�Ʈ ����
	FILE* f_PDF; // ������ PDF�� ������ ����
	FILE* f_equalized_PDF_gray; //equalization�� ������ ���� PDF�� ������ ����
	FILE* f_trans_func_eq; // transfer function�� ������ ����

	fopen_s(&f_PDF, "PDF.txt", "w+"); // �а� ���� ���� "PDF.txt"��� �� ������ ����. (������ ���� �ÿ��� ����.)
	fopen_s(&f_equalized_PDF_gray, "equalized_PDF_gray.txt", "w+");
	fopen_s(&f_trans_func_eq, "trans_func_eq.txt", "w+");

	float* PDF = cal_PDF(input_gray);	// PDF of Input image(Grayscale) : [L]
	float* CDF = cal_CDF(input_gray);	// CDF of Input image(Grayscale) : [L]

	G trans_func_eq[L] = { 0 };			// transfer function �迭

	// ��� �̹����� histogram equalization ����
	hist_eq(input_gray, equalized, trans_func_eq, CDF);
	// HE ���� ���� PDF�� ����Ͽ� equalized_PDF_gray�� ����
	float* equalized_PDF_gray = cal_PDF(equalized);

	for (int i = 0; i < L; i++) {
		// ��� ��Ʈ��(text ����)�� ������׷��� �ۼ�
		fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);
		fprintf(f_equalized_PDF_gray, "%d\t%f\n", i, equalized_PDF_gray[i]);

		// ��� ��Ʈ��(text ����)�� transfer function�� �ۼ�
		fprintf(f_trans_func_eq, "%d\t%d\n", i, trans_func_eq[i]);
	}

	// �޸𸮸� �ݳ��ϰ� ������ �۾� ����
	free(PDF);
	free(CDF);
	fclose(f_PDF);
	fclose(f_equalized_PDF_gray);
	fclose(f_trans_func_eq);

	////////////////////// Show each image ///////////////////////

	namedWindow("Grayscale", WINDOW_AUTOSIZE); // â �̸� "Grayscale"
	imshow("Grayscale", input_gray); // ��� �̹��� ����

	namedWindow("Equalized", WINDOW_AUTOSIZE);
	imshow("Equalized", equalized); // HE ������ �̹��� ����

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram equalization
// input: eq �����ϰ��� �ϴ� �̹���, equalized: eq ���� ���� ����� ���⿡ ����
// trans_func: equalization ������ ���� �Լ�, CDF: �߰��� CDF ����� �̿�Ǿ� ������ �ҷ���
void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF) {

	// compute transfer function
	for (int i = 0; i < L; i++)
		trans_func[i] = (G)((L - 1) * CDF[i]);  // output = (L-1) * CDF(input)

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			// input�� transfer function ������� equalized�� ����
			equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}
