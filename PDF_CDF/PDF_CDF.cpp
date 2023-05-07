#include "hist_func.h" // hist_func.h ������� ���� ����


int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR); // �÷� �̹��� �޾ƿ�
	Mat input_gray;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// RGB�� GrayScale���� ��ȯ

	FILE* f_PDF, * f_CDF; // PDF, CDF �ؽ�Ʈ ����

	fopen_s(&f_PDF, "PDF.txt", "w+"); // �а� ���� ���� "PDF_RGB.txt"��� �� ������ ����. (������ ���� �ÿ��� ����.)
	fopen_s(&f_CDF, "CDF.txt", "w+");

	// ��� input�� PDF, CDF�� ����Ѵ�.
	// hist_func.h ��������� ��� �Լ��� ����Ѵ�.
	float* PDF = cal_PDF(input_gray);		// PDF of Input image(Grayscale) : [L]
	float* CDF = cal_CDF(input_gray);		// CDF of Input image(Grayscale) : [L]

	for (int i = 0; i < L; i++) {
		// ��� ��Ʈ��(text ����)�� ������׷��� �ۼ�
		fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);
		fprintf(f_CDF, "%d\t%f\n", i, CDF[i]);
	}

	// �Է� �̹����� ���� ������׷��� ���
	int hist[256] = { 0 }; // ������׷� �迭 ����, 0���� �ʱ�ȭ
	for (int i = 0; i < input_gray.rows; i++) {
		for (int j = 0; j < input_gray.cols; j++) {
			int intensity = (int)input_gray.at<G>(i, j); // ��� �ȼ��� ��ȸ�ϸ�, �ش� �ȼ��� intensity�� ����
			hist[intensity]++; // �ش� �ȼ��� intensity�� ���� ������׷� ���� ������Ŵ
		}
	}

	// ������׷� �׷��� �׸��� (plot it)
	// Draw histogram for PDF
	Mat hist_pdf = Mat::zeros(L, L, IM_TYPE); // �̹��� ũ���� hist_pdf �迭 ����
	for (int i = 0; i < L; i++) {
		float pdf_val = PDF[i] * (L - 1); // PDF ���� (����-1)�� ���Ͽ� pdf_val�� ����
		line(hist_pdf, Point(i, (L - 1)), Point(i, (L - 1) - pdf_val), Scalar(255, 255, 255), 1); // �̹����� ����(256)�� �������� ���� �׸�.
		//Scalar(255, 255, 255)�� ��� ���� �ǹ�
	}
	namedWindow("PDF Histogram", WINDOW_AUTOSIZE); // â �̸� "PDF Histogram"
	imshow("PDF Histogram", hist_pdf); // â�� ������׷� ����

	// Draw histogram for CDF
	// PDF ������׷� ��� ��İ� ����
	Mat hist_cdf = Mat::zeros(256, 256, IM_TYPE);
	for (int i = 0; i < L; i++) {
		float cdf_val = CDF[i] * (L - 1);
		line(hist_cdf, Point(i, (L - 1)), Point(i, (L - 1) - cdf_val), Scalar(255, 255, 255), 1);
	}
	namedWindow("CDF Histogram", WINDOW_AUTOSIZE);
	imshow("CDF Histogram", hist_cdf);

	// �޸𸮸� �ݳ��ϰ� ������ �۾� ����
	free(PDF);
	free(CDF);
	fclose(f_PDF);
	fclose(f_CDF);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE); // â �̸� "RGB"
	imshow("RGB", input); // ���� �̹��� ����

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray); // ��� �̹��� ����

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}