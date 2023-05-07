#include "hist_func.h" // hist_func.h ������� ���� ����

void hist_eq_Color(Mat& input, Mat& equalized, G(*trans_func)[3], float** CDF);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR); // �÷� �̹��� �޾ƿ�
	Mat equalized_RGB = input.clone(); // input matrix�� equalized_RGB matrix�� �����ص�

	// PDF, transfer function �ؽ�Ʈ ����
	FILE* f_equalized_PDF_RGB, * f_PDF_RGB; // ������ PDF�� equalization�� ������ ���� PDF�� ������ ���ϵ�
	FILE* f_trans_func_eq_RGB; // transfer function�� ������ ����

	fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+"); // �а� ���� ���� "PDF_RGB.txt"��� �� ������ ����. (������ ���� �ÿ��� ����.)
	fopen_s(&f_equalized_PDF_RGB, "equalized_PDF_RGB.txt", "w+");
	fopen_s(&f_trans_func_eq_RGB, "trans_func_eq_RGB.txt", "w+");

	float** PDF_RGB = cal_PDF_RGB(input);	// PDF of Input image(RGB) : [L][3]
	float** CDF_RGB = cal_CDF_RGB(input);	// CDF of Input image(RGB) : [L][3]

	G trans_func_eq_RGB[L][3] = { 0 };		// transfer function �迭

	// �÷� �̹����� histogram equalization ����
	hist_eq_Color(input, equalized_RGB, trans_func_eq_RGB, CDF_RGB);

	// HE ���� ���� PDF�� ����Ͽ� equalized_PDF_RGB�� ����
	float** equalized_PDF_RGB = cal_PDF_RGB(equalized_RGB);

	for (int i = 0; i < L; i++) {
		// ��� ��Ʈ��(text ����)�� RGB ������ ������׷��� �ۼ�
		fprintf(f_PDF_RGB, "%d\t%f\t%f\t%f\n", i, PDF_RGB[i][0], PDF_RGB[i][1], PDF_RGB[i][2]);
		fprintf(f_equalized_PDF_RGB, "%d\t%f\t%f\t%f\n", i, equalized_PDF_RGB[i][0], equalized_PDF_RGB[i][1], equalized_PDF_RGB[i][2]);

		// ��� ��Ʈ��(text ����)�� transfer function�� �ۼ�
		fprintf(f_trans_func_eq_RGB, "%d\t%f\t%f\t%f\n", i, trans_func_eq_RGB[i][0], trans_func_eq_RGB[i][1], trans_func_eq_RGB[i][2]);
	}

	// �޸𸮸� �ݳ��ϰ� ������ �۾� ����
	free(PDF_RGB);
	free(CDF_RGB);
	fclose(f_PDF_RGB);
	fclose(f_equalized_PDF_RGB);
	fclose(f_trans_func_eq_RGB);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE); // â �̸� "RGB"
	imshow("RGB", input); // ���� �̹��� ����

	namedWindow("Equalized_RGB", WINDOW_AUTOSIZE);
	imshow("Equalized_RGB", equalized_RGB);  // HE ������ �̹��� ����

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram equalization on 3 channel image
// input: eq �����ϰ��� �ϴ� �̹���, equalized: eq ���� ���� ����� ���⿡ ����
// trans_func: equalization ������ ���� �Լ�, CDF: �߰��� CDF ����� �̿�Ǿ� ������ �ҷ���
void hist_eq_Color(Mat& input, Mat& equalized, G(*trans_func)[3], float** CDF) {

	// compute transfer function
	for (int i = 0; i < L; i++) {
		for (int j = 0; j < 3; j++) { // ä�θ��� ���
			trans_func[i][j] = (G)((L - 1) * CDF[i][j]); // output = (L-1) * CDF(input)
		}
	}

	// perform the transfer function
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			//input RGB ������ transfer function�� ������� equalized RGB ������ ����
			equalized.at<C>(i, j)[0] = trans_func[input.at<C>(i, j)[0]][0];
			equalized.at<C>(i, j)[1] = trans_func[input.at<C>(i, j)[1]][1];
			equalized.at<C>(i, j)[2] = trans_func[input.at<C>(i, j)[2]][2];
		}
	}

}