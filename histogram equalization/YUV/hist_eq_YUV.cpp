#include "hist_func.h" // hist_func.h ������� ���� ����

void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR); // �÷� �̹��� �޾ƿ�
	Mat equalized_YUV;

	cvtColor(input, equalized_YUV, CV_RGB2YUV);	// RGB�� YUV�� ��ȯ�ϰ�, equalized_YUV�� ����

	// Y, U, V ä�� ������ ������
	Mat channels[3];
	split(equalized_YUV, channels);
	Mat Y = channels[0];						// U = channels[1], V = channels[2]

	// PDF, transfer function �ؽ�Ʈ ����
	FILE* f_equalized_PDF_YUV, * f_PDF_RGB; // ������ PDF�� equalization�� ������ ���� PDF�� ������ ���ϵ�
	FILE* f_trans_func_eq_YUV; //  transfer function�� ������ ����

	float** PDF_RGB = cal_PDF_RGB(input);		// PDF of Input image(RGB) : [L][3]
	float* CDF_YUV = cal_CDF(Y);				// CDF of Y channel image

	fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+"); // �а� ���� ���� "PDF_RGB.txt"��� �� ������ ����. (������ ���� �ÿ��� ����.)
	fopen_s(&f_equalized_PDF_YUV, "equalized_PDF_YUV.txt", "w+");
	fopen_s(&f_trans_func_eq_YUV, "trans_func_eq_YUV.txt", "w+");

	G trans_func_eq_YUV[L] = { 0 };			// transfer function �迭

	//  Y channel ������ histogram equalization ����
	hist_eq(Y, Y, trans_func_eq_YUV, CDF_YUV);

	// Y���� HE ���� �� �ٽ� ��ġ��(Y, U, V channels)
	merge(channels, 3, equalized_YUV);

	// ��ģ YUV�� �ٽ� RGB(use "CV_YUV2RGB" flag)
	cvtColor(equalized_YUV, equalized_YUV, CV_YUV2RGB);

	// �ٽ� ��ȯ�� RGB�� PDF�� ��� (RGB)�Ͽ� equalized_PDF_YUV�� ����
	float** equalized_PDF_YUV = cal_PDF_RGB(equalized_YUV);

	for (int i = 0; i < L; i++) {
		// ��� ��Ʈ��(text ����)�� ������׷��� �ۼ�
		fprintf(f_PDF_RGB, "%d\t%f\t%f\t%f\n", i, PDF_RGB[i][0], PDF_RGB[i][1], PDF_RGB[i][2]);
		fprintf(f_equalized_PDF_YUV, "%d\t%f\t%f\t%f\n", i, equalized_PDF_YUV[i][0], equalized_PDF_YUV[i][1], equalized_PDF_YUV[i][2]);
		// ��� ��Ʈ��(text ����)�� transfer function�� �ۼ�
		fprintf(f_trans_func_eq_YUV, "%d\t%d\n", i, trans_func_eq_YUV[i]);
	}

	// �޸𸮸� �ݳ��ϰ� ������ �۾� ����
	free(PDF_RGB);
	free(CDF_YUV);
	fclose(f_PDF_RGB);
	fclose(f_equalized_PDF_YUV);
	fclose(f_trans_func_eq_YUV);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE); // â �̸� "RGB"
	imshow("RGB", input); // ���� �̹��� ����

	namedWindow("Equalized_YUV", WINDOW_AUTOSIZE);
	imshow("Equalized_YUV", equalized_YUV);  // HE ������ �̹��� ����

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
		trans_func[i] = (G)((L - 1) * CDF[i]); // output = (L-1) * CDF(input)

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			// input�� transfer function ������� equalized�� ����
			equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}