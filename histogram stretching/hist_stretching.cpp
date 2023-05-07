#include "hist_func.h" // hist_func.h ������� ���� ����

void linear_stretching(Mat& input, Mat& stretched, G* trans_func, G x1, G x2, G y1, G y2);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR); // �÷� �̹��� �޾ƿ�
	Mat input_gray;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// RGB�� GrayScale���� ��ȯ

	Mat stretched = input_gray.clone(); // stretched matrix�� input_gray�� ������ ������ �ʱ�ȭ

	// PDF, transfer function �ؽ�Ʈ ����
	FILE* f_PDF; // ������ PDF�� ������ ����
	FILE* f_stretched_PDF; //stretching�� ������ ���� PDF�� ������ ����
	FILE* f_trans_func_stretch; // transfer function�� ������ ����

	fopen_s(&f_PDF, "PDF.txt", "w+"); // �а� ���� ���� "PDF.txt"��� �� ������ ����. (������ ���� �ÿ��� ����.)
	fopen_s(&f_stretched_PDF, "stretched_PDF.txt", "w+");
	fopen_s(&f_trans_func_stretch, "trans_func_stretch.txt", "w+");

	G trans_func_stretch[L] = { 0 }; // transfer function �迭

	float* PDF = cal_PDF(input_gray); // PDF of Input image(Grayscale) : [L]

	//  histogram stretching (50 ~ 110 -> 10 ~ 110)�� ����
	linear_stretching(input_gray, stretched, trans_func_stretch, 50, 110, 10, 110);
	// stretching ���� ���� PDF�� ����Ͽ� stretched_PDF�� ����
	float* stretched_PDF = cal_PDF(stretched);

	for (int i = 0; i < L; i++) {
		// ��� ��Ʈ��(text ����)�� ������׷��� �ۼ�
		fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);
		fprintf(f_stretched_PDF, "%d\t%f\n", i, stretched_PDF[i]);

		// ��� ��Ʈ��(text ����)�� transfer function�� �ۼ�
		fprintf(f_trans_func_stretch, "%d\t%d\n", i, trans_func_stretch[i]);
	}

	// �޸𸮸� �ݳ��ϰ� ������ �۾� ����
	free(PDF);
	free(stretched_PDF);
	fclose(f_PDF);
	fclose(f_stretched_PDF);
	fclose(f_trans_func_stretch);

	////////////////////// Show each image ///////////////////////

	namedWindow("Grayscale", WINDOW_AUTOSIZE); // â �̸� "Grayscale"
	imshow("Grayscale", input_gray); // ��� �̹��� ����

	namedWindow("Stretched", WINDOW_AUTOSIZE);
	imshow("Stretched", stretched); // stretching ������ �̹��� ����

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram stretching (linear method)
// input: stretching �����ϰ��� �ϴ� �̹���, stretched: stretching ���� ���� ����� ���⿡ ����
// trans_func: stretching ������ ���� �Լ�, (x1,x2) -> (y1, y2)�� stretch �ϰ��� ��
void linear_stretching(Mat& input, Mat& stretched, G* trans_func, G x1, G x2, G y1, G y2) {

	float constant = (y2 - y1) / (float)(x2 - x1); // linear stretching function�� ���⸦ constant�� ���� 

	// compute transfer function
	// linear stretching function�� ���� -> (a(i), b(i), a(i+1), b(i+2)) �� ���
	// y = {( b(i+1) - b(i) ) / ( a(i+1) - a(i) )} ( x - a(i) ) + b(i)
	for (int i = 0; i < L; i++) {
		if (i >= 0 && i <= x1) // x1 ���� ���� ���
			trans_func[i] = (G)(y1 / x1 * i); // 
		else if (i > x1 && i <= x2) // x1 ~ x2 ������ ���
			trans_func[i] = (G)(constant * (i - x1) + y1);
		else // x2 ���� ū ���
			trans_func[i] = (G)((L - 1 - x2) / (L - 1 - y2) * (i - x2) + y2);
	}

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			// input�� transfer function ������� stretched�� ����
			stretched.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}