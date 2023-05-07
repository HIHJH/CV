#include "hist_func.h" // hist_func.h ������� ���� ����

void hist_matching(Mat& input, Mat& reference, Mat& output, G* trans_func, float* input_CDF, float* ref_CDF);

int main() {

    Mat input = imread("input.jpg", CV_LOAD_IMAGE_GRAYSCALE); // ��� �̹��� �޾ƿ�
    Mat reference = imread("reference.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    
    Mat output = input.clone(); // output�� input�� ������ ������ �ʱ�ȭ

    // PDF, transfer function �ؽ�Ʈ ����
    FILE* f_input_PDF; // ����(��� �̹���)�� PDF�� ������ ����
    FILE* f_output_PDF; //HM�� ������ ���� PDF�� ������ ����
    FILE* f_trans_func; // transfer function�� ������ ����

    fopen_s(&f_input_PDF, "input_PDF.txt", "w+"); // �а� ���� ���� "input_PDF.txt"��� �� ������ ����. (������ ���� �ÿ��� ����.)
    fopen_s(&f_output_PDF, "output_PDF.txt", "w+");
    fopen_s(&f_trans_func, "trans_func.txt", "w+");

    float* input_PDF = cal_PDF(input); // PDF of Input image(Grayscale) : [L]
    float* input_CDF = cal_CDF(input); // CDF of Input image(Grayscale) : [L]
    float* ref_PDF = cal_PDF(reference); // PDF of Reference image(Grayscale) : [L]
    float* ref_CDF = cal_CDF(reference); // CDF of Reference image(Grayscale) : [L]
    
    G trans_func[L] = { 0 }; // transfer function �迭

    // ��� �̹����� histogram matching ����
    hist_matching(input, reference, output, trans_func, input_PDF, ref_PDF);
    // HM ���� ���� PDF�� ����Ͽ� output_PDF�� ����
    float* output_PDF = cal_PDF(output);

    for (int i = 0; i < L; i++) {
        // ��� ��Ʈ��(text ����)�� ������׷��� �ۼ�
        fprintf(f_input_PDF, "%d\t%f\n", i, input_PDF[i]);
        fprintf(f_output_PDF, "%d\t%f\n", i, output_PDF[i]);

        // ��� ��Ʈ��(text ����)�� transfer function�� �ۼ�
        fprintf(f_trans_func, "%d\t%d\n", i, trans_func[i]);
    }

    // �޸𸮸� �ݳ��ϰ� ������ �۾� ����
    free(input_PDF);
    free(input_CDF);
    free(ref_PDF);
    free(ref_CDF);
    free(output_PDF);
    fclose(f_input_PDF);
    fclose(f_output_PDF);
    fclose(f_trans_func);

    ////////////////////// Show each image ///////////////////////

    namedWindow("Input", WINDOW_AUTOSIZE); // â �̸� "Input"
    imshow("Input", input); // ��� ���� �̹��� ����

    namedWindow("Reference", WINDOW_AUTOSIZE);
    imshow("Reference", reference); // ��� ���۷��� �̹��� ����

    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow("Output", output); // HM ������ �̹��� ����

    //////////////////////////////////////////////////////////////

    waitKey(0);

    return 0;
}

//histogram matching
// input: matching �����ϰ��� �ϴ� �̹���, reference: matching�� ��ǥ�� �� �̹���
// output: matching ���� ���� ����� ���⿡ ����, trans_func: matching ������ ���� �Լ�
// input_CDF, ref_CDF: �߰��� CDF ����� �̿�Ǿ� ������ �ҷ���
void hist_matching(Mat& input, Mat& reference, Mat& output, G *trans_func, float *input_CDF, float *ref_CDF) {
    // compute the transfer function
    //  r: input, z: output, T,G: Histogram equalization func
    // s = T(r), s = G(z) �� ��, z = G^(-1)*(T(r))
    for (int i = 0; i < L; i++) {
        int j = L - 1;
        do {
            trans_func[i] = j;
            j--;
        } while (j >= 0 && ref_CDF[j] >= input_CDF[i]);
        // ref_CDF[j] �� input_CDF[i] ���� ���ϸ� trans_func �Լ��� �����Ͽ� ���Լ��� ����� �Ͱ� ���� ȿ���� ��
    }

    // perform the transfer function on input image
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            // input�� transfer function ������� output�� ����
            output.at<G>(i, j) = trans_func[input.at<G>(i, j)];
        }
    }
}