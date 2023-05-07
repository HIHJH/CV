#include "hist_func.h" // hist_func.h ������� ���� ����

void hist_matching(Mat& input, Mat& reference, Mat& output, G* trans_func, float* input_CDF, float* ref_CDF);

int main() {

    Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR); // �÷� �̹��� �޾ƿ�
    Mat reference = imread("reference.jpg", CV_LOAD_IMAGE_COLOR); 
    
    Mat matched_YUV, ref_YUV;
    cvtColor(input, matched_YUV, CV_RGB2YUV); // RGB�� YUV�� ��ȯ�ϰ� matched_YUV�� ����
    cvtColor(reference, ref_YUV, CV_RGB2YUV);

    // Y, U, V ä�� ������ ������
    Mat matched_channels[3], ref_channels[3];
    split(matched_YUV, matched_channels);
    split(ref_YUV, ref_channels);
    Mat matched_Y = matched_channels[0]; // U = matched_channels[1], V = matched_channels[2]
    Mat ref_Y = ref_channels[0]; // U = ref_channels[1], V = ref_channels[2]
    
    // PDF, transfer function �ؽ�Ʈ ����
    FILE* f_input_PDF_RGB; // ����(�÷� �̹���)�� PDF�� ������ ����
    FILE* f_output_PDF_RGB; //HM�� ������ ���� PDF�� ������ ����
    FILE* f_trans_func_YUV; // transfer function�� ������ ����

    float** input_PDF_RGB = cal_PDF_RGB(input); // PDF of Input image(RGB) : [L][3]
    float* Y_CDF_YUV = cal_CDF(matched_Y); // CDF of Y matched_channel image
    float** ref_PDF_RGB = cal_PDF_RGB(reference); // PDF of reference image(RGB) : [L][3]
    float* ref_CDF_YUV = cal_CDF(ref_Y); // CDF of Y ref_channel image

    fopen_s(&f_input_PDF_RGB, "input_PDF_RGB.txt", "w+"); // �а� ���� ���� "input_PDF.txt"��� �� ������ ����. (������ ���� �ÿ��� ����.)
    fopen_s(&f_output_PDF_RGB, "output_PDF_RGB.txt", "w+");
    fopen_s(&f_trans_func_YUV, "trans_func_YUV.txt", "w+");

    G trans_func_YUV[L] = { 0 }; // transfer function �迭

    // Y channel ������ histogram matching ����
    hist_matching(matched_Y, ref_Y, matched_Y, trans_func_YUV, Y_CDF_YUV, ref_CDF_YUV);

    // Y���� HE ���� �� �ٽ� ��ġ��(Y, U, V channels)
    merge(matched_channels, 3, matched_YUV);

    // ��ģ YUV�� �ٽ� RGB(use "CV_YUV2RGB" flag)
    cvtColor(matched_YUV, matched_YUV, COLOR_YUV2RGB);

    // �ٽ� ��ȯ�� RGB�� PDF�� ��� (RGB)�Ͽ� output_PDF_RGB�� ����
    float** output_PDF_RGB = cal_PDF_RGB(matched_YUV);

    for (int i = 0; i < L; i++) {
        // ��� ��Ʈ��(text ����)�� ������׷��� �ۼ�
        fprintf(f_input_PDF_RGB, "%d\t%f\t%f\t%f\n", i, input_PDF_RGB[i][0], input_PDF_RGB[i][1], input_PDF_RGB[i][2]);
        fprintf(f_output_PDF_RGB, "%d\t%f\t%f\t%f\n", i, output_PDF_RGB[i][0], output_PDF_RGB[i][1], output_PDF_RGB[i][2]);
        // ��� ��Ʈ��(text ����)�� transfer function�� �ۼ�
        fprintf(f_trans_func_YUV, "%d\t%d\n", i, trans_func_YUV[i]);
    }

    // �޸𸮸� �ݳ��ϰ� ������ �۾� ����
    free(input_PDF_RGB);
    free(Y_CDF_YUV);
    free(ref_PDF_RGB);
    free(ref_CDF_YUV);
    free(output_PDF_RGB);
    fclose(f_input_PDF_RGB);
    fclose(f_output_PDF_RGB);
    fclose(f_trans_func_YUV);

    ////////////////////// Show each image ///////////////////////

    namedWindow("Input", WINDOW_AUTOSIZE);  // â �̸� "input"
    imshow("Input", input); // ���� �̹��� ����

    namedWindow("Reference", WINDOW_AUTOSIZE);
    imshow("Reference", reference); // ���۷��� �̹��� ����

    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow("Output", matched_YUV); // HM ������ �̹��� ����

    //////////////////////////////////////////////////////////////

    waitKey(0);

    return 0;
}

//histogram matching
// input: matching �����ϰ��� �ϴ� �̹���, reference: matching�� ��ǥ�� �� �̹���
// output: matching ���� ���� ����� ���⿡ ����, trans_func: matching ������ ���� �Լ�
// input_CDF, ref_CDF: �߰��� CDF ����� �̿�Ǿ� ������ �ҷ���
void hist_matching(Mat& input, Mat& reference, Mat& output, G* trans_func, float* input_CDF, float* ref_CDF) {
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