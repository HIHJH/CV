#include "hist_func.h" // hist_func.h 헤더파일 내용 포함

void hist_matching(Mat& input, Mat& reference, Mat& output, G* trans_func, float* input_CDF, float* ref_CDF);

int main() {

    Mat input = imread("input.jpg", CV_LOAD_IMAGE_GRAYSCALE); // 흑백 이미지 받아옴
    Mat reference = imread("reference.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    
    Mat output = input.clone(); // output은 input을 복제한 것으로 초기화

    // PDF, transfer function 텍스트 파일
    FILE* f_input_PDF; // 원본(흑백 이미지)의 PDF를 저장할 파일
    FILE* f_output_PDF; //HM을 수행한 후의 PDF를 저장할 파일
    FILE* f_trans_func; // transfer function을 저장할 파일

    fopen_s(&f_input_PDF, "input_PDF.txt", "w+"); // 읽고 쓰기 위해 "input_PDF.txt"라는 빈 파일을 연다. (내용이 있을 시에는 비운다.)
    fopen_s(&f_output_PDF, "output_PDF.txt", "w+");
    fopen_s(&f_trans_func, "trans_func.txt", "w+");

    float* input_PDF = cal_PDF(input); // PDF of Input image(Grayscale) : [L]
    float* input_CDF = cal_CDF(input); // CDF of Input image(Grayscale) : [L]
    float* ref_PDF = cal_PDF(reference); // PDF of Reference image(Grayscale) : [L]
    float* ref_CDF = cal_CDF(reference); // CDF of Reference image(Grayscale) : [L]
    
    G trans_func[L] = { 0 }; // transfer function 배열

    // 흑백 이미지의 histogram matching 수행
    hist_matching(input, reference, output, trans_func, input_PDF, ref_PDF);
    // HM 수행 후의 PDF를 계산하여 output_PDF에 저장
    float* output_PDF = cal_PDF(output);

    for (int i = 0; i < L; i++) {
        // 출력 스트림(text 파일)에 히스토그램을 작성
        fprintf(f_input_PDF, "%d\t%f\n", i, input_PDF[i]);
        fprintf(f_output_PDF, "%d\t%f\n", i, output_PDF[i]);

        // 출력 스트림(text 파일)에 transfer function을 작성
        fprintf(f_trans_func, "%d\t%d\n", i, trans_func[i]);
    }

    // 메모리를 반납하고 수행한 작업 정리
    free(input_PDF);
    free(input_CDF);
    free(ref_PDF);
    free(ref_CDF);
    free(output_PDF);
    fclose(f_input_PDF);
    fclose(f_output_PDF);
    fclose(f_trans_func);

    ////////////////////// Show each image ///////////////////////

    namedWindow("Input", WINDOW_AUTOSIZE); // 창 이름 "Input"
    imshow("Input", input); // 흑백 원본 이미지 띄우기

    namedWindow("Reference", WINDOW_AUTOSIZE);
    imshow("Reference", reference); // 흑백 레퍼런스 이미지 띄우기

    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow("Output", output); // HM 수행한 이미지 띄우기

    //////////////////////////////////////////////////////////////

    waitKey(0);

    return 0;
}

//histogram matching
// input: matching 수행하고자 하는 이미지, reference: matching의 목표가 될 이미지
// output: matching 수행 후의 결과를 여기에 저장, trans_func: matching 과정을 담은 함수
// input_CDF, ref_CDF: 중간에 CDF 계산이 이용되어 포인터 불러옴
void hist_matching(Mat& input, Mat& reference, Mat& output, G *trans_func, float *input_CDF, float *ref_CDF) {
    // compute the transfer function
    //  r: input, z: output, T,G: Histogram equalization func
    // s = T(r), s = G(z) 일 때, z = G^(-1)*(T(r))
    for (int i = 0; i < L; i++) {
        int j = L - 1;
        do {
            trans_func[i] = j;
            j--;
        } while (j >= 0 && ref_CDF[j] >= input_CDF[i]);
        // ref_CDF[j] 와 input_CDF[i] 값을 비교하며 trans_func 함수를 설정하여 역함수를 계산한 것과 같은 효과를 냄
    }

    // perform the transfer function on input image
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            // input에 transfer function 적용시켜 output에 저장
            output.at<G>(i, j) = trans_func[input.at<G>(i, j)];
        }
    }
}