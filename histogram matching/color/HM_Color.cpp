#include "hist_func.h" // hist_func.h 헤더파일 내용 포함

void hist_matching(Mat& input, Mat& reference, Mat& output, G* trans_func, float* input_CDF, float* ref_CDF);

int main() {

    Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR); // 컬러 이미지 받아옴
    Mat reference = imread("reference.jpg", CV_LOAD_IMAGE_COLOR); 
    
    Mat matched_YUV, ref_YUV;
    cvtColor(input, matched_YUV, CV_RGB2YUV); // RGB를 YUV로 변환하고 matched_YUV에 저장
    cvtColor(reference, ref_YUV, CV_RGB2YUV);

    // Y, U, V 채널 각각을 나누기
    Mat matched_channels[3], ref_channels[3];
    split(matched_YUV, matched_channels);
    split(ref_YUV, ref_channels);
    Mat matched_Y = matched_channels[0]; // U = matched_channels[1], V = matched_channels[2]
    Mat ref_Y = ref_channels[0]; // U = ref_channels[1], V = ref_channels[2]
    
    // PDF, transfer function 텍스트 파일
    FILE* f_input_PDF_RGB; // 원본(컬러 이미지)의 PDF를 저장할 파일
    FILE* f_output_PDF_RGB; //HM을 수행한 후의 PDF를 저장할 파일
    FILE* f_trans_func_YUV; // transfer function을 저장할 파일

    float** input_PDF_RGB = cal_PDF_RGB(input); // PDF of Input image(RGB) : [L][3]
    float* Y_CDF_YUV = cal_CDF(matched_Y); // CDF of Y matched_channel image
    float** ref_PDF_RGB = cal_PDF_RGB(reference); // PDF of reference image(RGB) : [L][3]
    float* ref_CDF_YUV = cal_CDF(ref_Y); // CDF of Y ref_channel image

    fopen_s(&f_input_PDF_RGB, "input_PDF_RGB.txt", "w+"); // 읽고 쓰기 위해 "input_PDF.txt"라는 빈 파일을 연다. (내용이 있을 시에는 비운다.)
    fopen_s(&f_output_PDF_RGB, "output_PDF_RGB.txt", "w+");
    fopen_s(&f_trans_func_YUV, "trans_func_YUV.txt", "w+");

    G trans_func_YUV[L] = { 0 }; // transfer function 배열

    // Y channel 에서만 histogram matching 수행
    hist_matching(matched_Y, ref_Y, matched_Y, trans_func_YUV, Y_CDF_YUV, ref_CDF_YUV);

    // Y에서 HE 수행 후 다시 합치기(Y, U, V channels)
    merge(matched_channels, 3, matched_YUV);

    // 합친 YUV를 다시 RGB(use "CV_YUV2RGB" flag)
    cvtColor(matched_YUV, matched_YUV, COLOR_YUV2RGB);

    // 다시 변환한 RGB의 PDF를 계산 (RGB)하여 output_PDF_RGB에 저장
    float** output_PDF_RGB = cal_PDF_RGB(matched_YUV);

    for (int i = 0; i < L; i++) {
        // 출력 스트림(text 파일)에 히스토그램을 작성
        fprintf(f_input_PDF_RGB, "%d\t%f\t%f\t%f\n", i, input_PDF_RGB[i][0], input_PDF_RGB[i][1], input_PDF_RGB[i][2]);
        fprintf(f_output_PDF_RGB, "%d\t%f\t%f\t%f\n", i, output_PDF_RGB[i][0], output_PDF_RGB[i][1], output_PDF_RGB[i][2]);
        // 출력 스트림(text 파일)에 transfer function을 작성
        fprintf(f_trans_func_YUV, "%d\t%d\n", i, trans_func_YUV[i]);
    }

    // 메모리를 반납하고 수행한 작업 정리
    free(input_PDF_RGB);
    free(Y_CDF_YUV);
    free(ref_PDF_RGB);
    free(ref_CDF_YUV);
    free(output_PDF_RGB);
    fclose(f_input_PDF_RGB);
    fclose(f_output_PDF_RGB);
    fclose(f_trans_func_YUV);

    ////////////////////// Show each image ///////////////////////

    namedWindow("Input", WINDOW_AUTOSIZE);  // 창 이름 "input"
    imshow("Input", input); // 원본 이미지 띄우기

    namedWindow("Reference", WINDOW_AUTOSIZE);
    imshow("Reference", reference); // 레퍼런스 이미지 띄우기

    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow("Output", matched_YUV); // HM 수행한 이미지 띄우기

    //////////////////////////////////////////////////////////////

    waitKey(0);

    return 0;
}

//histogram matching
// input: matching 수행하고자 하는 이미지, reference: matching의 목표가 될 이미지
// output: matching 수행 후의 결과를 여기에 저장, trans_func: matching 과정을 담은 함수
// input_CDF, ref_CDF: 중간에 CDF 계산이 이용되어 포인터 불러옴
void hist_matching(Mat& input, Mat& reference, Mat& output, G* trans_func, float* input_CDF, float* ref_CDF) {
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