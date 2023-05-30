#include <include/contocr.h>
using namespace ContainerOCR;

int main()
{
    ContOCR ocr = ContOCR();
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat img = cv::imread("../../../assets/images/double_id.jpg", cv::IMREAD_COLOR);
    ocr.Run(img);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    std::cout << "Time taken by function: " << duration << " milliseconds" << std::endl;
}