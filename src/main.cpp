#include "include/contocr.h"
using namespace ContainerOCR;

int main(int argc, char** argv)
{
    if (torch::cuda::is_available()) {
        std::cout << "Only support GPU inference" << std::endl;
        ContOCR ocr = ContOCR();
        cv::VideoCapture vid_capture(argv[1]);
        if (!vid_capture.isOpened()) {
            std::cout << "Error opening video stream or file" << std::endl;
            return -1;
        }
        int i = 0;
        cv::Mat frame;
        while (vid_capture.isOpened()) {
            std::cout << "Frame: " << i << std::endl;
            vid_capture >> frame;
            if(frame.empty()) break;
            else {
                auto start = std::chrono::high_resolution_clock::now();
                auto result = ocr.Run(frame);
                //std::vector<std::pair<std::string, float>> result {{"OOLU9785423", 0.9654}};
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
                if (!result.empty()) {
                    for (const auto &code : result) {
                      std::cout << "Code: " << code << std::endl;
                      //cv::putText(frame, code.first, cv::Point(1035, 550), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(0,255,255), 1);
                    }
                    std::cout << "Time taken: " << duration << " milliseconds" << std::endl;
                }
                //std::cout << cv::getTextSize("OOLU9785423", cv::FONT_HERSHEY_DUPLEX, 0.8, 1, NULL) << std::endl;
                //cv::imshow("container", frame);
            }
            i += 1;
            if(cv::waitKey(30) >= 0) break;
        }
        vid_capture.release();
        cv::destroyAllWindows();
    }
    else {
        std::cerr << "GPU is not available" << std::endl;
    }
    return 0;
}