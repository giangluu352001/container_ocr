#include <include/contocr.h>
using namespace ContainerOCR;

int main()
{
    ContOCR ocr = ContOCR();
    /*cv::VideoCapture vid_capture("../../../assets/images/container.mp4");
    while (vid_capture.isOpened()) {
        cv::Mat frame;
        bool isSuccess = vid_capture.read(frame);
        if (isSuccess == true)
        {
            auto start = std::chrono::high_resolution_clock::now();
            ocr.Run(frame);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
            std::cout << "Time taken by function: " << duration << " milliseconds" << std::endl;
        }
        if (isSuccess == false) {
            std::cout << "Video camera is disconnected" << std::endl;
            break;
        }
        int key = cv::waitKey(20);
        if (key == 'q') {
            std::cout << "q key is pressed by the user. Stopping the video" << std::endl;
            break;
        }
    }
    vid_capture.release();
    cv::destroyAllWindows();*/
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat img = cv::imread("../../../assets/images/best.jpg", cv::IMREAD_COLOR);
    ocr.Run(img);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    std::cout << "Time taken by function: " << duration << " milliseconds" << std::endl;
    return 0;
}