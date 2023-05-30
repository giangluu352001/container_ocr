#include <include/preprocess.h>

namespace ContainerOCR {
    void Resizer::Run(cv::Mat &img, const int &resized_size) {
        int imgH = img.rows, imgW = img.cols;
        int resized_H, resized_W;
        if (imgH < imgW) {
            resized_H = int(ceilf(float(resized_size) / 32) * 32);
            resized_W = int(ceilf((resized_H / float(imgH)) * float(imgW) / 32) * 32);
        }
        else {
            resized_W = int(ceilf(float(resized_size) / 32) * 32);
            resized_H = int(ceilf((resized_W / float(imgW)) * float(imgH) / 32) * 32);
        }
        cv::resize(img, img, cv::Size(resized_W, resized_H));
    }
    void Normalizer::Run(cv::Mat &img, const double BGR_MEAN[3], const double &scale) {
        img.convertTo(img, CV_32FC3, 1.0 / 255.0);
        cv::subtract(img, cv::Scalar(BGR_MEAN[0], BGR_MEAN[1], BGR_MEAN[2]), img);
        img *= scale;
    }
    void PermuteBatch::Run(const cv::Mat &img, std::vector<torch::jit::IValue> &data) {
        int height = img.rows;
        int width = img.cols;
        std::vector<int64_t> sizes = { 1, height, width, img.channels() };
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input_tensor = torch::from_blob(img.data, at::IntList(sizes), options);
        data.push_back(input_tensor.permute({ 0, 3, 1, 2 }));
    }
}
