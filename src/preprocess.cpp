#include "include/preprocess.h"

namespace ContainerOCR {
    void Resizer::Run(cv::Mat &img, const std::pair<int, int> &resized_shape, const bool &keep_ratio) {
        int imgH = img.rows, imgW = img.cols;
        int resized_H, resized_W;
        if (keep_ratio) {
            int resized_size = std::min(resized_shape.first, resized_shape.second);
            if (imgH < imgW) {
                resized_H = int(ceilf(float(resized_size) / 32) * 32);
                resized_W = int(ceilf((resized_H / float(imgH)) * float(imgW) / 32) * 32);
            }
            else {
                resized_W = int(ceilf(float(resized_size) / 32) * 32);
                resized_H = int(ceilf((resized_W / float(imgW)) * float(imgH) / 32) * 32);
            }
        }
        else {
            resized_H = resized_shape.first;
            resized_W = resized_shape.second;
        }
        cv::resize(img, img, cv::Size(resized_W, resized_H));
    }
    void Normalizer::Run(cv::Mat &img, const double BGR_MEAN[3], const double &scale) {
        img.convertTo(img, CV_32FC3, 1.0 / 255.0);
        cv::subtract(img, cv::Scalar(BGR_MEAN[0], BGR_MEAN[1], BGR_MEAN[2]), img);
        img *= scale;
    }
    void PermuteBatch::Run(const std::vector<cv::Mat> &imgs, std::vector<torch::jit::IValue> &data) {
        std::vector<torch::Tensor> tensorVec;
	    torch::TensorOptions options = torch::TensorOptions{torch::kFloat32};
        for (auto& img : imgs) {
            int height = img.rows;
            int width = img.cols;
            std::vector<int64_t> sizes = {height, width, img.channels() };
            torch::Tensor input_tensor = torch::from_blob(img.data, at::IntList(sizes), options).to(at::kCUDA);
            tensorVec.push_back(input_tensor.permute({ 2, 0, 1 }));
        }
        data.push_back(torch::stack(tensorVec, 0));
    }
}
