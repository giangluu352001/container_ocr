#pragma once

#include <opencv2/opencv.hpp>
#include <torch/script.h>

namespace ContainerOCR {
	class Resizer {
		public:
			virtual void Run(cv::Mat &img, const std::pair<int, int> &resized_size, const bool &keep_ratio);
	};
	class Normalizer {
		public:
			virtual void Run(cv::Mat &img, const double BGR_MEAN[3], const double &scale);
	};
	class PermuteBatch {
		public:
			virtual void Run(const std::vector<cv::Mat> &imgs, std::vector<torch::jit::IValue> &data);
	};
}