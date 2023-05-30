#pragma once

#include "opencv2/opencv.hpp"
#include <torch/script.h>

namespace ContainerOCR {
	class Resizer {
		public:
			virtual void Run(cv::Mat &img, const int& resized_size);
	};
	class Normalizer {
		public:
			virtual void Run(cv::Mat &img, const double BGR_MEAN[3], const double &scale);
	};
	class PermuteBatch {
		public:
			virtual void Run(const cv::Mat &img, std::vector<torch::jit::IValue> &data);
	};
}