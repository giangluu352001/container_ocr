#pragma once

#include "opencv2/opencv.hpp"
#include <torch/script.h>
#include <torchvision/vision.h>

namespace ContainerOCR {
	struct CharacterReplacement {
		char c;
		float prob;
		int index;
		CharacterReplacement(const char& c, const float& prob, const int& index) :
			c(c), prob(prob), index(index) {}
		bool operator < (const CharacterReplacement &other) const {
			return prob < other.prob;
		}
	};
	struct OCRResult {
		std::vector<std::vector<int>> box;
		bool isVertical;
		std::vector<CharacterReplacement> replacement;
		std::pair<std::string, float> label;
		OCRResult(const std::vector<std::vector<int>> &box, const bool &isVertical,
			const std::vector<CharacterReplacement> &replacement, const std::pair<std::string, float> &label) :
			box(box), isVertical(isVertical), replacement(replacement), label(label) {}
	};

	class Utils {
		public:
			static std::vector<std::vector<float>> Mat2Vector(const cv::Mat &mat);
			static std::vector<std::vector<std::vector<float>>> Tensor3D2Vector(const at::Tensor &tensor);
			static cv::Mat get_rotate_crop_image(const cv::Mat &srcimage, const std::vector<std::vector<int>> &box);
			static void LoadModel(const std::string &model_dir, torch::jit::script::Module &module);
			static std::vector<int> argsort(const std::vector<float>& array);
	};
}