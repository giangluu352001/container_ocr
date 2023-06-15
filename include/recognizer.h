#pragma once

#include "include/preprocess.h"
#include "include/postprocess.h"
#include "include/utils.h"

namespace ContainerOCR {
	class SVTRRecognizer {
		public:
			explicit SVTRRecognizer(const std::string& model_path, const double& scale, const float &prob_thresh, const std::string& label_dict,
				const int &topklargest, const int &ignore_index, const int &rec_height, const int &rec_width) {
				this->scale = scale;
				this->label_dict = label_dict;
				this->ignore_index = ignore_index;
				this->topklargest = topklargest;
				this->prob_thresh = prob_thresh;
				this->rec_height = rec_height;
				this->rec_width = rec_width;
				Utils::LoadModel(model_path, this->predictor);
			}
			void Run(std::vector<cv::Mat> &imgs, std::vector<std::pair<std::string, float>> &labels,
				std::vector<std::vector<CharacterReplacement>> &replacements);
		private:
			double scale = 2;
			double mean[3] = { 0.5, 0.5, 0.5 };
			int ignore_index = 0;
			int topklargest = 5;
			float prob_thresh = 0;
			int rec_height = 64;
			int rec_width = 256;
			std::string label_dict;
			torch::jit::script::Module predictor;
			Resizer resizer;
			Normalizer normalizer;
			PermuteBatch permuter;
			SVTRPostProcessor post_processor;
	};
}