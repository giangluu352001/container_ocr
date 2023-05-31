#pragma once

#include <include/preprocess.h>
#include <include/postprocess.h>
#include <include/utils.h>

namespace ContainerOCR {
	class DBDetector {
		public:
			explicit DBDetector(const std::string &model_path, const double &thresh,
				const double &box_thresh, const int &max_candidates, const double &scale,
				const double &unclip_ratio, const int &min_size, const int &detection_size) {
				this->thresh = thresh;
				this->box_thresh = box_thresh;
				this->max_candidates = max_candidates;
				this->unclip_ratio = unclip_ratio;
				this->min_size = min_size;
				this->detection_size = detection_size;
				this->scale = scale;
				Utils::LoadModel(model_path, this->predictor);
			}
			void Run(cv::Mat& img, std::vector<std::vector<cv::Point2f>> &boxes);
		private:
			double thresh = 0.3;
			double box_thresh = 0.7;
			int max_candidates = 1000;
			double unclip_ratio = 1.5;
			int min_size = 3;
			int detection_size = 640;
			double mean[3] = { 0.4810937817254901, 0.4575245789019607, 0.4078705409019607 };
			double scale = 1;

			torch::jit::script::Module predictor;
			Resizer resizer;
			Normalizer normalizer;
			PermuteBatch permuter;
			DBPostProcessor post_processor;
	};
}