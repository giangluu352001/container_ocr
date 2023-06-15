#include "include/recognizer.h"

namespace ContainerOCR {
	void SVTRRecognizer::Run(std::vector<cv::Mat> &imgs, std::vector<std::pair<std::string, float>> &labels,
		std::vector<std::vector<CharacterReplacement>> &replacements) {
		for (auto& img : imgs) {
			this->resizer.Run(img, std::make_pair(this->rec_height, this->rec_width), false);
			this->normalizer.Run(img, this->mean, this->scale);
		}
		std::vector<torch::jit::IValue> input;
		this->permuter.Run(imgs, input);
		torch::NoGradGuard no_grad;
		//auto start1 = std::chrono::high_resolution_clock::now();
		at::Tensor output = this->predictor.forward(input).toTensor().to(at::kCPU);
		//auto stop1 = std::chrono::high_resolution_clock::now();
		//auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1).count();
		//std::cout << "Time taken by function forward recognizer: " << duration1 << " milliseconds" << std::endl;
		//auto start2 = std::chrono::high_resolution_clock::now();
		replacements = this->post_processor.decode(output, this->label_dict, 
			labels, this->topklargest, this->ignore_index, this->prob_thresh);
		//auto stop2 = std::chrono::high_resolution_clock::now();
		//auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2).count();
		//std::cout << "Time taken by function postprocessing recognizer: " << duration2 << " milliseconds" << std::endl;

	}
}