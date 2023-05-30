#include <include/recognizer.h>

namespace ContainerOCR {
	void SVTRRecognizer::Run(cv::Mat &img, std::vector<std::pair<std::string, float>> &labels,
		std::vector<std::vector<CharacterReplacement>> &replacements) {
		this->normalizer.Run(img, this->mean, this->scale);
		std::vector<torch::jit::IValue> input;
		this->permuter.Run(img, input);
		torch::NoGradGuard no_grad;
		at::Tensor output = this->predictor.forward(input).toTensor();
		replacements = this->post_processor.decode(output, this->label_dict, labels, this->topklargest, this->ignore_index, this->prob_thresh);
	}
}