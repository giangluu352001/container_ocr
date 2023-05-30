#include <include/detector.h>

namespace ContainerOCR {
	void DBDetector::Run(cv::Mat &img, std::vector<std::vector<std::vector<int>>> &boxes) {
		const std::vector<int> original_shape {img.rows, img.cols};
		this->resizer.Run(img, this->detection_size);
		this->normalizer.Run(img, this->mean, this->scale);
		std::vector<torch::jit::IValue> input;
		this->permuter.Run(img, input);
		int heightDB = img.rows;
		int widthDB = img.cols;
		torch::NoGradGuard no_grad;
		at::Tensor output = this->predictor.forward(input).toTensor();
		cv::Mat prob_map(heightDB, widthDB, CV_32FC1, output.data_ptr());
		cv::Mat segmentation = prob_map > this->thresh;
		boxes = this->post_processor.boxes_from_bitmap(
			prob_map, segmentation, original_shape[1], original_shape[0], 
			this->max_candidates, this->min_size, this->box_thresh, this->unclip_ratio);
	}
}