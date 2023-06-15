#include "include/detector.h"

namespace ContainerOCR {
	void DBDetector::Run(cv::Mat &img, std::vector<std::vector<cv::Point2f>> &boxes) {
		const std::vector<int> original_shape {img.rows, img.cols};
		this->resizer.Run(img, std::make_pair(this->detection_size, this->detection_size), true);
		this->normalizer.Run(img, this->mean, this->scale);
		std::vector<torch::jit::IValue> input;
		this->permuter.Run({ img }, input);
		int heightDB = img.rows;
		int widthDB = img.cols;
		torch::NoGradGuard no_grad;
		//auto start1 = std::chrono::high_resolution_clock::now();
		at::Tensor output = this->predictor.forward(input).toTensor().to(at::kCPU);
		//auto stop1 = std::chrono::high_resolution_clock::now();
		//auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1).count();
		//std::cout << "Time taken by function forward detector: " << duration1 << " milliseconds" << std::endl;
		//auto start2 = std::chrono::high_resolution_clock::now();
		cv::Mat prob_map(heightDB, widthDB, CV_32FC1, output.data_ptr());
		cv::Mat segmentation = prob_map > this->thresh;
		boxes = this->post_processor.boxes_from_bitmap(
			prob_map, segmentation, original_shape[1], original_shape[0], 
			this->max_candidates, this->min_size, this->box_thresh, this->unclip_ratio);
		//auto stop2 = std::chrono::high_resolution_clock::now();
		//auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2).count();
		//std::cout << "Time taken by function postprocessing detector: " << duration2 << " milliseconds" << std::endl;
	}
}