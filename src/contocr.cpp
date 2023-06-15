#include "include/params.h"
#include "include/contocr.h"

namespace ContainerOCR {
	ContOCR::ContOCR() {
		this->ratio_vertical = ratio_vertical;

		this->detector = new DBDetector(det_model_path, det_db_thresh, det_db_box_thresh,
			det_max_candidates, det_scale, det_db_unclip_ratio, det_text_min_size, det_db_size);
		this->horizontal_recognizer = new SVTRRecognizer(rec_horizontal_model_path, 
			rec_scale, prob_thesh, label_dict, topklargest, ignore_index, rec_height, rec_width);
		this->vertical_recognizer = new SVTRRecognizer(rec_vertical_model_path,
			rec_scale, prob_thesh, label_dict, topklargest, ignore_index, rec_height, rec_width);
		this->extraction = new ContExtraction(alphabet, cluster_max_chars, max_len_code, max_len_seri_number);
	}
	ContOCR::~ContOCR() {
		if (this->detector != nullptr) {
			delete this->detector;
		}
		if (this->horizontal_recognizer != nullptr) {
			delete this->horizontal_recognizer;
		}
		if (this->vertical_recognizer != nullptr) {
			delete this->vertical_recognizer;
		}
		if (this->extraction != nullptr) {
			delete this->extraction;
		}
	}
	std::vector<std::pair<std::string, float>> ContOCR::Run(const cv::Mat &img) {
		cv::Mat processedImage;
		img.copyTo(processedImage);
		std::vector<std::vector<cv::Point2f>> boxes;
		std::vector<OCRResult> results;
		//auto start0 = std::chrono::high_resolution_clock::now();
		this->detector->Run(processedImage, boxes);
		if (boxes.empty()) return {};
		//auto stop0 = std::chrono::high_resolution_clock::now();
		//auto duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(stop0 - start0).count();
		//std::cout << "Time taken by function detector: " << duration0 << " milliseconds" << std::endl;
		std::vector<cv::Mat> horizontal_images;
		std::vector<cv::Mat> vertical_images;
		std::vector<bool> isVertical;
		for (const auto &box: boxes) {
			auto cropped = Utils::get_rotate_crop_image(img, box);
			if (cropped.rows > this->ratio_vertical * cropped.cols) {
				cv::rotate(cropped, cropped, cv::ROTATE_90_COUNTERCLOCKWISE);
				vertical_images.push_back(cropped);
				isVertical.push_back(true);
			}
			else {
				horizontal_images.push_back(cropped);
				isVertical.push_back(false);
			}
		}
		std::vector<std::pair<std::string, float>> labels;
		std::vector<std::vector<CharacterReplacement>> replacements;
		//auto start1 = std::chrono::high_resolution_clock::now();
		if (!vertical_images.empty()) this->vertical_recognizer->Run(vertical_images, labels, replacements);
		if (!horizontal_images.empty()) this->horizontal_recognizer->Run(horizontal_images, labels, replacements);
		//auto stop1 = std::chrono::high_resolution_clock::now();
		//auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1).count();
		//std::cout << "Time taken by function recognizer: " << duration1 << " milliseconds" << std::endl;
		int num_of_ids = 0;
		//auto start2 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < labels.size(); i++) {
			OCRResult result = OCRResult(boxes[i], isVertical[i], replacements[i], labels[i]);
			num_of_ids += result.label.first.length();
			results.push_back(result);
		}
		auto codes = this->extraction->clusterBoxes(results, num_of_ids);
		//auto stop2 = std::chrono::high_resolution_clock::now();
		//auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2).count();
		//std::cout << "Time taken by function cluster: " << duration2 << " milliseconds" << std::endl;
		return codes;
	}
}