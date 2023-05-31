#include <include/params.h>
#include <include/contocr.h>

namespace ContainerOCR {
	ContOCR::ContOCR() {
		this->ratio_vertical = ratio_vertical;

		this->detector = new DBDetector(det_model_path, det_db_thresh, det_db_box_thresh,
			det_max_candidates, det_scale, det_db_unclip_ratio, det_text_min_size, det_db_size);
		this->horizontal_recognizer = new SVTRRecognizer(rec_horizontal_model_path, 
			rec_scale, prob_thesh, label_dict, topklargest, ignore_index);
		this->vertical_recognizer = new SVTRRecognizer(rec_vertical_model_path,
			rec_scale, prob_thesh, label_dict, topklargest, ignore_index);
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
	void ContOCR::Run(cv::Mat img) {
		cv::Mat originalImage;
		img.copyTo(originalImage);
		std::vector<std::vector<cv::Point2f>> boxes;
		std::vector<OCRResult> results;
		auto start0 = std::chrono::high_resolution_clock::now();
		this->detector->Run(img, boxes);
		auto stop0 = std::chrono::high_resolution_clock::now();
		auto duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(stop0 - start0).count();
		std::cout << "Time taken by function detector: " << duration0 << " milliseconds" << std::endl;
		auto start1 = std::chrono::high_resolution_clock::now();
		for (const auto &box : boxes) {
			auto cropped = Utils::get_rotate_crop_image(originalImage, box);
			bool isVertical = false;
			if (cropped.rows > this->ratio_vertical * cropped.cols) {
				cv::rotate(cropped, cropped, cv::ROTATE_90_COUNTERCLOCKWISE);
				isVertical = true;
			}
			std::vector<std::pair<std::string, float>> labels;
			std::vector<std::vector<CharacterReplacement>> replacements;
			if (isVertical) this->vertical_recognizer->Run(cropped, labels, replacements);
			else this->horizontal_recognizer->Run(cropped, labels, replacements);
			OCRResult result = OCRResult(box, isVertical, replacements[0], labels[0]);
			results.push_back(result);
		}
		auto stop1 = std::chrono::high_resolution_clock::now();
		auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1).count();
		std::cout << "Time taken by function recognizer: " << duration1 << " milliseconds" << std::endl;
		auto start2 = std::chrono::high_resolution_clock::now();
		auto final = this->extraction->clusterBoxes(results);
		auto stop2 = std::chrono::high_resolution_clock::now();
		auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2).count();
		std::cout << "Time taken by function cluster: " << duration2 << " milliseconds" << std::endl;
		std::cout << "Number of codes: " << final.size() << std::endl;
		for (const auto &x : final)
			std::cout << "Code: " << x << std::endl;
	}
}