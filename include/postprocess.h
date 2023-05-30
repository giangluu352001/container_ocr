#pragma once

#include "opencv2/opencv.hpp"
#include <torch/script.h>
#include <vector>
#include <include/utils.h>

namespace ContainerOCR {
    class DBPostProcessor {
        public:
            std::vector<cv::Point2f> unclip(const std::vector<std::vector<float>>& box, const double& unclip_ratio);
            double box_score_fast(const cv::Mat& bitmap, const std::vector<std::vector<float>>& _box);
            std::vector<std::vector<float>> get_mini_boxes(const cv::InputArray& box, float& sside);
            std::vector<std::vector<std::vector<int>>> boxes_from_bitmap(const cv::Mat& pred, const cv::Mat& bitmap, 
                const int& dest_width, const int& dest_height, const int& max_candidates, 
                const int& min_size, const double& box_thresh, const double &unclip_ratio);
        };

    class SVTRPostProcessor {
        public:
            std::vector<std::vector<CharacterReplacement>> decode(const at::Tensor &pred, const std::string &dict,
                std::vector<std::pair<std::string, float>> &result_list, const int& topk, const int &ignore_index, const float &prob_thresh);
    };
}