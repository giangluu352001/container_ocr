#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include "include/utils.h"

namespace ContainerOCR {
    class DBPostProcessor {
        public:
            std::vector<cv::Point2f> unclip(const std::vector<cv::Point2f> &box, const double& unclip_ratio);
            double box_score_fast(const cv::Mat& bitmap, const std::vector<cv::Point2f> &_box);
            std::vector<cv::Point2f> get_mini_boxes(const cv::InputArray& box, float& sside);
            std::vector<std::vector<cv::Point2f>> boxes_from_bitmap(const cv::Mat &pred, 
                const cv::Mat &bitmap, const int& dest_width, const int& dest_height, 
                const int& max_candidates, const int& min_size, const double& box_thresh, const double &unclip_ratio);
        };

    class SVTRPostProcessor {
        public:
            std::vector<std::vector<CharacterReplacement>> decode(const at::Tensor &pred, const std::string &dict,
                std::vector<std::pair<std::string, float>> &result_list, const int& topk, const int &ignore_index, const float &prob_thresh);
    };
}