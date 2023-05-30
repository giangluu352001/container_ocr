#include <include/postprocess.h>
#include <include/utils.h>
#include <clipper2/clipper.h>

namespace ContainerOCR {
    std::vector<cv::Point2f> DBPostProcessor::unclip(const std::vector<std::vector<float>> &box, const double &unclip_ratio) {
        Clipper2Lib::Path64 poly;
        for (int i = 0; i < box.size(); i++) {
            poly.push_back(Clipper2Lib::Point64(int(box[i][0]), int(box[i][1])));
        }
        double distance = Area(poly) * unclip_ratio / Clipper2Lib::Length(poly);
        Clipper2Lib::ClipperOffset offset;
        offset.AddPath(poly, Clipper2Lib::JoinType::Round, Clipper2Lib::EndType::Polygon);
        Clipper2Lib::Paths64 expanded;
        offset.Execute(distance, expanded);
        std::vector<cv::Point2f> points;
        for (int j = 0; j < expanded.size(); j++) {
            for (int i = 0; i < expanded[expanded.size() - 1].size(); i++) {
                points.emplace_back(cv::Point2f(static_cast<float>(expanded[j][i].x),
                    static_cast<float>(expanded[j][i].y)));
            }
        }
        return points;
    }
    double DBPostProcessor::box_score_fast(const cv::Mat &bitmap, const std::vector<std::vector<float>> &box) {
        int h = bitmap.rows, w = bitmap.cols;
        float box_x[4] = { box[0][0], box[1][0], box[2][0], box[3][0] };
        float box_y[4] = { box[0][1], box[1][1], box[2][1], box[3][1] };

        int xmin = std::clamp(int(floor(*(std::min_element(box_x, box_x + 4)))), 0, w - 1);
        int xmax = std::clamp(int(ceil(*(std::max_element(box_x, box_x + 4)))), 0, w - 1);
        int ymin = std::clamp(int(floor(*(std::min_element(box_y, box_y + 4)))), 0, h - 1);
        int ymax = std::clamp(int(ceil(*(std::max_element(box_y, box_y + 4)))), 0, h - 1);
        
        int mask_width = xmax - xmin + 1;
        int mask_height = ymax - ymin + 1;
        cv::Mat mask = cv::Mat::zeros(mask_height, mask_width, CV_8UC1);

        cv::Point root_point[4];
        root_point[0] = cv::Point(int(box[0][0]) - xmin, int(box[0][1]) - ymin);
        root_point[1] = cv::Point(int(box[1][0]) - xmin, int(box[1][1]) - ymin);
        root_point[2] = cv::Point(int(box[2][0]) - xmin, int(box[2][1]) - ymin);
        root_point[3] = cv::Point(int(box[3][0]) - xmin, int(box[3][1]) - ymin);
        const cv::Point* ppt[1] = { root_point };
        int npt[] = { 4 };
        cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));
        cv::Mat croppedImg = bitmap(cv::Rect(xmin, ymin, mask_width, mask_height));
        auto score = cv::mean(croppedImg, mask)[0];
        return score;
    }
    std::vector<std::vector<float>> DBPostProcessor::get_mini_boxes(const cv::InputArray &box, float &sside) {
        cv::RotatedRect bounding_box = cv::minAreaRect(box);
        sside = std::min(bounding_box.size.width, bounding_box.size.height);
        cv::Mat points;
        cv::boxPoints(bounding_box, points);
        auto array = Utils::Mat2Vector(points);
        std::sort(array.begin(), array.end(),
            [](const std::vector<float> &a, const std::vector<float> &b) {
                return a[1] > b[1];
        });
        float leftNeighbor = std::atan2f(std::abs(array[1][1] - array[0][1]), std::abs(array[1][0] - array[0][0]));
        float rightNeighbor = std::atan2f(std::abs(array[2][1] - array[0][1]), std::abs(array[2][0] - array[0][0]));
        std::vector<std::vector<float>> bottom;
        std::vector<std::vector<float>> top;
        if (rightNeighbor < leftNeighbor) {
            bottom = { array[0], array[2] };
            top = { array[3], array[1] };
        }
        else {
            bottom = { array[0], array[1] };
            top = { array[3], array[2] };
        }
        sort(top.begin(), top.end(),
            [](const std::vector<float> &a, const std::vector<float> &b) {
                return a[0] < b[0];
        });
        sort(bottom.begin(), bottom.end(),
            [](const std::vector<float> &a, const std::vector<float> &b) {
                return a[0] < b[0];
        });
        array = { top[0], top[1], bottom[1], bottom[0] };
        return array;
    }
    std::vector<std::vector<std::vector<int>>> DBPostProcessor::boxes_from_bitmap(const cv::Mat &pred, 
        const cv::Mat &bitmap, const int &dest_width, const int &dest_height, 
        const int &max_candidates, const int &min_size, const double &box_thresh, const double &unclip_ratio) {
        const int width = bitmap.cols, height = bitmap.rows;
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        const size_t num_contours = std::min(contours.size(), static_cast<size_t>(max_candidates));
        std::vector<std::vector<std::vector<int>>> boxes;
        boxes.reserve(num_contours);
        for (const auto &contour : contours) {
            float sside;
            const auto points = this->get_mini_boxes(contour, sside);
            if (sside < min_size) {
                continue;
            }
            const auto score = this->box_score_fast(pred, points);
            if (box_thresh > score) {
                continue;
            }
            const auto clipbox = this->unclip(points, unclip_ratio);
            const auto box = this->get_mini_boxes(clipbox, sside);
            if (sside < min_size + 2) {
                continue;
            }

            std::vector<std::vector<int>> temp_box;
            temp_box.reserve(4);
            const float width_ratio = float(dest_width) / width;
            const float height_ratio = float(dest_height) / height;
            for (const auto &point : box) {
                const int x = std::clamp(int(point[0] * width_ratio), 0, dest_width);
                const int y = std::clamp(int(point[1] * height_ratio), 0, dest_height);
                temp_box.push_back({ x, y });
            }
            boxes.push_back(temp_box);
        }
        return boxes;
    }

    std::vector<std::vector<CharacterReplacement>> SVTRPostProcessor::decode(const at::Tensor &pred, const std::string& dict,
        std::vector<std::pair<std::string, float>> &result_list, const int &topk, const int &ignore_index, const float &prob_thresh) {
        std::vector<std::vector<std::vector<float>>> probs_sequence = Utils::Tensor3D2Vector(pred);
        std::vector<std::vector<CharacterReplacement>> replacements;
        std::vector<int> TempIndices(topk, 0);
        for (const auto& batch : probs_sequence) {
            std::vector<char> char_list;
            std::vector<float> conf_list;
            std::vector<CharacterReplacement> replacement;
            int index = 0;
            for (int batch_idx = 0; batch_idx < batch.size(); batch_idx++) {
                auto indices = Utils::argsort(batch[batch_idx]);
                std::vector<int> TopkIndices(indices.begin(), indices.begin() + topk);
                if (TopkIndices[0] == ignore_index || (batch_idx > 0 && 
                    TempIndices[0] == TopkIndices[0])) {
                    TempIndices = TopkIndices;
                    continue;
                }
                for (int idx = 0; idx < topk; idx++) {
                    if (idx == 0) {
                        char_list.push_back(dict[TopkIndices[idx]]);
                        if (batch[batch_idx][TopkIndices[idx]]) {
                            conf_list.push_back(batch[batch_idx][TopkIndices[idx]]);
                        }
                        else conf_list.push_back(1);
                    }
                    else {
                        if (TopkIndices[idx] == ignore_index || batch[batch_idx][TopkIndices[idx]] < prob_thresh) continue;
                        CharacterReplacement replace_chars(dict[TopkIndices[idx]], batch[batch_idx][TopkIndices[idx]], index);
                        replacement.push_back(replace_chars);
                    }
                }
                index += 1;
                TempIndices = TopkIndices;
            }
            replacements.push_back(replacement);
            std::string text = std::accumulate(char_list.begin(), char_list.end(), std::string());
            if (text.length() > 0) {
                double sum = std::accumulate(conf_list.begin(), conf_list.end(), 0.0);
                float conf = sum / static_cast<double>(conf_list.size());
                result_list.emplace_back(text, conf);
            }
        }
        return replacements;
    }

}