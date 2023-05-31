#include <include/postprocess.h>
#include <include/utils.h>
#include <clipper2/clipper.h>

namespace ContainerOCR {
    std::vector<cv::Point2f> DBPostProcessor::unclip(const std::vector<cv::Point2f> &box, const double &unclip_ratio) {
        Clipper2Lib::Path64 poly;
        for (int i = 0; i < box.size(); i++) {
            poly.push_back(Clipper2Lib::Point64(int(box[i].x), int(box[i].y)));
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
    double DBPostProcessor::box_score_fast(const cv::Mat &bitmap, const std::vector<cv::Point2f> &box) {
        int h = bitmap.rows, w = bitmap.cols;
        const int rotated_rectangle_size = 4;
        float box_x[4] = { box[0].x, box[1].x, box[2].x, box[3].x };
        float box_y[4] = { box[0].y, box[1].y, box[2].y, box[3].y };

        int xmin = std::clamp(int(floor(*(std::min_element(box_x, box_x + 4)))), 0, w - 1);
        int xmax = std::clamp(int(ceil(*(std::max_element(box_x, box_x + 4)))), 0, w - 1);
        int ymin = std::clamp(int(floor(*(std::min_element(box_y, box_y + 4)))), 0, h - 1);
        int ymax = std::clamp(int(ceil(*(std::max_element(box_y, box_y + 4)))), 0, h - 1);
        
        int mask_width = xmax - xmin + 1;
        int mask_height = ymax - ymin + 1;
        cv::Mat mask = cv::Mat::zeros(mask_height, mask_width, CV_8UC1);

        cv::Point root_point[rotated_rectangle_size];
        for (int i = 0; i < rotated_rectangle_size; i++) {
            root_point[i] = cv::Point(int(box[i].x) - xmin, int(box[i].y) - ymin);
        }
        const cv::Point* ppt[1] = { root_point };
        int npt[] = { rotated_rectangle_size };
        cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));
        cv::Mat croppedImg = bitmap(cv::Rect(xmin, ymin, mask_width, mask_height));
        auto score = cv::mean(croppedImg, mask)[0];
        return score;
    }
    std::vector<cv::Point2f> DBPostProcessor::get_mini_boxes(const cv::InputArray &box, float &sside) {
        cv::RotatedRect bounding_box = cv::minAreaRect(box);
        sside = std::min(bounding_box.size.width, bounding_box.size.height);
        cv::Mat points;
        cv::boxPoints(bounding_box, points);
        auto pts = Utils::Mat2Points(points);
        Utils::sortPointsByY(pts);
        float leftNeighbor = std::atan2f(std::abs(pts[1].y - pts[0].y), 
            std::abs(pts[1].x - pts[0].x));
        float rightNeighbor = std::atan2f(std::abs(pts[2].y - pts[0].y), 
            std::abs(pts[2].x - pts[0].x));
        std::vector<cv::Point2f> bottom;
        std::vector<cv::Point2f> top;
        if (rightNeighbor < leftNeighbor) {
            bottom = {pts[0], pts[2] };
            top = { pts[3], pts[1] };
        }
        else {
            bottom = { pts[0], pts[1] };
            top = { pts[3], pts[2] };
        }
        Utils::sortPointsByX(top);
        Utils::sortPointsByX(bottom);
        pts = { top[0], top[1], bottom[1], bottom[0] };
        return pts;
    }
    std::vector<std::vector<cv::Point2f>> DBPostProcessor::boxes_from_bitmap(const cv::Mat &pred, 
        const cv::Mat &bitmap, const int &dest_width, const int &dest_height, 
        const int &max_candidates, const int &min_size, const double &box_thresh, const double &unclip_ratio) {
        const int width = bitmap.cols, height = bitmap.rows;
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        const size_t num_contours = std::min(contours.size(), static_cast<size_t>(max_candidates));
        std::vector<std::vector<cv::Point2f>> boxes;
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

            std::vector<cv::Point2f> temp_box;
            temp_box.reserve(4);
            const float width_ratio = float(dest_width) / width;
            const float height_ratio = float(dest_height) / height;
            for (const auto &point : box) {
                const int x = std::clamp(int(point.x * width_ratio), 0, dest_width);
                const int y = std::clamp(int(point.y * height_ratio), 0, dest_height);
                temp_box.push_back(cv::Point2f(x, y));
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