#include <cmath>
#include <include/extraction.h>
#include "opencv2/opencv.hpp"

namespace ContainerOCR {
    bool ContExtraction::checkConstraintCode(const int &idx, const char &c) {
        int len_owner_code = this->max_len_code - this->max_len_seri_number;
        if ((idx < len_owner_code && !std::isupper(c)) ||
            (idx >= len_owner_code && !std::isdigit(c))) {
            return false;
        }
        return true;
    }
    char ContExtraction::calculate_check_digit(const std::string &code) {
        int sum = 0;
        int len_owner_code = this->max_len_code - this->max_len_seri_number;
        for (size_t i = 0; i < code.length(); i++) {
            if (!this->checkConstraintCode(i, code[i])) {
                return '\0';
            }
            char n = code[i];
            int index = this->alphabet.find(n);
            sum += index * std::pow(2, i);
        }
        char c = '0' + sum % this->max_len_code % (this->max_len_code - 1);
        return c;
    }

    bool ContExtraction::isISO_6346(const std::string &code) {
        int size = code.length();
        if (this->calculate_check_digit(code.substr(0, size - 1)) != code[size - 1])
            return false;
        return true;
    }

    std::vector<std::pair<std::string, float>> ContExtraction::clusterBoxes(std::vector<OCRResult> &codes) {
        //for (auto x : codes) {
        //    std::cout << x.label << " ___ " << std::endl;
        //}
        std::vector<std::pair<std::string, float>> results;
        int num_of_ids = 0;
        for (const auto &id : codes) {
            num_of_ids += id.label.first.length();
        }
        int k = std::ceil(static_cast<float>(num_of_ids) / this->cluster_max_chars);
        //std::cout << "k: " << k << std::endl;
        if (k > 1) {
            int numPoints = codes.size();
            cv::Mat data(numPoints, 2, CV_32F);
            for (int i = 0; i < numPoints; i++) {
                auto box = codes[i].box;

                data.at<float>(i, 0) = float(box[0].x + box[1].x) / 2;
                data.at<float>(i, 1) = float(box[0].y + box[2].y) / 2;
            }
            cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0);
            cv::Mat labels, centers;
            cv::kmeans(data, k, labels, criteria, 10, cv::KMEANS_RANDOM_CENTERS, centers);
            for (int i = 0; i < k; i++) {
                std::vector<OCRResult> region;
                for (int j = 0; j < labels.rows; j++) {
                    if (labels.at<int>(j) == i) {
                        region.push_back(codes[j]);
                    }
                }
                auto final_code = this->merge_code(region);
                results.push_back(final_code);
            }
        }
        else {
            auto final_code = this->merge_code(codes);
            results.push_back(final_code);
        }
        return results;
    };
    
    std::pair<std::string, float> ContExtraction::merge_code(std::vector<OCRResult> &codes) {
        if (!std::all_of(codes.begin(), codes.end(),
            [](const auto& code) { return code.isVertical; })) {
            Utils::sortBoxesByY(codes);
        }
        else {
            Utils::sortBoxesByX(codes);
            for (int i = 0; i < codes.size(); i++) {
                if (codes[i].label.first.length() == this->max_len_code) {
                    std::iter_swap(codes.begin(), codes.begin() + i);
                }
                else if (codes[i].label.first.length() == this->max_len_seri_number && i == 0) {
                    std::iter_swap(codes.begin() + 1, codes.begin() + i);
                }
            }
        }

        std::string code = "";
        float confident = 0.0;
        std::priority_queue<CharacterReplacement> replacements;
        int num_parts = 0;
        for (int i = 0; i < codes.size(); i++) {
            int inc = code.length();
            for (auto &replacement : codes[i].replacement) {
                replacement.index += inc;
                if (!this->checkConstraintCode(replacement.index, replacement.c)) {
                    continue;
                }
                replacements.push(replacement);
            }
            code += codes[i].label.first;
            confident += codes[i].label.second;
            num_parts += 1;
            if (code.length() >= this->max_len_code) break;
        }
        confident = num_parts != 0 ? confident / num_parts : 0;
        if (code.length() != this->max_len_code) {
            return std::make_pair(code, confident);
        }
        else {
            std::string tempCode = code;
            this->replace_code(code, replacements);
            if (tempCode != code)
                return std::make_pair(code, -1);
            else return std::make_pair(code, confident);
        }
    }
    void ContExtraction::replaceTwoChars(std::vector<CharacterReplacement> &replacements, std::string &code) {
        for (int i = 0; i < replacements.size() - 1; i++) {
            for (int j = 1; j < replacements.size(); j++) {
                if (replacements[i].index != replacements[j].index) {
                    std::string tempCode = code;
                    tempCode[replacements[i].index] = replacements[i].c;
                    tempCode[replacements[j].index] = replacements[j].c;
                    if (this->isISO_6346(tempCode)) {
                        code = tempCode;
                        return;
                    }
                }
            }
        }
    }
    void ContExtraction::replace_code(std::string &code, 
        std::priority_queue<CharacterReplacement> &replacements) {
        if (!this->isISO_6346(code)) {
            std::priority_queue<CharacterReplacement> tempReplacements = replacements;
            while (!replacements.empty()) {
                std::string tempCode = code;
                auto best_replacement = replacements.top();
                tempCode[best_replacement.index] = best_replacement.c;
                replacements.pop();
                if (this->isISO_6346(tempCode)) {
                    code = tempCode;
                    return;
                }
            }
            std::vector<CharacterReplacement> vec;
            while (!tempReplacements.empty()) {
                vec.push_back(tempReplacements.top());
                tempReplacements.pop();
            }
            if (vec.size() > 1) 
                this->replaceTwoChars(vec, code);
        }
    }

}