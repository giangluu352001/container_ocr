#pragma once

#include <vector>
#include <string>
#include "include/utils.h"

namespace ContainerOCR {
    class ContExtraction {
        public:
            explicit ContExtraction(const std::string &alphabet, const int &cluster_max_chars, 
                const int &max_len_code, const int &max_len_seri_number) {
                this->alphabet = alphabet;
                this->cluster_max_chars = cluster_max_chars;
                this->max_len_code = max_len_code;
                this->max_len_seri_number = max_len_seri_number;
            }
            std::vector<std::pair<std::string, float>> clusterBoxes(std::vector<OCRResult> &codes, const int &num_of_ids);
        private:
            std::string alphabet = "0123456789A BCDEFGHIJK LMNOPQRSTU VWXYZ";
            int cluster_max_chars = 15;
            int max_len_code = 11;
            int max_len_seri_number = 7;
            bool checkConstraintCode(const int &index, const char &c);
            char calculate_check_digit(const std::string &code);
            std::pair<std::string, float> merge_code(std::vector<OCRResult> &codes);
            bool isISO_6346(const std::string &code);
            void replaceTwoChars(std::vector<CharacterReplacement> &replacements, std::string &code);
            void replace_code(std::string &code, std::priority_queue<CharacterReplacement> &replacements);
    };
}