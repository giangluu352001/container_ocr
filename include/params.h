#pragma once

#include <string>

std::string det_model_path = "../assets/weights/resnetDB.pt";
int det_db_size = 640;
double det_db_thresh = 0.2;
double det_db_box_thresh = 0.5;
double det_db_unclip_ratio = 1.5;
int det_max_candidates = 100;
int det_text_min_size = 3;
double det_scale = 1;
std::string rec_horizontal_model_path = "../assets/weights/horizontal_svtr.pt";
std::string rec_vertical_model_path = "../assets/weights/vertical_svtr.pt";
std::string label_dict = "~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
int ignore_index = 0;
int topklargest = 3;
double rec_scale = 2;
float prob_thesh = 0.01;
std::string alphabet = "0123456789A BCDEFGHIJK LMNOPQRSTU VWXYZ";
int cluster_max_chars = 15;
int max_len_code = 11;
int max_len_seri_number = 7;
float ratio_vertical = 3;
int rec_height = 64;
int rec_width = 256;