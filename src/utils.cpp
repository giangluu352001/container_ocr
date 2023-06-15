#include "include/utils.h"

namespace ContainerOCR {
    void Utils::sortPointsByX(std::vector<cv::Point2f> &points) {
        std::sort(points.begin(), points.end(),
            [](const cv::Point2f &a, const cv::Point2f &b) {
                return a.x < b.x;
            });
    }
    void Utils::sortPointsByY(std::vector<cv::Point2f>& points) {
        std::sort(points.begin(), points.end(),
            [](const cv::Point2f& a, const cv::Point2f& b) {
                return a.y > b.y;
            });
    }
    void Utils::sortBoxesByX(std::vector<OCRResult>& codes) {
        std::sort(codes.begin(), codes.end(),
            [](const OCRResult& a, const OCRResult& b) {
                return a.box[0].x < b.box[0].x;
            });
    }
    void Utils::sortBoxesByY(std::vector<OCRResult> &codes) {
        std::sort(codes.begin(), codes.end(), 
            [](const OCRResult& a, const OCRResult& b) {
            return a.box[0].y < b.box[0].y;
        });
    }
	std::vector<std::vector<float>> Utils::Mat2Vector(const cv::Mat& mat) {
        std::vector<std::vector<float>> img_vec(mat.rows, std::vector<float>(mat.cols));
        for (int i = 0; i < mat.rows; ++i) {
            const float* row_ptr = mat.ptr<float>(i);
            for (int j = 0; j < mat.cols; ++j) {
                img_vec[i][j] = row_ptr[j];
            }
        }
        return img_vec;
	}
    std::vector<cv::Point2f> Utils::Mat2Points(const cv::Mat &mat) {
        std::vector<cv::Point2f> points;
        for (int i = 0; i < mat.rows; ++i) {
            const float* row_ptr = mat.ptr<float>(i);
            for (int j = 0; j < mat.cols; j += 2) {
                cv::Point2f point(row_ptr[j], row_ptr[j + 1]);
                points.push_back(point);
            }
        }
        return points;
    }
    std::vector<std::vector<std::vector<float>>> Utils::Tensor3D2Vector(const at::Tensor& tensor)
    {
        int batch_size = tensor.size(0);
        int num_rows = tensor.size(1);
        int num_cols = tensor.size(2);

        std::vector<std::vector<std::vector<float>>> seq_vec;
        seq_vec.reserve(batch_size);

        auto accessor = tensor.accessor<float, 3>();
        for (int b = 0; b < batch_size; ++b) {
            std::vector<std::vector<float>> row_vec;
            row_vec.reserve(num_rows);
            for (int r = 0; r < num_rows; ++r) {
                const float* row_ptr = accessor[b][r].data();
                std::vector<float> col_vec(row_ptr, row_ptr + num_cols);
                row_vec.push_back(std::move(col_vec));
            }
            seq_vec.push_back(std::move(row_vec));
        }

        return seq_vec;
    }

    cv::Mat Utils::get_rotate_crop_image(const cv::Mat &srcimage, const std::vector<cv::Point2f> &box) {

        cv::Point2f tl = box[0];
        cv::Point2f tr = box[1];
        cv::Point2f br = box[2];
        cv::Point2f bl = box[3];

        float widthA = std::sqrt((br.x - bl.x) * (br.x - bl.x) + (br.y - bl.y) * (br.y - bl.y));
        float widthB = std::sqrt((tr.x - tl.x) * (tr.x - tl.x) + (tr.y - tl.y) * (tr.y - tl.y));
        int maxWidth = std::max(int(widthA), int(widthB));

        float heightA = std::sqrt((tr.x - br.x) * (tr.x - br.x) + (tr.y - br.y) * (tr.y - br.y));
        float heightB = std::sqrt((tl.x - bl.x) * (tl.x - bl.x) + (tl.y - bl.y) * (tl.y - bl.y));
        int maxHeight = std::max(int(heightA), int(heightB));

        std::vector<cv::Point2f> dstPoints(4);
        dstPoints[0] = cv::Point2f(0, 0);
        dstPoints[1] = cv::Point2f(static_cast<float>(maxWidth - 1), 0);
        dstPoints[2] = cv::Point2f(static_cast<float>(maxWidth - 1), static_cast<float>(maxHeight - 1));
        dstPoints[3] = cv::Point2f(0, static_cast<float>(maxHeight - 1));

        cv::Mat M = cv::getPerspectiveTransform(box, dstPoints);
        cv::Mat dst_img;
        cv::warpPerspective(srcimage, dst_img, M, cv::Size(maxWidth, maxHeight),
            cv::INTER_CUBIC, cv::BORDER_REPLICATE);
        return dst_img;
    }

    void Utils::LoadModel(const std::string& model_path, torch::jit::script::Module &module) {
  
        try {
            std::cout << "Loading model at: " << model_path << std::endl;
            module = torch::jit::load(model_path);
        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model\n" << e.msg();
            return;
        }
        module.eval();
    }

    std::vector<int> Utils::argsort(const std::vector<float>& array) {
        const int array_len = array.size();
        std::vector<int> array_index;
        array_index.reserve(array_len);
        for (int i = 0; i < array_len; ++i)
            array_index.push_back(i);

        std::sort(array_index.begin(), array_index.end(),
            [array](int pos1, int pos2) { return array[pos1] > array[pos2]; });

        return array_index;
    }

}