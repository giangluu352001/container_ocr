#pragma once

#include <include/detector.h>
#include <include/recognizer.h>
#include <include/extraction.h>

namespace ContainerOCR {
	class ContOCR {
		public:
			explicit ContOCR();
			~ContOCR();
			void Run(cv::Mat img);
		private:
			float ratio_vertical = 3;

			DBDetector *detector = nullptr;
			SVTRRecognizer *horizontal_recognizer = nullptr;
			SVTRRecognizer *vertical_recognizer = nullptr;
			ContExtraction *extraction = nullptr;
	};
}