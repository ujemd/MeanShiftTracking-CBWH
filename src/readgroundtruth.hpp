/*
 * readgroundtruth.hpp
 *
 *  Created on: May 5, 2019
 *      Author: david
 */

#ifndef SRC_READGROUNDTRUTH_HPP_
#define SRC_READGROUNDTRUTH_HPP_

#include "opencv2/opencv.hpp"

using namespace cv;

// function for reading ground truth bounding boxes
std::vector<Rect> readGroundTruthFile(std::string ground_truth_path);

#endif /* SRC_READGROUNDTRUTH_HPP_ */
