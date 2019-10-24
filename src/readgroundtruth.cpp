/*
 * readgroundtruth.cpp
 *
 *  Created on: May 5, 2019
 *      Author: david
 */

#include "readgroundtruth.hpp"

/**
 * Reads a text file where each row contains comma separated values of
 * corners of ground truth bounding boxes. Returns a list of cv::Rect with
 * the bounding boxes data.
 * @param ground_truth_path: full path to ground truth text file
 * @return bbox_list: list of ground truth bounding boxes of class Rect
 */
std::vector<Rect> readGroundTruthFile(std::string ground_truth_path)
{
	// for reading text file
	std::ifstream inFile; //file stream
	std::string bbox_str; //line of file containing bounding box data
	std::string bbox_value; // bounding box data
	int value, xmin, ymin, width, height; // bounding box data
	std::vector<int> x_values, y_values; // list of coordinates for bounding box

	// for storing bounding box data
	Rect bbox;
	std::vector<Rect> bbox_list;

	// open text file
	inFile.open(ground_truth_path.c_str());
	if(!inFile) {
			std::cout << "Could not open ground truth file " << ground_truth_path << std::endl;
	}

	// Read line of groundtruth.txt
	while(std::getline(inFile, bbox_str)){
		// Clear vector of values
		x_values.clear();
		y_values.clear();
		int line_ctr = 0;
		std::stringstream linestream(bbox_str);
		// Read comma separated values of groundtruth.txt
		while(std::getline(linestream, bbox_value, ',')){
			std::istringstream ss(bbox_value);
			ss >> value;
			if(line_ctr%2 == 0) x_values.push_back(value);
			else y_values.push_back(value);
			line_ctr++;
		}
		// Initialize a cv::Rect for a bounding box
		//Store to bounding box list
		xmin = *std::min_element(x_values.begin(), x_values.end());
		ymin = *std::min_element(y_values.begin(), y_values.end());
		width = *std::max_element(x_values.begin(), x_values.end()) - xmin;
		height = *std::max_element(y_values.begin(), y_values.end()) - ymin;
		bbox_list.push_back(Rect(xmin, ymin, width, height));
	}
	inFile.close();

	return bbox_list;
}
