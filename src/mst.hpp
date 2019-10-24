/*
 * mst.hpp
 *
 *  Created on: May 5, 2019
 *      Author: david
 */

#ifndef SRC_MST_HPP_
#define SRC_MST_HPP_

#include "opencv2/opencv.hpp"

using namespace cv;

typedef enum {
	UNIFORM=0,
	EPANECHNIKOV=1
} KERNEL_TYPE;

Mat getKernel(int width, int height, KERNEL_TYPE kernel_type);
Mat createIndexedRGBColours(int max_val, int numbins);
Mat estimatePDF(Mat indexedROI, Mat kernel, int numbins);
Mat estimateBackgroundPDF(Mat indexedFrame, Rect roi, int numbins);
Mat estimateNewModelPDF(Mat model_pdf, Mat bkg_pdf);
Mat computeWeights(Mat model_pdf, Mat candidate_pdf, Mat roi, int numbins, Mat kernel);
bool findNextLocation(Mat weights, Rect &candidate_roi, float eps, float &prevDist, Size frame_size);
void updateModelPDF(Mat model_pdf, Mat &new_model_pdf, Mat &old_bkg_pdf, Mat indexedFrame, Rect roi,
		int numbins, float eps);
void meanShiftTrack(Mat indexedFrame, Mat model_pdf, Mat &new_model_pdf, Mat &bkg_pdf,
		Mat kernel, Mat uniKernel, Rect &target_roi, Size framesize,
		float eps1, float eps2, int numbins, int maxIter, int plot);
Mat plotHist(Mat model_pdf, Mat candidate_pdf, Mat bkg_pdf);

#endif /* SRC_MST_HPP_ */
