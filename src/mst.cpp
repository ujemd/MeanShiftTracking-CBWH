/*
 * mst.cpp
 *
 *  Created on: May 5, 2019
 *      Author: david
 */


#include "mst.hpp"

/**
 * Creates a kernel profile given the desired dimensions and type.
 * @param width: the width of the kernel.
 * @param height: the height of the kernel.
 * @param kernel_type: an enum parameter specifying the kernel type. Can take values EPANECHNIKOV,
 * default is UNIFORM.
 * @return a Mat with the kernel.
 */
Mat getKernel(int width, int height, KERNEL_TYPE kernel_type)
{
	// initialize kernel
	Mat k(height, width, CV_32FC1, Scalar::all(0));

	switch(kernel_type)
	{
		case EPANECHNIKOV:
		{
			// define kernel per pixel
			for (int x=0;x<k.cols;x++)
			for (int y=0;y<k.rows;y++)
			{
				// define according to norm
				k.at<float>(y,x)= 1 - pow((2*((float)x+1)/width) - 1,2) - pow((2*((float)y+1)/height) - 1,2);
				// pixels outside the ellipse are set as 0
				if (k.at<float>(y,x) < 0)
					k.at<float>(y,x) = 0;
			}
		}
		break;
		default: //uniform
			// define kernel per pixel
			for (int x=0;x<k.cols;x++)
			for (int y=0;y<k.rows;y++)
			{
				// pixels inside ellipse are set as 1
				float r = pow((2*((float)x+1)/width)-1,2) + pow((2*((float)y+1)/height)-1,2);
				if (r <= 1)
					k.at<float>(y,x) = 1;
			}
	}

	return k;
}

/**
 * Returns a LUT of color bins given the maximum possible value of a color and
 * number of bins per color channel
 * @param max_val: maximum value of pixel (255 for uint8 images)
 * @param numbins: number of bins for a channel.
 * @return Mat look up table that maps from color to histogram bin.
 */
Mat createIndexedRGBColours(int max_val, int numbins)
{
	// LUT function takes as input a look up table of 256 elements
	// This look up table maps elements from a color range to a bin range.
	Mat lookUpTable(1, 256, CV_8U, Scalar::all(0));
	int den = max_val + 1;

	// used for filling LUT
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i <= max_val; ++i)
    	p[i] = (uchar)(i*numbins/den); //p[i] is in range [0, numbins]
	return lookUpTable;
}

/**
 * Calculates a histogram of the color distribution of an image (without normalizing).
 * @param indexedROI: ROI of target whose pixel values contain their respective bin indices
 * @param kernel: kernel profile
 * @param numbins: number of bins per color channel
 * @return Mat histogram of indexedROI
 */
Mat estimatePDF(Mat indexedROI, Mat kernel, int numbins)
{
	// bin1-3: have bins for each color channel
	// bin is the color bin, has values between 0 and numbins^3-1
	int bin, bin1, bin2, bin3;

	// indexedROI is in range [0, numbins]
	// im1-3 have each a color channel
	Mat im1(indexedROI.rows, indexedROI.cols, CV_8UC1);
	Mat im2(indexedROI.rows, indexedROI.cols, CV_8UC1);
	Mat im3(indexedROI.rows, indexedROI.cols, CV_8UC1);
	Mat channels[3] = {im1, im2, im3};

	// split indexedROI channels
	split(indexedROI, channels);

	// histogram of numbins^3 bins
	Mat pdf;
	pdf = Mat::zeros(1, numbins*numbins*numbins, CV_32FC1);

	// get color bins per pixel
	for (int x=0;x<indexedROI.cols;x++)
	for (int y=0;y<indexedROI.rows;y++)
	{
		// each position (y,x) contains a bin, which helps incrementing the pdf for that bin
		bin1 = (int)im1.at<uchar>(y,x); //get bin for pixel
		bin2 = (int)im2.at<uchar>(y,x);
		bin3 = (int)im3.at<uchar>(y,x);

		// get combined color bin
		bin = bin1*numbins*numbins + bin2*numbins + bin3;
		//std::cout << "bin: " << (int)bin << std::endl;

		// increase histogram count for that bin
		pdf.at<float>(0,bin) += kernel.at<float>(y,x);
	}

	return pdf;
}

/**
 * Computes histogram of area around object.
 * @param indexedFrame: current frame [0, numbins-1]
 * @param roi: target ROI.
 * @param numbins: number of bins per color channel.
 * @return Mat histogram of background.
 */
Mat estimateBackgroundPDF(Mat indexedFrame, Rect roi, int numbins)
{
	// Pixel positions defining outer background region
	int xmin = std::max(0, roi.x - roi.width/2);
	int xmax = std::min(indexedFrame.cols, roi.x + 3*roi.width/2);
	int ymin = std::max(0, roi.y - roi.height/2);
	int ymax = std::min(indexedFrame.rows, roi.y + 3*roi.height/2);

	//imshow("bkg", numbins*indexedFrame(Rect(xmin, ymin, xmax-xmin, ymax-ymin)));

	// bin1-3: have bins for each color channel
	// bin is the color bin, has values between 0 and numbins^3-1
	int bin, bin1, bin2, bin3;

	// indexedFrame is in range [0, numbins]
	// im1-3 have each a color channel
	Mat im1(indexedFrame.rows, indexedFrame.cols, CV_8UC1);
	Mat im2(indexedFrame.rows, indexedFrame.cols, CV_8UC1);
	Mat im3(indexedFrame.rows, indexedFrame.cols, CV_8UC1);
	Mat channels[3] = {im1, im2, im3};

	// split indexedROI channels
	split(indexedFrame, channels);

	// histogram of numbins^3 bins
	Mat bkg_pdf;
	bkg_pdf = Mat::zeros(1, numbins*numbins*numbins, CV_32FC1);

	// get color bins per pixel
	for (int x=xmin;x<=xmax;x++)
	for (int y=ymin;y<=ymax;y++)
	{
		Point pos(x,y);
		if(!roi.contains(pos))
		{
			// each position (y,x) contains a bin, which helps incrementing the pdf for that bin
			bin1 = (int)im1.at<uchar>(y,x); //get bin for pixel
			bin2 = (int)im2.at<uchar>(y,x);
			bin3 = (int)im3.at<uchar>(y,x);

			// get combined color bin
			bin = bin1*numbins*numbins + bin2*numbins + bin3;
			//std::cout << "bin: " << (int)bin << std::endl;

			// increase histogram count for that bin
			bkg_pdf.at<float>(0,bin) += 1;
		}
	}

	// normalize histogram
	float s = sum(bkg_pdf)[0];
	bkg_pdf.convertTo(bkg_pdf, bkg_pdf.type(), 1./s, 0);

	return bkg_pdf;
}

/**
 * Estimates weighted model histogram (eq 7).
 * @param model_pdf: histogram of target model (q) without normalization.
 * @param bkg_pdf: background histogram.
 * @return weighted histogram of model.
 */
Mat estimateNewModelPDF(Mat model_pdf, Mat bkg_pdf)
{
	// Get min non zero value (o*)
	Mat mask;
	mask = bkg_pdf > 0;
	double minNonZero;
	minMaxLoc(bkg_pdf, &minNonZero, NULL, NULL, NULL, mask);

	// Get coefficients (v)
	Mat v;
	pow(bkg_pdf + 0.000001,-1, v); // o^-1
	v = minNonZero*v; //o*/o
	v = min(v, 1); // min(o*/o,1)

	// Get new model histogram (q') eq 7
	Mat new_model_pdf;
	multiply(v, model_pdf, new_model_pdf);

	// normalize histogram
	float s = sum(new_model_pdf)[0];
	new_model_pdf.convertTo(new_model_pdf, new_model_pdf.type(), 1./s, 0);

	return new_model_pdf;
}

/**
 * Returns weights of every pixel according to equation 10 in paper.
 * @param model_pdf: histogram of target model (q)
 * @param candidate_pdf: histogram of target candidate (p)
 * @param indexedROI: ROI of target whose pixel values contain their respective bin indices
 * @param numbins: number of bins per color channel
 * @return Mat weights of each pixel (w_i)
 */
Mat computeWeights(Mat new_model_pdf, Mat candidate_pdf, Mat indexedROI, int numbins, Mat kernel)
{
	// probabilities of a single bin for model q_u and candidate p_u
	float model_prob, candidate_prob;

	// bin1-3: have bins for each color channel
	// bin is the color bin, has values between 0 and numbins^3-1
	int bin, bin1, bin2, bin3;

	// indexedROI is in range [0, numbins]
	// im1-3 have each a color channel
	Mat im1(indexedROI.rows, indexedROI.cols, CV_8UC1);
	Mat im2(indexedROI.rows, indexedROI.cols, CV_8UC1);
	Mat im3(indexedROI.rows, indexedROI.cols, CV_8UC1);
	Mat channels[3] = {im1, im2, im3};

	// split indexedROI channels
	split(indexedROI, channels);

	//output weights matrix with same size of indexedROI
	Mat weights(indexedROI.rows, indexedROI.cols, CV_32FC1, Scalar::all(0));

	// get color bins per pixel
	for (int x=0;x<indexedROI.cols;x++)
	for (int y=0;y<indexedROI.rows;y++)
	{
		// each position (y,x) contains a bin
		bin1 = (int)im1.at<uchar>(y,x); //get bin for pixel
		bin2 = (int)im2.at<uchar>(y,x);
		bin3 = (int)im3.at<uchar>(y,x);

		// get combined color bin
		bin = bin1*numbins*numbins + bin2*numbins + bin3;

		// get probabilities q_u and p_u
		model_prob = new_model_pdf.at<float>(0,bin);
		candidate_prob = candidate_pdf.at<float>(0,bin);

		// compute weight of a pixel
		if (kernel.at<float>(y,x) == 1)
			weights.at<float>(y,x) = sqrtf(model_prob/(candidate_prob+0.000001));
	}

	return weights;
}

/**
 * Estimates the center location of target.
 * @param weights: weights of pixel locations (w_i)
 * @param candidate_roi: bounding box of target candidate
 * @param eps: epsilon for convergence condition
 * @param framesize: size of current frame
 * @return 1 if mean shift converged, 0 otherwise
 */
bool findNextLocation(Mat weights, Rect &candidate_roi, float eps, float &prevDist, Size framesize)
{
	// 1D matrices of horizontal and vertical locations
	Mat xlocs(1, candidate_roi.width, CV_32FC1);
	Mat ylocs(candidate_roi.height, 1, CV_32FC1);

	// fill values of x and y locations
	for (int x=0; x < candidate_roi.width; x++)
		xlocs.at<float>(x) = x -candidate_roi.width/2; // x = -center:center
	for (int y=0; y < candidate_roi.height; y++)
		ylocs.at<float>(y) = y -candidate_roi.height/2; // y = -center:center

	// replicated matrices of locations in x and y
	Mat xlocs2D, ylocs2D;
	repeat(xlocs, candidate_roi.height, 1, xlocs2D);
	repeat(ylocs, 1, candidate_roi.width, ylocs2D);

	// multiplication of weights by locations
	Mat weightedX, weightedY;
	multiply(xlocs2D, weights, weightedX);
	multiply(ylocs2D, weights, weightedY);

	// sum of weighted locations
	float sumx = sum(weightedX)[0];
	float sumy = sum(weightedY)[0];

	// sum of weights
	float den = sum(weights)[0] +0.000001;

	// get former top left position
	int prevX = candidate_roi.x;
	int prevY = candidate_roi.y;

	// calculation of new target position (eq 13 in paper - only valid for Epanechnikov kernel)
	float newX = prevX + sumx/den;
	float newY = prevY + sumy/den;

	// compute distance between centers of previous and new locations
	float distance = sqrtf(pow(newX -  prevX,2) + pow(newY - prevY,2));

	// return 1 to break mean shift loop if distance is less than epsilon
	if (distance < eps)
		return true;
	else if (distance > prevDist)
	{
		//std::cout << "distance > prevDist " << std::endl;
		return true;
	}

	prevDist = distance;

	// given new target center, get top left corner position
	int x = cvFloor(newX);
	int y = cvFloor(newY);

	// ensure that new top left position has legal values
	if (x<0)
		x = 0;
	if (y<0)
		y = 0;
	if (x + candidate_roi.width > framesize.width-1)
		x -= x + candidate_roi.width - (framesize.width-1);
	if (y + candidate_roi.height > framesize.height-1)
		y -= y + candidate_roi.height - (framesize.height-1);

	// update candidate top left corner position
	candidate_roi.x = (int) x;
	candidate_roi.y = (int) y;

	// return 0 to continue mean shift loop
	return false;
}

/**
 * Updates weighted histogram of target model if similarity between background histograms < eps2.
 * @param model_pdf: histogram of target model (q)
 * @param new_model_pdf: weighted histogram of target model given background
 * @param old_bkg_pdf: previous background histogram
 * @param indexedFrame: current frame [0, numbins-1]
 * @param roi: bounding box of target candidate
 * @param numbins: number of bins per color channel
 * @param eps: epsilon for updating weighted histogram of target model given background
 */
void updateModelPDF(Mat model_pdf, Mat &new_model_pdf, Mat &old_bkg_pdf, Mat indexedFrame, Rect roi, int numbins, float eps)
{
	// estimate background model histogram (o)
	Mat bkg_pdf;
	bkg_pdf = estimateBackgroundPDF(indexedFrame, roi, numbins);

	// compute similarity
	Mat tmp;
	multiply(old_bkg_pdf, bkg_pdf, tmp); //o.*o'
	pow(tmp,0.5, tmp); // sqrt(o.*o')
	float sim = sum(tmp)[0];

	if(sim < eps)
	{
		new_model_pdf = estimateNewModelPDF(model_pdf, bkg_pdf);
		bkg_pdf.copyTo(old_bkg_pdf);
	}
}

/**
 * Mean shift algorithm for tracking a single object.
 * @param indexedFrame: current frame [0, numbins-1]
 * @param model_pdf: histogram of target model (q)
 * @param new_model_pdf: weighted histogram of target model given background
 * @param bkg_pdf: background histogram
 * @param kernel: kernel profile
 * @param uniKernel: uniform kernel profile
 * @param target_roi: bounding box of target candidate
 * @param framesize: size of input frame
 * @param eps1: epsilon for convergence condition
 * @param eps2: epsilon for updating weighted histogram of target model given background
 * @param numbins: number of bins per color channel
 * @param maxIter: maximum number of mean shift iterations
 * @param plot: true for plotting model and candidate histograms
 */
void meanShiftTrack(Mat indexedFrame, Mat model_pdf, Mat &new_model_pdf, Mat &bkg_pdf,
		Mat kernel, Mat uniKernel, Rect &target_roi, Size framesize,
		float eps1, float eps2, int numbins, int maxIter, int plot)
{
	// mean shift variables
	Mat target_pdf, // target candidate histogram
		weights, // weights matrix
		histImage; // plot of histogram(s)
	int condition = 0; //0: mean shift has not converged. 1: mean shift has converged
	float distance = 1000.0f; // initial distance between candidate centers
	float s;

	// mean shift loop
	for (int iter = 0; iter < maxIter; iter++)
	{
		// Get target candidate histogram (p) without normalization
		target_pdf = estimatePDF(indexedFrame(target_roi), kernel, numbins);
		// normalize target candidate histogram
		s = sum(target_pdf)[0];
		target_pdf.convertTo(target_pdf, target_pdf.type(), 1./s, 0);

		// get weights wi for each pixel location
		weights = computeWeights(new_model_pdf, target_pdf, indexedFrame(target_roi), numbins, uniKernel);
		//imshow("w", weights);

		// get next target center location
		condition = findNextLocation(weights, target_roi, eps1, distance, framesize);
		// condition is true if distance between previous center location
		// and current center location is minimal
		if(condition)
			break;
	}

	// Update new_model_pdf (q')
	updateModelPDF(model_pdf, new_model_pdf, bkg_pdf, indexedFrame, target_roi, numbins, eps2);

	// Plot histogram
	if(plot == 1)
	{
		histImage = plotHist(new_model_pdf, target_pdf, bkg_pdf);
		imshow("hist", histImage);
	}
}

/**
 * Draws three histograms.
 * @param model_pdf: target model histogram
 * @param candidate_pdf: target candidate histogram
 * @param bkg_pdf: target background histogram
 * @return Mat image plot with three histograms
 */
Mat plotHist(Mat model_pdf, Mat candidate_pdf, Mat bkg_pdf)
{
	Mat model_normHist, candidate_normHist, bkg_normHist;

	// Draw the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvCeil( (double) hist_w/model_pdf.cols);

	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	normalize(model_pdf, model_normHist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(candidate_pdf, candidate_normHist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(bkg_pdf, bkg_normHist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	// Draw
	for( int i = 1; i < model_pdf.cols; i++ )
	{
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(model_normHist.at<float>(0,i-1)) ) ,
				Point( bin_w*(i), hist_h - cvRound(model_normHist.at<float>(0,i)) ),
				Scalar( 0, 255, 0), 2, 8, 0  );
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(candidate_normHist.at<float>(0,i-1)) ) ,
				Point( bin_w*(i), hist_h - cvRound(candidate_normHist.at<float>(0,i)) ),
				Scalar( 0, 0, 255), 2, 8, 0  );
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(bkg_normHist.at<float>(0,i-1)) ) ,
				Point( bin_w*(i), hist_h - cvRound(bkg_normHist.at<float>(0,i)) ),
				Scalar( 255, 0, 0), 2, 8, 0  );
	}

	return histImage;
}
