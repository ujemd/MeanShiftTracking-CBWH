/* Applied Video Analysis of Sequences (AVSA)
 *
 *	Sample Opencv project for lab assignment 3
 * 
 *  INSTRUCTIONS
 *  1) To compile this project, right click on the project name (right or left window) and "build"
 *  2) To run this project, please click on the 'green' play button
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 */

//system libraries C/C++
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>

//opencv libraries
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

//include for mean shift related functions
#include "mst.hpp"

//include for reading ground truth data
#include "readgroundtruth.hpp"

//namespaces
using namespace cv; //avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)

//main function
int main(int argc, char ** argv) 
{
	Mat frame; // current Frame

	double t = 0; //variable for execution time	

	// paths for the dataset
	std::string dataset_path = "../task2_videos_meanshift";
	std::string baseline_seq[6] = {"ball1", "ball2", "basketball", "bolt1", "glove", "road"};
	std::string image_path = "%08d.jpg"; //path to images
	std::string ground_truth_path = "groundtruth.txt"; //path to ground truth data

	//for bounding boxes
	std::vector<Rect> bbox_list;
	int frame_idx;

	//      For mean shift: default values
	// kernel parameters
	KERNEL_TYPE kernel_type = EPANECHNIKOV; // kernel profile type
	// histogram binning parameters
	int numbins = 16; //example in paper: 16 bins per color channel
	int maxIter = 10; //maximum mean shift iterations
	float eps1 = 0.1; // epsilon for location convergence
	float eps2 = 0.5; // epsilon for updating background and model histograms
	int plot = 1;
	int max_val = 255; //max pixel value for uint8 images

	// user input
	if (argc != 1 && argc != 6)
	{
		std::cout << "Please, enter 0 or 5 arguments only." << std::endl;
		return 0;
	}
	//user sets numbins, maxIter, eps, plot. No need to set max_val for now.
	//kernel type is EPANECHNIKOV by default.
	if(argc == 6)
	{
		// histogram binning parameters
		numbins = atoi(argv[1]); //example in paper: 16 bins per color channel
		maxIter = atoi(argv[2]); //maximum mean shift iterations
		eps1 = atof(argv[3]); // epsilon for location convergence
		eps2 = atof(argv[4]); // epsilon for updating background and model histograms
		plot = atoi(argv[5]);
		std::cout << "Plot=" << plot << std::endl;

		if(numbins < 0 || maxIter < 0 || eps1 < 0 || eps2 < 0)
		{
			std::cout << "Please, enter positive arguments only." << std::endl;
			return -1;
		}
	}

	int NumSeq = sizeof(baseline_seq)/sizeof(baseline_seq[0]);  //number of sequence per category

	//Loop for all sequence of each category
	for (int s=0; s<NumSeq; s++ )
	{
		VideoCapture cap;//reader to grab videoframes

		//Compose full path of images
		std::string inputvideo = dataset_path + "/" + baseline_seq[s] + "/" + image_path;
		std::cout << "Displaying sequence at " << inputvideo << std::endl;

		//open the video file to check if it exists
		cap.open(inputvideo);
		if (!cap.isOpened()) {
			std::cout << "Could not open video file " << inputvideo << std::endl;
			return -1;
		}

		//Read ground truth file and store bounding boxes
		std::string inputGroundTruth = dataset_path + "/" + baseline_seq[s] + "/" + ground_truth_path;

		// Get ground truth bounding boxes
		bbox_list = readGroundTruthFile(inputGroundTruth);

		//get first frame to represent model
		cap >> frame;

		// Use first bounding box as initialization
		Rect target_roi = bbox_list[0];

		imshow("model", frame(target_roi));

		// Get kernel profile
		Mat kernel = getKernel(target_roi.width, target_roi.height, kernel_type);
		//imshow("k", kernel);
		// Uniform kernel for weights computation
		Mat uniKernel = getKernel(target_roi.width, target_roi.height, UNIFORM);
		//imshow("k", uniKernel);

		// Define look up table for mapping colors to histogram bins
		Mat lookUpTable = createIndexedRGBColours(max_val, numbins);

		// Map colors of frame to histogram bins
		Mat indexedFrame;
		LUT(frame, lookUpTable, indexedFrame);
		//imshow("indexed frame", numbins*indexedFrame);

		// Get target model histogram (q) without normalization
		Mat model_pdf = estimatePDF(indexedFrame(target_roi), kernel, numbins);
		// Get background histogram (o)
		Mat bkg_pdf = estimateBackgroundPDF(indexedFrame, target_roi, numbins);
		// Get new model histogram (q', eq 7)
		Mat new_model_pdf = estimateNewModelPDF(model_pdf, bkg_pdf);

		//main loop
		for (;;) {
			//get frame & check if we achieved the end of the file (e.g. frame.data is empty)
			cap >> frame;
			if (!frame.data)
				break;

			//Time measurement
			t = (double)getTickCount();

			// Do stuff here

			// ins: roi_0, target_hist, frame, epsilon
			// outs: roi_1
			// Map colors of frame to histogram bins
			LUT(frame, lookUpTable, indexedFrame);

			// mean shift main loop
			meanShiftTrack(indexedFrame, model_pdf, new_model_pdf, bkg_pdf,
					kernel, uniKernel, target_roi, frame.size(),
					eps1, eps2, numbins, maxIter, plot);

			//new_bbox = roi_1;
			//mybbox_list.push_back(new_bbox);

			// End of stuff

			//get the frame number and write it on the current frame
			std::stringstream ss;
			ss << cap.get(cv::CAP_PROP_POS_FRAMES); //uncomment this line for opencv 3.1++

			putText(frame, ss.str().c_str(), cv::Point(15, 15),FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255)); //text in red

			// use frame number to retrieve ground truth bounding box
			ss >> frame_idx;
			rectangle(frame, bbox_list[frame_idx], Scalar(0, 255, 0), 3.5);
			rectangle(frame, target_roi, Scalar(0, 0, 255), 3.5);
			imshow("frame", frame);

			//imwrite(format("mst_gif/mst_seq%d_%08d.jpeg",s,frame_idx), frame);


			//Time measurement
			t = (double)getTickCount() - t;
			printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());

			//exit if ESC key is pressed
			if(frame_idx == 2){
				while(1){
					if(waitKey(30) == 27) break;
				}
			}
			if(waitKey(30) == 27) break;
			//if(waitKey(-1) == 'c') continue;
		}

		//release all resource
		cap.release();
		destroyAllWindows();
	}

	return 0;
}
