## Mean Shift Tracking with Corrected Background

Simple mean shift base tracking using background information, based on the paper _Robust mean-shift tracking with corrected
background-weighted histogram_ by Ning et al.

After compiling using the makefile, the program can be run in the command line by setting the following parameters in the provided order: Number of bins per color channel, Maximum number of iterations of the mean shift step, epsilon for defining the convergence condition of mean shift, threshold for background model similarity, plot which if set to 1, displays the model histogram, the candidate histogram and the background histogram for each frame.
