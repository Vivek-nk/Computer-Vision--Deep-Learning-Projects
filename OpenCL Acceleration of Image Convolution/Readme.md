# OpenCL Acceleration of Image Convolution

This is a study of optimization techniques for image convolution in OpenCL  A series of OpenCL programs (kernels) will be deployed with increasing level of optimization to attain a faster convolution performance, and discuss the attained improvement. First, the base model describes a 'global memory' implementation where every thread directly accesses the global memory without coordination with other threads. The base OpenCL implementation is slowly refined and optimized to get a reasonable performance and will be compared to the CPU implementation. After that, a 'local memory' implementation that minimizes access to global memory by sharing data between threads. This implementation conforms to standard best-practices when writing OpenCL programs. 

The Code has been run on python and timing results are generated for CPU and 5 OpenCL implementations for
image size = [ 512x512, 1024x1204, 2048x2048]  and Filter Size = [3x3, 5x5, 7x7, 9x9, 11x11, 13x13].

Python Code named runexp.py is used to run all the Image Size and Filter Size.

Command: python runexp.py will run all the Image and Filter Size (in the server) without the requirement to change the Image Size and Filter Size Manually. 

Plots have been developed using EXCEL. 


