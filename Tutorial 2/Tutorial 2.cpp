#include <iostream>
#include <vector>
#include "Utils.h"
#include "CImg.h"


//program uses modified workshop code to accomplish follwing tasks 
// convert to grey scale image - map
// produce histogram -atomic increment
// cumlative histogram - actomic add for scan
// LUT scaling simple map 
// re-projection simple map
// event time for image to buffer transfers
// event time for kernals
// colour images that are inputted will be converted to greyscale
// fixed bin size of 256 (0-255)



using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	//change file name here for different images
	string image_filename = "test.ppm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input,"input");

	
		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device //enable command queue profiling
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);


		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//declaring size of histogram
		typedef int mytype;
		std::vector<mytype> histogram(256); //range of intensities 0-255
		std::vector<mytype> CUM(256);
		std::vector<mytype> LUT(256);//LUT =look up tabel 
		size_t histogram_size = histogram.size() * sizeof(mytype);//size in bytes

	

		//Part 4 - device operations

		//device - buffers ------------------------------------------------------------------------------------------------------------------------
		// name (context , flag , bytes)
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_grey(context, CL_MEM_READ_WRITE, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image (final image
		cl::Buffer histogram_buffer(context, CL_MEM_READ_WRITE, histogram_size);
		cl::Buffer CUM_buffer(context, CL_MEM_READ_WRITE, histogram_size);
		cl::Buffer LUT_buffer(context, CL_MEM_READ_WRITE, histogram_size);


		//4.1 Copy images to device memory// intialise the histgram buffers ------------------------------------------------------------------------
		cl::Event image_write_buffer;
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0], NULL, &image_write_buffer);
		queue.enqueueFillBuffer(histogram_buffer, 0, 0, histogram_size);//zero histogram buffer on device memory
		queue.enqueueFillBuffer(CUM_buffer, 0, 0, histogram_size);//zero histogram buffer on device memory
		queue.enqueueFillBuffer(LUT_buffer, 0, 0, histogram_size);
		
		

		
		//4.2 Setup and execute the kernels (i.e. device code)

		//convert image to grey scale
		cl::Event grey_scale_kernal;
		cl::Kernel kernel_grey = cl::Kernel(program, "grey");
		kernel_grey.setArg(0, dev_image_input);
		kernel_grey.setArg(1, dev_image_grey);

		queue.enqueueNDRangeKernel(kernel_grey, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &grey_scale_kernal);
		
		
		//histogram kernal (uses the grey scale image) ----------------------------------------------------------------------------
		cl::Event hist_kernal_time;
		cl::Kernel kernel_hist = cl::Kernel(program, "hist_simple");
		kernel_hist.setArg(0, dev_image_grey); 
		kernel_hist.setArg(1, histogram_buffer);
		//launch kernal. local size not defined going with complier determined work group
		//global work group is the size of image as pixel = work item
		queue.enqueueNDRangeKernel(kernel_hist, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &hist_kernal_time);
		//takes resulting histgram from the device buffer and brings it to host
		//queue.enqueueReadBuffer(histogram_buffer, CL_TRUE, 0, histogram_size, &histogram[0]);
		//std::cout << "hist size = " << histogram.size() << std::endl;
		//std::cout << "hist = " << histogram << std::endl;
		std::cout << "Histogram ready" << std::endl;


		//Cumlative histogram kernal --------------------------------------------------------------------------------------------
		cl::Event kernal_CUM_time;
		cl::Kernel kernel_CUM = cl::Kernel(program, "scan_add_atomic");
		kernel_CUM.setArg(0, histogram_buffer);
		kernel_CUM.setArg(1, CUM_buffer);
		
		queue.enqueueNDRangeKernel(kernel_CUM, cl::NullRange, cl::NDRange(histogram_size), cl::NullRange, NULL, &kernal_CUM_time);
		//queue.enqueueReadBuffer(CUM_buffer, CL_TRUE, 0, histogram_size, &CUM[0]);
		//std::cout << "CUM = " << CUM << std::endl;
		std::cout << "CUM histogram ready" << std::endl;

		
		//LUT // mapping--------------------------------------------------------------------------------------------------------
		cl::Event kernal_LUT_time;
		cl::Kernel kernel_LUT = cl::Kernel(program, "LUT");
		kernel_LUT.setArg(0, CUM_buffer);
		kernel_LUT.setArg(1, LUT_buffer);
		
		queue.enqueueNDRangeKernel(kernel_LUT, cl::NullRange, cl::NDRange(histogram_size), cl::NullRange, NULL, &kernal_LUT_time);
		//queue.enqueueReadBuffer(LUT_buffer, CL_TRUE, 0, histogram_size, &LUT[0]);
		//std::cout << "LUT = " << LUT << std::endl;
		std::cout << "LUT ready" << std::endl;

		//reprojection // mapping ---------------------------------------------------------------------------------------------

		cl::Event kernal_PROJECT_time;
		cl::Kernel kernel_PROJECT = cl::Kernel(program, "PROJECT");
		kernel_PROJECT.setArg(0, dev_image_grey);
		kernel_PROJECT.setArg(1, LUT_buffer);
		kernel_PROJECT.setArg(2, dev_image_output);
		queue.enqueueNDRangeKernel(kernel_PROJECT, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &kernal_PROJECT_time);
		std::cout << "projection ready" << std::endl;

		
		//DISPLAY IMAGES------------------------------------------------------------------------------------------------------
		vector<unsigned char> output_buffer(image_input.size());
		//4.3 Copy the result from device to host
		cl::Event output_from_buffer;
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0], NULL, &output_from_buffer);
		//host output array is the output_buffer
		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image,"output");



		// display times
		std::cout << "input_image to buffer[ns]:" << (image_write_buffer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - image_write_buffer.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;
		std::cout << "grey_scale kernal[ns]:" << (grey_scale_kernal.getProfilingInfo<CL_PROFILING_COMMAND_END>() - grey_scale_kernal.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;
		std::cout << "histogram kernal[ns]:" << (hist_kernal_time.getProfilingInfo<CL_PROFILING_COMMAND_END>() - hist_kernal_time.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;
		std::cout << "Cumalative histogram kernal[ns]:" << (kernal_CUM_time.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernal_CUM_time.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;
		std::cout << "LUT kernal[ns]:" << (kernal_LUT_time.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernal_LUT_time.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;
		std::cout << "Re-Project kernal[ns]:" << (kernal_PROJECT_time.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernal_PROJECT_time.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;
		std::cout << "output_image from buffer[ns]:" << (output_from_buffer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - output_from_buffer.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;
 		
		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }		

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
