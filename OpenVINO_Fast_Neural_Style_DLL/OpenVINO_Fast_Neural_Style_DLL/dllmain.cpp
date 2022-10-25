// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"

// Create a macro to quickly mark a function for export
#define DLLExport __declspec (dllexport)

// Wrap code to prevent name-mangling issues
extern "C" { 

	ov::Core core;
	std::shared_ptr<ov::Model> model;
	ov::CompiledModel compiled_model;

	// List of available compute devices
	std::vector<std::string> available_devices;

	ov::InferRequest infer_request;
	ov::Tensor input_tensor;
	float* input_data;

	int input_w;
	int input_h;
	int nPixels;
	// The number of color channels 
	int num_channels = 3;

	float mean[] = { 0.485, 0.456, 0.406 };
	float standard_dev[] = { 0.229, 0.224, 0.225 };


	DLLExport int GetDeviceCount() {

		available_devices.clear();

		for (std::string device : core.get_available_devices()) {
			if (device.find("GNA") == std::string::npos) {
				available_devices.push_back(device);
			}
		}

		return available_devices.size();
	}

	DLLExport std::string* GetDeviceName(int index) {
		return &available_devices[index];
	}


	// Load a torchscript model from the specified file path
	DLLExport int LoadModel(char* modelPath, int index, int inputDims[2]) {

		int return_val = 0;
		core.set_property("GPU", ov::cache_dir("cache"));

		// Try loading the specified model
		try { model = core.read_model(modelPath); }
		catch (...) { return 1; }

		// Try updating the model input dimensions
		try { model->reshape({ 1, 3, inputDims[1], inputDims[0] }); }
		catch (...) { return_val = 2; }

		auto compiled_model = core.compile_model(model, "MULTI",
			ov::device::priorities(available_devices[index]),
			ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
			ov::hint::inference_precision(ov::element::f32));

		infer_request = compiled_model.create_infer_request();

		// Get input tensor by index
		input_tensor = infer_request.get_input_tensor(0);

		input_w = input_tensor.get_shape()[3];
		input_h = input_tensor.get_shape()[2];
		nPixels = input_w * input_h;

		inputDims[0] = input_w;
		inputDims[1] = input_h;

		input_data = input_tensor.data<float>();

		// Return a value of 0 if the model loads successfully
		return return_val;
	}

	// Perform inference with the provided texture data
	DLLExport void PerformInference(uchar* inputData) {

		try {

			// Store the pixel data for the source input image in an OpenCV Mat
			cv::Mat texture = cv::Mat(input_h, input_w, CV_8UC4, inputData);
			// Remove the alpha channel
			cv::cvtColor(texture, texture, cv::COLOR_RGBA2RGB);

			// Iterate over each pixel in image
			for (int p = 0; p < nPixels; p++) {
				// Iterate over each color channel for each pixel in image
				for (int ch = 0; ch < num_channels; ++ch) {
					input_data[ch * nPixels + p] = texture.data[p * num_channels + ch] / 255.0f;
				}
			}

			infer_request.infer();

			// model has only one output
			ov::Tensor output_tensor = infer_request.get_output_tensor();
			// IR v10 works with converted precisions (i64 -> i32)
			auto out_data = output_tensor.data<float>();

			// Iterate over each pixel in image
			for (int p = 0; p < nPixels; p++) {
				// Iterate over each color channel for each pixel in image
				for (int ch = 0; ch < num_channels; ++ch) {
					float val = out_data[ch * nPixels + p];
					val = ((val * standard_dev[ch]) + mean[ch]) * 255.0f;
					val = std::min(std::max(val, (float)0.), (float)255.);
					texture.data[p * num_channels + ch] = val;
				}
			}

			// Add alpha channel
			cv::cvtColor(texture, texture, cv::COLOR_RGB2RGBA);
			// Copy values form the OpenCV Mat back to inputData
			std::memcpy(inputData, texture.data, texture.total() * texture.channels());
		}
		catch (...) {}
	}
}