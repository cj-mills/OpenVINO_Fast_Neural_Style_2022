// OpenVINO_Fast_Neural_Style_EXE.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"
#include <opencv2/opencv.hpp>

void printInputAndOutputsInfo(const ov::Model& network) {
    std::cout << "model name: " << network.get_friendly_name() << std::endl;

    const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
    for (const ov::Output<const ov::Node> input : inputs) {
        std::cout << "    inputs" << std::endl;

        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        std::cout << "        input name: " << name << std::endl;

        const ov::element::Type type = input.get_element_type();
        std::cout << "        input type: " << type << std::endl;

        const ov::Shape shape = input.get_shape();
        std::cout << "        input shape: " << shape << std::endl;
    }

    const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
    for (const ov::Output<const ov::Node> output : outputs) {
        std::cout << "    outputs" << std::endl;

        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        std::cout << "        output name: " << name << std::endl;

        const ov::element::Type type = output.get_element_type();
        std::cout << "        output type: " << type << std::endl;

        const ov::Shape shape = output.get_shape();
        std::cout << "        output shape: " << shape << std::endl;
    }
}

int main(int argc, const char* argv[])
{
    std::string model_path = argv[1];
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    core.set_property("GPU", ov::cache_dir("cache"));

    cv::Mat image;
    std::string file_name = argv[2];

    image = cv::imread(file_name);
    cv::Mat out_img(image.rows, image.cols, CV_8UC3);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    model->reshape({ 1, 3, image.rows, image.cols });

    auto compiled_model = core.compile_model(model, "MULTI",
        ov::device::priorities(argv[3]),
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
        ov::hint::inference_precision(ov::element::f32));

    printInputAndOutputsInfo(*model);

    std::vector<std::string> available_devices = core.get_available_devices();

    std::cout << "Available Devices:" << std::endl;
    for (std::string device : available_devices) {
        std::cout << device << std::endl;
    }

    std::cout << "Create inference request" << std::endl;
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Get input tensor by index
    ov::Tensor input_tensor = infer_request.get_input_tensor(0);
    // IR v10 works with converted precisions (i64 -> i32)
    auto input_data = input_tensor.data<float>();

    // The number of color channels 
    int num_channels = image.channels();
    // Get the number of pixels in the input image
    int H = input_tensor.get_shape()[2];
    int W = input_tensor.get_shape()[3];
    std::cout << "Height: " << H << std::endl;
    std::cout << "Width: " << W << std::endl;
    int nPixels = W * H;

    float mean[] = { 0.485, 0.456, 0.406 };
    float std[] = { 0.229, 0.224, 0.225 };

    for (int i = 0; i < 5; i++) {

        auto start = std::chrono::high_resolution_clock::now();

        // Iterate over each pixel in image
        for (int p = 0; p < nPixels; p++) {
            // Iterate over each color channel for each pixel in image
            for (int ch = 0; ch < num_channels; ++ch) {
                input_data[ch * nPixels + p] = image.data[p * num_channels + ch] / 255.0f;
            }
        }

        //std::cout << "Before Inference" << std::endl;
        infer_request.infer();

        //std::cout << "Get Output" << std::endl;
        // model has only one output
        ov::Tensor output_tensor = infer_request.get_output_tensor();
        // IR v10 works with converted precisions (i64 -> i32)
        auto out_data = output_tensor.data<float>();

        

        // Iterate over each pixel in image
        for (int p = 0; p < nPixels; p++) {
            // Iterate over each color channel for each pixel in image
            for (int ch = 0; ch < num_channels; ++ch) {
                float val = out_data[ch * nPixels + p];
                val = ((val * std[ch]) + mean[ch]) * 255.0f;
                val = std::min(std::max(val, (float)0.), (float)255.);
                out_img.data[p * num_channels + ch] = val;
            }
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        float fps = (1000.0 / duration.count());
        std::cout << "Inference time: " << duration.count() << "ms (" << fps << "fps)" << std::endl;

        
    }

    cv::cvtColor(out_img, out_img, cv::COLOR_RGB2BGR);
    cv::imwrite("output.png", out_img);

    std::cout << "End" << std::endl;
}
