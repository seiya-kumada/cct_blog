//
//  main.cpp
//  text_detection
//
//  Created by 熊田聖也 on 2018/10/03.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;


template<typename T>
void print_arg(const cv::CommandLineParser& parser, const std::string& name) {
    std::cout << name << ": " << parser.get<T>(name) << std::endl;
}


void print_args(const cv::CommandLineParser& parser) {
    print_arg<float>(parser, "thr");
    print_arg<float>(parser, "nms");
    print_arg<int>(parser, "width");
    print_arg<int>(parser, "height");
    print_arg<std::string>(parser, "model");
    print_arg<std::string>(parser, "input");
}


std::vector<cv::Mat> detect_text(const cv::Mat& image, int inpWidth, int inpHeight, cv::dnn::Net& net)
{
    // construct a blob from the image
    // the blob is a preprocessed input for the neural network
    cv::Mat blob;
    cv::dnn::blobFromImage(
        // input image
        image,
                           
        // 4-dimensional blob
        blob,
                           
        // multiplier for image values
        1.0,
                           
        // spatial size for output image
        cv::Size(inpWidth, inpHeight),
                           
        // scalar with mean values which are subtracted from channels
        cv::Scalar(123.68, 116.78, 103.94),
                           
        // flag which indicates that swap first and last channels in 3-channel image is necessary
        // whether Blue and Red channels are swapped or not
        true,
                           
        // flag which indicates whether image will be cropped after resize or not
        // if crop is false, direct resize is performed without cropping and with preserving aspect ratio
        false);

//    std::cout << "AAA " << blob.size[0] << std::endl; // 1
//    std::cout << "AAA " << blob.size[1] << std::endl; // 3 BGR
//    std::cout << "AAA " << blob.size[2] << std::endl; // 320 rows
//    std::cout << "AAA " << blob.size[3] << std::endl; // 320 cols
    
    
    // set the input to the dnn
    net.setInput(blob);
    
    // perform a forward pass of the model and extract features out of two layers
    std::vector<cv::Mat> outs;
    std::vector<cv::String> outNames {"feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"};
    /*
     feature_fusion/Conv_7/Sigmoid: probability that a region contains text
     feature_fusion/concat_3: geometry of the image
     */
    net.forward(outs, outNames);
    return outs;
}


void decode(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh,
            std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences)
{
    detections.clear();
    CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
    CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
    CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);
    
    const int height = scores.size[2];
    const int width = scores.size[3];
    for (int y = 0; y < height; ++y)
    {
        const float* scoresData = scores.ptr<float>(0, 0, y);
        const float* x0_data = geometry.ptr<float>(0, 0, y);
        const float* x1_data = geometry.ptr<float>(0, 1, y);
        const float* x2_data = geometry.ptr<float>(0, 2, y);
        const float* x3_data = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x)
        {
            float score = scoresData[x];
            if (score < scoreThresh)
                continue;
            
            // Decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];
            
            cv::Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                           offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            cv::Point2f p1 = cv::Point2f(-sinA * h, -cosA * h) + offset;
            cv::Point2f p3 = cv::Point2f(-cosA * w, sinA * w) + offset;
            cv::RotatedRect r(0.5f * (p1 + p3), cv::Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            detections.push_back(r);
            confidences.push_back(score);
        }
    }
}


void render_detections(const cv::Mat& image, int inpHeight, int inpWidth, const std::vector<int>& indices, const std::vector<cv::RotatedRect>& boxes, cv::dnn::Net& net)
{
    cv::Point2f ratio((float)image.cols / inpWidth, (float)image.rows / inpHeight);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        const cv::RotatedRect& box = boxes[indices[i]];
        
        cv::Point2f vertices[4];
        box.points(vertices);
        for (int j = 0; j < 4; ++j)
        {
            vertices[j].x *= ratio.x;
            vertices[j].y *= ratio.y;
        }
        for (int j = 0; j < 4; ++j)
            line(image, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 1);
    }
    
    // Put efficiency information.
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = cv::format("Inference time: %.2f ms", t);
    putText(image, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    
    static const std::string kWinName {"EAST: An Efficient and Accurate Scene Text Detector"};
    cv::namedWindow(kWinName, cv::WINDOW_NORMAL);
    cv::imshow(kWinName, image);
    cv::waitKey();
}

const po::variables_map parse_command_lines(int argc, const char * argv[])
{
    po::options_description desc("Use this script to run TensorFlow implementation (https://github.com/argman/EAST) of "
                                 "EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)");
    desc.add_options()
        ("help,h", "Print help message.")
        ("input_path,i", po::value<std::string>(), "Path to input image or video file. Skip this argument to capture frames from a camera.")
        ("model_path,m", po::value<std::string>(), "Path to a binary .pb file contains trained network.")
        ("width", po::value<int>()->default_value(320), "Preprocess input image by resizing to a specific width. It should be multiple by 32.")
        ("height", po::value<int>()->default_value(320), "Preprocess input image by resizing to a specific height. It should be multiple by 32.")
        ("thr", po::value<float>()->default_value(0.5), "Confidence threshold.")
        ("nms", po::value<float>()->default_value(0.4), "Non-maximum suppression threshold.")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    return vm;
}

int main(int argc, const char * argv[]) {
    try {
        // load command lines
        const auto vm = parse_command_lines(argc, argv);
        const auto input_path = vm["input_path"].as<std::string>();
        const auto confThreshold = vm["thr"].as<float>();
        const auto nmsThreshold = vm["nms"].as<float>();
        const auto inpWidth = vm["width"].as<int>();
        const auto inpHeight = vm["height"].as<int>();
        const auto model_path = vm["model_path"].as<std::string>();
        
        // Open a video file or an image file or a camera stream.
        const auto image = cv::imread(input_path);
        
        // obtain the two output layer sets
        // Load network.
        auto net = cv::dnn::readNet(model_path);
        const auto outs = detect_text(image, inpWidth, inpHeight, net);
        const auto& scores = outs[0];
        const auto& geometry = outs[1];
        
        // decode predicted bounding boxes and return results.
        std::vector<cv::RotatedRect> boxes {};
        std::vector<float> confidences {};
        decode(scores, geometry, confThreshold, boxes, confidences);

        // Apply non-maximum suppression procedure.
        std::vector<int> indices {};
        cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        
        // render detections
        render_detections(image, inpHeight, inpWidth, indices, boxes, net);
    
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    
    return 0;
}
