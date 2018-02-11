//
//  main.cpp
//  sample
//
//  Created by 熊田聖也 on 2018/02/11.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include <iostream>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <boost/format.hpp>
#include <vector>
#include "normal.hpp"
#include "halide.hpp"
#include <chrono>

int normal_process(const char* argv[])
{
    const std::string src_path {argv[1]};
    const std::string dst_path {argv[2]};
    
    const int dst_width {atoi(argv[3])};
    const int dst_height {atoi(argv[4])};
    
    // load a source image
    const cv::Mat src_image {cv::imread(src_path)};
    
    if (src_image.empty())
    {
        std::cout << "file not found\n";
        return 1;
    }
    
    std::cout << boost::format("(src_width, src_height) = (%1%, %2%)") % src_image.cols % src_image.rows << std::endl;
    
    // make buffer for output image
    cv::Mat dst_image(dst_height, dst_width, CV_8UC3);
    std::cout << boost::format("(dst_width, dst_height) = (%1%, %2%)") % dst_image.cols % dst_image.rows << std::endl;
    
    // use cv access
    {
        auto start = std::chrono::system_clock::now();
        constexpr int NUM = 10;
        for (auto i = 0; i < NUM; ++i)
        {
            resize_with_cv_access(src_image, dst_image);
        }
        auto end = std::chrono::system_clock::now();
        std::cout << boost::format("cv access: %1% msec\n") % (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / NUM);
    }
    
    // use raw access
    {
        auto start = std::chrono::system_clock::now();
        constexpr int NUM = 10;
        for (auto i = 0; i < NUM; ++i)
        {
            resize_with_raw_access(src_image, dst_image);
        }
        auto end = std::chrono::system_clock::now();
        std::cout << boost::format("raw access: %1% msec\n") % (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / NUM);
    }
    
    // save the dst image
    cv::imwrite(dst_path, dst_image);
    return 0;
}

int halide_process(const char* argv[])
{
    const std::string src_path {argv[1]};
    const std::string dst_path {argv[2]};
    
    const int dst_width {atoi(argv[3])};
    const int dst_height {atoi(argv[4])};

    resize_with_halide(src_path, dst_width, dst_height, dst_path);
    return 0;
}

int main(int argc, const char * argv[])
{
//    normal_process(argv);
    halide_process(argv);
    return 0;
}
