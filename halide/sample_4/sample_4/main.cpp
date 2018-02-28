//
//  main.cpp
//  sample_4
//
//  Created by 熊田聖也 on 2018/02/28.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//
#include <string>
#include <iostream>
#include <Halide.h>
#include <halide_image_io.h>
#include "resize.h"
#include <boost/format.hpp>

int resize_with_halide(const std::string& src_path, int dst_width, int dst_height, const std::string& dst_path)
{
    Halide::Runtime::Buffer<uint8_t> input = Halide::Tools::load_image(src_path);
    const int src_cols = input.width();
    const int src_rows = input.height();
    std::cout << src_cols << ", " << src_rows << std::endl;
    
    
    Halide::Runtime::Buffer<uint8_t> output(dst_width, dst_height, 3);
    std::cout << output.width() << ", " << output.height() << ", " << output.channels() << std::endl;
    
    constexpr auto ITERATIONS = 10;
    for (auto i = 0; i < ITERATIONS; ++i)
    {
        auto start = std::chrono::system_clock::now();
        resize(input, src_rows, src_cols, dst_height, dst_width, output);
        auto end = std::chrono::system_clock::now();
        std::cout << boost::format("halide access[%1%]: %2% msec\n")
        % i
        % std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    Halide::Tools::save(output, dst_path);
    return 0;
}

int halide_process(const char* argv[])
{
    const std::string src_path {argv[1]};
    const std::string dst_path {argv[2]};
    
    const int dst_width {atoi(argv[3])};
    const int dst_height {atoi(argv[4])};
    
    return resize_with_halide(src_path, dst_width, dst_height, dst_path);
}

int main(int argc, const char * argv[])
{
    if (argc != 5)
    {
        std::cout << "unvalid sequence of arguments\n";
        return 1;
    }
    return halide_process(argv);
}
