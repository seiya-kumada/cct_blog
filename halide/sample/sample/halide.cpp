//
//  halide.cpp
//  sample
//
//  Created by 熊田聖也 on 2018/02/11.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "halide.hpp"
#include <Halide.h>
#include <halide_image_io.h>
#include <boost/format.hpp>
#include <chrono>

//https://www.slideshare.net/fixstars/halide-82788728

int blur_with_halide_2(const std::string& src_path, int dst_cols, int dst_rows, const std::string& dst_path)
{
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image(src_path);
    Halide::Func input_bounded = Halide::BoundaryConditions::repeat_edge(input);

    Halide::Func input_16 {};
    Halide::Var x {};
    Halide::Var y {};
    Halide::Var c {};
    input_16(x, y, c) = Halide::cast<uint16_t>(input_bounded(x, y, c));

    Halide::Func blur_x {};
    blur_x(x, y, c) = (input_16(x - 1, y, c) + 2 * input_16(x, y, c) + input_16(x + 1, y, c)) / 4;
    
    Halide::Func blur_y {};
    blur_y(x, y, c) = (blur_x(x, y - 1, c) + 2 * blur_x(x, y, c) + blur_x(x, y + 1, c)) / 4;
    
    Halide::Func output {};
    output(x, y, c) = Halide::cast<uint8_t>(blur_y(x, y, c));

    
    output.compute_root().vectorize(x, 8).parallel(y);
    blur_x.compute_at(output, y).vectorize(x, 8);
    
    
    auto start = std::chrono::system_clock::now();
    Halide::Buffer<uint8_t> result = output.realize(input.width(), input.height(), input.channels());
    auto end = std::chrono::system_clock::now();
    std::cout << boost::format("halide access: %1% msec\n") % std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    Halide::Tools::save(result, dst_path);
    return 1;
}

int blur_with_halide(const std::string& src_path, int dst_cols, int dst_rows, const std::string& dst_path)
{
    //_/_/_/ load a source image and repeat its edges
    
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image(src_path);
    Halide::Func src_image_8 {"src_image_8"};
    src_image_8 = Halide::BoundaryConditions::repeat_edge(input);
    
    Halide::Func src_image_16 {"src_image_16"};
    Halide::Var x {"x"};
    Halide::Var y {"y"};
    Halide::Var c {"c"};
    src_image_16(x, y, c) = Halide::cast<uint16_t>(src_image_8(x, y, c));
    
    //_/_/_/ describe algorithm

    Halide::Func blur {"blur"};
    blur(x, y, c) = (src_image_16(x - 1, y, c) + 4 * src_image_16(x, y, c) + src_image_16(x + 1, y, c) + src_image_16(x, y - 1, c) + src_image_16(x, y + 1, c)) / 8;
    
    // convert back to 8-bit
    Halide::Func dst_image {"dst_image"};
    dst_image(x, y, c) = Halide::cast<uint8_t>(blur(x, y, c));
    
    //_/_/_/ describe scheduling
    
    
    dst_image.vectorize(x, 4).parallel(y);
    
    //    Halide::Var i_inner, j_inner;
    //    dst_image.tile(i, j, i_inner, j_inner, 256, 32).vectorize(i_inner, 8).parallel(j);

    
    //_/_/_/ run
    
    auto start = std::chrono::system_clock::now();
    Halide::Buffer<uint8_t> output = dst_image.realize(input.width(), input.height(), input.channels());
    auto end = std::chrono::system_clock::now();
    std::cout << boost::format("halide access: %1% msec\n") % std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    //_/_/_/ save it
    
    Halide::Tools::save(output, dst_path);

    
    return 1;
}

int resize_with_halide(const std::string& src_path, int dst_cols, int dst_rows, const std::string& dst_path)
{
    //_/_/_/ load a source image and repeat its edges
    
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image(src_path);
    Halide::Func src_image {};
    src_image = Halide::BoundaryConditions::repeat_edge(input);
    
    //_/_/_/ describe algorithm
    
    const int src_cols = input.width();
    const int src_rows = input.height();

    const float sc = static_cast<float>(src_cols) / dst_cols;
    const float sr = static_cast<float>(src_rows) / dst_rows;
    
    Halide::Var i {};
    Halide::Var j {};
    Halide::Var c {};
    
    auto fj = j * sr;
    auto cj0 = Halide::cast<int>(fj);
    auto cj1 = cj0 + 1;
    auto dj = fj - cj0;
    
    auto fi = i * sc;
    auto ci0 = Halide::cast<int>(fi);
    auto ci1 = ci0 + 1;
    auto di = fi - ci0;
    
    const auto c0 = (1.0f - dj) * (1.0f - di);
    const auto c1 = (1.0f - dj) * di;
    const auto c2 = dj * (1.0f - di);
    const auto c3 = dj * di;

    const auto& src_pixel0 = src_image(ci0, cj0, c);
    const auto& src_pixel1 = src_image(ci1, cj0, c);
    const auto& src_pixel2 = src_image(ci0, cj1, c);
    const auto& src_pixel3 = src_image(ci1, cj1, c);

    Halide::Func dst_image {};
    dst_image(i, j, c) = Halide::saturating_cast<uint8_t>(c0 * src_pixel0 + c1 * src_pixel1 + c2 * src_pixel2 + c3 * src_pixel3);

    //_/_/_/ describe scheduling
    
//    dst_image.vectorize(i, 4).parallel(j);
    
    Halide::Var i_inner, j_inner;
    dst_image.tile(i, j, i_inner, j_inner, 256, 32).vectorize(i_inner, 8).parallel(j);
    
    //_/_/_/ run
    
    auto start = std::chrono::system_clock::now();
    Halide::Buffer<uint8_t> output = dst_image.realize(dst_cols, dst_rows, input.channels());
    auto end = std::chrono::system_clock::now();
    std::cout << boost::format("halide access: %1% msec\n") % std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    //_/_/_/ save it
    
//    Halide::Tools::save(output, dst_path);

    return 1;
}
