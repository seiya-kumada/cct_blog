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
    
    Halide::Var i_inner, j_inner;
    auto x_vector_size = 64;
    dst_image.compute_root();
    dst_image.tile(i, j, i_inner, j_inner, x_vector_size, 4).vectorize(i_inner, 16).parallel(j);

    //_/_/_/ run
    
    Halide::Buffer<uint8_t> output {dst_cols, dst_rows, input.channels()};
    constexpr auto ITERATIONS = 10;
    for (auto i = 0; i < ITERATIONS; ++i )
    {
        auto start = std::chrono::system_clock::now();
        dst_image.realize(output);
        auto end = std::chrono::system_clock::now();
        std::cout << boost::format("halide access[%1%]: %2% msec\n")
            % i
            % std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    
    //_/_/_/ save it
    
    Halide::Tools::save(output, dst_path);

    return 1;
}
