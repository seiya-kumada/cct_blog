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
    //load a source image
    Halide::Buffer<uint8_t> src_image = Halide::Tools::load_image(src_path);
    
    //_/_/_/ describe algorithm
    
    const int src_cols = src_image.width();
    const int src_rows = src_image.height();
    
    const float sc = static_cast<float>(src_cols) / dst_cols;
    const float sr = static_cast<float>(src_rows) / dst_rows;
    
    Halide::Var i {};
    Halide::Var j {};
    Halide::Var c {};
    
    auto fj = j * sr;
    auto cj0 = Halide::cast<int>(fj);
    auto cj1 = Halide::clamp(cj0 + 1, 0, src_rows - 1);
    auto dj = fj - cj0;
    
    auto fi = i * sc;
    auto ci0 = Halide::cast<int>(fi);
    auto ci1 = Halide::clamp(ci0 + 1, 0, src_cols - 1);
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
    dst_image(i, j, c) = Halide::cast<uint8_t>(c0 * src_pixel0 + c1 * src_pixel1 + c2 * src_pixel2 + c3 * src_pixel3);
    
    //_/_/_/ describe scheduling
    
//    dst_image.parallel(j);
//    Halide::Var inner_i {};
//    Halide::Var outer_i {};
//    dst_image.split(i, outer_i, inner_i, 4);
//    dst_image.vectorize(inner_i);

//    Halide::Var i_inner {};
//    Halide::Var i_outer {};
//    Halide::Var j_inner {};
//    Halide::Var j_outer {};
//    Halide::Var tile_index {};
//    dst_image
//        .tile(i, j, i_outer, j_outer, i_inner, j_inner, 32, 32)
//        .fuse(i_outer, j_outer, tile_index)
//        .parallel(tile_index);

    dst_image.vectorize(i, 4).parallel(j);
    
    //_/_/_/ run
    
    auto start = std::chrono::system_clock::now();
    Halide::Buffer<uint8_t> output = dst_image.realize(dst_cols, dst_rows, src_image.channels());
    auto end = std::chrono::system_clock::now();
    std::cout << boost::format("halide access: %1% msec\n") % std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    //_/_/_/ save it
    
    Halide::Tools::save(output, dst_path);

    return 1;
}
