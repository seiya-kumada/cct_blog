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
#include <boost/range/irange.hpp>
#include <boost/range/algorithm/for_each.hpp>

//https://www.slideshare.net/fixstars/halide-82788728

int resize_with_halide()
{
    Halide::ImageParam input {Halide::type_of<uint8_t>(), 3};
    
    //_/_/_/ load a source image and repeat its edges
    
    Halide::Func src_image {};
    src_image = Halide::BoundaryConditions::repeat_edge(input);
    
    //_/_/_/ describe algorithm
    Halide::Param<int> src_rows {};
    Halide::Param<int> src_cols {};
    Halide::Param<int> dst_rows {};
    Halide::Param<int> dst_cols {};
    
    const float sc = static_cast<float>(src_cols.get()) / dst_cols.get();
    const float sr = static_cast<float>(src_rows.get()) / dst_rows.get();
    
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

    //_/_/_/ save a static library
    const auto path = "/Users/uu103907/Projects/cct_blog/halide/sample_3/sample_3/resize";
    dst_image.compile_to_static_library(
        path,
        {input, src_rows, src_cols, dst_rows, dst_cols},
        "resize");
    
    return 1;
}
