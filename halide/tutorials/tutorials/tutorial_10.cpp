//
//  tutotial_10.cpp
//  tutorials
//
//  Created by 熊田聖也 on 2018/02/10.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "tutorial_10.hpp"
#include <Halide.h>

int tutorial_10()
{
    Halide::Func brigher {};
    Halide::Var x {};
    Halide::Var y {};
    
    Halide::Param<uint8_t> offset {};
    
    Halide::ImageParam input {Halide::type_of<uint8_t>(), 2};
    
    brigher(x, y) = input(x, y) + offset;
    
    brigher.vectorize(x, 16).parallel(y);
    
    const auto lib_path = "/Users/uu103907/Projects/cct_blog/halide/tutorials/tutorials/lesson_10_halide";
    brigher.compile_to_static_library(lib_path, {input, offset}, "brighter");
    
    return 1;
}
