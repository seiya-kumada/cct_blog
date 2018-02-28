//
//  tutorial_10_2.cpp
//  tutorials
//
//  Created by 熊田聖也 on 2018/02/10.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "tutorial_10_2.hpp"
#include "lesson_10_halide.h"
#include <Halide.h>

int tutorial_10_2()
{
    Halide::Runtime::Buffer<uint8_t> input(640, 480), output(640, 480);
    int offset = 5;
    int error = brighter(input, offset, output);
    
    if (error)
    {
        std::cout << "Halide returned an error:\n";
        return -1;
    }
    
    std::cout << "Success!\n";
    return 1;
}
