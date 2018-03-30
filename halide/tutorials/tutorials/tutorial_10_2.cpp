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
    
    for (int y = 0; y < 480; y++) {
        for (int x = 0; x < 640; x++) {
            uint8_t input_val = input(x, y);
            uint8_t output_val = output(x, y);
            uint8_t correct_val = input_val + offset;
            if (output_val != correct_val) {
                printf("output(%d, %d) was %d instead of %d\n",
                       x, y, output_val, correct_val);
                return -1;
            }
        }
    }
    std::cout << "Success!\n";
    return 1;
}

