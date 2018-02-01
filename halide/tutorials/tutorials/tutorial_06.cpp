//
//  tutorial_06.cpp
//  tutorials
//
//  Created by 熊田聖也 on 2018/02/01.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "tutorial_06.hpp"
#include <Halide.h>

int tutorial_06()
{
    Halide::Func gradient {"gradient"};
    Halide::Var x {"x"};
    Halide::Var y {"y"};
    gradient(x, y) = x + y;
    
    gradient.trace_stores();
    
//    Halide::Buffer<int> result {5, 3};
//    gradient.realize(result);
    
    constexpr int WINDOW_WIDTH {5};
    constexpr int WINDOW_HEIGHT {7};
    Halide::Buffer<int> shifted {WINDOW_WIDTH, WINDOW_HEIGHT};
    constexpr int LEFT = 100;
    constexpr int TOP = 50;
    shifted.set_min(LEFT, TOP);
    
    gradient.realize(shifted);
    
    for (auto y = TOP; y < TOP + WINDOW_HEIGHT; ++y)
    {
        for (auto x = LEFT; x < LEFT + WINDOW_WIDTH; ++x)
        {
            if (shifted(x, y) != x + y)
            {
                std::cout << "Something went wrong!\n";
            }
        }
    }
    
    return 1;
}
