//
//  tutorial_03.cpp
//  tutorials
//
//  Created by 熊田聖也 on 2018/01/21.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "tutorial_03.hpp"
#include <Halide.h>

const std::string DST_PATH {"/Users/kumada/Projects/cct_blog/halide/images/outputs/gradient.html"};


int tutorial_03()
{
    Halide::Func gradient {"gradient"};
    Halide::Var x {"x"};
    Halide::Var y {"y"};
    
    gradient(x, y) = x + y;
    
//    Halide::Buffer<int> output {gradient.realize(8, 8)};
    // 疑似コードが出力される。
    gradient.compile_to_lowered_stmt(DST_PATH, {}, Halide::HTML);
    
    return 1;
}
