//
//  tutorial_09.cpp
//  tutorials
//
//  Created by 熊田聖也 on 2018/02/04.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "tutorial_09.hpp"
#include <Halide.h>
#include <iostream>
#include <array>

#ifdef __SSE2__
#include <emmintrin.h>
#endif
#include <halide_image_io.h>

namespace Tools = Halide::Tools;

int tutorial_09()
{
    std::cout << __SSE2__ << std::endl;
    Halide::Var x {"x"};
    Halide::Var y {"y"};
    
    const std::string SRC_PATH {"/Users/kumada/Projects/cct_blog/halide/images/inputs/gray.jpg"};
    Halide::Buffer<uint8_t> input = Tools::load_image(SRC_PATH);
    {
        Halide::Func f {};
        f(x, y) = x + y;
        
        f(3, 7) = 42;
        f(x, y) = f(x, y) + 17;
        f(x, 3) = f(x, 0) * f(x, 10);
        f(0, y) = f(0, y) / f(3, y);
        f.realize(100, 101);

        Halide::Func g {"g"};
        g(x, y) = x + y;
        g(2, 1) = 42;
        g(x, 0) = g(x, 1);
        g.trace_loads();
        g.trace_stores();
        
        g.realize(4, 4);
    }
    
    {
        Halide::Func f {};
        f(x, y) = (x + y) / 100.0f;
        
        Halide::RDom r {0, 50};
        f(x, r) = f(x, r) * f(x, r);
        Halide::Buffer<float> halide_result = f.realize(100, 100);
        
    }
    
    {
        Halide::Func histogram {"histogram"};
        histogram(x) = 0;
        
        Halide::Var r_x {};
        Halide::Var r_y {};
        Halide::RDom r {0, input.width(), 0, input.height()};
        histogram(input(r.x, r.y)) += 1;
        Halide::Buffer<int> halide_result = histogram.realize(256);
        
        std::array<int, 256> c_result {};
        for (auto x = 0; x < 256; ++x)
        {
            c_result[x] = 0;
        }
        
        for (auto r_y = 0; r_y < input.height(); ++r_y)
        {
            for (auto r_x = 0; r_x < input.width(); ++r_x)
            {
                c_result[input(r_x, r_y)] += 1;
            }
        }
        
        for (auto x = 0; x < 256; ++x)
        {
            if (c_result[x] != halide_result(x))
            {
                std::cout << "error\n";
                return -1;
            }
        }
    }
    
    {
        Halide::Func f {};
        f(x, y) = x * y;
        f.trace_stores();
//        f(x, 0) = f(x, 8);
//        f(0, y) = f(0, y) + 2;
//
//        f.vectorize(x, 4).parallel(y);
        f.vectorize(x, 4);
        Halide::Buffer<int> halide_result = f.realize(4, 4);
        
        
    }
    {
        Halide::Func f {};
        f.trace_stores();
        
        // x軸方向にベクトル化、y軸方向に並列化。
        f(x, y) = x * y;
        f.vectorize(x, 4).parallel(y);

        // 最初の更新では、x軸方向にベクトル化。
        f(x, 1) = f(x, 2);
        f.update(0).vectorize(x, 4);
        
        // 次の更新では、y軸方向に分離と並列化。
        f(0, y) = f(0, y) + 2;
        Halide::Var yo {};
        Halide::Var yi {};
        f.update(1).split(y, yo, yi, 4).parallel(yo);
        
        Halide::Buffer<int> halide_result = f.realize(4, 4);
        
        
    }

    return 1;
}
