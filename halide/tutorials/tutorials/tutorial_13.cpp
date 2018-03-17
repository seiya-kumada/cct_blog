//
//  tutorial_13.cpp
//  tutorials
//
//  Created by 熊田聖也 on 2018/03/17.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "tutorial_13.hpp"
#include <Halide.h>
#include <halide_image_io.h>

int tutorial_13()
{
    Halide::Func single_valued {};
    Halide::Var x {"x"};
    Halide::Var y {"y"};
    single_valued(x, y) = x + y;
    
    Halide::Func color_image {"color_image"};
    Halide::Var c {"c"};

    color_image(x, y, c) = Halide::cast<uint8_t>(Halide::select(
        c == 0, 245, // red
        c == 1, 42,  // green
        132));       // blue
//    color_image(x, y, 0) = 255;
    Halide::Buffer<uint8_t> output {100, 100, 3};
    color_image.realize(output);
    Halide::Tools::save(output, "/Users/kumada/Projects/cct_blog/halide/images/outputs/hoge.png");

    Halide::Func brighter {};
    brighter(x, y, c) = color_image(x, y, c) + 10;
  
    // cを0から3まで展開する。
//    brighter.bound(c, 0, 3).unroll(c);
    
    // こうやって書くとスケジューリングが難しくなる。
    Halide::Func func_array[3];
    func_array[0](x, y) = x + y;
    func_array[1](x, y) = Halide::sin(x);
    func_array[2](x, y) = Halide::cos(y);
    
    Halide::Func multi_valued {};
    multi_valued(x, y) = Halide::Tuple(x + y, Halide::sin(x * y));
    
    {
        Halide::Realization r = multi_valued.realize(80, 60);
        if (r.size() == 2)
        {
            std::cout << "OK\n";
        }
        const Halide::Buffer<int>& im0 = r[0];
        const Halide::Buffer<float>& im1 = r[1];
        if (im0(30, 40) == 30 + 40)
        {
            std::cout << "OK\n";
        }
        
        if (im1(30, 40) == std::sinf(30 * 40))
        {
            std::cout << "OK\n";
        }
        
    }
    
    
    Halide::Func multi_valued_2 {};
    multi_valued_2(x, y) = {x + y, Halide::sin(x * y)};
    Halide::Expr integer_part = multi_valued_2(x, y)[0];
    Halide::Expr floating_part = multi_valued_2(x, y)[1];
    Halide::Func consumer {};
    consumer(x, y) = {integer_part + 10, floating_part + 10.0f};
    
    {
        Halide::Func input_func {};
        input_func(x) = Halide::sin(x);
        Halide::Buffer<float> input = input_func.realize(100);
        
        Halide::Func arg_max {};
        arg_max() = {0, input(0)};
        
        Halide::RDom r{1, 99}; // Reduction Domain
        auto old_index = arg_max()[0];
        auto old_max = arg_max()[1];
        auto new_index = Halide::select(old_max < input(r), r, old_index);
        auto new_max = Halide::max(input(r), old_max);
        arg_max() = {new_index, new_max};
        
        
        int arg_max_0 = 0;
        float arg_max_1 = input(0);
        for (auto r = 1; r < 100; ++r)
        {
            int old_index = arg_max_0;
            float old_max = arg_max_1;
            int new_index = old_max < input(r) ? r : old_index;
            float new_max = std::max(input(r), old_max);
            arg_max_0 = new_index;
            arg_max_1 = new_max;
            
        }
        
        {
            Halide::Realization r = arg_max.realize();
            const Halide::Buffer<int>& r0 = r[0];
            const Halide::Buffer<float>& r1 = r[1];
            assert(arg_max_0 == r0(0));
            assert(arg_max_1 == r1(0));
        }
    }
    // Tuples for user-defined types.
    return 1;
}
