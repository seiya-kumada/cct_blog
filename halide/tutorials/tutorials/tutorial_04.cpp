//
//  tutorial_04.cpp
//  tutorials
//
//  Created by 熊田聖也 on 2018/01/21.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "tutorial_04.hpp"
#include <Halide.h>
#include <chrono>
// Printing additional context.から
int tutorial_04()
{
    Halide::Var x {"x"};
    Halide::Var y {"y"};

    auto start = std::chrono::system_clock::now();
    {
        Halide::Func gradient {"gradient"};
        gradient(x, y) = x + y;
//        gradient.trace_stores();
//
//        std::cout << "Evaluating gradient\n";
        Halide::Buffer<int> output {gradient.realize(8, 8)};
        std::cout << &output << std::endl;
//
    }
    auto end = std::chrono::system_clock::now();
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << msec << "[msec]\n";
    
    start = end;
    {
        Halide::Func parallel_gradient {"parallel_gradinet"};
        parallel_gradient(x, y) = x + y;
//        parallel_gradient.trace_stores();

        // スケジューリングの決定
        // yのループを並列化する。
        parallel_gradient.parallel(y);

        Halide::Buffer<int> output {parallel_gradient.realize(8, 8)};
        std::cout << &output << std::endl;
    }
    end = std::chrono::system_clock::now();
    msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << msec << "[msec]\n";
    
    {
        Halide::Func f {};
        f(x, y) = Halide::sin(x) + Halide::cos(y);
        
        Halide::Func g {};
        g(x, y) = Halide::sin(x) + Halide::print(Halide::cos(y));
        g.realize(4, 4);
    }
    return 1;
}
