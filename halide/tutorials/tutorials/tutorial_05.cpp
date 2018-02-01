//
//  tutorial_05.cpp
//  tutorials
//
//  Created by 熊田聖也 on 2018/01/27.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "tutorial_05.hpp"
#include <chrono>
#include <Halide.h>
#include <boost/format.hpp>

int tutorial_05()
{
    Halide::Var x {"x"};
    Halide::Var y {"y"};
    
    {
        Halide::Func gradient {"gradient"};
        gradient(x, y) = x + y;
        gradient.trace_stores();
        
        std::cout << "Evaluating gradient row-major\n";
        Halide::Buffer<int> output {gradient.realize(4, 4)};
        
        std::cout << "Equivalent C:\n";
        for (auto y = 0; y < 4; ++y)
        {
            for (auto x = 0; x < 4; ++x)
            {
                std::cout << boost::format("Evaluating at x = %1%, y = %2%: %3%") %x %y %(x + y) << std::endl;
            }
        }
        
        std::cout << "Pseudo-code for the schedule\n";
        gradient.print_loop_nest();
    }
    
    {
        Halide::Func gradient {"gradiet_col_major"};
        gradient(x, y) = x + y;
        gradient.trace_stores();
        
        gradient.reorder(y, x);
        std::cout << "Evaluating gradient column-major\n";
        Halide::Buffer<int> output {gradient.realize(4, 4)};
        
    }
    
    {
        Halide::Func gradient {"gradient_split"};
        gradient(x, y) = x + y;
        gradient.trace_stores();
        
        Halide::Var x_outer {};
        Halide::Var x_innter {};
        gradient.split(x, x_outer, x_innter, 2);
        
        
        std::cout << "Equivalent C:\n";
        for (auto y = 0; y < 4; ++y)
        {
            for (auto x_outer = 0; x_outer < 2; ++x_outer)
            {
                for (auto x_innter = 0; x_innter < 2; ++x_innter)
                {
                    auto x = x_outer * 2 + x_innter;
                    std::cout << boost::format("Evaluatinng at x = %1%, y = %2%: %3%") %x %y %(x + y) << std::endl;
                }
            }
        }
    }
    
    {
        Halide::Func gradient {"gradient_fused"};
        gradient(x, y) = x + y;
        
        Halide::Var fused {};
        gradient.fuse(x, y, fused);
        std::cout << "Evaluating gradient with x and y fused\n";
        Halide::Buffer<int> output {gradient.realize(4, 4)};
        
        for (auto fused = 0; fused < 4 * 4; ++fused)
        {
            auto y = fused / 4;
            auto x = fused % 4;
            std::cout << boost::format("Evaluatinng at x = %1%, y = %2%: %3%") %x %y %(x + y) << std::endl;
        }
    }
    
    {
        Halide::Func gradient {"gradient_tiled"};
        gradient(x, y) = x + y;
        gradient.trace_stores();
        
        Halide::Var x_outer {};
        Halide::Var x_inner {};
        Halide::Var y_outer {};
        Halide::Var y_inner {};
//        gradient.split(x, x_outer, x_inner, 4);
//        gradient.split(y, y_outer, y_inner, 4);
//        gradient.reorder(x_inner, y_inner, x_outer, y_outer);
        gradient.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4);
        Halide::Buffer<int> output {gradient.realize(8, 8)};
        
    }
    
    {
        Halide::Func gradient {"gradient_in_vectors"};
        gradient(x, y) = x + y;
        gradient.trace_stores();
        
        constexpr int X_INNER_SIZE {4};
        Halide::Var x_outer {};
        Halide::Var x_inner {};
        gradient.split(x, x_outer, x_inner, X_INNER_SIZE);
        gradient.vectorize(x_inner);
        
        Halide::Buffer<int> output {gradient.realize(8, 4)};
        
    
    }
    
    {
        Halide::Func gradient {"gradient_unroll"};
        gradient(x, y) = x + y;
        gradient.trace_stores();
        
        constexpr int SPLIT_FACTOR {2};
        Halide::Var x_outer {};
        Halide::Var x_inner {};
        gradient.split(x, x_outer, x_inner, SPLIT_FACTOR);
        gradient.unroll(x_inner);
        
        constexpr int HEIGHT {3};
        constexpr int WIDTH {4};
        Halide::Buffer<int> result {gradient.realize(WIDTH, HEIGHT)};
        
        for (auto y = 0; y < HEIGHT; ++y)
        {
            for (auto x_outer = 0; x_outer < WIDTH / SPLIT_FACTOR; ++x_outer)
            {
                {
                    constexpr auto x_inner {0};
                    const auto x {x_outer * SPLIT_FACTOR + x_inner};
                    std::cout << boost::format("Evaluating at x = %1%, y = %2%: %3%\n") %x %y %(x + y);
                }
                {
                    constexpr auto x_inner {1};
                    const auto x {x_outer * SPLIT_FACTOR + x_inner};
                    std::cout << boost::format("Evaluating at x = %1%, y = %2%: %3%\n") %x %y %(x + y);
                }
            }
        }
        
    }
    
    {
        Halide::Func gradient {"gradient_split_7x2"};
        gradient(x, y) = x + y;
        gradient.trace_stores();
        
        Halide::Var x_outer {};
        Halide::Var x_inner {};
        gradient.split(x, x_outer, x_inner, 3);
        Halide::Buffer<int> output {gradient.realize(7, 2)};
        
        
    }
    
    {
        Halide::Func gradient {"gradient_fused_tiles"};
        gradient(x, y) = x + y;
        gradient.trace_stores();
        
        Halide::Var x_outer {};
        Halide::Var y_outer {};
        Halide::Var x_inner {};
        Halide::Var y_inner {};
        Halide::Var tile_index {};
        
        gradient.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4);
        gradient.fuse(x_outer, y_outer, tile_index);
        gradient.parallel(tile_index);
        
        Halide::Buffer<int> output {gradient.realize(8, 8)};
        
//        for (auto tile_index = 0; tile_index < 4; ++tile_index)
//        {
//            auto y_outer = tile_index / 2; // 0,0,1,1
//            auto x_outer = tile_index % 2; // 0,1,0,1
//
//            for (auto y_inner = 0; y_inner < 4; ++y_inner)
//            {
//                auto y = y_outer * 4 + y_inner;
//                for (auto x_inner = 0; x_inner < 4; ++x_inner)
//                {
//                    auto x = x_outer * 4 + x_inner;
//                    std::cout << boost::format("Evaluating at x = %1%, y = %2%: %3%") %x %y %(x + y) << std::endl;
//                }
//            }
//
//        }
        
    }
    
    {
        Halide::Func gradient_fast {"gradient_fast"};
        gradient_fast(x, y) = x + y;
        
        Halide::Var x_outer {};
        Halide::Var y_outer {};
        Halide::Var x_inner {};
        Halide::Var y_inner {};
        Halide::Var tile_index {};
        
        gradient_fast
            .tile(x, y, x_outer, y_outer, x_inner, y_inner, 64, 64)
            .fuse(x_outer, y_outer, tile_index)
            .parallel(tile_index);
        
        Halide::Var x_inner_outer {};
        Halide::Var y_inner_outer {};
        Halide::Var x_vectors {};
        Halide::Var y_pairs {};
        
        gradient_fast
            .tile(x_inner, y_inner, x_inner_outer, y_inner_outer, x_vectors, y_pairs, 4, 2)
            .vectorize(x_vectors)
            .unroll(y_pairs);
        
        Halide::Buffer<int> result {gradient_fast.realize(350, 250)};
    }
    
    {
        constexpr int WIDTH {4};
        constexpr int HEIGHT {6};
        
        Halide::Func gradient {"gradient_tiled"};
        gradient(x, y) = x + y;
        gradient.trace_stores();
        
        Halide::Var x_outer {};
        Halide::Var x_inner {};
        Halide::Var y_outer {};
        Halide::Var y_inner {};
        constexpr int TILE_WIDRH {2};
        constexpr int TILE_HEIGHT {3};
        gradient.split(x, x_outer, x_inner, TILE_WIDRH);
        gradient.split(y, y_outer, y_inner, TILE_HEIGHT);
        gradient.reorder(x_inner, y_inner, x_outer, y_outer);
        
        Halide::Buffer<int> output {gradient.realize(WIDTH, HEIGHT)};
    }

    {
        constexpr int WIDTH {4};
        constexpr int HEIGHT {6};
        
        Halide::Func gradient {"gradient_tiled"};
        gradient(x, y) = x + y;
        gradient.trace_stores();
        
        Halide::Var x_outer {};
        Halide::Var x_inner {};
        Halide::Var y_outer {};
        Halide::Var y_inner {};
        constexpr int TILE_WIDTH {2};
        constexpr int TILE_HEIGHT {3};
        gradient.tile(x, y, x_outer, y_outer, x_inner, y_inner, TILE_WIDTH, TILE_HEIGHT);
        
//        gradient.split(x, x_outer, x_inner, TILE_WIDRH);
//        gradient.split(y, y_outer, y_inner, TILE_HEIGHT);
//        gradient.reorder(x_inner, y_inner, x_outer, y_outer);
        
        Halide::Buffer<int> output {gradient.realize(WIDTH, HEIGHT)};
    }

    return 1;
}

//int tutorial_055()
//{
//    Halide::Var x {"x"};
//    Halide::Var y {"y"};
//    constexpr int SIZE = 4000;
//
//    {
//        Halide::Func gradient {"gradiet_col_major"};
//        gradient(x, y) = x + y;
//        gradient.reorder(y, x);
//        std::cout << "> Evaluating gradient column-major\n";
//
//        auto start = std::chrono::system_clock::now();
//        Halide::Buffer<int> output {gradient.realize(SIZE, SIZE)};
//        auto end = std::chrono::system_clock::now();
//        auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//        std::cout << msec << "[msec]\n";
//
//    }
//
//    {
//        Halide::Func gradient {"gradient"};
//        gradient(x, y) = x + y;
//
//        std::cout << "> Evaluating gradient row-major\n";
//        auto start = std::chrono::system_clock::now();
//        Halide::Buffer<int> output {gradient.realize(SIZE, SIZE)};
//        auto end = std::chrono::system_clock::now();
//        auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//        std::cout << msec << "[msec]\n";
//    }
//
//
//    {
//        Halide::Func gradient {"gradient_split"};
//        gradient(x, y) = x + y;
//
//        Halide::Var x_outer {};
//        Halide::Var x_innter {};
//        gradient.split(x, x_outer, x_innter, 2);
//
//        std::cout << "> Split a variable into two\n";
//
//        auto start = std::chrono::system_clock::now();
//        Halide::Buffer<int> output {gradient.realize(SIZE, SIZE)};
//        auto end = std::chrono::system_clock::now();
//        auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//        std::cout << msec << "[msec]\n";
//
//    }
//
//    return 1;
//}


