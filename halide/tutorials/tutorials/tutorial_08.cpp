//
//  tutorial_08.cpp
//  tutorials
//
//  Created by 熊田聖也 on 2018/02/03.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "tutorial_08.hpp"
#include <Halide.h>
#include <array>

int tutorial_08()
{
    Halide::Var x {"x"};
    Halide::Var y {"y"};
    
    {
        Halide::Func producer {"producer_default"};
        Halide::Func consumer {"consumer_default"};
        
        // the first stage
        producer(x, y) = Halide::sin(x * y);
        
        // the second stage
        consumer(x, y) = (producer(x, y) + producer(x, y + 1) + producer(x + 1, y) + producer(x + 1, y + 1)) / 4;
    
        // turn on tracing
        consumer.trace_stores();
        producer.trace_stores();
        
        std::cout << "evaluate producer-consumer pipeline with default schedule\n";
        consumer.realize(4, 4);
    }
    
    {
        Halide::Func producer {"producer_default"};
        Halide::Func consumer {"consumer_default"};
        
        // the first stage
        producer(x, y) = Halide::sin(x * y);
        
        // the second stage
        consumer(x, y) = (producer(x, y) + producer(x, y + 1) + producer(x + 1, y) + producer(x + 1, y + 1)) / 4;

        producer.compute_root();
        
        consumer.trace_stores();
        producer.trace_stores();
        
        std::cout << "evaluate producer-consumer pipeline with default schedule\n";
        consumer.realize(4, 4);
    }
    
    {
        Halide::Func producer {"producer_default"};
        Halide::Func consumer {"consumer_default"};
        
        // the first stage
        producer(x, y) = Halide::sin(x * y);
        
        // the second stage
        consumer(x, y) = (producer(x, y) + producer(x, y + 1) + producer(x + 1, y) + producer(x + 1, y + 1)) / 4;

        producer.compute_at(consumer, y);
        
        std::array<std::array<float, 4>, 4> result {};
        for (auto y = 0; y < 4; ++y)
        {
            std::array<std::array<float, 2>, 5> producer_storage {};
            for (auto py = y; py < y + 2; ++py)
            {
                for (auto px = 0; px < 5; ++px)
                {
                    producer_storage[py - y][px] = sin(px * py);
                }
            }
            
            for (auto x = 0; x < 4; ++x)
            {
                result[y][x] = (producer_storage[0][x] + producer_storage[1][x] + producer_storage[0][x + 1] + producer_storage[1][1 + x]) / 4;
            }
        }
    }
    
    {
        Halide::Func producer {"producer_default"};
        Halide::Func consumer {"consumer_default"};
        
        // the first stage
        producer(x, y) = Halide::sin(x * y);
        
        // the second stage
        consumer(x, y) = (producer(x, y) + producer(x, y + 1) + producer(x + 1, y) + producer(x + 1, y + 1)) / 4;

        producer.store_root();
        producer.compute_at(consumer, y);
        
        producer.trace_stores();
        consumer.trace_stores();
        
        consumer.realize(4, 4);
        
    }
    
    {
        Halide::Func producer {"producer_default"};
        Halide::Func consumer {"consumer_default"};
        
        // the first stage
        producer(x, y) = Halide::sin(x * y);
        
        // the second stage
        consumer(x, y) = (producer(x, y) + producer(x, y + 1) + producer(x + 1, y) + producer(x + 1, y + 1)) / 4;

        producer.store_root().compute_at(consumer, x);
     
        producer.trace_stores();
        consumer.trace_stores();
        
        std::cout << "HHH\n";
        consumer.realize(4, 4);
    }
    
    {
        Halide::Func producer {"producer_default"};
        Halide::Func consumer {"consumer_default"};
        
        // the first stage
        producer(x, y) = Halide::sin(x * y);
        
        // the second stage
        consumer(x, y) = (producer(x, y) + producer(x, y + 1) + producer(x + 1, y) + producer(x + 1, y + 1)) / 4;

        Halide::Var x_outer {};
        Halide::Var y_outer {};
        Halide::Var x_inner {};
        Halide::Var y_inner {};
        consumer.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4);
        producer.compute_at(consumer, x_outer);
        
        producer.trace_stores();
        consumer.trace_stores();
        consumer.realize(8, 8);
        
        
    }
    
    {
        Halide::Func producer {"producer_default"};
        Halide::Func consumer {"consumer_default"};
        
        // the first stage
        producer(x, y) = Halide::sin(x * y);
        
        // the second stage
        consumer(x, y) = (producer(x, y) + producer(x, y + 1) + producer(x + 1, y) + producer(x + 1, y + 1)) / 4;

        Halide::Var yo {};
        Halide::Var yi {};
        consumer.split(y, yo, yi, 16);
        consumer.parallel(yo);
        consumer.vectorize(x, 4);
        
        producer.store_at(consumer, yo);
        producer.compute_at(consumer, yi);
        producer.vectorize(x, 4);
        
        Halide::Buffer<float> halide_result = consumer.realize(160, 160);
        
    }
    return 1;
}
