//
//  tutorial_15.cpp
//  tutorials
//
//  Created by 熊田　聖也 on 2018/03/27.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "tutorial_15.hpp"
#include <Halide.h>
using namespace Halide;
class MyFirstGenerator : public Halide::Generator<MyFirstGenerator>
{
public:
    Input<uint8_t> offset {"offset"};
    Input<Halide::Buffer<uint8_t>> input {"input", 2};
    
    Output<Halide::Buffer<uint8_t>> brighter {"brighter", 2};
    
    Halide::Var x {"x"};
    Halide::Var y {"y"};
    
    void generate()
    {
        brighter(x, y) = input(x, y) + offset;
        brighter.vectorize(x, 16).parallel(y);
    }
};

HALIDE_REGISTER_GENERATOR(MyFirstGenerator, my_first_generator)
//int tutorial_15(int argc, char * argv[])
//{
//    return Halide::Internal::generate_filter_main(argc, argv, std::cerr);
//}

