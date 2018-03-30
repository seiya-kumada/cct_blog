//
//  tutorial_14.cpp
//  tutorials
//
//  Created by 熊田　聖也 on 2018/03/27.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "tutorial_14.hpp"
#include <Halide.h>

Halide::Type valid_halide_types[] = {
    
    Halide::UInt(8),
    Halide::UInt(16),
    Halide::UInt(32),
    Halide::UInt(64),
    Halide::Int(8),
    Halide::Int(16),
    Halide::Int(32),
    Halide::Int(64),
    Halide::Float(32),
    Halide::Float(64),
    Halide::Handle(),
};

int tutorial_14()
{
    assert(Halide::UInt(8).bits() == 8);
    assert(Halide::Int(8).is_int());
    
    Halide::Type t {Halide::UInt(8)};
    t = t.with_bits(t.bits() * 2);
    assert(t == Halide::UInt(16));
    
    assert(Halide::type_of<float>() == Halide::Float(32));
    
    return 0;
}
