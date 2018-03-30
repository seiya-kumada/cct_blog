//
//  main.cpp
//  sample_3
//
//  Created by 熊田　聖也 on 2018/02/28.
//  Copyright © 2018年 熊田　聖也. All rights reserved.
//

#include "halide.hpp"

int main(int argc, const char * argv[])
{
    make_halide_static_library_for_resize();
    return 0;
}

