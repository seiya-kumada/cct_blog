//
//  main.cpp
//  sample_3
//
//  Created by 熊田　聖也 on 2018/02/28.
//  Copyright © 2018年 熊田　聖也. All rights reserved.
//

#include <iostream>

#include <iostream>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <boost/format.hpp>
#include <vector>
#include "normal.hpp"
#include "halide.hpp"
#include <chrono>

int halide_process(const char* argv[])
{
    resize_with_halide();
    return 0;
}

int main(int argc, const char * argv[])
{
    halide_process(argv);
    return 0;
}

