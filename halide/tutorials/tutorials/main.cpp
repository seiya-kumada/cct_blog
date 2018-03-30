//
//  main.cpp
//  tutorials
//
//  Created by 熊田聖也 on 2018/01/15.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//
//  halideのtutorial

#include <Halide.h>


int main(int argc, char * argv[]) {
    return  Halide::Internal::generate_filter_main(argc, argv, std::cerr);
}
