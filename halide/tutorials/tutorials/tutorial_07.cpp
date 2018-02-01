//
//  tutorial_07.cpp
//  tutorials
//
//  Created by 熊田聖也 on 2018/02/01.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "tutorial_07.hpp"
#include <Halide.h>
#include <halide_image_io.h>
namespace Tools = Halide::Tools;

int tutorial_07()
{
    Halide::Var x {"x"};
    Halide::Var y {"y"};
    Halide::Var c {"c"};
    
    const std::string SRC_PATH {"../images/inputs/tutorial_02.jpg"};
    const std::string DST_PATH {"../images/outputs/tutorial_07.jpg"};
    
    // 最初に、水平方向にぼかして、次に垂直方向にぼかす。
    {
        Halide::Buffer<uint8_t> input = Tools::load_image(SRC_PATH);
        
        // 16bitに引き上げる。
        Halide::Func input_16 {"input_16"};
        input_16(x, y, c) = Halide::cast<uint16_t>(input(x, y, c));
        
        // 水平方向にぼかす。
        Halide::Func blur_x {"blur_x"};
        blur_x(x, y, c) = (input_16(x - 1, y, c) + 2 * input_16(x, y, c) + input_16(x + 1, y, c)) / 4;
        
        // 垂直方向にぼかす。
        Halide::Func blur_y {"blur_y"};
        blur_y(x, y, c) = (blur_x(x, y - 1, c) + 2 * blur_x(x, y, c) + blur_x(x, y + 1, c)) / 4;
        
        // 8bitに戻す。
        Halide::Func output {"output"};
        output(x, y, c) = Halide::cast<uint8_t>(blur_y(x, y, c));
        
        // パイプラインは常にfeed-forwar graphである。
        
//        Halide::Buffer<uint8_t> result {output.realize(input.width(), input.height(), input.channels())};
    
        const int window_width {input.width() - 2};
        const int window_height {input.height() - 2};
        Halide::Buffer<uint8_t> window {window_width, window_height, input.channels()};
        constexpr int LEFT = 1;
        constexpr int TOP = 1;
        window.set_min(LEFT, TOP);
        output.realize(window);
        
        Tools::save_image(window, DST_PATH);
    }
    return 1;
}
