//
//  tutorial_02.cpp
//  tutorials
//
//  Created by 熊田聖也 on 2018/01/21.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "tutorial_02.hpp"
#include <Halide.h>
#include <halide_image_io.h>

const std::string SRC_PATH {"/Users/kumada/Projects/cct_blog/halide/images/inputs/tutorial_02.jpg"};
const std::string DST_PATH {"/Users/kumada/Projects/cct_blog/halide/images/outputs/tutorial_02.jpg"};


int tutorial_022()
{
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image(SRC_PATH);
    
    // pipelineを表現するオブジェクトを作る。
    Halide::Func brighter {};
    
    // 座標(x,y)とチャネルcを表現するオブジェクトを作る。
    Halide::Var x {};
    Halide::Var y {};
    Halide::Var c {};

    // アルゴリズムを記述する。
    brighter(x, y, c) = Halide::cast<uint8_t>(min(input(x, y, c) * 1.5f, 255));
    
    // 実行する。
    Halide::Buffer<uint8_t> output {
        brighter.realize(input.width(), input.height(), input.channels())
    };
    
    // 保存する。
    Halide::Tools::save(output, DST_PATH);

    
    return 1;
}
// 画像を明るくする。
int tutorial_02()
{
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image(SRC_PATH);
    
    // pipelineを表現するオブジェクトを作る。
    Halide::Func brighter {};
    
    // 座標(x,y)とチャネルcを表現するオブジェクトを作る。
    Halide::Var x {};
    Halide::Var y {};
    Halide::Var c {};
    
    // 画像の各画素値は以下のように表現される。
    Halide::Expr value {input(x, y, c)};
    
    // これをfloat型に変換する。
    value = Halide::cast<float>(value);
    
    // 画素値を1.5倍する。
    value *= 1.5f;
    
    // 255に収める。
    value = Halide::min(value, 255.0f);
    
    // uint8_tに変換する。
    value = Halide::cast<uint8_t>(value);
    
    // 最終的な画像を定義する。
    brighter(x, y, c) = value;
    
    // 実行する。
    Halide::Buffer<uint8_t> output {
        brighter.realize(input.width(), input.height(), input.channels())
    };
    
    // 保存する。
    Halide::Tools::save(output, DST_PATH);
    
    
    return 1;
}
