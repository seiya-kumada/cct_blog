//
//  tutorial_12.cpp
//  tutorials
//
//  Created by 熊田聖也 on 2018/02/10.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "tutorial_12.hpp"
#include <Halide.h>
#include <halide_image_io.h>
namespace Tools = Halide::Tools;

//Halide::Var x {};
//Halide::Var y {};
//Halide::Var c {};
//Halide::Var i {};
//Halide::Var ii {};
//Halide::Var xo {};
//Halide::Var yo {};
//Halide::Var xi {};
//Halide::Var yi {};

class MyPipeline
{
private:
    static Halide::Var x_;
    static Halide::Var y_;
    static Halide::Var c_;
    static Halide::Var i_;
    static Halide::Var ii_;
    static Halide::Var xo_;
    static Halide::Var yo_;
    static Halide::Var xi_;
    static Halide::Var yi_;

public:
    Halide::Func lut_;
    Halide::Func padded_;
    Halide::Func padded16_;
    Halide::Func sharpen_;
    Halide::Func curved_;
    Halide::Buffer<uint8_t> input_;
    
    MyPipeline(Halide::Buffer<uint8_t> in)
        : lut_{}
        , padded_{}
        , padded16_{}
        , sharpen_{}
        , curved_{}
        , input_{in}
    {
        lut_(i_) = Halide::cast<uint8_t>(Halide::clamp(Halide::pow(i_ / 255.0f, 1.2f) * 255.0f, 0, 255));
        
        // これって何してるの？
        // なくても良いような。
        padded_(x_, y_, c_) = input_(
            Halide::clamp(x_, 0, input_.width() - 1),
            Halide::clamp(y_, 0, input_.height() - 1),
            c_
        );
        
        padded16_(x_, y_, c_) = Halide::cast<uint16_t>(padded_(x_, y_, c_));
        
        sharpen_(x_, y_, c_) =
            padded16_(x_, y_, c_) * 2 -
            (
                padded16_(x_ - 1, y_, c_) +
                padded16_(x_, y_ - 1, c_) +
                padded16_(x_ + 1, y_, c_) +
                padded16_(x_, y_ + 1, c_)
            ) / 4;
        
        curved_(x_, y_, c_) = lut_(sharpen_(x_, y_, c_));
        
    }
    
    void schedule_for_cpu()
    {
        lut_.compute_root();
        curved_.reorder(c_, x_, y_).bound(c_, 0, 3).unroll(c_);
        
        Halide::Var yo {};
        Halide::Var yi {};
        curved_.split(y_, yo_, yi_, 16).parallel(yo_);
        
        // curved_のyiが計算される前に、sharpen_が計算される。
        sharpen_.compute_at(curved_, yi);
        
        sharpen_.vectorize(x_, 8);
        
        // curved_のyo計算に必要なpadded_の箇所だけが保持される。
        // この状態でcurved_のyiを計算する。
        padded_.store_at(curved_, yo).compute_at(curved_, yi);
        
        padded_.vectorize(x_, 16);
        
        curved_.compile_jit();
    }
};

Halide::Var MyPipeline::x_ {};
Halide::Var MyPipeline::y_ {};
Halide::Var MyPipeline::c_ {};
Halide::Var MyPipeline::i_ {};
Halide::Var MyPipeline::ii_ {};
Halide::Var MyPipeline::xo_ {};
Halide::Var MyPipeline::yo_ {};
Halide::Var MyPipeline::xi_ {};
Halide::Var MyPipeline::yi_ {};

int tutorial_12()
{
    return 1;
}
