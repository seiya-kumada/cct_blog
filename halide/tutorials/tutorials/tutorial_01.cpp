//
//  tutorial_01.cpp
//  tutorials
//
//  Created by 熊田聖也 on 2018/01/15.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "tutorial_01.hpp"
#include <Halide.h>

template<typename T>
struct Type;
/*
 HalideはC++に埋め込まれた言語である。
 Halideを用いたコードをpipelineと呼ぶ。
 通常のコンパイラは、テキストファイルを受け取り、機械語あるいは中間言語に変換するプログラムである。
 HalideのコンパイラはC言語で記述されており、pipelineを含むコードのコンパイルの時に呼び出される。
 pipelineはテキストファイルとしてHalideコンパイラに渡されるわけではない。
 pipelineはC++のデータ構造としてHalideコンパイラに渡される。
 通常のコンパイラのようにテキストファイルを処理するのではない。
 プログラマが記述したpipelineはC++のデータ構造に変換され、
 通常のコンパイル時に、そのデータ構造へのポインタがHalideコンパイラに渡される。
 Halideはライブラリをリンクして使われる。
 
 2通りにコンパイル形式がある。
 1. ディスク上のファイルんいコードを出力する。Cコンパイラの出力ファイルと同じ形式のもの。
 2. メモリ上に直接機械語を出力する。
 
 アルゴリズムと計算法（並列化、ベクトル化、ブロック化）を分離する。
 */
int tutorial_011()
{
    // アルゴリズムを記述する。
    // Define
    Halide::Var x {};
    Halide::Var y {};
    Halide::Func gradient {};
    gradient(x, y) = x + y;
    
    // (800,600)の画像を作成する。
    // Run
    Halide::Buffer<int32_t> output {gradient.realize(800, 600)};
    
    // 結果を検証する。
    for (auto j = 0; j < output.height(); ++j)
    {
        for (auto i = 0; i < output.width(); ++i)
        {
            if (output(i, j) != i + j)
            {
                std::cout << "Something went wrong!\n";
                return -1;
            }
        }
    }
    return 0;
}

int tutorial_01()
{
    // クラスFuncのオブジェクトを作る。これはパイプラインステージを表す。
    // 純粋関数である。これは、各画素が持つべき値を定義する。
    // 計算される画像とみなすことができる。
    Halide::Func gradient {};
    
    // クラスVarのオブジェクトを作る。
    // パイプラインステージにて使われる変数を表す。
    Halide::Var x {};
    Halide::Var y {};
    
    // xとyは、画像のx,y軸の位置を表す。
    
    // 位置x,yにある画素値の和を表現する式を定義する。
    // Varオブジェクトは2項演算子+をサポートする。
    Halide::Expr e {x + y};
    
    // 位置(x,y)の画素値はeである。
    gradient(x, y) = e;
    
    // この時点ではなにも計算されない。
    // 実現すべき計算を定義しただけである。
    
    // この行で実行時コンパイルされ、実際の計算が行われる。
    // (800,600)の画像が作られる。
    // x,yの値の型はint32_tとなり、eの値の型もint32_tとなる。
    // graidentはint32_tの値を持つ画像になる。
    Halide::Buffer<int32_t> output {gradient.realize(800, 600)};
    
    // 結果を検証する。
    for (auto j = 0; j < output.height(); ++j)
    {
        for (auto i = 0; i < output.width(); ++i)
        {
            if (output(i, j) != i + j)
            {
                std::cout << "Something went wrong!\n";
                return -1;
            }
        }
    }
    std::cout << "Success\n";
    return 0;
}
