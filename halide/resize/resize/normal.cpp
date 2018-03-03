//
//  normal.cpp
//  sample
//
//  Created by 熊田聖也 on 2018/02/11.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "normal.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

inline const uint8_t* get_pixel(const uint8_t* data, int row, int col, size_t step, size_t elem_size)
{
    return data + (row * step + col * elem_size);
}

inline uint8_t interpolate(float c0, float c1, float c2, float c3, const uint8_t* p0, const uint8_t* p1, const uint8_t* p2, const uint8_t* p3, int i)
{
    return c0 * p0[i] + c1 * p1[i] + c2 * p2[i] + c3 * p3[i];
}

void resize_with_raw_access(const cv::Mat& src_image, cv::Mat& dst_image)
{
    //_/_/_/ 画像の読み込みと保存はOpenCVを使用。
    
    // 入力画像の各種値・ポインタを取り出す。
    const int src_cols = src_image.cols;
    const int src_rows = src_image.rows;
    const size_t src_step = src_image.step;
    const size_t src_elem_size = src_image.elemSize();
    const uint8_t* src_data = src_image.data;

    // 出力画像の各種値・ポインタを取り出す。
    const int dst_cols = dst_image.cols;
    const int dst_rows = dst_image.rows;
    const size_t dst_step = dst_image.step;
    const size_t dst_elem_size = dst_image.elemSize();
    uint8_t* dst_data = dst_image.data;
    
    // 拡大縮小率の逆数を計算する。
    const float sc = static_cast<float>(src_cols) / dst_cols;
    const float sr = static_cast<float>(src_rows) / dst_rows;
    
    // 拡大縮小後の画素を左上から順に走査する。
    // y軸方向の走査
    for (auto j = 0, jd = 0; j < dst_rows; ++j, jd += dst_step)
    {
        // 元画像における位置yを算出する。
        const float fj = j * sr;
        const int cj0 = static_cast<int>(fj); // 端数を切り捨てる。
        const float dj = fj - cj0;
        
        // +1した値が画像内に収まるように調整する。
        const int cj1 = cj0 + 1 >= src_rows ? cj0 : cj0 + 1;
        
        // 拡大縮小後画像へのポインタの位置を更新する。
        uint8_t* dst_p = dst_data + jd;
        
        // x軸方向の走査
        for (auto i = 0, id = 0; i < dst_cols; ++i, id += dst_elem_size)
        {
            // 元画像における位置xを算出する。
            const float fi = i * sc;
            const int ci0 = static_cast<int>(fi);
            const float di = fi - ci0; // dx

            const int ci1 = ci0 + 1 >= src_cols ? ci0 : ci0 + 1;

            // 面積を計算する。
            const float c0 = (1.0f - dj) * (1.0f - di);
            const float c1 = (1.0f - dj) * di;
            const float c2 = dj * (1.0f - di);
            const float c3 = dj * di;

            // 周辺画素を取り出す。
            const uint8_t* src_pixel0 = get_pixel(src_data, cj0, ci0, src_step, src_elem_size);
            const uint8_t* src_pixel1 = get_pixel(src_data, cj0, ci1, src_step, src_elem_size);
            const uint8_t* src_pixel2 = get_pixel(src_data, cj1, ci0, src_step, src_elem_size);
            const uint8_t* src_pixel3 = get_pixel(src_data, cj1, ci1, src_step, src_elem_size);

            // ポインタ位置を更新する。
            uint8_t* dst_pixel = dst_p + id;

            // RGB値を計算する。
            dst_pixel[0] = interpolate(c0, c1, c2, c3, src_pixel0, src_pixel1, src_pixel2, src_pixel3, 0);
            dst_pixel[1] = interpolate(c0, c1, c2, c3, src_pixel0, src_pixel1, src_pixel2, src_pixel3, 1);
            dst_pixel[2] = interpolate(c0, c1, c2, c3, src_pixel0, src_pixel1, src_pixel2, src_pixel3, 2);
        }
    }
}
