//
//  normal.cpp
//  sample
//
//  Created by 熊田聖也 on 2018/02/11.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include "normal.hpp"
#include <iostream>

inline const cv::Vec3b& get_pixel(const cv::Mat& img, int row, int col)
{
    return img.at<cv::Vec3b>(row, col);
}

inline const uint8_t* get_pixel_(const uint8_t* data, int row, int col, size_t step, size_t elem_size)
{
    return data + (row * step + col * elem_size);
}

inline uint8_t interpolate(float c0, float c1, float c2, float c3, const cv::Vec3b& p0, const cv::Vec3b& p1, const cv::Vec3b& p2, const cv::Vec3b& p3, int i)
{
    return c0 * p0[i] + c1 * p1[i] + c2 * p2[i] + c3 * p3[i];
}

inline uint8_t interpolate_(float c0, float c1, float c2, float c3, const uint8_t* p0, const uint8_t* p1, const uint8_t* p2, const uint8_t* p3, int i)
{
    return c0 * p0[i] + c1 * p1[i] + c2 * p2[i] + c3 * p3[i];
}

void resize_with_cv_access(const cv::Mat& src_image, cv::Mat& dst_image)
{
    const int src_cols = src_image.cols;
    const int src_rows = src_image.rows;
    const int dst_cols = dst_image.cols;
    const int dst_rows = dst_image.rows;
    const float sc = static_cast<float>(src_cols) / dst_cols;
    const float sr = static_cast<float>(src_rows) / dst_rows;
    
    for (auto j = 0; j < dst_rows; ++j)
    {
        // position at src image
        const float fj = j * sr; // precise y position
        const int cj0 = static_cast<int>(fj); // round it
        const float dj = fj - cj0; // dy

        // check whether the position is outside the image or not
        const int cj1 = cj0 + 1 >= src_rows ? cj0 : cj0 + 1;

        cv::Vec3b* dst_p = dst_image.ptr<cv::Vec3b>(j);
        
        for (auto i = 0; i < dst_cols; ++i)
        {
            const float fi = i * sc; // x
            const int ci0 = static_cast<int>(fi);
            const float di = fi - ci0; // dx
            
            const int ci1 = ci0 + 1 >= src_cols ? ci0 : ci0 + 1;
            
            const float c0 = (1.0f - dj) * (1.0f - di);
            const float c1 = (1.0f - dj) * di;
            const float c2 = dj * (1.0f - di);
            const float c3 = dj * di;
            
            const cv::Vec3b& src_pixel0 = get_pixel(src_image, cj0, ci0);
            const cv::Vec3b& src_pixel1 = get_pixel(src_image, cj0, ci1);
            const cv::Vec3b& src_pixel2 = get_pixel(src_image, cj1, ci0);
            const cv::Vec3b& src_pixel3 = get_pixel(src_image, cj1, ci1);
        
            cv::Vec3b& dst_pixel = dst_p[i];
            
            dst_pixel[0] = interpolate(c0, c1, c2, c3, src_pixel0, src_pixel1, src_pixel2, src_pixel3, 0);
            dst_pixel[1] = interpolate(c0, c1, c2, c3, src_pixel0, src_pixel1, src_pixel2, src_pixel3, 1);
            dst_pixel[2] = interpolate(c0, c1, c2, c3, src_pixel0, src_pixel1, src_pixel2, src_pixel3, 2);
        }
    }
}

void resize_with_raw_access(const cv::Mat& src_image, cv::Mat& dst_image)
{
    const int src_cols = src_image.cols;
    const int src_rows = src_image.rows;
    const size_t src_step = src_image.step;
    const size_t src_elem_size = src_image.elemSize();
    const uint8_t* src_data = src_image.data;
    
    const int dst_cols = dst_image.cols;
    const int dst_rows = dst_image.rows;
    const size_t dst_step = dst_image.step;
    const size_t dst_elem_size = dst_image.elemSize();
    uint8_t* dst_data = dst_image.data;
    
    const float sc = static_cast<float>(src_cols) / dst_cols;
    const float sr = static_cast<float>(src_rows) / dst_rows;
    
    for (auto j = 0, jd = 0; j < dst_rows; ++j, jd += dst_step)
    {
        // position at src image
        const float fj = j * sr; // precise y position
        const int cj0 = static_cast<int>(fj); // round it
        const float dj = fj - cj0; // dy
        
        // check whether the position is outside the image or not
        const int cj1 = cj0 + 1 >= src_rows ? cj0 : cj0 + 1;
        
        uint8_t* dst_p = dst_data + jd;
        for (auto i = 0, id = 0; i < dst_cols; ++i, id += dst_elem_size)
        {
            const float fi = i * sc; // x
            const int ci0 = static_cast<int>(fi);
            const float di = fi - ci0; // dx

            const int ci1 = ci0 + 1 >= src_cols ? ci0 : ci0 + 1;

            const float c0 = (1.0f - dj) * (1.0f - di);
            const float c1 = (1.0f - dj) * di;
            const float c2 = dj * (1.0f - di);
            const float c3 = dj * di;

            const uint8_t* src_pixel0 = get_pixel_(src_data, cj0, ci0, src_step, src_elem_size);
            const uint8_t* src_pixel1 = get_pixel_(src_data, cj0, ci1, src_step, src_elem_size);
            const uint8_t* src_pixel2 = get_pixel_(src_data, cj1, ci0, src_step, src_elem_size);
            const uint8_t* src_pixel3 = get_pixel_(src_data, cj1, ci1, src_step, src_elem_size);

            uint8_t* dst_pixel = dst_p + id;

            dst_pixel[0] = interpolate_(c0, c1, c2, c3, src_pixel0, src_pixel1, src_pixel2, src_pixel3, 0);
            dst_pixel[1] = interpolate_(c0, c1, c2, c3, src_pixel0, src_pixel1, src_pixel2, src_pixel3, 1);
            dst_pixel[2] = interpolate_(c0, c1, c2, c3, src_pixel0, src_pixel1, src_pixel2, src_pixel3, 2);
        }
    }
}
