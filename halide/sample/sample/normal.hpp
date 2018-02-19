//
//  normal.hpp
//  sample
//
//  Created by 熊田聖也 on 2018/02/11.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#ifndef normal_hpp
#define normal_hpp
#include <opencv2/imgcodecs.hpp>

void resize_with_cv_access(const cv::Mat& src_image, cv::Mat& dst_image);
void blur_with_cv_access(const cv::Mat& src_image, cv::Mat& dst_image);
void resize_with_raw_access(const cv::Mat& src_image, cv::Mat& dst_image);

#endif /* normal_hpp */
