//
//  main.cpp
//  sample_2
//
//  Created by 熊田聖也 on 2018/02/25.
//  Copyright © 2018年 熊田聖也. All rights reserved.
//

#include <Halide.h>
using namespace Halide;

#include <iostream>

// Some support code for timing and loading/saving images
#include <halide_image_io.h>
#include <clock.h>

// Include OpenCV for timing comparison
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char **argv) {
    Buffer<uint8_t> in_8 = Tools::load_image("/Users/kumada/Projects/cct_blog/halide/images/inputs/src_image.jpg");
    Func in_bounded = BoundaryConditions::repeat_edge(in_8);
    Var x, y, c;
    Func in;
    in(x, y, c) = cast<float>(in_bounded(x, y, c));
    
    // Define a 7x7 Gaussian Blur with a repeat-edge boundary condition.
    float sigma = 1.5f;
    
    Func kernel;
    kernel(x) = exp(-x*x/(2*sigma*sigma)) / (sqrtf(2*M_PI)*sigma);
    
    
    
    Func blur_y;
    blur_y(x, y, c) = (kernel(0) * in_bounded(x, y, c) +
                       kernel(1) * (in_bounded(x, y-1, c) +
                                    in_bounded(x, y+1, c)) +
                       kernel(2) * (in_bounded(x, y-2, c) +
                                    in_bounded(x, y+2, c)) +
                       kernel(3) * (in_bounded(x, y-3, c) +
                                    in_bounded(x, y+3, c)));
    
    Func blur_x;
    blur_x(x, y, c) = (kernel(0) * blur_y(x, y, c) +
                       kernel(1) * (blur_y(x-1, y, c) +
                                    blur_y(x+1, y, c)) +
                       kernel(2) * (blur_y(x-2, y, c) +
                                    blur_y(x+2, y, c)) +
                       kernel(3) * (blur_y(x-3, y, c) +
                                    blur_y(x+3, y, c)));
    
    
    // Schedule it.
    blur_x.compute_root().vectorize(x, 8).parallel(y);
    blur_y.compute_at(blur_x, y).vectorize(x, 8);

    // Print out pseudocode for the pipeline.
//    blur_x.compile_to_lowered_stmt("blur.html", {in}, HTML);
    
    // Benchmark the pipeline.
    Buffer<float> output(in_8.width(),
                        in_8.height(),
                        in_8.channels());
    for (int i = 0; i < 10; i++) {
        double t1 = current_time();
        blur_x.realize(output);
        double t2 = current_time();
        std::cout << "Time: " << (t2 - t1) << '\n';
    }
    
//    Tools::save_image(output, "output.png");
    
    // Time OpenCV doing the same thing.
    {
        cv::Mat input_image = cv::Mat::zeros(in_8.width(), in_8.height(), CV_32FC3);
        cv::Mat output_image = cv::Mat::zeros(in_8.width(), in_8.height(), CV_32FC3);
        
        double best = 1e10;
        for (int i = 0; i < 10; i++) {
            double t1 = current_time();
            GaussianBlur(input_image, output_image, cv::Size(7, 7),
                         1.5f, 1.5f, cv::BORDER_REPLICATE);
            double t2 = current_time();
            best = std::min(best, t2 - t1);
        }
        std::cout << "OpenCV time: " << best << "\n";
    }
    
    return 0;
}
