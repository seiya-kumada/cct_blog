#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector_types.h>
#include <iostream>
#include <list>
#include <thrust/transform.h>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <utility>
// https://stackoverflow.com/questions/36551469/triggering-c11-support-in-nvcc-with-cmake
// https://qiita.com/pollenjp/items/391afc3e9f93006b83ba
// https://qiita.com/shinya_ohtani/items/1c5f3731e91b2d006297
// https://stackoverflow.com/questions/7678995/from-thrustdevice-vector-to-raw-pointer-and-back


namespace
{
    void test_cpu();
    void test_gpu();
}

int main(void)
{
    test_cpu();
    test_gpu();
    return 0;
}

namespace
{
    const std::string PATH = "/home/ubuntu/data/thrust/image.jpg";
    constexpr int ITERATIONS = 1000;

    struct gray_converter
    {
        const float r_factor_ {0.3};
        const float g_factor_ {0.59};
        const float b_factor_ {0.11};

        __host__ __device__
        uchar operator()(const uchar3& rgb) const 
        {
            float g = r_factor_ * rgb.x + g_factor_ * rgb.y + b_factor_ * rgb.z; 
            if (g < 0)
            {
                g = 0;
            }
            if (g > 255)
            {
                g = 255;
            }
            return (uchar)(g);
        }
    };
    
    void test_cpu()
    {
        // load RGB image
        auto src_image = cv::imread(PATH);
        auto rows = src_image.rows;
        auto cols = src_image.cols;

        // copy image to source buffer 
        // uchar3 means a sequence of (uchar,uchar,uchar)
        uchar3* ptr = reinterpret_cast<uchar3*>(src_image.data);
        std::vector<uchar3> src_buffer(ptr, ptr + rows * cols);

        // make destination buffer
        std::vector<uchar> dst_buffer(rows * cols);

        auto start = std::chrono::system_clock::now();
        for (auto i = 0; i < ITERATIONS; ++i)
        {
            std::transform(src_buffer.begin(), src_buffer.end(), dst_buffer.begin(), gray_converter());
        }
        auto end = std::chrono::system_clock::now();

        cv::Mat gray_image(rows, cols, CV_8UC1, dst_buffer.data());
        cv::imwrite("./hoge.jpg", gray_image);

        // display time 
        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << t << " [ms]" << std::endl; 
    }

    void test_gpu()
    {
        // load RGB image
        auto src_image = cv::imread(PATH);
        auto rows = src_image.rows;
        auto cols = src_image.cols;

        // copy image to container on GPU device
        // uchar3 means a sequence of (uchar,uchar,uchar)
        uchar3* ptr = reinterpret_cast<uchar3*>(src_image.data);
        thrust::device_vector<uchar3> src_buffer(ptr, ptr + rows * cols);
        
        // make ouput buffer on GPU device
        thrust::device_vector<uchar> dst_buffer(rows * cols);

        auto start = std::chrono::system_clock::now();
        for (auto i = 0; i < ITERATIONS; ++i)
        {
            thrust::transform(src_buffer.begin(), src_buffer.end(), dst_buffer.begin(), gray_converter());
        }
        auto end = std::chrono::system_clock::now();

        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << t << " [ms]" << std::endl; 
    }
}
