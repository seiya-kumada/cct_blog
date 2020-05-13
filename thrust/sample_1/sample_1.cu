#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector_types.h>
#include <iostream>
#include <list>
#include <thrust/transform.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <utility>
// https://stackoverflow.com/questions/36551469/triggering-c11-support-in-nvcc-with-cmake
// https://qiita.com/pollenjp/items/391afc3e9f93006b83ba
// https://qiita.com/shinya_ohtani/items/1c5f3731e91b2d006297
// https://stackoverflow.com/questions/7678995/from-thrustdevice-vector-to-raw-pointer-and-back


namespace
{
    void test_container();
    void test_stl_algorithm();
    void test_algorithm();
    void test_opencv();
    void test_cpu();
    void test_gpu();
}

int main(void)
{
    //test_container();
    //test_stl_algorithm();
    //test_algorithm();
    //test_opencv();
    test_cpu();
    //test_gpu();
    return 0;
}

namespace
{
    void test_container()
    {
        //_/_/_/ CPUメモリ上に確保する。
        
        // thrust::host_vector<int> hv {1, 2, 3}; <-- エラーになります。
        // 以下を真似することはできません。
        std::vector<int> sv {1, 2, 3};

        // ひとつずつ代入します。
        thrust::host_vector<int> hv(3);
        hv[0] = 1;
        hv[1] = 2;
        hv[2] = 3;

        //_/_/_/ GPUメモリ上に確保する。
        
        // 簡単です。 
        thrust::device_vector<int> dv(3);
        dv[0] = 1;
        dv[1] = 2;
        dv[2] = 3;

        //_/_/_/ CPUメモリ上のバッファをGPUメモリ上へ転送する。

        // 簡単です。
        dv = hv;
        
        //_/_/_/ GPUメモリ上のバッファをCPUメモリ上へ転送する。

        // copy関数を使います。
        thrust::copy(dv.begin(), dv.end(), hv.begin());
    }

    void test_stl_algorithm()
    {
        std::vector<int> sv {1, 2, 3};
        std::cout << sv[0] << " " << sv[1] << " " << sv[2] << std::endl;

        std::vector<int> dv(3);
        std::transform(sv.begin(), sv.end(), dv.begin(), std::negate<int>());    
        std::cout << dv[0] << " " << dv[1] << " " << dv[2] << std::endl;
    }


    void test_algorithm()
    {
        thrust::device_vector<int> sv(3);
        thrust::sequence(sv.begin(), sv.end(), 1);
        std::cout << sv[0] << " " << sv[1] << " " << sv[2] << std::endl;

        thrust::device_vector<int> dv(3);
        thrust::transform(sv.begin(), sv.end(), dv.begin(), thrust::negate<int>());    
        std::cout << dv[0] << " " << dv[1] << " " << dv[2] << std::endl;
    }

    const std::string PATH = "/home/ubuntu/data/thrust/image.jpg";
    constexpr int ITERATIONS = 1000;

    struct gray_converter
    {
        const float r_factor_ {0.299};
        const float g_factor_ {0.587};
        const float b_factor_ {0.114};

        __host__ __device__
        inline uchar operator()(const uchar3& rgb) const 
        {
            float g = r_factor_ * rgb.z + g_factor_ * rgb.y + b_factor_ * rgb.x; 
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

    struct binary_generator
    {
        const uint8_t threshold_;
        
        binary_generator(uint8_t threshold)
            : threshold_{threshold} {}

        __host__ __device__
        inline uchar operator()(const uchar v) const 
        {
            return v > threshold_ ? 255 : 0;
        }
    };

    void test_opencv()
    {
        // load RGB image
        auto src_image = cv::imread(PATH);

        cv::Mat gray_image {};
        cv::Mat binary_image {};
        auto start = std::chrono::system_clock::now();
        for (auto i = 0; i < ITERATIONS; ++i)
        {
            cv::cvtColor(src_image, gray_image, cv::COLOR_BGR2GRAY);
            cv::threshold(gray_image, binary_image, 128, 255, cv::THRESH_BINARY);
        }
        auto end = std::chrono::system_clock::now();

        cv::imwrite("opencv_hoge.jpg", binary_image);

        // display time 
        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "opencv " << t << " [ms]" << std::endl; 
 
    }

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
        std::vector<uchar> dst_buffer(src_buffer.size());
        
        cv::Mat gray_image {};
        auto gc = gray_converter();
        auto bg = binary_generator(128);
        auto start = std::chrono::system_clock::now();
        for (auto i = 0; i < ITERATIONS; ++i)
        {
            std::transform(src_buffer.begin(), src_buffer.end(), dst_buffer.begin(), gc);
            std::transform(dst_buffer.begin(), dst_buffer.end(), dst_buffer.begin(), bg);
            gray_image = cv::Mat(rows, cols, CV_8UC1, dst_buffer.data());
        }
        auto end = std::chrono::system_clock::now();

        cv::imwrite("cpu_hoge.jpg", gray_image);

        // display time 
        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "cpu " << t << " [ms]" << std::endl; 
    }

    void test_gpu()
    {
        // load RGB image
        auto src_image = cv::imread(PATH);
        auto rows = src_image.rows;
        auto cols = src_image.cols;

        // make ouput buffer on GPU device
        thrust::device_vector<uchar> dst_buffer(rows * cols);
        
        thrust::host_vector<uchar> host_image(rows * cols);
        cv::Mat gray_image {};
        auto gc = gray_converter();
        auto bg = binary_generator(128);
        auto start = std::chrono::system_clock::now();
        
        // copy image to container on GPU device
        // uchar3 means a sequence of (uchar,uchar,uchar)
        uchar3* ptr = reinterpret_cast<uchar3*>(src_image.data);
        thrust::device_vector<uchar3> src_buffer(ptr, ptr + rows * cols);
 
        for (auto i = 0; i < ITERATIONS; ++i)
        {
            thrust::transform(src_buffer.begin(), src_buffer.end(), dst_buffer.begin(), gc);
            thrust::transform(dst_buffer.begin(), dst_buffer.end(), dst_buffer.begin(), bg);
            thrust::copy(dst_buffer.begin(), dst_buffer.end(), host_image.begin());
            gray_image = cv::Mat(rows, cols, CV_8UC1, host_image.data());
        }
        auto end = std::chrono::system_clock::now();

        cv::imwrite("gpu_hoge.jpg", gray_image);
        
        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "gpu " << t << " [ms]" << std::endl; 
    }
}
