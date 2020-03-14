#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <list>
#include <thrust/transform.h>
#include <chrono>
// https://stackoverflow.com/questions/36551469/triggering-c11-support-in-nvcc-with-cmake
// https://qiita.com/pollenjp/items/391afc3e9f93006b83ba

namespace
{
    //void test_0();
    //void test_1();
    //void test_2();
    void test_3();
}

int main(void)
{
    test_3();
    return 0;
}

namespace
{
#if 0
    void test_0()
    {
        // H has storage for 4 integers
        thrust::host_vector<int> hs(4);

        // initialize individual elements
        hs[0] = 14;
        hs[1] = 20;
        hs[2] = 38;
        hs[3] = 46;
    
        for (auto h : hs)
        {
            std::cout << h << std::endl;
        }

        // H.size() returns the size of vector H
        std::cout << "hs has size " << hs.size() << std::endl;

        // resize H
        hs.resize(2);
        std::cout << "hs now has size " << hs.size() << std::endl;
        for (auto h : hs)
        {
            std::cout << h << std::endl;
        }

        // Copy host_vector hs to device_vector ds
        thrust::device_vector<int> ds = hs;
        for (auto d : ds)
        {
            std::cout << d << std::endl;
        }

        // elements of ds can be modified
        ds[0] = 99;
        ds[1] = 88;

        for (auto d : ds)
        {
            std::cout << d << std::endl;
        }

        // hs and ds are automatically destroyed when the function returns
    }

    void test_1()
    {
        thrust::device_vector<int> ds(10, 1);
        thrust::fill(std::begin(ds), std::begin(ds) + 7, 9);
        for (auto d : ds)
        {
            std::cout << d << " ";
        } 
        std::cout << std::endl;

        thrust::host_vector<int> hs(ds.begin(), ds.begin() + 5);
        for (auto h : hs)
        {
            std::cout << h << " ";
        } 
        std::cout << std::endl;

        thrust::sequence(hs.begin(), hs.end(), 2);
        for (auto h : hs)
        {
            std::cout << h << " ";
        } 
        std::cout << std::endl;

        thrust::copy(hs.begin(), hs.end(), ds.begin());
        for (auto d : ds)
        {
            std::cout << d << " ";
        } 
        std::cout << std::endl;
    }

    void test_2()
    {
        std::list<int> stl_list {};
        for (auto i = 0; i < 4; ++i)
        {
            stl_list.push_back(10 * i);
        }

        thrust::device_vector<int> ds {std::begin(stl_list), std::end(stl_list)};
        for (auto d : ds)
        {
            std::cout << d << " ";
        }
        std::cout << std::endl;

        std::vector<int> dst {};
        thrust::copy(ds.begin(), ds.end(), std::back_inserter(dst));
        for (auto d : dst)
        {
            std::cout << d << " ";
        }
        std::cout << std::endl;
    }
    
#endif
    struct saxpy_functor
    {
        const float a;
        saxpy_functor(float _a) : a(_a) {}

        __host__ __device__
        float operator()(const float& x, const float& y) const
        {
            return a * x + y;
        }
    };

    struct saxpy_fast_
    {
        const float a;
        saxpy_fast_(float _a) : a(_a) {}

        __host__ __device__
        void operator()(const thrust::device_ptr<float>& xs, const thrust::device_ptr<float>& ys, int n) const 
        {
            for (auto i = 0; i < n; ++i)
            {
               ys[i] = ys[i] + a * xs[i];
            }
        }
    };

    void saxpy_fast(float a, thrust::device_vector<float>& x, thrust::device_vector<float>& y)
    {
        thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), saxpy_functor(a));
    }

	void saxpy_slow(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
	{
    	thrust::device_vector<float> temp(X.size());
   
    	// temp <- A
    	thrust::fill(temp.begin(), temp.end(), A);
    
    	// temp <- A * X
    	thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());

    	// Y <- A * X + Y
    	thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
	}
    void test_3()
    {
        int N = 30000000;
        thrust::device_vector<float> src(N);
        thrust::sequence(src.begin(), src.end());

        thrust::device_vector<float> dst(N);
        thrust::fill(dst.begin(), dst.end(), 5);
		{        
            auto s = std::chrono::system_clock::now();
            saxpy_fast(1, src, dst);
            auto e = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
            std::cout << "fast: " << elapsed << " [ms]" << std::endl;
        }

        {
            auto s = std::chrono::system_clock::now();
            saxpy_slow(1, src, dst);
            auto e = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
            std::cout << "slow: " << elapsed << " [ms]" << std::endl;
        }

        {
            auto s = std::chrono::system_clock::now();
            saxpy_fast_ sf(1);
            sf(src.data(), src.data(), N);
            auto e = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
            std::cout << "slow: " << elapsed << " [ms]" << std::endl;
        }
    }
}
