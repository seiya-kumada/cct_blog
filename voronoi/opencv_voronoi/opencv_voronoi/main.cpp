//
//  main.cpp
//  opencv_voronoi
//
//  Created by 熊田聖也 on 2019/05/17.
//  Copyright © 2019 熊田聖也. All rights reserved.
//

#include <iostream>
#include <random>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cpplinq.hpp>
#include <boost/range/irange.hpp>

// http://schima.hatenablog.com/entry/2014/01/24/205517

constexpr int N_POINTS{1000};

cv::Mat initialize_canvas(const std::vector<cv::Point2f>& points);
void calculate_voronoi(cv::Subdiv2D& subdiv, const std::vector<cv::Point2f>& points, cv::Mat& canvas);
void calculate_delaunay(cv::Subdiv2D& subdiv, const std::vector<cv::Point2f>& points, cv::Mat& canvas);

int main(int argc, const char * argv[]) {
    
    // make random points
    std::mt19937 mt{};
    std::uniform_int_distribution<int> distribution{0, N_POINTS - 1};
    std::vector<cv::Point2f> points{};
    points.reserve(N_POINTS);
    for ([[maybe_unused]] auto i : boost::irange(0, N_POINTS))
    {
        auto x = distribution(mt);
        auto y = distribution(mt);
        points.emplace_back(x, y);
    }
    
    // make an object
    cv::Subdiv2D subdiv{};
    subdiv.initDelaunay(cv::Rect(0, 0, N_POINTS, N_POINTS));
    subdiv.insert(points);
  
    // make a delaunay triangulation
    auto canvas = initialize_canvas(points);
    calculate_delaunay(subdiv, points, canvas);
    cv::imwrite("/Users/kumada/Projects/cct_blog/voronoi/opencv_voronoi/opencv_voronoi/delaunay.jpg", canvas);
    
    // make a voronoi diagram
    canvas = initialize_canvas(points);
    calculate_voronoi(subdiv, points, canvas);
    cv::imwrite("/Users/kumada/Projects/cct_blog/voronoi/opencv_voronoi/opencv_voronoi/voronoi.jpg", canvas);
    
    return 0;
}

cv::Mat initialize_canvas(const std::vector<cv::Point2f>& points)
{
    // display random points
    cv::Mat canvas = cv::Mat::zeros(N_POINTS, N_POINTS, CV_8UC3);
    for (const auto& point: points)
    {
        cv::circle(canvas, point, 4, cv::Scalar(0, 0, 255), -1);
    }
    return canvas;
}

void calculate_voronoi(cv::Subdiv2D& subdiv, const std::vector<cv::Point2f>& points, cv::Mat& canvas)
{

    // prepare output containers
    std::vector<int> idx{};
    std::vector<std::vector<cv::Point2f>> facet_lists{};
    std::vector<cv::Point2f> facet_cernters{};

    // calculcate voronoi
    subdiv.getVoronoiFacetList(idx, facet_lists, facet_cernters);

    // display voronoi structure
    for (const auto& list: facet_lists)
    {
        auto before = list.back();
        for (const auto& point: list)
        {
            cv::Point p1{static_cast<int>(before.x), static_cast<int>(before.y)};
            cv::Point p2{static_cast<int>(point.x), static_cast<int>(point.y)};
            cv::line(canvas, p1, p2, cv::Scalar(64, 255, 128));
            before = std::move(point);
        }
    }
}

void calculate_delaunay(cv::Subdiv2D& subdiv, const std::vector<cv::Point2f>& points, cv::Mat& canvas)
{
    std::vector<cv::Vec4f> edge_list{};
    subdiv.getEdgeList(edge_list);
    
    for (const auto& edge: edge_list)
    {
        cv::Point p1{static_cast<int>(edge.val[0]), static_cast<int>(edge.val[1])};
        cv::Point p2{static_cast<int>(edge.val[2]), static_cast<int>(edge.val[3])};
        cv::line(canvas, p1, p2, cv::Scalar(48, 128, 48));
    }
}
