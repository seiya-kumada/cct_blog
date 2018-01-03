//
//  main.cpp
//  EmAlgorithm
//
//  Created by 熊田聖也 on 2017/12/23.
//  Copyright © 2017年 熊田聖也. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <numeric>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/algorithm/for_each.hpp>
#include <boost/format.hpp>
#include <fstream>

constexpr int DIMENSION     {3};
constexpr int CLUSTER_NUM   {10};
constexpr double EPSILON    {1.0e-08};

// 各画素にCLUSTER_NUM個の値が割り振られている。
// これらはどのクラスタに属するかを決める確率であるから和は1である。
void observe_probs(const cv::Mat& probs)
{
    const auto range = boost::irange(0, probs.rows);
    boost::for_each(range, [&probs](const auto& n){
        const auto gamma_n = probs.ptr<double>(n);
        const auto total = std::accumulate(gamma_n, gamma_n + CLUSTER_NUM, 0.0);
        if (std::abs(total - 1.0) >= EPSILON) {
            std::cout << "ERROR\n";
        }
    });
}

// 重みの和は1である。
void observe_weights(const cv::Mat& weights)
{
    const auto total = std::accumulate(weights.begin<double>(), weights.end<double>(), 0.0);
    if (std::abs(total - 1.0) >= EPSILON) {
        std::cout << "ERROR\n";
    }
}

void save_means(const cv::Mat& means)
{
    auto path = boost::format("/Users/kumada/Projects/cct_blog/em_algorithm/means_%1%.txt") % CLUSTER_NUM;
    auto fout = std::ofstream(path.str());
    const auto range = boost::irange(0, means.rows);
    boost::for_each(range, [&means, &fout](const auto& i){
        const auto ptr = means.ptr<cv::Vec3b>(i);
        fout << *ptr << std::endl;
    });

}

void observe_labels_and_means(const cv::Mat& labels, const cv::Mat& means, int height, int width)
{
    // 新規の画像（出力）
    cv::Mat rgb_image(height, width, CV_8UC3);

    // 浮動小数点数をuint_8に、チャンネル数を3に変換する。
    cv::Mat means_u8 {};
    means.convertTo(means_u8, CV_8UC1, 255.0);
    cv::Mat means_u8c3 = means_u8.reshape(DIMENSION);
    save_means(means_u8c3);

    boost::copy(
        boost::make_iterator_range(labels.begin<int>(), labels.end<int>()) |
        boost::adaptors::transformed(
            [&means_u8c3](const auto& label){
                return *means_u8c3.ptr<cv::Vec3b>(label);
            }
        ),
        rgb_image.begin<cv::Vec3b>()
    );

    cv::imshow("tmp", rgb_image);
    auto path = boost::format("/Users/kumada/Projects/cct_blog/em_algorithm/em_result_%1%.jpg") % CLUSTER_NUM;
    cv::imwrite(path.str(), rgb_image);
    cv::waitKey();
}

void save_labels(const cv::Mat& labels)
{
    auto path = boost::format("/Users/kumada/Projects/cct_blog/em_algorithm/labels_%1%.txt") % CLUSTER_NUM;
    auto fout = std::ofstream(path.str());
    const auto range = boost::irange(0, labels.rows);
    boost::for_each(range, [&labels, &fout](const auto& i){
        const auto ptr = labels.ptr<int>(i);
        fout << *ptr << std::endl;
    });
}


int main(int argc, const char * argv[]) {
    
    // 画像を読み込む。
    auto image = cv::imread(argv[1]);
    assert(image.type() == CV_8UC3);
    std::cout << image.rows << ", " << image.cols << std::endl;
    
    const auto image_rows = image.rows;
    const auto image_cols = image.cols;

    // 形状を変える。
    auto reshaped_image = image.reshape(1, image_rows * image_cols);
    assert(reshaped_image.type() == CV_8UC1);
    
    // 行数は画素数、列数は3である。
    assert(reshaped_image.rows == image_rows * image_cols);
    assert(reshaped_image.cols == DIMENSION);
    
    // EMアルゴリズムの入力を作る。
    cv::Mat samples;
    reshaped_image.convertTo(samples, CV_64FC1, 1.0 / 255.0);
    assert(samples.type() == CV_64FC1);
    assert(samples.rows == image_rows * image_cols);
    assert(samples.cols == DIMENSION);
    std::cout << samples.at<cv::Vec3d>(21*81) << std::endl;

    // EMアルゴリズムを初期化する。
    auto model = cv::ml::EM::create();
    // ガウス関数の個数(Kに相当する)
    model->setClustersNumber(CLUSTER_NUM);
    // 共分散行列として正の値を持つ対角行列を仮定する。
    model->setCovarianceMatrixType(cv::ml::EM::Types::COV_MAT_DIAGONAL);
    
    // 出力を用意する。
    cv::Mat labels {};
    cv::Mat probs  {};
    cv::Mat log_likelihoods {};
    
    // 実行する。
    // 初期値はK-meansクラスタリングにより決まる。
    model->trainEM(samples, log_likelihoods, labels, probs);
    
    // 出力を確認する。各画素の対数尤度の値
    assert(log_likelihoods.type() == CV_64FC1);
    assert(log_likelihoods.rows == image_rows * image_cols);
    assert(log_likelihoods.cols == 1);
    
    // 出力を確認する。各画素に割り振られたラベルの値
    assert(labels.type() == CV_32SC1);
    assert(labels.rows == image_rows * image_cols);
    assert(labels.cols == 1);
    save_labels(labels);
    
    // 出力を確認する。各画素に割り振られた事後確率の値
    assert(probs.type() == CV_64FC1);
    assert(probs.rows == image_rows * image_cols);
    assert(probs.cols == CLUSTER_NUM);
    observe_probs(probs);
    
    // 出力を確認する。各ガウス関数の平均値
    const auto& means = model->getMeans();
    assert(means.type() == CV_64FC1);
    assert(means.rows == CLUSTER_NUM);
    assert(means.cols == DIMENSION);
    observe_labels_and_means(labels, means, image_rows, image_cols);
    
    // 出力を確認する。各ガウス関数の重み
    const cv::Mat& weights = model->getWeights();
    assert(weights.type() == CV_64FC1);
    assert(weights.rows == 1);
    assert(weights.cols == CLUSTER_NUM);
    observe_weights(weights);
    
    return 0;
}
