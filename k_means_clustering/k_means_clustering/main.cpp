//
//  main.cpp
//  k_means_clustering
//
//  Created by 熊田聖也 on 2017/10/19.
//  Copyright © 2017年 熊田聖也. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/iterator.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/format.hpp>

void show_result(const cv::Mat& labels, const cv::Mat& centers, int cluster_numger, int height, int width);

int main(int argc, const char * argv[])
{
    // 最初の引数を見て、画像を読み込む
    auto image = cv::imread(argv[1]);
    if (image.empty())
    {
        std::cout << "unable to load an input image\n";
        return 1;
    }
    std::cout << "image: (" << image.rows << "," << image.cols << ")" << std::endl;
    
    // RGB画像であることを確認する。
    if (image.type() != CV_8UC3)
    {
        std::cout << "non-color image\n";
    }

    /*
     cv::kmeansは、各行に点の座標を持つ行列を要求する。したがって、入力画像を保持する行列imageの形状を
     (H*W, 3)に変更する。
     */
    auto reshaped_image = image.reshape(1, image.rows * image.cols);
    assert(reshaped_image.type() == CV_8UC1);
    std::cout << "reshaped_image: (" << reshaped_image.rows << ", " << reshaped_image.cols << ")" << std::endl;
    
    /*
     cv::kmeansは浮動小数点の行列を要求するので、型を変更する。同時に255で規格化しておく。
     */
    cv::Mat reshaped_image32f {};
    reshaped_image.convertTo(reshaped_image32f, CV_32FC1, 1.0/255.0);
    assert(reshaped_image32f.type() == CV_32FC1);
    
    // クラスタ数を引数から取り出す。
    const int cluster_number {atoi(argv[2])};
    
    // 出力値を初期化する。
    cv::Mat labels {};
    cv::Mat centers {};
    
    /*
     K-meansクラスタリングを停止させる条件を表現したクラスcv::TermCriteriaのインスタンスを
     作成する。今回は最大繰り返し数を100とした。最後の引数の1はここではダミーである。
     */
    cv::TermCriteria criteria {cv::TermCriteria::COUNT, 100, 1};

    /*
     K-meansクラスタリングを実行する。
     cv::kmeansの第6引数にcv::KMEANS_RANDOM_CENTERSを指定した。これは、クラスタの中心座標の初期値を乱数で
     決めることを意味する。
     */
    cv::kmeans(reshaped_image32f, cluster_number, labels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centers);
    
    // 結果を表示する。
    show_result(labels, centers, cluster_number, image.rows, image.cols);
    
    return 0;
}

void show_result(const cv::Mat& labels, const cv::Mat& centers, int cluster_number, int height, int width)
{
    /*
     画像の形状を(H,W)とすると、labelsの形状は(H*W,1)である。各行にはラベルの値が整数値で収められている。
     cluster_numberを3とすれば、0,1,2のいずれかとなる。
     centersの形状は(cluster_numeber,3)である。各行にクラスタの中心座標が収められている。今の場合、(B,G,R)の
     値が並ぶことになる。
     */
    std::cout << "labels: " << "(" << labels.rows << ", " << labels.cols << ")" << std::endl;
    std::cout << "centers: " << "(" << centers.rows << ", " << centers.cols << ")" << std::endl;
    assert(labels.type() == CV_32SC1);
    assert(centers.type() == CV_32FC1);
    
    /*
     centersの要素の型は浮動小数点である。これを0から255までのstd::uint8_tの型に変更する。
     さらに、チャンネル数を3に変更する。
     */
    cv::Mat centers_u8 {};
    centers.convertTo(centers_u8, CV_8UC1, 255.0);
    const auto centers_u8c3 = centers_u8.reshape(3);
    assert(centers_u8c3.type() == CV_8UC3);

    // K-meansクラスタリングの結果を画像に変換して表示するので、入れ物を初期化する。
    cv::Mat rgb_image {height, width, CV_8UC3};

    /*
     画像に変換する。
        1. labelsからひとつずつ値を取り出す。
        2. そのラベルに相当するRGB値を取り出して、rgb_imageへコピーする。
     cv::Matの要素へのアクセスは下記のようにポインタを使うのが最速である。
     */
    boost::copy(
        boost::make_iterator_range(labels.begin<int>(), labels.end<int>())
            | boost::adaptors::transformed([&centers_u8c3](const auto& label){ return *centers_u8c3.ptr<cv::Vec3b>(label); }),
        rgb_image.begin<cv::Vec3b>()
    );
    
    // 表示する。
    cv::imshow("result", rgb_image);
    
    // 出力パスを作り、保存する。
    const auto output_path = (boost::format("%1%_result.jpg") % cluster_number).str();
    cv::imwrite(output_path, rgb_image);
    
    cv::waitKey();
}
