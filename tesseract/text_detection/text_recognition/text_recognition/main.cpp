//
//  main.cpp
//  text_recognition
//
//  Created by 熊田聖也 on 2018/10/19.
//  Copyright © 2018年 熊田聖也. All rights reserved.
// https://www.learnopencv.com/deep-learning-based-text-recognition-ocr-using-tesseract-and-opencv/
#include <string>
#include <iostream>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//const std::string PATH = "/Users/kumada/Projects/text_detection/opencv-text-recognition/images/sample2.png";
const std::string PATH = "/Users/kumada/Projects/text_detection/opencv-text-detection/images/sample_3.png";

template<typename T>
struct Type;

int sample_0(int argc, const char * argv[])
{
    // create tesseract object
    auto ocr = std::make_unique<tesseract::TessBaseAPI>();
    
    // initialize tesseract to use English (eng) and the LSTM OCR engine.
    ocr->Init(nullptr, "jpn", tesseract::OEM_LSTM_ONLY);
    
    // set page segmentation mode to PSM_AUTO (3)
    ocr->SetPageSegMode(tesseract::PSM_AUTO);
    
    // open input image using OpenCV
    auto im = cv::imread(PATH, cv::IMREAD_COLOR);
    
    // set image data
    ocr->SetImage(im.data, im.cols, im.rows, 3, static_cast<int>(im.step));
    
    // run tesseract OCR on image
    auto out_text = ocr->GetUTF8Text();
    std::cout << out_text << std::endl;
    
    //    ocr->End();
    return 0;
}

int sample_1(int argc, const char * argv[])
{
    // create tesseract object
    auto ocr = std::make_unique<tesseract::TessBaseAPI>();
    
    ocr->InitForAnalysePage();
    
    // open input image using OpenCV
    auto im = cv::imread(PATH, cv::IMREAD_COLOR);
    
    // set image data
    ocr->SetImage(im.data, im.cols, im.rows, 3, static_cast<int>(im.step));

    tesseract::PageIterator *iter = ocr->AnalyseLayout();
    
    int left, top, right, bottom;
    while (iter->Next(tesseract::RIL_PARA))
    {
        iter->BoundingBox(tesseract::RIL_PARA, &left, &top, &right, &bottom);
        std::cout << left << ", " << top << ", " << right << ", " << bottom << std::endl;
        cv::rectangle(im, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 200), 3, 4);
    }
    
    cv::imwrite("./result.jpg", im);
    
    return 0;
}

int main(int argc, const char * argv[])
{
    return sample_1(argc, argv);
}
