/**
 * This file is to test its performance.
 * Test 1: simply query a image with highest similarity score compared to the specified image in KITTI color sequence 00.
 *  With no other constraint.
 * calculate the mean and standard variance of the time cost
 *  -- time cost for loading the vocabulary
 *  -- time cost for extracting features (without transform features to BoW vector)
 *  -- time cost for extracting BoW vector (extract features + transform features to BoW vector)
 *  -- time cost for adding one image's BoW to database
 *  -- time cost for querying one image's BoW
 */
 
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip> // for 'setw'
// DBoW3
#include "DBoW3.h" // defines OrbVocabulary and OrbDatabase
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "DescManip.h"

using namespace std;


void ExtractFeaturesFromOneImage(const cv::Mat &imgGray, cv::Ptr<cv::ORB> &orb, 
    cv::Mat &vfeaturesInOneImg);

void LoadImages(const string &strPathToSequence, 
    vector<string> &vstrImageFilenames, vector<double> &vTimestamps);

std::pair<double, double> VectorMeanAndVariance(const std::vector<double> v);



// ----------------------------------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv){
    if(argc != 3){
        std::cerr << std::endl << "Usage: ./mytest/test1   path_to_sequence  path_to_vocabulary" << std::endl;
        return 1;
    }

    std::chrono::steady_clock::time_point t1;
    std::chrono::steady_clock::time_point t2;
    std::chrono::duration<double> time_used;


    // ---------------------------------- Load the images' path -----------------------------------------
    std::string strPathToSequence = argv[1];
    std::vector<string> vstrImageFilenames;
    std::vector<double> vTimestamps;
    LoadImages(strPathToSequence,  vstrImageFilenames, vTimestamps);

    std::string strPathToVocabulary = argv[2];

    // ---------------------------------- check if the vocabulary file exists or not----------------------------------
    fstream vocFile;
    vocFile.open(strPathToVocabulary);
    if(!vocFile){
        std::cerr << "error: cannot open the vocabulary file " << strPathToVocabulary << endl;
        return -1;
    }
    vocFile.close();

    // ---------------------------------- create the file to store test results ----------------------------------
    ofstream f;
    std::string filename = "mytest/result/test1_time_cost_result.txt";
    f.open(filename.c_str());
    f << fixed;

    // ---------------------------------- load & create KF Database ----------------------------------
    std::cout << "load the vocabulary ..." << std::endl;
    t1 = std::chrono::steady_clock::now();
    DBoW3::Vocabulary voc(strPathToVocabulary);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
    f  <<  setprecision(3) << " -- time cost for loading the vocabulary " << time_used.count() << "s" << std::endl << std::endl; 
    std::cout << "has done." << endl;

    std::cout << "create the database ..." << std::endl;
    DBoW3::Database db(voc, true, 1);
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<double> vTimeAddToDatabase;
    vTimeAddToDatabase.reserve(vstrImageFilenames.size());
    std::vector<double> vTimeExtractBoW;
    vTimeExtractBoW.reserve(vstrImageFilenames.size());
    for(size_t i = 0,  nImgs = vstrImageFilenames.size(); i < nImgs; i+=10)
    {
        cv::Mat originalImg = cv::imread(vstrImageFilenames[i], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imgGray;
        cv::cvtColor(originalImg, imgGray, cv::COLOR_BGR2GRAY); 

        t1 = std::chrono::steady_clock::now();  // time cost for extracting BoW vector (extract features + transform features to BoW vector)
        cv::Mat vfeaturesInOneImg;
        ExtractFeaturesFromOneImage(imgGray, orb, vfeaturesInOneImg);
        DBoW3::BowVector bowVector;
        voc.transform(vfeaturesInOneImg, bowVector);
        t2 = std::chrono::steady_clock::now();
        time_used = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
        vTimeExtractBoW.push_back(time_used.count());

        t1 = std::chrono::steady_clock::now();  // time cost for add one image's BoW to databse
        db.add(bowVector);
        t2 = std::chrono::steady_clock::now();
        time_used = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
        vTimeAddToDatabase.push_back(time_used.count());
    }
    std::cout << "has done." << endl;


    // ---------------------------------- query the specified image in the database & record the time cost ----------------------------------
    std::cout << "start test: time cost of querying a single image ... " << endl;
    std::vector<double> vTimeExtractFeatures(vstrImageFilenames.size());
    std::vector<double> vTimeQuerying(vstrImageFilenames.size());
    // do the test for every image in the sequence, and calculate the mean and standard variance of the time cost
    for(size_t nImgId = 0,  nImgs = vstrImageFilenames.size(); nImgId < nImgs; nImgId++){
        cv::Mat originalImg = cv::imread(vstrImageFilenames[nImgId], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imgGray;
        cv::cvtColor(originalImg, imgGray, cv::COLOR_BGR2GRAY); 

        t1 = std::chrono::steady_clock::now(); // time cost for extracting features (without transform features to BoW vector)
        cv::Mat vfeaturesInOneImg;
        ExtractFeaturesFromOneImage(imgGray, orb, vfeaturesInOneImg);
        t2 = std::chrono::steady_clock::now();
        time_used = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
        // std::cout << " -- time cost for extracting features: " << time_used.count() << "s" << std::endl; 
        vTimeExtractFeatures[nImgId] = time_used.count();

        DBoW3::BowVector bowVector;
        voc.transform(vfeaturesInOneImg, bowVector);

        t1 = std::chrono::steady_clock::now(); // time cost for querying one image's BoW
        DBoW3::QueryResults queryResult;
        db.query(bowVector, queryResult, 1, -1); // max_result=1, max_id=-1 unlimited
        t2 = std::chrono::steady_clock::now();
        time_used = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
        // std::cout << " -- time cost for querying: " << time_used.count() << "s" << std::endl; 
        // std::cout << "Searching for Image " << nImgId << ": " << queryResult << std::endl;
        vTimeQuerying[nImgId] = time_used.count();
    }

    // ---------------------------------- deal with the test results ----------------------------------
    // calculate the mean and standard variance of time cost
    // then save them to file
    std::pair<double, double>  pairMeanAndVariance;
    pairMeanAndVariance = VectorMeanAndVariance(vTimeExtractFeatures);
    f <<  setprecision(6) << " -- time cost for extracting features: "<< std::endl << " mean: " << pairMeanAndVariance.first 
        << "s, stdev: " <<  pairMeanAndVariance.second << "s" << std::endl << std::endl; 
    
    pairMeanAndVariance = VectorMeanAndVariance(vTimeExtractBoW);
    f <<  setprecision(6) << " -- time cost for extracting BoW vector: "<< std::endl << " mean: " << pairMeanAndVariance.first 
        << "s, stdev: " <<  pairMeanAndVariance.second << "s" << std::endl << std::endl; 

    pairMeanAndVariance = VectorMeanAndVariance(vTimeAddToDatabase);
    f <<  setprecision(6) << " -- time cost for adding one image's BoW to database: "<< std::endl << " mean: " << pairMeanAndVariance.first 
        << "s, stdev: " <<  pairMeanAndVariance.second << "s" << std::endl << std::endl; 

    pairMeanAndVariance = VectorMeanAndVariance(vTimeQuerying);
    f <<  setprecision(6) << " -- time cost for querying: "<< std::endl << " mean: " << pairMeanAndVariance.first 
        << "s, stdev: " <<  pairMeanAndVariance.second << "s" << std::endl << std::endl; 

    f.close();
    std::cout << "time cost test result has been saved to  "  << filename << "." << std::endl;

    
    return 0;
}



// --------------------------------------------------------------------------------------------------------------------------
std::pair<double, double> VectorMeanAndVariance(const std::vector<double> v){
    double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
    double m =  sum / v.size();
    double accum = 0.0;
    std::for_each (std::begin(v), std::end(v), [&](const double d) {
        accum += (d - m) * (d - m);
    });
    double stdev = sqrt(accum / (v.size()-1));

    std::pair<double, double> pairMeanAndVariance;
    pairMeanAndVariance.first = m;
    pairMeanAndVariance.second = stdev;

    return pairMeanAndVariance;
}


// --------------------------------------------------------------------------------------------------------------------------
void ExtractFeaturesFromOneImage(const cv::Mat &imgGray, cv::Ptr<cv::ORB> &orb, 
    cv::Mat &descriptors){
     // extract the KeyPoints and corresponding descripors
    cv::Mat mask;
    std::vector<cv::KeyPoint> keypoints;
    orb->detectAndCompute(imgGray, mask, keypoints, descriptors);
}



// --------------------------------------------------------------------------------------------------------------------------
void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }
    fTimes.close();

    string strPrefixLeft = strPathToSequence + "/image_2/";  // 使用 rgb 图

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}


// --------------------------------------------------------------------------------------------------------------------------
// void SaveTestResult(const string &filename, std::vector<double> &vTimeExtractFeatures,
//     std::vector<double> &vTimeExtractBoW, std::vector<double> &vTimeAddToDatabase,
//     std::vector<double> &vTimeQuerying){
    
//     ofstream f;
//     f.open(filename.c_str());
//     f << fixed; 

//     for(size_t i = 0, n = vTimeExtractFeatures.size(); i < n; i++){
//          f <<  setprecision(6) 

//     }
    
// }