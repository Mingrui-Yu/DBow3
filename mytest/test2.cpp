/**
 *  测试在 KITTI 数据集上的 PR 曲线
 * 以 位姿 ground truth 作为判断是否在同一地点的标准
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

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "DescManip.h"

using namespace std;

typedef Eigen::Matrix<double, 3, 3> Mat33;
typedef Eigen::Matrix<double, 3, 1> Vec3;


void ExtractFeaturesFromOneImage(const cv::Mat &imgGray, cv::Ptr<cv::ORB> &orb, 
    cv::Mat &vfeaturesInOneImg);

void LoadImages(const string &strPathToSequence, 
    vector<string> &vstrImageFilenames, vector<double> &vTimestamps);

void LoadGroundtruthPose(const string &strPose, 
    std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t);

std::pair<double, double> VectorMeanAndVariance(const std::vector<double> v);

bool PoseIsVeryNear(std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t,
    int current_id, int loop_id);

bool PoseIsAcceptablyNear(std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t,
    int current_id, int loop_id);



// ----------------------------------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv){
    if(argc != 5){
        std::cerr << std::endl << 
            "Usage: ./mytest/test1   PATH_TO_SEQUENCE  PATH_TO_VOCABULARY PATH_TO_GROUNDTRUTH_POSE  PATH_TO_RESULT" 
            << std::endl;
        return 1;
    }


    // ---------------------------------- Load the images' path -----------------------------------------
    std::string strPathToSequence = argv[1];
    std::vector<string> vstrImageFilenames;
    std::vector<double> vTimestamps;
    LoadImages(strPathToSequence,  vstrImageFilenames, vTimestamps);

    // ---------------------------------- Load the groud_truth file -----------------------------------------
    std::string strPathToPose = argv[3];
    std::vector<Mat33, Eigen::aligned_allocator<Mat33>> vPoses_R;
    std::vector<Vec3, Eigen::aligned_allocator<Vec3>> vPoses_t;
    LoadGroundtruthPose(strPathToPose,  vPoses_R, vPoses_t);

    // ---------------------------------- check if the vocabulary file exists or not----------------------------------
    std::string strPathToVocabulary = argv[2];

    fstream vocFile;
    vocFile.open(strPathToVocabulary);
    if(!vocFile){
        std::cerr << "error: cannot open the vocabulary file " << strPathToVocabulary << endl;
        return -1;
    }
    vocFile.close();

    // ---------------------------------- create the file to store test results ----------------------------------
    ofstream f;
    std::string filename = argv[4];
    f.open(filename.c_str());
    f << fixed;

    // ----------------------------------------- 一些数据初始化 ----------------------------------------------
    int totalNumGroundTruthP = 0;
    // 存储不同 threshold 下的结果
    std::vector<double> vThreshold;
    std::vector<int> vTotalNumTP;
    std::vector<int> vTotalNumFP;
    std::vector<int> vTotalNumFN;
    for(double thres = 0.10; thres > -1e-6; thres -= 0.002){
        vThreshold.push_back(thres);
        vTotalNumTP.push_back(0);
        vTotalNumFP.push_back(0);
        vTotalNumFN.push_back(0);
    }

    // ---------------------------------- load & create KF Database ----------------------------------
    std::cout << "load the vocabulary ..." << std::endl;
    DBoW3::Vocabulary voc(strPathToVocabulary);
    std::cout << "has done." << endl;

    // 记录 存储在 Database 中的 frame 对应的 frame id
    std::vector<unsigned long> vFrameIdInDatabase; 

    std::cout << "create the database ..." << std::endl;
    DBoW3::Database db(voc, true, 1);
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    for(size_t i = 0,  nImgs = vstrImageFilenames.size(); i < nImgs; i+=10)
    {
        cv::Mat originalImg = cv::imread(vstrImageFilenames[i], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imgGray;
        cv::cvtColor(originalImg, imgGray, cv::COLOR_BGR2GRAY); 

        cv::Mat vfeaturesInOneImg;
        ExtractFeaturesFromOneImage(imgGray, orb, vfeaturesInOneImg);
        DBoW3::BowVector bowVector;
        voc.transform(vfeaturesInOneImg, bowVector);

        db.add(bowVector);
        vFrameIdInDatabase.push_back(i);
    }
    std::cout << "has done." << endl;


    // ---------------------------------- query the specified image in the database & record the time cost ----------------------------------
    std::cout << "start test: querying ... " << endl;

    for(size_t nImgId = 0,  nImgs = vstrImageFilenames.size(); nImgId < nImgs; nImgId++){
        int current_id = nImgId;
        cv::Mat originalImg = cv::imread(vstrImageFilenames[nImgId], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imgGray;
        cv::cvtColor(originalImg, imgGray, cv::COLOR_BGR2GRAY); 

        // ------------------   根据 groundtruth 位姿找出数据集中总共存在的正确匹配的个数
        int numGroundTruthP = 0;
        for(auto &db_id: vFrameIdInDatabase){

            if( std::abs((int)current_id - (int)db_id) < 100){ // 时间上临近的就不算 LOOP 了
                continue;
            }
            bool bGroundTruthCorrect = PoseIsVeryNear(vPoses_R, vPoses_t, current_id, db_id);
            
            if (bGroundTruthCorrect){
                std::cout << std::setprecision(3) << "GroundTruth for Image " 
                    << current_id << ":  db id: " << db_id  << std::endl;
                numGroundTruthP++;
            }
        }
        totalNumGroundTruthP += numGroundTruthP;

        // DBoW3 查询
        cv::Mat vfeaturesInOneImg;
        ExtractFeaturesFromOneImage(imgGray, orb, vfeaturesInOneImg);
        DBoW3::BowVector bowVector;
        voc.transform(vfeaturesInOneImg, bowVector);

        DBoW3::QueryResults queryResult;
        db.query(bowVector, queryResult, vFrameIdInDatabase.size(), -1); // max_result=1, max_id=-1 unlimited

        for(size_t index = 0; index < vThreshold.size(); index++){
            int numTP = 0;
            int numFP = 0;
            for(auto qit = queryResult.begin(); qit != queryResult.end(); qit++){
                int loop_id =  vFrameIdInDatabase[qit->Id];
                if(qit->Score < vThreshold[index]){
                    continue;
                }
                if( std::abs((int)current_id - (int)loop_id) < 100){ // 时间上临近的就不算 LOOP 了
                    continue;
                }
                bool bCorrect = PoseIsVeryNear(vPoses_R, vPoses_t, current_id, loop_id);         
                bool bAcceptable = PoseIsAcceptablyNear(vPoses_R, vPoses_t, current_id, loop_id);

                 if (bCorrect){
                    numTP++;
                } else if ( ! bAcceptable) {
                    numFP++;
                }
                std::cout << std::setprecision(3) << "Searching for Image " << current_id << ":  loop id: " << loop_id
                        << ", score: " << qit->Score  << ", bAcceptable: " << bAcceptable << std::endl;
            }

            vTotalNumTP[index] += numTP;
            vTotalNumFP[index] += numFP;
        }

    }


   // 结果输出 保存至 txt 文件
    for(size_t i = 0; i < vThreshold.size(); i++){
        vTotalNumFN[i] = totalNumGroundTruthP - vTotalNumTP[i];

        double precision  = ((double)vTotalNumTP[i]) / (double)(vTotalNumTP[i] + vTotalNumFP[i]);
        double recall = (double)vTotalNumTP[i] / (double)(vTotalNumTP[i] + vTotalNumFN[i]);

        std::cout << "threshold:" << vThreshold[i] <<", precision: " << precision << ", recall: " << recall << std::endl;
        f << std::setprecision(6) << vThreshold[i] << " " << precision << " " << recall << std::endl;
    }
    
    f.close();
    std::cout << "result has been saved to  "  << filename << "." << std::endl;

    
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

// ---------------------------------------------------------------------------------------

void LoadGroundtruthPose(const string &strPose, 
    std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t){

    vPoses_R.clear();
    vPoses_t.clear();

    ifstream fPoses;
    std::string strPathPoseFile = strPose;
    fPoses.open(strPathPoseFile.c_str());
    while(!fPoses.eof())
    {
        double R0, R1, R2, R3, R4, R5, R6, R7, R8;
        double t0, t1, t2;
        fPoses >> R0 >> R1 >> R2 >> t0
                      >> R3 >> R4 >> R5 >> t1
                      >> R6 >> R7 >> R8 >> t2;

        Mat33 R; 
        R << R0, R1, R2,
                  R3, R4, R5,
                  R6, R7, R8;
        Vec3 t; 
        t << t0, t1, t2;

        vPoses_R.push_back(R);
        vPoses_t.push_back(t);
    }

    fPoses.close();
}

// --------------------------------------------------------------------------------------------------------------------------

bool PoseIsVeryNear(std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t,
    int current_id, int loop_id){

    Eigen::AngleAxisd rotation_vector;
    rotation_vector.fromRotationMatrix(vPoses_R[current_id].inverse() * vPoses_R[loop_id]);

    bool bCorrect = ((vPoses_t[current_id] - vPoses_t[loop_id]).norm() < 5)
                                        && ((std::abs(rotation_vector.angle()) < 3.14 / 12));
                                    
    return bCorrect;
}

// --------------------------------------------------------------------------------------------------------------------------

bool PoseIsAcceptablyNear(std::vector<Mat33, Eigen::aligned_allocator<Mat33> > &vPoses_R,
    std::vector<Vec3, Eigen::aligned_allocator<Vec3> > &vPoses_t,
    int current_id, int loop_id){

    Eigen::AngleAxisd rotation_vector;
    rotation_vector.fromRotationMatrix(vPoses_R[current_id].inverse() * vPoses_R[loop_id]);

    bool bCorrect = ((vPoses_t[current_id] - vPoses_t[loop_id]).norm() < 20)
                                        && ((std::abs(rotation_vector.angle()) < 3.14 / 6));
                                    
    return bCorrect;
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