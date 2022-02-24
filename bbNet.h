#ifndef sensSLAM_bbNet
#define sensSLAM_bbNet

#include "boundBox.h"

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class bbNet{
private:

    cv::dnn::Net net;

    float scoreThresh = 0.2;
    float nmsThresh = 0.4;
    float confThresh = 0.4;
    std::vector<std::string> labels;
    std::string name;
    int netChoice;
    
    void detectSSD(const cv::Mat& imgL, std::vector<boundBox>& bbL,
        const cv::Mat& imgR, std::vector<boundBox>& bbR);

    void detectYOLO(const cv::Mat& imgL, std::vector<boundBox>& bbL,
        const cv::Mat& imgR, std::vector<boundBox>& bbR);

    cv::Mat stackImgBlob(const cv::Mat& imgL,const cv::Mat& imgR) const;
    cv::Mat batchImgBlob(const cv::Mat& imgL,const cv::Mat& imgR) const;
    cv::Mat singleImgBlob(const cv::Mat& img) const;

public:
    
    bbNet(int netChoice=0){
        //this will set name/labels/netChoice and read net
        set_netChoice(netChoice);
        //changing the netchoice is essentially just making
        //a new instance of this class..
    }
    
    //getters
    std::string get_name() const {return name;}
    std::string get_label(int i) const {return labels[i];}
    std::string get_label(size_t i) const {return labels[i];}
    int get_netChoice() const {return netChoice;}
    float get_scoreThresh() const {return scoreThresh;}
    float get_nmsThresh() const {return nmsThresh;}
    float get_confThresh() const {return confThresh;}

    //setters
    void set_netChoice(int netChoice);
    void set_scoreThresh(float thresh){this->scoreThresh = thresh;}
    void set_nmsThresh(float thresh){this->nmsThresh = thresh;}
    void set_confThresh(float thresh){this->confThresh = thresh;}

    void detect(const cv::Mat& imgL, std::vector<boundBox>& bbL,
        const cv::Mat& imgR, std::vector<boundBox>& bbR) {
        if(netChoice == 0){
            return detectSSD(imgL,bbL,imgR,bbR);
        }else if(netChoice < 99){
            return detectYOLO(imgL,bbL,imgR,bbR);
        }else{
            return;
        }
    }


};

#endif