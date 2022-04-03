#ifndef semSLAM_dataframe
#define semSLAM_dataframe

#include "boundBox.h"
#include "bbNet.h"
#include "constsUtils.h"

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam_quadrics/geometry/ConstrainedDualQuadric.h>

class dataframe{
private:
  // image left
  // probably not necessary to store,
  // honestly, they'll only be looked at
  // once I should think.
  cv::Mat imgL; 
  // image right
  // probably not necessary to store,
  // honestly, they'll only be looked at
  // once I should think.
  cv::Mat imgR; 
  
  // path to left image file
  std::string camLPath; 
  // path to right image file
  std::string camRPath; 
  // path to left data file
  std::string dataLPath; 
  // path to right data file
  std::string dataRPath; 

  //color images?
  bool color;

  // baseline between images
  double baseline;

  // frame number
  size_t frame;
public:  
  // set of measurements from left and right
  std::vector< boundBox > bbL;
  std::vector< boundBox > bbR;

  // vector to hold dual quadric estimates
  std::vector< gtsam_quadrics::ConstrainedDualQuadric > quadEst;

  // constructor
  dataframe(const std::string& toSeq, double baseline, int frameN, bool color);

  // compute the bounding boxes (extract them, align them, and match them)
  void computeBoundingBoxes(bbNet& imageNet, const semConsts& runConsts);

  int matchingBox(const boundBox& box, const std::vector<boundBox>& candidates);

  // for testing purposes, creates calibration objects
  void calibObjs(const gtsam::Pose3& P,const gtsam::Cal3_S2::shared_ptr& K);

  // using the computed bounding boxes, computes quadrics for each bb pair
  void estQuadrics(const gtsam::Pose3& Pl,const gtsam::Cal3_S2::shared_ptr& K);

  // Getters
  cv::Mat getImgL(){return imgL;}
  cv::Mat getImgR(){return imgR;}
  size_t getFrame()const {return frame;}


  // Debugging / Saving functions
  void showImg(int wait = 20){cv::imshow("img",imgL); cv::waitKey(wait);}
  void showBoxes(const std::string& imgText);
  void saveBoxes(const std::string& imgText,size_t frame,const gtsam::Cal3_S2::shared_ptr& K,
    std::vector<gtsam_quadrics::ConstrainedDualQuadric> landmarks = std::vector<gtsam_quadrics::ConstrainedDualQuadric>(),
    std::vector<gtsam::Key> landmarkKeys = std::vector<gtsam::Key>(),gtsam::Pose3 Pl = gtsam::Pose3());
  bool isColor(){return color;}
  void saveData();

};

#endif