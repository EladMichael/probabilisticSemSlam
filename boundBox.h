#ifndef sensSLAM_boundBox
#define sensSLAM_boundBox

#include <vector>
#include <opencv2/opencv.hpp>
#include <gtsam_quadrics/geometry/AlignedBox2.h>
#include <gtsam/geometry/Point3.h>

//wrapper for gtsam_quadrics::AlignedBox2

class boundBox{
public:
  gtsam_quadrics::AlignedBox2 aBox; //aligned box data (x,y min x,y max)
  int class_id; //semantic class
  float conf; //confidence in class id from NN
  int ID; //some ID, i.e. row from detection matrix or place in vector
  double xOffset; //Offset to align with other image (stereo correlation)
  int boxCorrelation; //Index of correlated detection in other image
  gtsam::Point3 X; //estimate of 3d location RELATIVE TO CAMERA

  double xmin() const {return aBox.xmin();}
  double xmax() const {return aBox.xmax();}
  double ymin() const {return aBox.ymin();}
  double ymax() const {return aBox.ymax();}
  double area() const {return aBox.width()*aBox.height();}
  std::vector<double> getVec() const {return std::vector<double>{xmin(),ymin(),xmax(),ymax()};}

  boundBox(){}
  
  boundBox(int i, const cv::Mat& img, const cv::Mat& det){
    class_id = static_cast<int>(det.at<float>(i,1));
    conf = det.at<float>(i,2);
    ID = i;
    double x0  = static_cast<double>(det.at<float>(i, 3)*img.cols);
    double y0  = static_cast<double>(det.at<float>(i, 4)*img.rows);
    double x1  = static_cast<double>(det.at<float>(i, 5)*img.cols);
    double y1  = static_cast<double>(det.at<float>(i, 6)*img.rows);
    aBox = gtsam_quadrics::AlignedBox2(x0,y0,x1,y1);
  }

  boundBox(float conf, int x0, int y0, int x1, int y1, int class_id, int ID){
    this->conf = conf;
    this->class_id = class_id;
    this->ID = ID;
    aBox = gtsam_quadrics::AlignedBox2(x0,y0,x1,y1);
  }

  boundBox(int i, const std::vector<double>& bb){
    ID = i;
    aBox = gtsam_quadrics::AlignedBox2(bb[0],bb[1],bb[2],bb[3]);
  }

  boundBox(gtsam_quadrics::AlignedBox2 aBox){
    this->aBox = aBox;
  }
  
  inline bool isIn(cv::Point2f p) const {
    return  ((xmin() < p.x) && (p.x < xmax()) &&
            (ymin() < p.y) && (p.y < ymax())); 
  }

  double IoU(const boundBox& other) const {
    double l = std::max(xmin()+xOffset,other.xmin());
    double r = std::min(xmax()+xOffset,other.xmax());
    // using image convention, top left is origin
    double t = std::max(ymin(),other.ymin());
    double b = std::min(ymax(),other.ymax());

    if((l >= r) || (t >= b)){
        // boxes do not overlap
        return 0;
    }
    double interA = (r-l)*(b-t);
    return interA/( area() + other.area() - interA);
  }
};

#endif