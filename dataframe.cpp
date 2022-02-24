#include "dataframe.h"
#include "assignment.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/triangulation.h>
#include <gtsam_quadrics/geometry/QuadricCamera.h>

inline bool goodMatch(cv::Point2f p0, cv::Point2f p1){
    return ((std::abs(p0.x-p1.x) < 100) && (std::abs(p0.y-p1.y) < 5));
}

dataframe::dataframe(const std::string& toSeq, double baseline, int frameN, bool color)
{

    this->baseline = baseline;
    this->color = color;
    this->frame = frameN;
    char imgSpecifier[20]; //longer than it should ever need to be
    char dataSpecifier[20]; //longer than it should ever need to be
    sprintf(imgSpecifier,"%06i.png",frameN);
    sprintf(dataSpecifier,"%06i",frameN);
    std::string dataDir("/home/emextern/Desktop/codeStorage/semSLAM/data/");
    // grab sequence number, i.e. 00, and / from end of toSeq
    dataLPath = dataDir + toSeq.substr(toSeq.size()-3,3);
    dataRPath = dataDir + toSeq.substr(toSeq.size()-3,3);
    if(!color){
        // grayscale
        camLPath = toSeq + std::string("image_0/") + imgSpecifier;
        camRPath = toSeq + std::string("image_1/") + imgSpecifier;
        dataLPath += std::string("image_0/") + dataSpecifier;
        dataRPath += std::string("image_1/") + dataSpecifier;
    }else{
        camLPath = toSeq + std::string("image_2/") + imgSpecifier;
        camRPath = toSeq + std::string("image_3/") + imgSpecifier;
        dataLPath += std::string("image_2/") + dataSpecifier;
        dataRPath += std::string("image_3/") + dataSpecifier;
    }
    if(file_exists(camLPath) && file_exists(camRPath)){
        imgL = cv::imread(camLPath);
        imgR = cv::imread(camRPath);
    }else{
        std::string errMsg("Image file could not be found!\n Path to images L and R: \n");
        errMsg += camLPath + std::string("\n") + camRPath + std::string("\n");
        throw std::runtime_error(errMsg);
    }
}


void dataframe::computeBoundingBoxes(bbNet& imageNet, const semConsts& runConsts)
{
    bool verbose = true;

    if(imageNet.get_netChoice() == 99){bbL.clear();bbR.clear();return;}

    dataLPath += "_"+std::to_string(imageNet.get_netChoice())+".tt";
    dataRPath += "_"+std::to_string(imageNet.get_netChoice())+".tt";

    // if the file doesn't exist, save the data, to aovid recomputation next time.
    // this will majorly speed up runs in the future, for prototyping purposes.
    bool saveDataFlag = false;
    if(file_exists(dataLPath) && file_exists(dataRPath)){
        //load the data from file
        std::ifstream ifs;
        ifs.open(dataLPath, std::ifstream::in);
        size_t obj = 0;
        while(ifs.peek() != std::ifstream::traits_type::eof()){
            bbL.push_back(boundBox());
            ifs.read((char*)&bbL[obj], sizeof(bbL[obj]));
            obj++;
        }
        ifs.close();
        // bbL.pop_back();


        ifs.open(dataRPath, std::ifstream::in);
        obj = 0;
        while(ifs.peek() != std::ifstream::traits_type::eof()){
            bbR.push_back(boundBox());
            ifs.read((char*)&bbR[obj], sizeof(bbR[obj]));
            obj++;
        }
        ifs.close(); 
        // bbR.pop_back();
        return;
    }

    imageNet.detect(imgL,bbL,imgR,bbR);

    if(verbose){
        cv::Mat bigImg,bigImgMask;
        cv::vconcat(imgL,imgR,bigImg);
        int offset = imgL.rows;
        for(size_t b = 0; b < bbL.size(); b++){
            int conf = static_cast<int>(100*bbL[b].conf);

            cv::putText(bigImg, imageNet.get_label(bbL[b].class_id)+" : 0."+std::to_string(conf), cv::Point(bbL[b].xmin(), bbL[b].ymin()-5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,0), 3);
            cv::putText(bigImg, imageNet.get_label(bbL[b].class_id)+" : 0."+std::to_string(conf), cv::Point(bbL[b].xmin(), bbL[b].ymin()-5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);

            cv::rectangle(bigImg, cv::Point(bbL[b].xmin(),bbL[b].ymin()), cv::Point(bbL[b].xmax(),bbL[b].ymax()), cv::Scalar(255,0,0), 1);
        }
        for(size_t b = 0; b < bbR.size(); b++){
            cv::rectangle(bigImg, cv::Point(bbR[b].xmin(),bbR[b].ymin()+offset), cv::Point(bbR[b].xmax(),bbR[b].ymax()+offset), cv::Scalar(255,0,0), 1);
        }
        cv::imshow("boxes",bigImg);

        cv::imwrite(dataLPath+std::string("_boxes.jpg"), bigImg);
        cv::waitKey(1000);
    }

    // if no candidates left, return empty handed
    if(bbL.empty() || bbR.empty()){
        bbL.clear();
        bbR.clear();
        if(saveDataFlag){saveData();}
        return;
    }

    if(verbose){
        std::cout<<"Initialized boxes #L, #R : "<<bbL.size()<<" , "<<bbR.size()<<std::endl;
        // for(size_t b = 0; b < bbL.size(); b++){
        //     std::cout<<"----- box "<<b<<" from img L ------\n";
        //     std::cout<<"x0,x1  y0,y1 : "<<bbL[b].xmin()<<" , "<<bbL[b].xmax();
        //     std::cout<<"    "<<bbL[b].ymin()<<" , "<<bbL[b].ymax()<<std::endl;
        // }
        // for(size_t b = 0; b < bbR.size(); b++){   
        //     std::cout<<"----- box "<<b<<" from img R ------\n";
        //     std::cout<<"x0,x1  y0,y1 : "<<bbR[b].xmin()<<" , "<<bbR[b].xmax();
        //     std::cout<<"    "<<bbR[b].ymin()<<" , "<<bbR[b].ymax()<<std::endl;
        // }
    }
    // For each detection, cut a "hole" in the mask 
    // to allow the detection of ORB keypoints in the 
    // bounding box, for correlation. zero is matrix type
    cv::Mat maskL = cv::Mat::zeros(imgL.size(),0);
    cv::Mat maskR = cv::Mat::zeros(imgR.size(),0);
    for(auto box = bbL.begin(); box != bbL.end(); box++){
        cv::rectangle(maskL, cv::Point(box->xmin(),box->ymin()), cv::Point(box->xmax(),box->ymax()), cv::Scalar(255), -1);
    }
    for(auto box = bbR.begin(); box != bbR.end(); box++){
        cv::rectangle(maskR, cv::Point(box->xmin(),box->ymin()), cv::Point(box->xmax(),box->ymax()), cv::Scalar(255), -1);
    }

    // Variables to store keypoints and descriptors
    cv::Mat descriptorsL,descriptorsR;
    std::vector<cv::KeyPoint> keypointsL, keypointsR;

    // Detect ORB features and compute descriptors.
    int featPerDet = 100;
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(featPerDet*std::max(bbL.size(),bbR.size()));
    if(isColor()){
        cv::Mat grayL,grayR;
        cv::cvtColor(imgL,grayL,cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgR,grayR,cv::COLOR_BGR2GRAY);
        orb->detectAndCompute(grayL, maskL, keypointsL, descriptorsL);
        orb->detectAndCompute(grayR, maskR, keypointsR, descriptorsR);
    }else{
        orb->detectAndCompute(imgL, maskL, keypointsL, descriptorsL);
        orb->detectAndCompute(imgR, maskR, keypointsR, descriptorsR);        
    }

    if(descriptorsL.empty() || descriptorsR.empty()){
        bbL.clear();
        bbR.clear();
        if(saveDataFlag){saveData();}
        return;
    }

    // Match features.
    std::vector<cv::DMatch> matches;
    // std::sort(matches.begin(), matches.end());
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING,true);
    matcher->match(descriptorsL, descriptorsR, matches, cv::Mat());
    if(false){
        std::cout<<"Got "<<matches.size()<<" matches out of ( "<<keypointsL.size()<<" , "<<keypointsR.size()<<" ) points."<<std::endl;
        cv::Mat imMatches;
        cv::drawMatches(imgL, keypointsL, imgR, keypointsR, matches, imMatches);
        cv::imshow("orb matches",imMatches);
        cv::waitKey(100);
        // std::cin.get();
    }

    // std::cin.get();
    // Find ORB points in each box, and where they map to
    // this is used to compute the stereo offset
    // For each detection, get the average xOffset of the matched ORB points
    // large x,y offsets are rejected (images are rectified, objects are farish)
    for(auto box = bbL.begin(); box != bbL.end();){
        int goodMatches = 0;
        box->xOffset = 0;
        for( size_t i = 0; i < matches.size(); i++ ){
            cv::Point2f p0 = keypointsL[matches[i].queryIdx].pt;
            cv::Point2f p1 = keypointsR[matches[i].trainIdx].pt;
            if(box->isIn(p0) && goodMatch(p0,p1)){
                box->xOffset += p1.x - p0.x;
                goodMatches++;
            }
        }
        if(goodMatches > 0){
            // If there is a suitable offset found,
            // check if there is a good correlation
            box->xOffset = box->xOffset / goodMatches;
            box++;
        }else{
            // if no suitable offset is found,
            // delete the box!
            bbL.erase(box);
        }
    }

    for(auto box = bbR.begin(); box != bbR.end(); ){
        int goodMatches = 0;
        box->xOffset = 0;
        for( size_t i = 0; i < matches.size(); i++ ){
            cv::Point2f p0 = keypointsL[matches[i].queryIdx].pt;
            cv::Point2f p1 = keypointsR[matches[i].trainIdx].pt;
            if(box->isIn(p1) && goodMatch(p0,p1)){
                box->xOffset += p0.x - p1.x;
                goodMatches++;
            }
        }
        if(goodMatches > 0){
            // If there is a suitable offset found,
            // store it and iterate to the next
            box->xOffset = box->xOffset / goodMatches;
            box++;
        }else{
            // if no suitable offset is found,
            // delete the box!
            bbR.erase(box);
        }
    }

    if(verbose){
        std::cout<<"Found matching ORB points and xOffsets."<<std::endl;
        std::cout<<"Boxes remaining: "<<bbL.size()<<" , "<<bbR.size()<<std::endl;
        // std::cout<<"Found xOffsets"<<std::endl;
        // std::cout<<"xOffsets: "<<std::endl;
        // for(auto box = bbL.begin(); box != bbL.end(); box++){
        //     std::cout<<box->xOffset<<" , ";
        // }
        // for(auto box = bbR.begin(); box != bbR.end(); box++){
        //     std::cout<<box->xOffset<<" , ";
        // }

        cv::Mat bigImg;
        cv::vconcat(imgL,imgR,bigImg);
        int offset = imgL.rows;
        for(size_t b = 0; b < bbL.size(); b++){
            cv::rectangle(bigImg, cv::Point(bbL[b].xmin(),bbL[b].ymin()), cv::Point(bbL[b].xmax(),bbL[b].ymax()), cv::Scalar(255,0,0), 1);
            cv::line(bigImg, cv::Point(0.5*(bbL[b].xmin()+bbL[b].xmax()),0.5*(bbL[b].ymin()+bbL[b].ymax())),
                             cv::Point(0.5*(bbL[b].xmin()+bbL[b].xmax())+bbL[b].xOffset,0.5*(bbL[b].ymin()+bbL[b].ymax())+offset),cv::Scalar(0,0,255),2);
        }
        for(size_t b = 0; b < bbR.size(); b++){
            cv::rectangle(bigImg, cv::Point(bbR[b].xmin(),bbR[b].ymin()+offset), cv::Point(bbR[b].xmax(),bbR[b].ymax()+offset), cv::Scalar(255,0,0), 1);
            // cv::line(bigImg, cv::Point(0.5*(bbR[b].xmin()+bbR[b].xmax()),0.5*(bbR[b].ymin()+bbR[b].ymax())+offset),
            //                  cv::Point(0.5*(bbR[b].xmin()+bbR[b].xmax())+bbR[b].xOffset,0.5*(bbR[b].ymin()+bbR[b].ymax())),cv::Scalar(0,0,255),2);
        }
        cv::imshow("offset boxes",bigImg);
        cv::imwrite(dataLPath+std::string("_assoc.jpg"), bigImg);
        cv::waitKey(1000);

    }

    // if no candidates left, return empty handed
    if(bbL.empty() || bbR.empty()){
        bbL.clear();
        bbR.clear();
        if(saveDataFlag){saveData();}
        return;
    }

    // Get assignment of bounding boxes from one image 
    // to the other
    std::vector<int> asgn = asgnBB(bbL,bbR,runConsts);
    
    // align assigned left boxes, translated assignments
    // to ID assignments (independent of position in vector)
    size_t asgnBoxes = 0;
    for(size_t b = 0; b < bbL.size(); b++){
        if(asgn[b] != -1){
            bbL[b].boxCorrelation = bbR[asgn[b]].ID;
            bbR[asgn[b]].boxCorrelation = bbL[b].ID;
            std::swap(bbL[asgnBoxes],bbL[b]);
            asgnBoxes++;
        }
    }

    if(asgnBoxes == 0){
        bbL.clear();
        bbR.clear();
        if(saveDataFlag){saveData();}
        return;
    }
    
    // align right boxes and get semantic information
    for(size_t bl = 0; bl < asgnBoxes; bl++){
        int rID = bbL[bl].boxCorrelation;
        if(bbR[bl].ID != rID){
            for(size_t br = bl+1; br < bbR.size(); br++){
                if(bbR[br].ID == rID){
                    std::swap(bbR[bl],bbR[br]);
                }
            }
        }
    }

    //remove unassigned boxes from left and right
    bbL.erase(bbL.begin()+asgnBoxes,bbL.end());
    bbR.erase(bbR.begin()+asgnBoxes,bbR.end());

    if(saveDataFlag){
        saveData();
    }
    return;
}
//###########################################################################################
void dataframe::saveData(){
    std::ofstream ofsL;
    std::ofstream ofsR;
    ofsL.open(dataLPath, std::ofstream::out | std::ofstream::trunc);
    ofsR.open(dataRPath, std::ofstream::out | std::ofstream::trunc);
    for(size_t obj = 0; obj < bbL.size(); obj++){
        ofsL.write((char*)&bbL[obj], sizeof(bbL[obj]));
        ofsR.write((char*)&bbR[obj], sizeof(bbR[obj]));
    }
    ofsL.close();    
    ofsR.close();   
}
//###########################################################################################
gtsam::Point3 dataframe::triangulatePoint(const cv::Point2f& pl,const cv::Point2f& pr) const
{
    gtsam::Pose3 P0 = gtsam::Pose3();
    gtsam::Pose3 P1 = P0.compose(gtsam::Pose3(gtsam::Rot3(),gtsam::Point3(baseline,0,0)));
    Eigen::Matrix<double,4,4> Pl = P0.matrix();
    Eigen::Matrix<double,4,4> Pr = P1.matrix();

    Eigen::Matrix<double,4,4> A;    
    for( int k = 0; k < 4; k++ )
    {
        A(0, k) = pl.x * Pl(2,k) - Pl(0,k);
        A(1, k) = pl.y * Pl(2,k) - Pl(1,k);
        A(2, k) = pr.x * Pr(2,k) - Pr(0,k);
        A(3, k) = pr.y * Pr(2,k) - Pr(1,k);
    }

    /* Solve system for current point */
    Eigen::JacobiSVD<Eigen::Matrix<double,4,4>> svd(A, Eigen::ComputeFullV);
    Eigen::MatrixXd V = svd.matrixV();

    /* Copy computed point */
    gtsam::Point3 output = -V({0,1,2},3) / V(3,3);
    return output;
}
//###########################################################################################
gtsam::Point3 dataframe::triangulate(const Eigen::Matrix<double,3,4>& P0, const Eigen::Matrix<double,3,4>& P1,size_t obj) const
{
    Eigen::Matrix<double,4,4> A;    
    double x0 = 0.5*(bbL[obj].xmin()+bbL[obj].xmax());
    double y0 = 0.5*(bbL[obj].ymin()+bbL[obj].ymax());
    double x1 = x0 + bbL[obj].xOffset;
    double y1 = y0;
    for( int k = 0; k < 4; k++ )
    {
        A(0, k) = x0 * P0(2,k) - P0(0,k);
        A(1, k) = y0 * P0(2,k) - P0(1,k);
        A(2, k) = x1 * P1(2,k) - P1(0,k);
        A(3, k) = y1 * P1(2,k) - P1(1,k);
    }

    /* Solve system for current point */
    Eigen::JacobiSVD<Eigen::Matrix<double,4,4>> svd(A, Eigen::ComputeFullV);
    Eigen::MatrixXd V = svd.matrixV();

    /* Copy computed point */
    gtsam::Point3 output = -V({0,1,2},3) / V(3,3);
    return output;
}
//###########################################################################################
// std::vector<gtsam::Point3> dataframe::triangulateAll(const Eigen::Matrix<double,3,3>& K, const gtsam::Pose3& P) const {
//     std::vector<gtsam::Point3> priors;

//     gtsam::Pose3 rightNudge = gtsam::Pose3(gtsam::Rot3(),gtsam::Point3(-baseline,0,0));
//     std::vector< Eigen::Matrix<double,3,4>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4> > > projMatrices(2);
//     projMatrices[0] = K*(P.matrix().block<3,4>(0,0));
//     projMatrices[1] = K*((P.compose(rightNudge)).matrix().block<3,4>(0,0));

//     for(size_t obj = 0; obj < bbL.size(); obj++){
//         priors.push_back(triangulate(projMatrices,obj));
//     }
//     return priors;
// }
//###########################################################################################
void dataframe::calibObjs(const gtsam::Pose3& P,const gtsam::Cal3_S2::shared_ptr& K){

    gtsam::Pose3 rightNudge = gtsam::Pose3(gtsam::Rot3(),gtsam::Point3(baseline,0,0));
    
    std::vector<gtsam_quadrics::ConstrainedDualQuadric> quadrics;
    // these are the calibration objects
    quadrics.push_back(gtsam_quadrics::ConstrainedDualQuadric(gtsam::Pose3(gtsam::Rot3(),gtsam::Point3(1,-1,50)), gtsam::Vector3(1.,2.,2.)));
    quadrics.push_back(gtsam_quadrics::ConstrainedDualQuadric(gtsam::Pose3(gtsam::Rot3(),gtsam::Point3(-20,-1,60)), gtsam::Vector3(1.,2.,2.)));
    size_t nM = bbL.size();
    for (size_t q = 0; q<quadrics.size(); q++) {
        gtsam_quadrics::DualConic conicL = gtsam_quadrics::QuadricCamera::project(quadrics[q], P, K);
        gtsam_quadrics::DualConic conicR = gtsam_quadrics::QuadricCamera::project(quadrics[q], P.compose(rightNudge), K);
        bbL.push_back(boundBox(conicL.bounds()));
        bbR.push_back(boundBox(conicR.bounds()));
        bbL[nM+q].xOffset = bbR[nM+q].xmin()-bbL[nM+q].xmin();
        bbR[nM+q].xOffset = bbL[nM+q].xmin()-bbR[nM+q].xmin();
        bbL[nM+q].class_id = 2; //car
        bbR[nM+q].class_id = 2; //car
        size_t obj = nM+q;
    }


    return;
}
//###########################################################################################
gtsam::Point3 simpleStereo(double u, double v,double b,double disp, const gtsam::Cal3_S2::shared_ptr& K){
    gtsam::Point3 output;
    output(0) = b*(u-(K->px()))/disp;
    output(1) = (K->fx())*b*(v-(K->py()))/(K->fy()*disp);
    output(2) = b*(K->fx())/disp;
    return output;
}
//###########################################################################################
void dataframe::estQuadrics(const gtsam::Pose3& Pl,const gtsam::Cal3_S2::shared_ptr& K){

    quadEst.clear();

    bool verbose = false;
    // create right pose by nudging Pl to the right by baseline.
    gtsam::Pose3 Pr = Pl.compose(gtsam::Pose3(gtsam::Rot3(),gtsam::Point3(baseline,0,0)));    
    Eigen::Matrix<double,3,4> projL = K->K()*Pl.matrix().block<3,4>(0,0);
    Eigen::Matrix<double,3,4> projR = K->K()*Pr.matrix().block<3,4>(0,0);
    std::vector<gtsam::Pose3> poses{Pl,Pr};
    std::vector< gtsam::Point2 , Eigen::aligned_allocator<Eigen::Matrix<double, 2, 1> > > points(2);
    // for each correlated pair of bounding boxes, compute a quadric estimate
    // we assume the depth is 2 (meters?), but this should eventually be a semantically 
    // motivated prior (i.e. cars cannot be more than this deep, people this deep, etc etc)
    if(verbose){
        std::cout<<"Estimating quadrics!\n Pose Left: \n"<<Pl<<"\n Pose Right: \n"<<Pr<<std::endl;
        cv::Mat bigImg;
        cv::vconcat(imgL,imgR,bigImg);
        int offset = imgL.rows;
        for(size_t b = 0; b < bbL.size(); b++){
            cv::rectangle(bigImg, cv::Point(bbL[b].xmin(),bbL[b].ymin()), cv::Point(bbL[b].xmax(),bbL[b].ymax()), cv::Scalar(255,0,0), 1);
            cv::rectangle(bigImg, cv::Point(bbR[b].xmin(),bbR[b].ymin()+offset), cv::Point(bbR[b].xmax(),bbR[b].ymax()+offset), cv::Scalar(255,0,0), 1);

            cv::line(bigImg, cv::Point(0.5*(bbL[b].xmin()+bbL[b].xmax()),0.5*(bbL[b].ymin()+bbL[b].ymax())),
                             cv::Point(0.5*(bbR[b].xmin()+bbR[b].xmax()),0.5*(bbR[b].ymin()+bbR[b].ymax())+offset),cv::Scalar(0,0,255),2);
        }
        cv::imshow("Box Matches",bigImg);
        cv::waitKey(2000);
    }
    size_t nObj = bbL.size();
    gtsam::Point3 eye = Pl.translation();
    gtsam::PinholeCamera cam(Pl,*K);
    for(size_t obj = 0; obj < nObj; obj++){

        //semantic prior on depth
        double obj_depth;
        switch(bbL[obj].class_id){
            case 0: obj_depth = 0.5; break; //person
            case 1: obj_depth = 0.5; break; //bicycle
            case 2: obj_depth = 2; break; //car
            case 3: obj_depth = 1; break; //motorcycle
            case 4: obj_depth = 8; break; //airplane
            case 5: obj_depth = 8; break; //bus
            case 6: obj_depth = 8; break; //train
            case 7: obj_depth = 5; break; //truck
            case 8: obj_depth = 4; break; //boat
            case 9: obj_depth = 0.25; break; //traffic light
            case 10: obj_depth = 0.25; break; //fire hydrant
            case 11: obj_depth = 0.25; break; //street sign
            case 12: obj_depth = 0.25; break; //stop sign
            case 13: obj_depth = 0.25; break; //parking meter
            case 14: obj_depth = 0.5; break; //bench
            default: obj_depth = 2;
        }    
        // get center 
        // gtsam::Point3 center = triangulate(projL,projR,obj);

        double u = 0.5*(bbL[obj].xmin()+bbL[obj].xmax());
        double v = 0.5*(bbL[obj].ymin()+bbL[obj].ymax());
        gtsam::Point3 localCenter = simpleStereo(u,v,baseline,bbR[obj].xOffset,K);
        gtsam::Point3 xShift = simpleStereo(bbL[obj].xmin(),v,baseline,bbR[obj].xOffset,K);
        gtsam::Point3 yShift = simpleStereo(u,bbL[obj].ymin(),baseline,bbR[obj].xOffset,K);
        double rx = (localCenter-xShift).norm();
        double ry = (localCenter-yShift).norm();

        localCenter *= (1+0.5*obj_depth/localCenter.norm());

        gtsam::Point3 center = Pl.transformFrom(localCenter);

        gtsam::Rot3 quadRot = cam.Lookat(eye,center,gtsam::Point3(0,-1,0)).pose().rotation();

    
        //this shifts the center to the supposed center given the depth, rather than constructing
        //the ellipsoid centered on the planar surface in ``front'' which was observed
        // center(0) += (center(0)-eye(0))*0.5*obj_depth/depth; 
        // center(2) += (center(2)-eye(2))*0.5*obj_depth/depth; 

        // get the new depth, to the center of the eventual ellipsoid
        // depth = (center-eye).norm();

        gtsam::Vector3 radii(rx,ry,obj_depth);
        quadEst.push_back(gtsam_quadrics::ConstrainedDualQuadric(gtsam::Pose3(quadRot,center),radii));
        if(verbose){
            std::cout<<"Estimated center for object: "<<obj<<" is "<<center.transpose()<<std::endl;
            std::cout<<"Estimated rotation for object: "<<obj<<" is \n"<<quadRot<<std::endl;
            std::cout<<"estimated dual quadric"<<obj<<" has pose: \n"<<quadEst[obj].pose()<<std::endl;
            std::cout<<"estimated dual quadric"<<obj<<" has radii: \n"<<quadEst[obj].radii().transpose()<<std::endl;
        }
        // TODO: REPROJECT QUADRICS INTO THE RIGHT AND LEFT IMAGE, SHOW THE NEW BOXES WITH THE REPROJECTED LANDMARKS
    }


    bool checkReprojection = false;
    if(checkReprojection){
        std::cout<<"Checking reprojection!"<<std::endl;
        cv::Mat bigImg;
        cv::vconcat(imgL,imgR,bigImg);

        int offset = imgL.rows;
        for(size_t b = 0; b < bbL.size(); b++){
            cv::rectangle(bigImg, cv::Point(bbL[b].xmin(),bbL[b].ymin()), cv::Point(bbL[b].xmax(),bbL[b].ymax()), cv::Scalar(255,255,255), 1);
            cv::rectangle(bigImg, cv::Point(bbR[b].xmin(),bbR[b].ymin()+offset), cv::Point(bbR[b].xmax(),bbR[b].ymax()+offset), cv::Scalar(255,255,255), 1);

            gtsam_quadrics::DualConic conicL = gtsam_quadrics::QuadricCamera::project(quadEst[b], Pl, K);
            gtsam_quadrics::DualConic conicR = gtsam_quadrics::QuadricCamera::project(quadEst[b], Pr, K);
            gtsam_quadrics::AlignedBox2 lBox = conicL.bounds();
            gtsam_quadrics::AlignedBox2 rBox = conicR.bounds();

            cv::rectangle(bigImg, cv::Point(lBox.xmin(),lBox.ymin()), cv::Point(lBox.xmax(),lBox.ymax()), cv::Scalar(0,0,255), 1);
            cv::rectangle(bigImg, cv::Point(rBox.xmin(),rBox.ymin()+offset), cv::Point(rBox.xmax(),rBox.ymax()+offset), cv::Scalar(0,0,255), 1);

        }
        cv::imshow("White Measurements, Red Reprojection",bigImg);
        cv::waitKey(5000);
    }
        





}
// #####################################################################################################
void dataframe::showBoxes(const std::string& imgText){

    // GET THIS TO SHOW THE ASSOCIATIONS, SO YOU CAN SEE IF IT IS WORKING. 
    // MAYBE EVEN WITH A LITTLE PROBABILITY BREAKDOWN FOR EACH BOX? ALSO, GENERATE ABSOLUTE DISTANCE ERROR PLOTS
    // IMPLEMENT LANDMARK MUST BE SEEN TWICE BEFORE ENTERING GRAPH
    // CHECK TO SEE IF LANDMARK IS BEHIND CAMERA BEFORE ENTERING IN GRAPH...

    cv::Mat bigImg;
    cv::vconcat(imgL,imgR,bigImg);
    int offset = imgL.rows;
    for(size_t b = 0; b < bbL.size(); b++){

        cv::rectangle(bigImg, cv::Point(bbL[b].xmin(),bbL[b].ymin()), cv::Point(bbL[b].xmax(),bbL[b].ymax()), cv::Scalar(255,0,0), 1);
        cv::rectangle(bigImg, cv::Point(bbR[b].xmin(),bbR[b].ymin()+offset), cv::Point(bbR[b].xmax(),bbR[b].ymax()+offset), cv::Scalar(255,0,0), 1);

        cv::putText(bigImg, std::to_string(b), cv::Point(bbL[b].xmin(), bbL[b].ymin()-5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,0), 4);
        cv::putText(bigImg, std::to_string(b), cv::Point(bbL[b].xmin(), bbL[b].ymin()-5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 2);


        cv::line(bigImg, cv::Point(0.5*(bbL[b].xmin()+bbL[b].xmax()),0.5*(bbL[b].ymin()+bbL[b].ymax())),
                         cv::Point(0.5*(bbR[b].xmin()+bbR[b].xmax()),0.5*(bbR[b].ymin()+bbR[b].ymax())+offset),cv::Scalar(0,0,255),2);
    }
    cv::putText(bigImg, imgText.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    cv::imshow("Detected Boxes",bigImg);
    cv::waitKey(1);
    // std::cin.get();
}
// #####################################################################################################
void dataframe::saveBoxes(const std::string& imgText, size_t frame,const gtsam::Cal3_S2::shared_ptr& K, std::vector<gtsam_quadrics::ConstrainedDualQuadric> landmarks,
    std::vector<gtsam::Key> landmarkKeys,gtsam::Pose3 Pl){

    cv::Mat bigImg;
    cv::vconcat(imgL,imgR,bigImg);
    int offset = imgL.rows;
    for(size_t b = 0; b < bbL.size(); b++){

        cv::rectangle(bigImg, cv::Point(bbL[b].xmin(),bbL[b].ymin()), cv::Point(bbL[b].xmax(),bbL[b].ymax()), cv::Scalar(255,255,255), 1);
        cv::rectangle(bigImg, cv::Point(bbR[b].xmin(),bbR[b].ymin()+offset), cv::Point(bbR[b].xmax(),bbR[b].ymax()+offset), cv::Scalar(255,255,255), 1);

        cv::putText(bigImg, std::to_string(b), cv::Point(bbL[b].xmin(), bbL[b].ymin()-5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,0), 4);
        cv::putText(bigImg, std::to_string(b), cv::Point(bbL[b].xmin(), bbL[b].ymin()-5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 2);


        cv::line(bigImg, cv::Point(0.5*(bbL[b].xmin()+bbL[b].xmax()),0.5*(bbL[b].ymin()+bbL[b].ymax())),
                         cv::Point(0.5*(bbR[b].xmin()+bbR[b].xmax()),0.5*(bbR[b].ymin()+bbR[b].ymax())+offset),cv::Scalar(255,255,255),2);
    }
    
    cv::putText(bigImg, imgText.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);

    gtsam::Pose3 Pr = Pl.compose(gtsam::Pose3(gtsam::Rot3(),gtsam::Point3(baseline,0,0)));

    for(size_t l = 0; l < landmarks.size(); l++){
        
        if(landmarks[l].isBehind(Pl) || ((Pl.translation() - landmarks[l].centroid()).norm()>50) ){
            //don't draw landmarks behind the pose, or landmarks more than 50 meters away.
            continue;
        }

        gtsam_quadrics::DualConic conicL = gtsam_quadrics::QuadricCamera::project(landmarks[l], Pl, K);
        gtsam_quadrics::DualConic conicR = gtsam_quadrics::QuadricCamera::project(landmarks[l], Pr, K);
        gtsam_quadrics::AlignedBox2 lBox = conicL.bounds();
        gtsam_quadrics::AlignedBox2 rBox = conicR.bounds();

        double x0 = std::max(lBox.xmin(),0.0);
        double y0 = std::max(lBox.ymin(),0.0);
        double x1 = std::min(lBox.xmax(),static_cast<double>(imgL.cols));
        double y1 = std::min(lBox.ymax(),static_cast<double>(imgL.rows));
        cv::rectangle(bigImg, cv::Point(x0,y0), cv::Point(x1,y1), cv::Scalar(0,0,255), 1);

        x0 = std::max(rBox.xmin(),0.0);
        y0 = std::max(rBox.ymin(),0.0);
        x1 = std::min(rBox.xmax(),static_cast<double>(imgR.cols));
        y1 = std::min(rBox.ymax(),static_cast<double>(imgR.rows));
        cv::rectangle(bigImg, cv::Point(x0,y0+offset), cv::Point(x1,y1+offset), cv::Scalar(0,0,255), 1);

        int lIdx = gtsam::Symbol(landmarkKeys[l]).index();
        cv::putText(bigImg, std::to_string(lIdx), cv::Point(x0, y0-5+offset), 
            cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,0), 4);
        cv::putText(bigImg, std::to_string(lIdx), cv::Point(x0, y0-5+offset), 
            cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,255), 2);        
    }

    std::string fileName = std::string("ass/frame") + std::to_string(frame) + std::string(".jpg");
    cv::imwrite(fileName, bigImg);
}