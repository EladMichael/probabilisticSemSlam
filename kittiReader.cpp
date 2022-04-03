#include "kittiReader.h"
#include "constsUtils.h"

#include <sstream>
#include <sys/stat.h>
#include <fstream>
#include <cstdio>
#include <algorithm>
#include <Eigen/Dense>
#include <cmath>
#include <gtsam/inference/Symbol.h>

// Kitti directory structure is assumed to be the same as the distribution 
// originally, i.e.
// kittiDirectory - 
    // dataset - 
        // poses - 
        //     00.txt
        //     01.txt
        //     etc.
        // sequences - 
        //     00 - 
        //         calib.txt
        //         image_0 (left gray) - 
        //             000000.png
        //             000001.png
        //             etc
        //         image_1 (right gray) - 
        //
        //         image_2 (left color) - 
        // 
        //         image_3 (right color) -
    
    // devkit - 
// Path to kitti below should be the path to the directory ABOVE dataset, 
// i.e. "kittiDirectory" in the example above

const std::string pathToKitti("ENTER YOUR PATH HERE/");

// ###############################################################
kittiReader::kittiReader(int seqN, bool color){   
    this->seqN = seqN;
    this->color = color;

    if(pathToKitti == "ENTER YOUR PATH HERE/"){
        throw std::runtime_error("Fill in the path to the KITTI dataset in kittiReader.cpp!\n");
    }

    pathToData = pathToKitti + std::string("dataset/");
    pathToSeq = pathToData + std::string("sequences/");
    pathToPose = pathToData + std::string("poses/");

    pathToResults = pathToKitti + std::string("results");

    if(!file_exists(pathToResults)){
        std::cout<<"Making results directory: "<<pathToResults<<std::endl;
        mkdir(pathToResults.c_str(),0777);
    }

    char seqSpecifier[30];
    sprintf(seqSpecifier,"%02i/",seqN);
    this->seqSpec = std::string(seqSpecifier);
    pathToSeq += seqSpecifier;
    pathToResults += seqSpecifier;
    if(!file_exists(pathToResults)){
        std::cout<<"Making subresults directory for sequence: "<<seqSpecifier<<std::endl;
        mkdir(pathToResults.c_str(),0777);
    }
    sprintf(seqSpecifier,"%02i.txt",seqN);
    pathToPose += seqSpecifier;
    loadCalib();    
    f = matplot::figure(true);
    ax = f->current_axes();
}
// ###############################################################
void kittiReader::loadCalib(){

    if(!file_exists(pathToSeq + std::string("calib.txt"))){
        std::string errMsg("Kitti Calibration could not be found on path: ");
        errMsg += pathToSeq + std::string("calib.txt\n");
        throw std::runtime_error(errMsg);
    }

    std::ifstream myFile(pathToSeq + std::string("calib.txt"));
    std::string line;

    if(color){
        // first two lines are for gray
        std::getline(myFile,line);
        std::getline(myFile,line);
    }
    std::getline(myFile,line);
    std::stringstream ss(line);
    // each line begins with a label
    ss.ignore(4,' '); //ignore 4 chars, or until space
    double val;
    for(int count = 0; ss>>val ; count++){
        switch(count){
            case 0: fx = val; break;
            case 1: s = val; break;
            case 2: u0 = val; break;
            case 3: initL = val; break;
            case 5: fy = val; break;
            case 6: v0 = val; break;
            default: continue;
        }
    }
    std::getline(myFile,line);
    std::stringstream ss2(line);
    // each line begins with a label
    ss2.ignore(4,' '); //ignore 4 chars, or until space
    for(int count = 0; ss2>>val ; count++){
        if(count == 3){
            initR = val;
        }else if(count == 4){
            break;
        }
    }           
    initL = initL/fx;
    initR = initR/fx;
    b = std::abs(initR-initL);
    K = Eigen::Matrix<double,3,3>::Identity();
    K(0,0) = fx;
    K(1,1) = fy;
    K(0,1) = s;
    K(0,2) = u0;
    K(1,2) = v0;
    myFile.close();
}
// ###############################################################
gtsam::Pose3 kittiReader::groundTruth(int poseN){
    
    if(!file_exists(pathToPose)){
        std::string errMsg("Ground truth could not be found!\n Path to ground truth: \n");
        errMsg += pathToPose + std::string("\n");
        throw std::runtime_error(errMsg);
    }

    std::ifstream myFile(pathToPose);
    std::string line;
    int lineCounter = 0;
    while(lineCounter < poseN){
        lineCounter++;
        if(!std::getline(myFile,line)){
            std::string errMsg("Requested GT pose beyond end of file!\n");
            errMsg += std::string("File being read: ") + pathToPose + std::string("\n");
            errMsg += std::string("Lines read: ") + std::to_string(lineCounter); 
            errMsg += std::string(" Pose requested: ") + std::to_string(poseN); 
            errMsg += std::string("\n"); 
            throw std::runtime_error(errMsg);
        }
    }
    std::getline(myFile,line);
    std::stringstream ss(line);

    double gt[12];
    for(int i = 0; i < 12; i++){
        ss>>gt[i];
    }
    gtsam::Pose3 Pgt(gtsam::Rot3(gt[0],gt[1],gt[2],
                               gt[4],gt[5],gt[6],
                               gt[8],gt[9],gt[10]),
            gtsam::Point3(gt[3],gt[7],gt[11]));
    myFile.close(); 
    return Pgt;
}
// ###############################################################
void kittiReader::writePoses(const std::vector<gtsam::Pose3>& x, std::string ID){
    FILE *myFile;

    if(!file_exists(pathToResults)){
        std::string errMsg("Path to results directory could not be found!\nPath to directory: \n");
        errMsg += pathToResults;
        throw std::runtime_error(errMsg);
    }

    myFile = fopen((pathToResults + ID + std::string(".txt")).c_str(),"w");

    for(size_t p = 0; p < x.size(); p++){
        Eigen::Matrix<double,4,4> P = x[p].matrix();
        fprintf(myFile,"%f %f %f %f %f %f %f %f %f %f %f %f\n",P(0,0),P(0,1),P(0,2),
            P(0,3),P(1,0),P(1,1),P(1,2),P(1,3),P(2,0),P(2,1),P(2,2),P(2,3));
    }
    fclose(myFile);

}
// ###############################################################
void kittiReader::writeMetrics(const std::vector<double>& time4BBs,const std::vector<double>& time4Opt,
        const std::vector<double>& time4ProbWin,const std::vector<double>& slidingFPS,
        const std::vector<int>& numMeas,const std::vector<int>& numLand,std::string ID){


    FILE *myFile;

    if(!file_exists(pathToResults)){
        std::string errMsg("Path to results directory could not be found!\nPath to directory: \n");
        errMsg += pathToResults;
        throw std::runtime_error(errMsg);
    }

    myFile = fopen((pathToResults + ID + std::string("_metrics.txt")).c_str(),"w");

    for(size_t i = 0; i < time4BBs.size(); i++){
        fprintf(myFile,"%g %g %g %g %d %d\n",time4BBs[i],time4Opt[i],time4ProbWin[i],slidingFPS[i],numMeas[i],numLand[i]);
    }
    fclose(myFile);
}
// ###############################################################
void kittiReader::showMetric(const std::vector<double>& metric, const std::string& name){

    size_t numIter = metric.size(); //should all be the same size really...

    std::vector<int> iterations(numIter);
    for(size_t iter = 0; iter < numIter; iter++){
        iterations[iter] = iter;
    }

    matplot::plot(iterations, metric);
    matplot::xlabel("Iteration");
    matplot::ylabel(name);
    matplot::grid(matplot::on);

    matplot::show();
}
// ###############################################################
void kittiReader::updateTrajectory(const gtsam::Values& estimate){

    gtsam::KeyVector keys = estimate.keys();
    for(size_t k = 0; k < keys.size(); k++){
        gtsam::Symbol keyS(keys[k]);
        if( (keyS.chr() == 'x') && ((keyS.index() % 2) == 0) ) {
            //update the pose for this left camera index
            int idx = static_cast<int>(keyS.index()/2);
            gtsam::Vector3 t = estimate.at(keyS).cast<gtsam::Pose3>().translation();
            if(idx+1 > trajx.size()){
                //new pose
                int buffer = idx-trajx.size(); //this adds zeros until the next element is the right index
                for(int buf = 0; buf < buffer; buf++){
                    trajx.push_back(0);
                    trajz.push_back(0);
                }
                trajx.push_back(t(0));
                trajz.push_back(t(2));
            }else{
                //updating previously added pose
                trajx[idx] = t(0);
                trajz[idx] = t(2);
            }
        }
    }

    // ax->plot(gtx, gtz);
    // matplot::hold(matplot::on);
    ax->plot(trajx, trajz);
    // ax->xlim({-10, 100});
    // ax->ylim({-10, 100});

    // matplot::xlabel("X position [m]");
    // matplot::ylabel("Z position [m]");
    // matplot::grid(matplot::on);

    f->draw();
}
// ###############################################################
std::vector< gtsam::Pose3 > kittiReader::groundTruthSet()
{
    if(!file_exists(pathToPose)){
        std::string errMsg("Ground truth could not be found!\n Path to ground truth: \n");
        errMsg += pathToPose + std::string("\n");
        throw std::runtime_error(errMsg);
    }

    std::vector< gtsam::Pose3 > gts;
    // just a guess...
    gts.reserve(2000);
    gtx.reserve(2000);
    gtz.reserve(2000);

    std::ifstream myFile(pathToPose);
    std::string line;
    while(std::getline(myFile,line)){
        // this is a requested pose!
        std::stringstream ss(line);
        double gt[12];
        for(int i = 0; i < 12; i++){
            ss>>gt[i];
        }
        gtsam::Pose3 Pgt(gtsam::Rot3(gt[0],gt[1],gt[2],
                                     gt[4],gt[5],gt[6],
                                     gt[8],gt[9],gt[10]),
                       gtsam::Point3(gt[3],gt[7],gt[11]));
        gts.push_back(Pgt);  
        gtx.push_back(gt[3]);
        gtz.push_back(gt[11]);      
    }

    // if(!getAll){
    //     // if the loop exited, but there is still a pose request, then
    //     // it must have been beyond the end of the file!
    //     if(lineCount < posesN[counter]){
    //         std::string errMsg("Requested GT pose beyond end of file!\n");
    //         errMsg += std::string("File being read: ") + pathToPose + std::string("\n");
    //         errMsg += std::string("Lines read: ") + std::to_string(lineCount+1); 
    //         errMsg += std::string(" Pose requested: ") + std::to_string(posesN[counter]); 
    //         errMsg += std::string("\n"); 
    //         throw std::runtime_error(errMsg);
    //     }
    // }

    myFile.close(); 
    return gts;
}
