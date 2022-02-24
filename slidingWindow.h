#ifndef semSLAM_slidingWindow
#define semSLAM_slidingWindow

#include "dataframe.h"
#include "constsUtils.h"

#include <vector>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_quadrics/geometry/ConstrainedDualQuadric.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam_quadrics/geometry/BoundingBoxFactor.h>


#include <map>
#include <iostream>

class slidingWindow{
private:
    size_t optWin;
    size_t probWin;
    size_t k;
    bool usePerm;

public:

    slidingWindow();
    slidingWindow(size_t optWin,size_t probWin,size_t k,bool usePerm){
        this->optWin = optWin;
        this->probWin = probWin;
        this->k = k;
        this-> usePerm = usePerm;
        dFrames.reserve(2*optWin);
    }
    size_t getOptWin() const {return optWin;}
    size_t getProbWin() const {return probWin;}
    size_t getK() const {return k;}
    bool getPerm() const {return usePerm;}

    void setOptWin(size_t optWin){this->optWin = optWin;}
    void setProbWin(size_t probWin){this->probWin = probWin;}
    void setK(size_t k){this->k = k;}
    void setPerm(bool usePerm){this->usePerm = usePerm;}

    std::map< gtsam::Key, std::vector<gtsam_quadrics::BoundingBoxFactor> > queuedFactors;
    std::map< gtsam::Key, gtsam_quadrics::ConstrainedDualQuadric > queuedLandmarks;
    std::map< gtsam::Key, size_t> queuedFrame;

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values estimate;
    gtsam::Values trajectoryOld;
    gtsam::Values landmarksOld;
    gtsam::LevenbergMarquardtParams parameters;

    std::vector<dataframe> dFrames;

    void slide(size_t frameN,const semConsts& runConsts);
    void updateProb(size_t frameN,const gtsam::Cal3_S2::shared_ptr& K, int& qKeyFree, std::vector<gtsam::Key>& newL, const semConsts& runConsts);

};

#endif