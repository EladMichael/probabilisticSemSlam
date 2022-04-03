#include "constsUtils.h"
#include "bbNet.h"
#include "slidingWindow.h"
#include "kittiReader.h"
#include "assignment.h"
#include "dataframe.h"

#include <gtsam_quadrics/geometry/ConstrainedDualQuadric.h>
#include <gtsam_quadrics/geometry/DualConic.h>
#include <gtsam_quadrics/geometry/AlignedBox2.h>
#include <gtsam_quadrics/geometry/BoundingBoxFactor.h>
#include <gtsam_quadrics/geometry/QuadricCamera.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/CalibratedCamera.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>



#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <random>
#include <string>
#include <iostream>
#include <cmath>
#include <vector>
#include <time.h>
#include <thread>
#include <chrono>
#include <map>


int main(int argc, char** argv) {

    //check if cmdline arg for sequence was given
    //Run options but not really
    bool color = true;
    std::string calibDir("runOpts/");
    if(!file_exists(calibDir)){
        calibDir = "../"+calibDir; //if you're running within the build folder, it might need to look above
    }

    if(!file_exists(calibDir)){
        throw std::runtime_error("Can't find calibration files in path: "+calibDir);
    }
    // ################## RUN OPTION BLOCK ###################
    int seq = 0;
    std::string calibFile("calibSample.txt");
    size_t maxFrame = 0;
    bool verbose = false;
    bool verboseTiming = false;
    bool frameReport = false;

    if(argc>1){
        seq = atoi(argv[1]);
        if(argc>2){
            calibFile = argv[2];
        }
        if(argc>3){
            maxFrame = std::max(atoi(argv[3]),0);
        }
        if(argc>4){
            frameReport = (atoi(argv[4])>0);
        }
        if(argc>5){
            std::cout<<"Too many arguments"<<std::endl;
            return -1;
        }
    }else{
        std::cout<<"usage: ./semslamRun seq# calibFile maxFrame verbose\n";
        return 1;
    }

    const semConsts runConsts(calibDir+calibFile);

    char ID[80];
    sprintf(ID,"o%zd_p%zd_k%zd_perm%d_net%zu",runConsts.optWin,
        runConsts.probWin,runConsts.k,runConsts.usePerm,runConsts.netChoice);

    std::cout<<"Running on sequence: "<<seq<<" with verbose: "<<verbose<<" color: "<<color<<" and maxFrame: "<<maxFrame<<std::endl;
    std::cout<<"Run_ID: "<<ID<<std::endl;
    // ################## END RUN OPTION BLOCK ###################

    // load calibration, get access to ground truth, etc
    kittiReader kittiData(seq,color);
    
    bbNet imageNet(runConsts.netChoice);
    std::cout<<"Using image net: "<<imageNet.get_name()<<std::endl;

    // get vector of ground truth poses
    std::vector< gtsam::Pose3 > gts = kittiData.groundTruthSet();

    if(maxFrame == 0){
        maxFrame = gts.size();
    }

    // turn the poses into between factors (odometry)
    std::vector<gtsam::Pose3> gtOdom;
    std::vector<gtsam::Pose3> gtTraj;
    gtOdom.reserve(gts.size());

    //no odometry for frame 0 (initial frame)
    gtOdom.push_back(gtsam::Pose3()); 
    for(size_t frame = 0; frame < maxFrame; frame++){

        if(frame < maxFrame-1){
            // gtOdom.push_back((gts[frame].between(gts[frame+1])).compose(delta));
            gtOdom.push_back((gts[frame].between(gts[frame+1])));
        }
        gtTraj.push_back(gts[frame]);
    }

    // Define the camera calibration parameters
    gtsam::Cal3_S2::shared_ptr K(new gtsam::Cal3_S2(kittiData.fx, kittiData.fy, 
        kittiData.s, kittiData.u0, kittiData.v0));

    // this is for transforming the pose of the left camera to the right camera
    gtsam::Pose3 rightNudge = gtsam::Pose3(gtsam::Rot3(),gtsam::Point3(kittiData.b,0,0));
    // the relative pose of the right camera from the left should be VERY tightly coupled

    // ###################### POSE GRAPH BLOCK #############################
    slidingWindow window(runConsts.optWin,runConsts.probWin,runConsts.k,runConsts.usePerm);
    window.parameters.orderingType = gtsam::Ordering::METIS;
    window.parameters.setLinearSolverType("SEQUENTIAL_QR");
    // window.parameters.linearSolverType = gtsam::MULTIFRONTAL_QR;
    // window.parameters.setVerbosityLM("SILENT"); // SILENT = 0, SUMMARY, TERMINATION, LAMBDA, TRYLAMBDA, TRYCONFIG, DAMPED, TRYDELTA
    window.parameters.setMaxIterations(100); 
    window.parameters.setlambdaUpperBound(1e10); ///< defaults to 1e5
    // ###################### END POSE GRAPH BLOCK #############################

    // ###################### Metrics Block #############################
    std::vector<double> time4BBs(maxFrame,0);
    std::vector<double> time4Opt(maxFrame,0);
    std::vector<double> time4ProbWin(maxFrame,0);
    std::vector<double> slidingFPS(maxFrame,0);
    std::vector<int> numMeas(maxFrame,0);
    std::vector<int> numLand(maxFrame,0);
    // ###################### End Metrics Block #############################
    
    // ###################### File Management Block #############################
    if(!file_exists("generatedData")){
        std::cout<<"Making generatedData directory.\n";
        mkdir("generatedData",0777);
    }
    if(!file_exists("generatedData/"+kittiData.seqSpec)){
        std::cout<<"Making generatedData/seqSpecifier directory.\n";
        mkdir(("generatedData/"+kittiData.seqSpec).c_str(),0777);
    }
    if(!file_exists("generatedData/"+std::to_string(kittiData.seqN)+"/costMatrices")){
        std::cout<<"Making generatedData/seqSpecifier/costMatrices directory.\n";
        mkdir(("generatedData/"+std::to_string(kittiData.seqN)+"/costMatrices").c_str(),0777);
    }
    // ###################### End File Management Block #############################

    // this is for computing the fps, averaged over the previous 10 frames
    // the first element is set to 1, just to avoid dividing by zero
    std::vector<double> frameS(10,0); 
    frameS[0] = 1;

    auto totalTime = tic();

    //smallest int key which hasn't been used
    int qKeyFree = 0; 

    for(size_t frame = 0; frame < maxFrame; frame++){

        auto frameTime = tic();

        // symbol for this iteration
        gtsam::Key curKeyL = gtsam::Symbol('x',2*frame);
        gtsam::Key curKeyR = gtsam::Symbol('x',2*frame+1);

        //Left camera pose
        gtsam::Pose3 curPoseL = gtTraj[frame];
        //Right camera pose
        gtsam::Pose3 curPoseR = curPoseL.compose(rightNudge);
        
        // Insert the initial values for each pose
        window.estimate.insert(curKeyL,curPoseL);
        window.estimate.insert(curKeyR,curPoseR);

        // set stereo factor (hackish...)
        gtsam::BetweenFactor<gtsam::Pose3> curLR(curKeyL, curKeyR, rightNudge, runConsts.stereoNoise);
        window.graph.add(curLR);

        // Get bounding box measurements and quadric estimates using odometry estimate
        dataframe curFrame(kittiData.pathToSeq,kittiData.b,frame,color);

        // Compute the bounding boxes for the images
        curFrame.computeBoundingBoxes(imageNet,runConsts);  

        // the current landmarks in the map
        std::vector< gtsam_quadrics::ConstrainedDualQuadric > landmarks;
        std::vector< gtsam::Key > landmarkKeys;
        std::vector< gtsam::Key > newLandmarks; //we will use this to add priors to new landmarks!
        landmarkKeys.reserve(qKeyFree);
        landmarks.reserve(qKeyFree);

        if(frame == 0){
            // Add a prior on initial poses at origin
            window.graph.emplace_shared<gtsam::NonlinearEquality<gtsam::Pose3> >(curKeyL,curPoseL);

        }else{

            // get the previous pose to set the odometry factor
            gtsam::Key prevKeyL = gtsam::Symbol('x',2*(frame-1));

            // previous pose
            gtsam::Pose3 prevPoseL = window.estimate.at(prevKeyL).cast<gtsam::Pose3>();

            // set odometry factor
            gtsam::BetweenFactor<gtsam::Pose3> bf(prevKeyL, curKeyL, 
                                              gtOdom[frame], runConsts.odomNoise);
            window.graph.add(bf);


            //Also while here, extract the current landmarks
            for(int q = 0; q < qKeyFree; q++){
                // key to landmark
                gtsam::Key qKey = gtsam::Symbol('q',q);
                // extract and cast landmark state
                if(window.estimate.exists(qKey)){
                    // the landmark is in the ``current'' view/map
                    gtsam_quadrics::ConstrainedDualQuadric quad = window.estimate.at(qKey).cast<gtsam_quadrics::ConstrainedDualQuadric>();
                    landmarks.push_back(quad);
                    landmarkKeys.push_back(qKey);
                }else if(window.landmarksOld.exists(qKey)){
                    // landmark is on the old category (but could be used for loop closure)
                    gtsam_quadrics::ConstrainedDualQuadric quad = window.landmarksOld.at(qKey).cast<gtsam_quadrics::ConstrainedDualQuadric>();
                    landmarks.push_back(quad);
                    landmarkKeys.push_back(qKey);
                }else if(window.queuedLandmarks.find(qKey) != window.queuedLandmarks.end()){
                    // landmark is queued (has not been observed twice yet)
                    if(frame - window.queuedFrame[qKey] > runConsts.landmark_age_thresh){
                        // Landmark is being aged out, removed from everything
                        window.queuedFrame.erase(window.queuedFrame.find(qKey));
                        window.queuedLandmarks.erase(window.queuedLandmarks.find(qKey));
                        window.queuedFactors.erase(window.queuedFactors.find(qKey));
                    }else{
                        // Landmark has not aged out yet
                        landmarks.push_back(window.queuedLandmarks[qKey]);
                        landmarkKeys.push_back(qKey);
                    }
                }
                // if the key does not correspond to a landmark, it was not initialised on purpose
                // (likely never observed more than once.)
            }
        }

        // Build the quadrics based on the measurements, for assignment
        curFrame.estQuadrics(curPoseL,K);

        // estimate the assignment probability, based on the current catalog of landmarks
        // and the estimated quadrics from the measurements 
        std::vector< std::vector<double> > assignmentProbs = getAssignmentProbs(curFrame.quadEst,landmarks,runConsts);

        // save the assignment problem! This is for drawing, and for comparing the assignment methods
        std::string savePath("generatedData/"+std::to_string(kittiData.seqN)+"/costMatrices/"+ID+"_frame"+std::to_string(frame)+".dat");
        saveAssignmentProb(curFrame.quadEst,landmarks,runConsts,savePath);

        //if we have estimated quadrics, insert left and right bounding boxes
        size_t nL = landmarks.size();
        numMeas[frame] = curFrame.bbL.size();
        numLand[frame] = nL;

        for(size_t m = 0; m < curFrame.quadEst.size(); m++){
            // for each quadric estimate
            for(size_t q = 0; q < nL; q++){
                // check if it assigns significantly to this landmark
                if(assignmentProbs[m][q] < runConsts.NEW_FACTOR_PROB_THRESH){continue;}
                // if it does, assign it, with the scaled covariance
                gtsam::Key qKey = landmarkKeys[q];

                // add bounding box factors from stereo cameras 
                gtsam_quadrics::BoundingBoxFactor bbfL(curFrame.bbL[m].aBox,K,curKeyL,
                    qKey,runConsts.bNoise(assignmentProbs[m][q]));
                
                gtsam_quadrics::BoundingBoxFactor bbfR(curFrame.bbR[m].aBox,K,curKeyR,
                    qKey,runConsts.bNoise(assignmentProbs[m][q]));
                
                window.graph.add(bbfL);    
                window.graph.add(bbfR);

                //landmark was in the queue, now that it is reobserved, put it in!
                if(window.queuedLandmarks.find(qKey) != window.queuedLandmarks.end()){
                    window.estimate.insert(qKey,window.queuedLandmarks[qKey]);

                    //also insert queued factors
                    window.graph.add(window.queuedFactors[qKey][0]);
                    window.graph.add(window.queuedFactors[qKey][1]);

                    //if this was a queued landmark, that means it is new! We put a prior on landmarks, to prevent
                    newLandmarks.push_back(qKey);

                    //remove from queue
                    window.queuedFrame.erase(window.queuedFrame.find(qKey));
                    window.queuedLandmarks.erase(window.queuedLandmarks.find(qKey));
                    window.queuedFactors.erase(window.queuedFactors.find(qKey));
                }


                if(!window.estimate.exists(qKey)){
                    if(window.landmarksOld.exists(qKey)){
                        window.estimate.insert(qKey,window.landmarksOld.at(qKey));
                        newLandmarks.push_back(qKey); //gotta re-add the prior
                    }else{
                        throw std::runtime_error("Can't find this landmark, but it claims to exist");
                    }
                }  

            }

            // if the quadric was significantly assigned 
            // to the dummy ``non-assigned'' landmark, then it 
            // becomes a new quadric (if it passes the NEW_LANDMARK_THRESH)
            if(assignmentProbs[m][nL] > runConsts.NEW_LANDMARK_THRESH){
                gtsam::Key qKey = gtsam::Symbol('q',qKeyFree);           
                
                window.queuedLandmarks[qKey] = curFrame.quadEst[m];
                // window.estimate.insert(qKey,curFrame.quadEst[m]);     

                // add bounding box factors from stereo cameras 
                gtsam_quadrics::BoundingBoxFactor bbfL(curFrame.bbL[m].aBox,K,curKeyL,
                    qKey,runConsts.bNoise(assignmentProbs[m][nL]));

                gtsam_quadrics::BoundingBoxFactor bbfR(curFrame.bbR[m].aBox,K,curKeyR,
                    qKey,runConsts.bNoise(assignmentProbs[m][nL]));

                window.queuedFactors[qKey] = std::vector<gtsam_quadrics::BoundingBoxFactor>{bbfL,bbfR};
                window.queuedFrame[qKey] = frame;
                qKeyFree++;
            }
        }


        gtsam::LevenbergMarquardtOptimizer optimizer(window.graph, window.estimate, window.parameters);

        // optimise the graph
        window.estimate = optimizer.optimize();

        //add current frame to dataframe vector
        window.dFrames.push_back(curFrame);

        newLandmarks.clear();

        //using new estimate, re-compute assignment probabilities for all poses within the probWin window
        //and update the bounding box factors in the graph.
        auto tWin = tic();
        window.slide(frame,runConsts);
        window.updateProb(frame,K,qKeyFree,newLandmarks,runConsts);
        time4ProbWin[frame] = toc(tWin);

        double fps = std::reduce(frameS.begin(),frameS.end())/frameS.size();
        std::ostringstream fps_label;
        fps_label << std::fixed << std::setprecision(2);
        fps_label << "Frame: " << frame << " / " << maxFrame << "    ";
        fps_label << "FPS: " << 1.0/fps;

        curFrame.saveBoxes(fps_label.str(),frame,K,landmarks,landmarkKeys,curPoseL);

        std::cout<<"\r"<<fps_label.str();
        kittiData.updateTrajectory(window.estimate);
        
        frameS[frame%10] = toc(frameTime);
        if(frame < 10){
            slidingFPS[frame] = 1.0/frameS[frame];
        }else{
            slidingFPS[frame] = 1.0/fps;
        }
    }

    gtsam::LevenbergMarquardtOptimizer optimizer(window.graph, window.estimate, window.parameters);
    window.estimate = optimizer.optimize();
    std::vector<gtsam::Pose3> posesCalculated(maxFrame);
    double sTime = toc(totalTime);
    std::cout<<"\n===============================================================\n";
    std::cout<<"Finished processing "<<maxFrame<<" frames with ID: "<<ID<<std::endl;
    std::cout<<"using network: "<<imageNet.get_name()<<" total time: "<<sTime<<std::endl;
    std::cout<<"average FPS: "<<maxFrame/sTime<<std::endl;
    std::cout<<"===============================================================\n";
    for(size_t f = 0; f < maxFrame; f++){
        gtsam::Key fKey = gtsam::Symbol('x',2*f);
        if(window.trajectoryOld.exists(fKey)){
            posesCalculated[f] = window.trajectoryOld.at(fKey).cast<gtsam::Pose3>();
        }else{
            posesCalculated[f] = window.estimate.at(fKey).cast<gtsam::Pose3>();
        }
    }
    kittiData.showMetric(slidingFPS,std::string("Average FPS"));
    //was this a full run?
    if(maxFrame == gts.size()){
        kittiData.writePoses(posesCalculated,ID);
        // ###################### Metrics Block #############################
        // Write metrics
        kittiData.writeMetrics(time4BBs,time4Opt,time4ProbWin,slidingFPS,numMeas,numLand,ID);
        // ###################### End Metrics Block #############################
    }
}
