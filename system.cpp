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
    std::string calibDir("/home/emextern/Desktop/codeStorage/semSLAM/calib/");
    // ################## RUN OPTION BLOCK ###################
    int seq = 0;
    std::string calibFile("default.txt");
    size_t maxFrame = 0;
    bool verbose = false;
    bool verboseTiming = false;
    bool frameReport = false;
    int rndSeed = time(0);

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
            rndSeed = atoi(argv[5]);
        }
        if(argc>6){
            std::cout<<"Too many arguments"<<std::endl;
            return -1;
        }
    }else{
        std::cout<<"usage: ./semslamRun seq# calibFile maxFrame verbose rndSeed\n";
        return 1;
    }

    const semConsts runConsts(calibDir+calibFile);

    char ID[80];
    sprintf(ID,"o%zd_p%zd_k%zd_perm%d_net%zu_seed%d",runConsts.optWin,
        runConsts.probWin,runConsts.k,runConsts.usePerm,runConsts.netChoice,rndSeed);

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

    std::default_random_engine generator(rndSeed); 
    std::normal_distribution<double> odom_rot_dist(0.0, runConsts.ODOM_ROT_SD_GEN);  
    std::normal_distribution<double> odom_t_dist(0.0, runConsts.ODOM_T_SD_GEN);  
    // boost::shared_ptr<gtsam::noiseModel::Diagonal> odomNoiseModel = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Ones()*ODOM_SD);

    
    // boost::shared_ptr<gtsam::noiseModel::Diagonal> boxNoiseModel = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector4::Ones()*BOX_SD);

    // turn the poses into between factors (odometry)
    // add noise to ground truth for noisy odometry
    std::vector<gtsam::Pose3> noisyOdometry;
    std::vector<gtsam::Pose3> noisyTrajectory;
    noisyOdometry.reserve(gts.size());
    //no odometry for frame 0 (initial frame)
    noisyOdometry.push_back(gtsam::Pose3()); 
    for(size_t frame = 0; frame < maxFrame; frame++){
        // std::vector<double> noiseVector(6); 
        // for(size_t i = 0; i < noiseVector.size(); i++){
        //     if(i < 3){
        //         noiseVector[i] = odom_rot_dist(generator);
        //     }else{
        //         noiseVector[i] = odom_t_dist(generator);
        //     }
        // }
        // gtsam::Pose3 delta = gtsam::Pose3::Retract(gtsam::Vector6(noiseVector.data()));
        if(frame < maxFrame-1){
            // noisyOdometry.push_back((gts[frame].between(gts[frame+1])).compose(delta));
            noisyOdometry.push_back((gts[frame].between(gts[frame+1])));
        }
        if(frame > 10){
            // noisyTrajectory.push_back(gts[frame].compose(delta));
            noisyTrajectory.push_back(gts[frame]);
        }else{
            noisyTrajectory.push_back(gts[frame]);
        }
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


    std::vector<double> frameS(10,0);
    frameS[0] = 1;
    auto totalTime = tic();
    int qKeyFree = 0; //smallest int key which hasn't been used
    // size_t msgSize = 0;
    for(size_t frame = 0; frame < maxFrame; frame++){
        // std::cin.get();
        auto frameTime = tic();
        // std::string frameMsg;
        // if(frameMsgVerb){
        //     frameMsg += "Frame:  "+std::to_string(frame);
        // }
        if(verbose || verboseTiming || frameReport){
            std::cout<<"\n ################################ Frame "<<frame<<" ########################################## \n\n";
        }
        // symbol for this iteration
        gtsam::Key curKeyL = gtsam::Symbol('x',2*frame);
        gtsam::Key curKeyR = gtsam::Symbol('x',2*frame+1);
        gtsam::Pose3 curPoseL = noisyTrajectory[frame];
        gtsam::Pose3 curPoseR = curPoseL.compose(rightNudge);
        
        window.estimate.insert(curKeyL,curPoseL);
        window.estimate.insert(curKeyR,curPoseR);

        // set stereo factor (hackish...)
        gtsam::BetweenFactor<gtsam::Pose3> curLR(curKeyL, curKeyR, rightNudge, runConsts.stereoNoise);
        window.graph.add(curLR);

        // Get bounding box measurements and quadric estimates using odometry estimate
        dataframe curFrame(kittiData.pathToSeq,kittiData.b,frame,color);
        // curFrame.showImg();
        auto tBB = tic();
        curFrame.computeBoundingBoxes(imageNet,runConsts);  
        time4BBs[frame] = toc(tBB);

        // std::cout<<"Generating calibration objects as measurements "<<curFrame.bbL.size()<<" and "<<curFrame.bbL.size()+1<<std::endl;
        // curFrame.calibObjs(gts[frame],K);

        if(verboseTiming){
            std::cout<<"Time to compute "<<curFrame.bbL.size()<<" bounding boxes: "<<1000*time4BBs[frame]<<"ms\n";
        }
        // if(frameMsg){
        //     frameMsg += " time for "+std::to_string(curFrame.bbL.size())+" bbs: " + std::to_string(1000*t2) + "ms";
        // }


        if(verbose){
            std::cout<<"Current frame key: ";
            gtsam::PrintKey(curKeyL);
            std::cout<<"\n found "<<curFrame.bbL.size()<<" stereo bounding boxes.\n";
        }

        // gonna need these later, the current landmarks in the map
        std::vector< gtsam_quadrics::ConstrainedDualQuadric > landmarks;
        std::vector< gtsam::Key > landmarkKeys;
        std::vector< gtsam::Key > newLandmarks; //we will use this to add priors to new landmarks!
        landmarkKeys.reserve(qKeyFree);
        landmarks.reserve(qKeyFree);

        if(frame == 0){
            // Add a prior on initial poses at origin
            window.graph.emplace_shared<gtsam::NonlinearEquality<gtsam::Pose3> >(curKeyL,curPoseL);
            // window.graph.addPrior(curKeyL, curPoseL, runConsts.stereoNoiseModel);            
        }else{
            // window.graph.addPrior(curKeyL, curPoseL, runConsts.stereoNoise);
            // window.graph.addPrior(curKeyL, curPoseL, runConsts.stereoNoiseModel);
            gtsam::Key prevKeyL = gtsam::Symbol('x',2*(frame-1));
            if(verbose){
                std::cout<<"Previous frame key ";
                gtsam::PrintKey(prevKeyL); std::cout<<std::endl;
            }

            // previous pose
            gtsam::Pose3 prevPoseL = window.estimate.at(prevKeyL).cast<gtsam::Pose3>();

            // // best estimate of current location
            // curPoseL = prevPoseL.compose(noisyOdometry[frame]);
            // curPoseR = curPoseL.compose(rightNudge);
            // // insert estimate of current pose as initial value
            // window.estimate.insert(curKeyL,curPoseL);
            // window.estimate.insert(curKeyR,curPoseR);

            // set odometry factor
            gtsam::BetweenFactor<gtsam::Pose3> bf(prevKeyL, curKeyL, 
                                              noisyOdometry[frame], runConsts.odomNoise);
            window.graph.add(bf);


            // if(verbose){
            //     std::cout<<"Previous pose: "<<prevPose<<std::endl;
            //     std::cout<<"Current pose: \n"<<curPose<<std::endl;
            // }

            //Also while here, extract the current landmarks
            for(int q = 0; q < qKeyFree; q++){
                // key to landmark
                gtsam::Key qKey = gtsam::Symbol('q',q);
                // extract and cast landmark state
                if(window.estimate.exists(qKey)){
                    gtsam_quadrics::ConstrainedDualQuadric quad = window.estimate.at(qKey).cast<gtsam_quadrics::ConstrainedDualQuadric>();
                    landmarks.push_back(quad);
                    landmarkKeys.push_back(qKey);
                }else if(window.landmarksOld.exists(qKey)){
                    gtsam_quadrics::ConstrainedDualQuadric quad = window.landmarksOld.at(qKey).cast<gtsam_quadrics::ConstrainedDualQuadric>();
                    landmarks.push_back(quad);
                    landmarkKeys.push_back(qKey);
                }else if(window.queuedLandmarks.find(qKey) != window.queuedLandmarks.end()){
                    if(frame - window.queuedFrame[qKey] > runConsts.landmark_age_thresh){
                        if(verbose){
                            std::cout<<"Landmark "<<q<<" is being removed from the queue, origin frame "<<window.queuedFrame[qKey]<<std::endl;
                        }
                        window.queuedFrame.erase(window.queuedFrame.find(qKey));
                        window.queuedLandmarks.erase(window.queuedLandmarks.find(qKey));
                        window.queuedFactors.erase(window.queuedFactors.find(qKey));
                        if(verbose){
                            bool success = window.queuedFrame.find(qKey)==window.queuedFrame.end();
                            success = success && window.queuedLandmarks.find(qKey)==window.queuedLandmarks.end();
                            success = success && window.queuedFactors.find(qKey)==window.queuedFactors.end();
                            std::cout<<"Landmark removed: "<<success;
                        }
                    }else{
                        landmarks.push_back(window.queuedLandmarks[qKey]);
                        landmarkKeys.push_back(qKey);
                    }
                }
                // if the key does not correspond to a landmark, it was not initialised on purpose
                // (likely never observed more than once.)
            }
        }

        curFrame.estQuadrics(curPoseL,K);
        std::vector< std::vector<double> > assignmentProbs = getAssignmentProbs(curFrame.quadEst,landmarks,runConsts);

        // if(runConsts.usePerm){
        //     assignmentProbs = permanentProb(curFrame.quadEst,landmarks,runConsts);
        // }else{
        //     assignmentProbs = asgnProbQuad(curFrame.quadEst,landmarks,runConsts);
        // }
        saveAssignmentProb(curFrame.quadEst,landmarks,runConsts,ID,frame);

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

                    if(frameReport){
                        std::cout<<"Measurements "<<m<<" from frame "<<frame<<" are being attributed to landmark "<<gtsam::Symbol(qKey).index()<<std::endl;
                    }
                    //also insert queued factors
                    window.graph.add(window.queuedFactors[qKey][0]);
                    window.graph.add(window.queuedFactors[qKey][1]);

                    //if this was a queued landmark, that means it is new! We put a prior on landmarks, to prevent
                    newLandmarks.push_back(qKey);

                    //remove from queue
                    window.queuedFrame.erase(window.queuedFrame.find(qKey));
                    window.queuedLandmarks.erase(window.queuedLandmarks.find(qKey));
                    window.queuedFactors.erase(window.queuedFactors.find(qKey));
                    if(verbose){
                        bool success = window.queuedFrame.find(qKey)==window.queuedFrame.end();
                        success = success && window.queuedLandmarks.find(qKey)==window.queuedLandmarks.end();
                        success = success && window.queuedFactors.find(qKey)==window.queuedFactors.end();
                        std::cout<<"Landmark removed: "<<success;
                    }
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
                
                if(frameReport){
                    std::cout<<"Measurements "<<m<<" from frame "<<frame<<" are generating a new landmark with index "<<gtsam::Symbol(qKey).index()<<std::endl;
                }

                if(verbose){
                    std::cout<<"quadric "<<qKeyFree<<" initial estimate: \n";
                    std::cout<<"Pose: \n"<<curFrame.quadEst[m].pose()<<"\nradii: ";
                    std::cout<<curFrame.quadEst[m].radii().transpose()<<std::endl;
                    std::cout<<"Matrix: \n"<<curFrame.quadEst[m].matrix()<<std::endl;
                }                

                // add bounding box factors from stereo cameras 
                gtsam_quadrics::BoundingBoxFactor bbfL(curFrame.bbL[m].aBox,K,curKeyL,
                    qKey,runConsts.bNoise(assignmentProbs[m][nL]));

                gtsam_quadrics::BoundingBoxFactor bbfR(curFrame.bbR[m].aBox,K,curKeyR,
                    qKey,runConsts.bNoise(assignmentProbs[m][nL]));

                // window.graph.add(bbfL);    
                // window.graph.add(bbfR);    
                window.queuedFactors[qKey] = std::vector<gtsam_quadrics::BoundingBoxFactor>{bbfL,bbfR};
                window.queuedFrame[qKey] = frame;
                qKeyFree++;
            }
        }
        // Update with the new factors
        // if(verbose){
        //     std::cout<<"#################################################################\n";
        //     std::cout<<"#################################################################\n";
            // std::cout<<"graph: before update: \n";
            // window.graph.print("Graph Before Update");

        //     std::cout<<"Initial estimate before update: \n";
            // window.estimate.print("Estimate Before Update");
            
        //     std::cout<<"#################################################################\n";
        //     std::cout<<"#################################################################\n";
        // }

        // if((frame+1)%updateSkip == 0){

        // }
        // isam.update(graph, initialEstimate);
        // build optimiser


        auto tOpt = tic();
        gtsam::LevenbergMarquardtOptimizer optimizer(window.graph, window.estimate, window.parameters);
        // gtsam::DoglegOptimizer optimizer(window.graph, window.estimate);

        // optimise the graph
        window.estimate = optimizer.optimize();

        time4Opt[frame] = toc(tOpt);
        if(verboseTiming){
            std::cout<<"Time to optimize: "<<1000*time4Opt[frame]<<"ms\n";
        }

        //add current frame to dataframe vector
        window.dFrames.push_back(curFrame);

        // for(size_t q = 0; q < newLandmarks.size(); q++){
        //     window.graph.addPrior(newLandmarks[q], window.estimate.at(newLandmarks[q]).cast<gtsam_quadrics::ConstrainedDualQuadric>(), runConsts.landmarkPriorNoise);  
        // }
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
        // std::string fps_label_str = fps_label.str();

        // #####################################################################################################
        // ############################################## Frame Report #########################################
        // #####################################################################################################        
        if(frameReport){

            std::cout<<"Measured "<<curFrame.bbL.size()<<" quadrics, currently "<<landmarks.size()<<" landmarks\n";
            std::cout<<" Current estimated 3D position: "<<curPoseL.translation().transpose()<<std::endl;
            for(size_t q = 0; q < curFrame.quadEst.size(); q++){
                std::cout<<"Box pair "<<q<<" constructed a quadric with center: "<<curFrame.quadEst[q].centroid().transpose()<<std::endl;
                std::cout<<"                                   and radii: "<<curFrame.quadEst[q].radii().transpose()<<std::endl;
                std::cout<<"    Landmark correlations (qIdx, prob): ";
                std::string assoc;
                for(size_t l = 0; l < landmarks.size(); l++){
                    if(assignmentProbs[q][l] > runConsts.NEW_FACTOR_PROB_THRESH){
                        gtsam::Symbol qSym = gtsam::Symbol(landmarkKeys[l]);
                        char probAssoc[20];
                        sprintf(probAssoc,"(%zu,%.2g), ",qSym.index(),assignmentProbs[q][l]);
                        assoc += std::string(probAssoc);
                    }
                }
                std::cout<<assoc<<std::endl;
            }
            std::cout<<"---- Landmarks -----\n";

            landmarks.clear();
            landmarkKeys.clear();
            for(int q = 0; q < qKeyFree; q++){
                // key to landmark
                gtsam::Key qKey = gtsam::Symbol('q',q);
                // extract and cast landmark state
                if(window.estimate.exists(qKey)){
                    gtsam_quadrics::ConstrainedDualQuadric quad = window.estimate.at(qKey).cast<gtsam_quadrics::ConstrainedDualQuadric>();
                    landmarks.push_back(quad);
                    landmarkKeys.push_back(qKey);
                }else if(window.landmarksOld.exists(qKey)){
                    gtsam_quadrics::ConstrainedDualQuadric quad = window.landmarksOld.at(qKey).cast<gtsam_quadrics::ConstrainedDualQuadric>();
                    landmarks.push_back(quad);
                    landmarkKeys.push_back(qKey);
                }else if(window.queuedLandmarks.find(qKey) != window.queuedLandmarks.end()){
                    landmarks.push_back(window.queuedLandmarks[qKey]);
                    landmarkKeys.push_back(qKey);
                }
            }

            for(size_t l = 0; l < landmarks.size(); l++){
                gtsam::Symbol qSym = gtsam::Symbol(landmarkKeys[l]);
                if(window.queuedLandmarks.find(landmarkKeys[l]) != window.queuedLandmarks.end()){
                    std::cout<<"QUEUED ";
                }
                std::cout<<"Landmark "<<qSym.index()<<" centroid: "<<landmarks[l].centroid().transpose()<<std::endl;
                std::cout<<"    with radii: "<<landmarks[l].radii().transpose()<<std::endl;
            }

            bool checkReprojection = true;
            if(!checkReprojection){
                curFrame.saveBoxes(fps_label.str(),frame,K);
            }else{
                curFrame.saveBoxes(fps_label.str(),frame,K,landmarks,landmarkKeys,curPoseL);
            }

        }
        // #####################################################################################################
        // ########################################## End Frame Report #########################################
        // ##################################################################################################### 
        curFrame.saveBoxes(fps_label.str(),frame,K,landmarks,landmarkKeys,curPoseL);

        if(false){
            curFrame.showBoxes(fps_label.str());
            kittiData.updateTrajectory(window.estimate);
        }else{
            std::cout<<"\r"<<fps_label.str();
            kittiData.updateTrajectory(window.estimate);
        }
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

    // Create an iSAM2 object. Unlike iSAM1, which performs periodic batch steps
    // to maintain proper linearization and efficient variable ordering, iSAM2
    // performs partial relinearization/reordering at each step. A parameter
    // structure is available that allows the user to set various properties, such
    // as the relinearization threshold and type of linear solver. For this
    // example, we we set the relinearization threshold small so the iSAM2 result
    // will approach the batch result.
    // gtsam::ISAM2Params params;
    // params.optimizationParams = ISAM2DoglegParams();
    // params.relinearizeThreshold = 0.01;
    // params.relinearizeSkip = 1;
    // // // parameters.print("");
    // gtsam::ISAM2 isam(params);
    // Create a Factor Graph and Values to hold the new data
    // gtsam::NonlinearFactorGraph graph;
    // gtsam::Values initialEstimate;
    //     isam.update(graph,initialEstimate);
    //     graph.resize(0);
    //     initialEstimate.clear();
    // Each call to iSAM2 update(*) performs one iteration of the iterative
    // nonlinear solver. If accuracy is desired at the expense of time,
    // update(*) can be called additional times to perform multiple optimizer
    // iterations every step.
    // isam.update();