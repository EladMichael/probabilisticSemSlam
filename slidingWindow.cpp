#include "slidingWindow.h"
#include "assignment.h"

#include <gtsam/slam/PriorFactor.h>

#include <cmath>
#include <unordered_set>


//####################################################################
void slidingWindow::slide(size_t frameN, const semConsts& runConsts){
/*
    Keep all landmarks observed in poses within the optimisation window, and
    all poses which observe those landmarks (regardless of position within the
    window or out), as well as all poses within the window. The oldest pose 
    requires a prior as well, to anchor the optimisation. So the optimisation
    window dictates the poses which, when they observe an object, make that 
    object special, and anybody who observes it gets to stay. 
*/

    bool verbose = false;

    if(frameN+1 < runConsts.optWin){return;}
    size_t inWindow = frameN+1-runConsts.optWin;
    //iterate through the graph, find which quadrics have been observed 
    // within the window
    std::unordered_set<int> observedQuadrics;
    for(auto iter = graph.begin(); iter!=graph.end();iter++){
        if((*iter)->keys().size() == 2){
            // this is either a bounding box factor,
            // or an odometry factor. Check the second key;
            gtsam::Symbol k0((*iter)->keys()[0]);
            size_t keyFrame = static_cast<size_t>(floor(k0.index()/2.0));
            gtsam::Symbol k1((*iter)->keys()[1]);
            if((k1.chr() == 'q') && (keyFrame >= inWindow)){
                //bounding box factor, from a pose within the window
                observedQuadrics.insert(k1.index());
                if(verbose){
                    std::cout<<"Quadric: "<<k1.index()<<" was observed in frame "<<keyFrame<<std::endl;
                }
            }
        }
    }

    // archive unobserved Quadrics
    gtsam::KeyVector keys = estimate.keys();
    for(size_t k = 0; k < keys.size(); k++){
        gtsam::Symbol keyS(keys[k]);
        if( (keyS.chr() == 'q') && (observedQuadrics.find(keyS.index()) == observedQuadrics.end()) ) {
            if(verbose){
                std::cout<<"Quadric: "<<keyS.index()<<" is being archived, it was not observed in window\n";
            }
            //this is the key to a quadric estimate which isn't observed by any poses in the runConsts.optWin
            if(landmarksOld.exists(keyS)){
                landmarksOld.update(keyS,estimate.at(keyS));
            }else{
                landmarksOld.insert(keyS,estimate.at(keyS));
            }
            estimate.erase(keyS);
        }
    }

    //Remove bounding box factors and priors for quadrics NOT observed find which quadrics have been observed 
    // within the window, and find oldest pose which observes any chosen quadrics
    for(auto iter = graph.begin(); iter!=graph.end();){
        gtsam::Symbol k0((*iter)->keys()[0]);
        if((*iter)->keys().size() == 1){
            if( ( k0.chr() == 'x' ) || ( observedQuadrics.find(k0.index()) != observedQuadrics.end() )  ){
                iter++;
            }else{
                //if we're here then k0.chr() == 'q' (the only other option), AND it's not in observed quadrics, so it should be archived.
                iter = graph.erase(iter);
            }
        }else{
            // this is either a bounding box factor,
            // or an odometry factor. Check the second key;
            gtsam::Symbol k1((*iter)->keys()[1]);
            if(k1.chr() == 'q'){
                //bounding box factor!
                if((observedQuadrics.find(k1.index()) == observedQuadrics.end())){
                    //not an observed quadric
                    iter = graph.erase(iter);
                    if(verbose){
                        std::cout<<"The bbf from pose "<<k0.index()<<" to quadric "<<k1.index();
                        std::cout<<" has been removed.\n";
                    }
                }else{
                    // this is a good quadric, mark the 
                    // frame from which this measurement was taken
                    // for preservation
                    inWindow = std::min( inWindow , static_cast<size_t>(floor(k0.index()/2.0)) );
                    iter++;
                    if(verbose){
                        std::cout<<"The bbf from pose "<<k0.index()<<" to quadric "<<k1.index();
                        std::cout<<" is marked for preservation, new inWindow: "<<inWindow<<std::endl;
                    }
                }
            }else{
                //this was an odometry factor
                iter++;
            }
        }
    }

    //iterate through the graph one more time, remove the odometry factors
    //before the inWindow marker
    bool newPrior = false;
    for(auto iter = graph.begin(); iter!=graph.end();){

        gtsam::Symbol key0((*iter)->keys()[0]);
        int frame0 = floor(key0.index()/2.0);
        if(frame0<inWindow){
            //this factor is out of window
            if((*iter)->keys().size() == 1){
                //prior factor
                iter = graph.erase(iter);
                newPrior = true;
                if(verbose){
                    std::cout<<"The prior on pose "<<key0.index()<<" was removed.\n";
                }
            }else{
                // this is either a bounding box factor,
                // or an odometry factor. Check the second key; 
                gtsam::Symbol key1((*iter)->keys()[1]);
                if((key1.chr() == 'x')){
                    //odometry factor outside of the window
                    iter = graph.erase(iter);
                    if(verbose){
                        std::cout<<"The odometry from "<<key0.index()<<" to ";
                        std::cout<<key1.index()<<" has been removed\n";
                    }
                    //archive and remove pose estimates of out of window poses
                    //frame0 is already checked to be out of window
                    if(!trajectoryOld.exists(key0)){
                        trajectoryOld.insert(key0,estimate.at(key0));
                        if(verbose){
                            std::cout<<"Pose "<<key0.index()<<" has been archived.\n";
                        }
                    }

                    if(estimate.exists(key0)){
                        estimate.erase(key0);
                        if(verbose){
                            std::cout<<"Pose "<<key0.index()<<" has been removed from estimates.\n";
                        }
                    }
                    

                    int frame1 = floor(key1.index()/2.0);
                    if(frame1<inWindow){
                        if(!trajectoryOld.exists(key1)){
                            trajectoryOld.insert(key1,estimate.at(key1));
                            if(verbose){
                                std::cout<<"Pose "<<key0.index()<<" has been archived.\n";
                            }
                        }

                        if(estimate.exists(key1)){
                            estimate.erase(key1);
                            if(verbose){
                                std::cout<<"Pose "<<key0.index()<<" has been removed from estimates.\n";
                            }
                        }
                    }
                }else{
                    iter++;
                } 
            }
        }else{
            //in window, no prob
            iter++;
        }
    }    

    //place a prior on the oldest remaining pose, to anchor the odometry
    if(newPrior){
        if(verbose){
            std::cout<<"Placing a new prior on pose: "<<2*inWindow<<std::endl;
        }
        gtsam::Key gKey = gtsam::Symbol('x',2*inWindow);
        graph.addPrior(gKey, estimate.at(gKey).cast<gtsam::Pose3>(),runConsts.odomNoise);
    }

}
//#####################################################################
void slidingWindow::updateProb(size_t frameN,const gtsam::Cal3_S2::shared_ptr& K, int& qKeyFree,
    std::vector<gtsam::Key>& newLandmarks,const semConsts& runConsts){
    // each dFrame (within the window) needs to update it's OWN quadric estimates outside of here
    if(runConsts.probWin == 0){dFrames.clear(); return;}

    bool verbose = false;
    if(verbose){
        std::cout<<"\n-----------------  Sliding Window :: Update Probabilities ------------------- \n";
        // std::cout<<"Graph before update: \n";
        // graph.print("");
    }

    //iterate through the graph, if a bounding box factor is within the 
    // runConsts.probWin, but not new, remove it. It will be updated. Also, build a 
    // reference table to check if any quadrics have NO remaining factors
    // (need to be removed)
    std::map<int,bool> quadricMeas;

    // for(size_t factorIdx = 0; factorIdx < graph.size(); factorIdx++){
    for(auto iter = graph.begin(); iter != graph.end();){
        gtsam::Symbol S0((*iter)->keys()[0]);
        if( ((*iter)->keys().size() == 1) ){
            // this is a prior factor
            iter++;
            continue;
        }else{
            gtsam::Symbol S1((*iter)->keys()[1]);
            if(S1.chr() == 'x'){
                //odometry
                iter++; 
                continue;
            } 

            if(quadricMeas.find(S1.index()) == quadricMeas.end()){
                // this is a quadric constraint for a quadric which hasn't been
                // seen before in this loop, all quadrics should be marked as bad initially
                quadricMeas[S1.index()] = false;
            }

            int factorFrame = floor(S0.index()/2.0);
            if(factorFrame == frameN){
                //don't update new factors
                quadricMeas[S1.index()] = true;
                iter++;
                if(verbose){
                    std::cout<<"Found a bbf to quadric: "<<S1.index()<<" which is new, i.e. constructed in frame "<<factorFrame<<std::endl;
                }
                continue;
            }

            if( (frameN-factorFrame) > runConsts.probWin-1){
                // factor is outside runConsts.probWin
                // this quadric is good (has a factor)
                if(verbose){
                    std::cout<<"Found a bbf to quadric: "<<S1.index()<<" which was constructed in frame "<<factorFrame<<std::endl;
                }
                quadricMeas[S1.index()] = true;
                iter++;
                continue;
            }
            // getting here means this is a bounding box factor within the prob window, not new,
            // so it has been recomputed using the new quadric estimates and will be reinserted 
            // in the next loop (or not, as the case may be)
            iter = graph.erase(iter);
            if(verbose){
                std::cout<<"Removing a bbf from frame "<<factorFrame<<" to quadric "<<S1.index()<<std::endl;
            }
        }
    }

    //build maps of factors so they can be updated/removed/added responsibly
    //add all updated factors    
    // std::map< std::pair<gtsam::Key,gtsam::Key> , gtsam_quadrics::BoundingBoxFactor > updatedFactors;

    for(auto iter = dFrames.begin(); iter != dFrames.end();){
        
        std::vector< std::vector<double> > probs;
        size_t dfFrame = iter->getFrame();

        if(verbose){
            std::cout<<"Loaded frame: "<<dfFrame<<std::endl;
        }

        if(frameN == dfFrame){
            if(verbose){
                std::cout<<"not updating the most recent frame\n";
            }
            iter++;
            continue;
        }

        if((frameN - dfFrame) > runConsts.probWin-1){
            if(verbose){
                std::cout<<"removing dataframe "<<dfFrame<<std::endl;
                std::cout<<"because current frame is: "<<frameN<<" and runConsts.probWin: "<<runConsts.probWin<<std::endl;
            }
            iter = dFrames.erase(iter);
            continue;
        }

        //it is dumb to do this over and over again
        std::vector<gtsam_quadrics::ConstrainedDualQuadric> landmarks;
        std::vector<gtsam::Key> landmarkKeys;
        //extract the current landmarks
        for(int q = 0; q < qKeyFree; q++){
            // key to landmark
            gtsam::Key qKey = gtsam::Symbol('q',q);
            // extract and cast landmark state
            if(estimate.exists(qKey)){
                gtsam_quadrics::ConstrainedDualQuadric quad = estimate.at(qKey).cast<gtsam_quadrics::ConstrainedDualQuadric>();
                landmarks.push_back(quad);
                landmarkKeys.push_back(qKey);
                if(verbose){
                    std::cout<<"Landmark "<<q<<" is in estimate and has centroid: "<<landmarks[landmarks.size()-1].centroid().transpose()<<std::endl;
                }
            }else if(landmarksOld.exists(qKey)){
                gtsam_quadrics::ConstrainedDualQuadric quad = landmarksOld.at(qKey).cast<gtsam_quadrics::ConstrainedDualQuadric>();
                landmarks.push_back(quad);
                landmarkKeys.push_back(qKey);
                if(verbose){
                    std::cout<<"Landmark "<<q<<" is archived and has centroid: "<<landmarks[landmarks.size()-1].centroid().transpose()<<std::endl;
                }
            }else if(queuedLandmarks.find(qKey) != queuedLandmarks.end()){
                if(dfFrame > queuedFrame[qKey] + runConsts.landmark_age_thresh){
                    if(verbose){
                        std::cout<<"Landmark "<<q<<" is being removed from the queue, origin frame "<<queuedFrame[qKey]<<std::endl;
                    }
                    queuedFrame.erase(queuedFrame.find(qKey));
                    queuedLandmarks.erase(queuedLandmarks.find(qKey));
                    queuedFactors.erase(queuedFactors.find(qKey));
                    if(verbose){
                        bool success = queuedFrame.find(qKey)==queuedFrame.end();
                        success = success && queuedLandmarks.find(qKey)==queuedLandmarks.end();
                        success = success && queuedFactors.find(qKey)==queuedFactors.end();
                        std::cout<<"Landmark removed: "<<success<<std::endl;
                    }
                }else{
                    landmarks.push_back(queuedLandmarks[qKey]);
                    landmarkKeys.push_back(qKey);
                    if(verbose){
                        std::cout<<"Landmark "<<q<<" is queued and has centroid: "<<landmarks[landmarks.size()-1].centroid().transpose()<<std::endl;
                    }
                }
            }
        }
        // if the key does not correspond to a landmark, it was not initialised on purpose
        // (likely never observed more than once.)

        gtsam::Pose3 p = estimate.at(gtsam::Symbol('x',2*dfFrame)).cast<gtsam::Pose3>();
        if(verbose){
            std::cout<<"Estimate of position in frame "<<dfFrame<<" is "<<p.translation().transpose()<<std::endl;
        }
        iter->estQuadrics(p,K);
        probs = getAssignmentProbs(iter->quadEst,landmarks,runConsts);
        // if(runConsts.usePerm){
        //     probs = permanentProb(iter->quadEst,landmarks,runConsts);
        // }else{
        //     probs = asgnProbQuad(iter->quadEst,landmarks,runConsts);
        // }

        size_t nL = landmarks.size();

        gtsam::Key curKeyL = gtsam::Symbol('x',2*dfFrame);
        gtsam::Key curKeyR = gtsam::Symbol('x',2*dfFrame+1);
        //for each quadric estimate
        for(size_t m = 0; m < iter->quadEst.size(); m++){
            // for each landmark
            for(size_t q = 0; q < nL; q++){
                // check if it assigns significantly to this landmark
                if(probs[m][q] < runConsts.NEW_FACTOR_PROB_THRESH){continue;}

                // if it does, assign it, with the scaled covariance
                gtsam::Key qKey = landmarkKeys[q];

                //this quadric is constrained by this factor, no need to remove
                quadricMeas[gtsam::Symbol(qKey).index()] = true;
                
                // add bounding box factors from stereo cameras 
                gtsam_quadrics::BoundingBoxFactor bbfL(iter->bbL[m].aBox,K,curKeyL,qKey,runConsts.bNoise(probs[m][q]));
                    // gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector4::Ones()*BOX_SD/probs[df][m][q]));
                
                gtsam_quadrics::BoundingBoxFactor bbfR(iter->bbR[m].aBox,K,curKeyR,qKey,runConsts.bNoise(probs[m][q]));
                    // gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector4::Ones()*BOX_SD/probs[df][m][q]));

                //this should be added to the graph if it is assigning to a non-queued landmark
                if(queuedLandmarks.find(qKey) == queuedLandmarks.end()){
                    graph.add(bbfL);    
                    graph.add(bbfR);

                    if(verbose){
                        std::cout<<"Measurements "<<m<<" from frame "<<dfFrame<<" are being "<<probs[m][q]<<" attributed to landmark "<<gtsam::Symbol(qKey).index()<<std::endl;
                    }

                    if(!estimate.exists(qKey)){
                        if(landmarksOld.exists(qKey)){
                            estimate.insert(qKey,landmarksOld.at(qKey));

                            newLandmarks.push_back(qKey); //this will ensure it gets a prior, next iteration
                            
                            if(verbose){
                                std::cout<<"Using archived estimate with centroid: "<<estimate.at(qKey).cast<gtsam_quadrics::ConstrainedDualQuadric>().centroid().transpose()<<std::endl;
                            }
                        }else{
                            throw std::runtime_error("Where is the estimate, if not live (in estimate), landmarksOld, or queued?");
                            // estimate.insert(qKey,iter->quadEst[m]);
                            // if(verbose){
                            //     std::cout<<"Using current estimate with centroid: "<<iter->quadEst[m].centroid().transpose()<<std::endl;
                            // }
                        }
                    } 

                }else if(queuedFrame[qKey] != dfFrame){
                    //this was a queued landmark, but did not originate in this frame
                    graph.add(bbfL);    
                    graph.add(bbfR);
                    graph.add(queuedFactors[qKey][0]);
                    graph.add(queuedFactors[qKey][1]);
                    
                    estimate.insert(qKey,queuedLandmarks[qKey]);

                    newLandmarks.push_back(qKey); //this will ensure it gets a prior, next iteration

                    if(verbose){
                        std::cout<<"Measurements "<<m<<" from frame "<<dfFrame<<" are being "<<probs[m][q]<<" attributed to QUEUED landmark "<<gtsam::Symbol(qKey).index()<<std::endl;
                        std::cout<<"Along with queued factors\n";
                        std::cout<<"Using queued estimate with centroid: "<<queuedLandmarks[qKey].centroid().transpose()<<std::endl;
                    }

                    queuedFrame.erase(queuedFrame.find(qKey));
                    queuedLandmarks.erase(queuedLandmarks.find(qKey));
                    queuedFactors.erase(queuedFactors.find(qKey));
                    if(verbose){
                        bool success = queuedFrame.find(qKey)==queuedFrame.end();
                        success = success && queuedLandmarks.find(qKey)==queuedLandmarks.end();
                        success = success && queuedFactors.find(qKey)==queuedFactors.end();
                        std::cout<<"Landmark removed from queue: "<<success<<std::endl;
                    }

                    

                }else{
                    //update the queued landmark, do not add to graph yet
                    if(verbose){
                        std::cout<<"Updating factors from measurements "<<m<<" in frame "<<dfFrame<<" to QUEUED landmark "<<gtsam::Symbol(qKey).index();
                        std::cout<<" with updated centroid: "<<iter->quadEst[m].centroid().transpose()<<std::endl;
                    }
                    queuedLandmarks[qKey] = iter->quadEst[m];
                    queuedFactors[qKey] = std::vector<gtsam_quadrics::BoundingBoxFactor>{bbfL,bbfR};
                }
            }

            // if the quadric was significantly assigned 
            // to the dummy ``non-assigned'' landmark, then it 
            // becomes a new quadric (if it passes the NEW_LANDMARK_THRESH)
            if(probs[m][nL] > runConsts.NEW_LANDMARK_THRESH){
                gtsam::Key qKey = gtsam::Symbol('q',qKeyFree);           
                
                // estimate.insert(qKey,dFrames[df].quadEst[m]);             
                queuedLandmarks[qKey] = iter->quadEst[m];

                // add bounding box factors from stereo cameras 
                gtsam_quadrics::BoundingBoxFactor bbfL(iter->bbL[m].aBox,K,curKeyL,qKey,runConsts.bNoise(probs[m][nL]));
                    // gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector4::Ones()*BOX_SD/probs[df][m][nL]));

                gtsam_quadrics::BoundingBoxFactor bbfR(iter->bbR[m].aBox,K,curKeyR,qKey,runConsts.bNoise(probs[m][nL]));
                    // gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector4::Ones()*BOX_SD/probs[df][m][nL]));

                queuedFactors[qKey] = std::vector<gtsam_quadrics::BoundingBoxFactor>{bbfL,bbfR};
                queuedFrame[qKey] = dfFrame;
                if(verbose){
                    std::cout<<"Measurements "<<m<<" from frame "<<dfFrame<<" generated landmark "<<gtsam::Symbol(qKey).index();
                    std::cout<<" w/ centroid "<<iter->quadEst[m].centroid().transpose()<<std::endl;
                }
                // updatedFactors[std::pair<gtsam::Key,gtsam::Key>(curKeyL,qKey)] = bbfL;
                // updatedFactors[std::pair<gtsam::Key,gtsam::Key>(curKeyR,qKey)] = bbfR;   
                qKeyFree++;
            }
        }
        iter++;
    }


    // perform consistency check, if any quadrics have no bounding box factors, they 
    // should be archived, to avoid inconsistency errors from gtsam
    for(auto iter = quadricMeas.begin(); iter != quadricMeas.end(); iter++){
        if(!iter->second){
            //this quadric is not constrained, archive it.
            gtsam::Key qKey = gtsam::Symbol('q',iter->first);
            if(landmarksOld.exists(qKey)){
                landmarksOld.update(qKey,estimate.at(qKey));
            }else{
                landmarksOld.insert(qKey,estimate.at(qKey));
            }
            estimate.erase(qKey);
            if(verbose){
                std::cout<<"Archived quadric: "<<iter->first<<" as it was not constrained anymore.\n";
            }
        }
    }

    //remove priors of archived quadrics
    for(auto iter = graph.begin(); iter != graph.end();){
        gtsam::Symbol S0((*iter)->keys()[0]);
        if( ((*iter)->keys().size() == 1)  && (S0.chr()=='q')){
            // this is a quadric prior factor
            if(!quadricMeas[S0.index()]){
                //unconstrained, remove it!
                iter = graph.erase(iter);
            }else{
                iter++;
            }
        }else{
            iter++;
        }
    }


    if(verbose){
        // std::cout<<"Graph after update: \n";
        // graph.print("");
        std::cout<<"\n-----------------  End Sliding Window :: Update Probabilities ------------------- \n";
    }
}

