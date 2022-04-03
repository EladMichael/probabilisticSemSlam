#include "assignment.h"

#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <matplot/matplot.h>
#include <algorithm>

#include "progressBar.hpp"
using progresscpp::ProgressBar;

bool getCosts(const std::string& id, size_t frame, std::vector< std::vector<double> >& costs);

void timePlots(const std::vector<double>& assignmentTimes,const std::vector<double>& approxPermTimes,
    const std::vector<double>& exactPermTimes,const std::vector<double>& fastestPermTimes,const std::vector<size_t>& rows,
    const std::vector<size_t>& cols);

void errPlots(size_t k, const std::vector<double>& maxAssignmentErr,const std::vector<double>& k20Errs,
    const std::vector<double>& maxApproxPermErr,const std::vector<double>& maxExactPermErr,
    const std::vector<double>& maxFastPermErr,const std::vector<size_t>& rows,
    const std::vector<size_t>& cols);

void sweepKPlots(std::vector<double>& k1Times,std::vector<double>& k20Times,
    std::vector<double>& k1000Times,std::vector<double>& k100Times,
    std::vector<double>& k200Times,std::vector<double>& k1Errs,
    std::vector<double>& k20Errs,std::vector<double>& k1000Errs,
    std::vector<double>& k100Errs,std::vector<double>& k200Errs);


bool getCosts(const std::string& id, size_t frame, std::vector< std::vector<double> >& costs){
    std::string dir("generatedData/00/costMatrices/");
    std::string fileName = id + "_frame"+std::to_string(frame)+".dat";

    if(!file_exists(dir+fileName)){
        return false;
    }
    std::ifstream myfile(dir+fileName);
    std::vector< std::vector<double> > costMatrix;

    std::string line;
    for(size_t row = 0; std::getline(myfile,line); row++){
        costMatrix.push_back(std::vector<double>());
        std::stringstream ss(line);
        std::string val;
        while(std::getline(ss,val,',')){
            if(val[0]=='i'){
                costMatrix[row].push_back(inf_d);
            }else{
                costMatrix[row].push_back(std::stod(val));
            }
        }
    }
    costMatrix.swap(costs);
    return true;
}




int main(int argc, char** argv){
    if(argc < 2){
        std::cout<<"Usage: ./comparison idString k(200 default) maxFrame(500 default) startFrame(0 default)\n";
        return 1;
    }
    std::string idString = argv[1];
    size_t k = 100;
    size_t maxFrame = 4541;
    size_t startFrame = 0;
    if(argc>2){
        k = atoi(argv[2]);
    }
    if(argc>3){
        maxFrame = atoi(argv[3]);
    }
    if(argc>4){
        startFrame = atoi(argv[4]);
    }
    std::cout<<"Loading idString: "<<idString<<" using k: "<<k<<" with maxFrame: "<<maxFrame<<" and start frame: "<<startFrame<<std::endl;


    std::vector<double> k1Times;
    std::vector<double> k20Times;
    std::vector<double> k1000Times;
    std::vector<double> k100Times;
    std::vector<double> k200Times;
    std::vector<double> k1Errs;
    std::vector<double> k20Errs;
    std::vector<double> k1000Errs;
    std::vector<double> k100Errs;
    std::vector<double> k200Errs;


    std::vector<double> assignmentTimes;
    std::vector<double> approxPermTimes;
    std::vector<double> exactPermTimes;
    // std::vector<double> exactLongPermTimes;
    std::vector<double> fastestPermTimes;

    std::vector<double> maxAssignmentErr;
    std::vector<double> maxApproxPermErr;
    std::vector<double> maxExactPermErr;
    // std::vector<double> maxExactLongPermErr;
    std::vector<double> maxFastPermErr;

    std::vector<size_t> rows;
    std::vector<size_t> cols;
    std::vector<double> sparsity;

    // std::vector<double> avgAssignmentErr;
    // std::vector<double> avgApproxPermErr;

    size_t frame=0;
    std::vector< std::vector<double> > costs;
    int counter = -1;

    ProgressBar bar(maxFrame,60);
    std::vector<int> frameCount;
    bool verbose = false;
    for(size_t frame = startFrame; frame < maxFrame ;frame++){
        ++bar;

        if(!getCosts(idString,frame+1,costs)){
            continue;
        }

        if(verbose){
            std::cout<<"=========================== Frame "<<frame<<" =================================="<<std::endl;
            std::cout<<"Cost Matrix:\n";
            for(size_t r = 0; r < costs.size(); r++){
                for(size_t c = 0; c < costs[0].size(); c++){
                    std::cout<<costs[r][c]<<" ";
                }
                std::cout<<std::endl;
            }
        }else{

            bar.display();
        }

        counter++;

        frameCount.push_back(frame);

        size_t nCols = costs[0].size();
        size_t nRows = costs.size();
        size_t nM = costs[0].size(); //measurements
        size_t nL = costs.size()-nM; //landmarks, excluding dummy rows

        std::vector<double> costsUnrolled(nCols*nRows);
        for(size_t r = 0; r < nRows; r++){
            for(size_t c = 0; c < nCols; c++){
                costsUnrolled[c*nRows+r] = costs[r][c];
            }
        }

        // row Idx would be used to unpack the solution back into the 
        // original (unconditioned, overlarge, etc) cost matrix shape
        std::vector<ptrdiff_t> rowIdx; 
        std::vector<double> conditionedCosts = conditionCosts(costsUnrolled,nL+nM,nM,rowIdx);

        // numer of landmarks that might actually be assigned
        size_t condL = (conditionedCosts.size()/nM)-nM;
        rows.push_back(condL+nM);
        cols.push_back(nM);
        size_t nFinite = 0;
        for(size_t i = 0; i < conditionedCosts.size(); i++){
            if(conditionedCosts[i]<inf_d){nFinite++;}
        }
        sparsity.push_back(nFinite/conditionedCosts.size());

        if(verbose){

            for(size_t r = 0; r < condL+nM; r++){
                for(size_t c = 0; c < nM; c++){
                    if(conditionedCosts[c*(condL+nM)+r]<inf_d){
                        conditionedCosts[c*(condL+nM)+r] = round(conditionedCosts[c*(condL+nM)+r]);
                    }
                }
            }

            std::cout<<"\n Conditioned Cost Matrix:\n";
            for(size_t r = 0; r < condL+nM; r++){
                for(size_t c = 0; c < nM; c++){
                    std::cout<<conditionedCosts[c*(condL+nM)+r]<<" ";
                }
                std::cout<<std::endl;
            }
        }

        std::vector< std::vector<double> > truthProbs = bruteForceProb(conditionedCosts,condL,nM);

        auto t1 = tic();
        std::vector< std::vector<double> > k1Probs = assignmentProb(conditionedCosts,condL,nM,1);
        auto t2 = toc(t1);
        k1Times.push_back(1000*t2);

        t1 = tic();
        std::vector< std::vector<double> > k10Probs = assignmentProb(conditionedCosts,condL,nM,20);
        t2 = toc(t1);
        k20Times.push_back(1000*t2);

        t1 = tic();
        std::vector< std::vector<double> > k1000Probs = assignmentProb(conditionedCosts,condL,nM,1000);
        t2 = toc(t1);
        k1000Times.push_back(1000*t2);

        t1 = tic();
        std::vector< std::vector<double> > k100Probs = assignmentProb(conditionedCosts,condL,nM,100);
        t2 = toc(t1);
        k100Times.push_back(1000*t2);

        t1 = tic();
        std::vector< std::vector<double> > k200Probs = assignmentProb(conditionedCosts,condL,nM,200);
        t2 = toc(t1);
        k200Times.push_back(1000*t2);

        t1 = tic();
        std::vector< std::vector<double> > assignProbs = assignmentProb(conditionedCosts,condL,nM,k);
        t2 = toc(t1);
        assignmentTimes.push_back(1000*t2);

        t1 = tic();
        std::vector< std::vector<double> > pApprxProbs = permanentProb(conditionedCosts,condL,nM,0);
        t2 = toc(t1);
        approxPermTimes.push_back(1000*t2);
        
        std::vector< std::vector<double> > pExactProbs;
        // std::vector< std::vector<double> > pExactLongProbs;
        if(condL+nM < 32){
            t1 = tic();
            pExactProbs = permanentProb(conditionedCosts,condL,nM,1);
            t2 = toc(t1);
            exactPermTimes.push_back(1000*t2);

            // t1 = tic();
            // pExactLongProbs = permanentProb(conditionedCosts,condL,nM,2);
            // t2 = toc(t1);
            // exactLongPermTimes.push_back(1000*t2);

        }else{
            pExactProbs = assignmentProb(conditionedCosts,condL,nM,1000); //this should be basically 100% accuracy, benefit of the fuckin' doubt
            // pExactLongProbs = assignmentProb(conditionedCosts,condL,nM,1000); //this should be basically 100% accuracy, benefit of the fuckin' doubt
            // exactLongPermTimes.push_back(1000); //these times will just be clipped off the graph
            exactPermTimes.push_back(1000); //these times will just be clipped off the graph
        }


        k1Errs.push_back(1e-32);
        k20Errs.push_back(1e-32);
        k1000Errs.push_back(1e-32);
        k100Errs.push_back(1e-32);
        k200Errs.push_back(1e-32);

        maxAssignmentErr.push_back(1e-32);
        maxApproxPermErr.push_back(1e-32);
        maxExactPermErr.push_back(1e-32);
        // maxExactLongPermErr.push_back(1e-32);

        for(size_t m = 0; m < truthProbs.size(); m++){
            for(size_t l = 0; l < truthProbs[0].size(); l++){

                k1Errs[counter] = std::max(k1Errs[counter],std::abs(k1Probs[m][l]-truthProbs[m][l]));
                k20Errs[counter] = std::max(k20Errs[counter],std::abs(k10Probs[m][l]-truthProbs[m][l]));
                k1000Errs[counter] = std::max(k1000Errs[counter],std::abs(k1000Probs[m][l]-truthProbs[m][l]));
                k100Errs[counter] = std::max(k100Errs[counter],std::abs(k100Probs[m][l]-truthProbs[m][l]));
                k200Errs[counter] = std::max(k200Errs[counter],std::abs(k200Probs[m][l]-truthProbs[m][l]));

                maxAssignmentErr[counter] = std::max(maxAssignmentErr[counter],std::abs(assignProbs[m][l]-truthProbs[m][l]));
                maxApproxPermErr[counter] = std::max(maxApproxPermErr[counter],std::abs(pApprxProbs[m][l]-truthProbs[m][l]));
                maxExactPermErr[counter] = std::max(maxExactPermErr[counter],std::abs(pExactProbs[m][l]-truthProbs[m][l]));
                // maxExactLongPermErr[counter] = std::max(maxExactLongPermErr[counter],std::abs(pExactLongProbs[m][l]-truthProbs[m][l]));
            }
        }

        // if((exactLongPermTimes[counter] <= approxPermTimes[counter]) && (exactLongPermTimes[counter] <= exactPermTimes[counter])){
        //     maxFastPermErr.push_back(maxExactLongPermErr[counter]);
        //     fastestPermTimes.push_back(exactLongPermTimes[counter]);
        // }else if((exactPermTimes[counter] <= approxPermTimes[counter]) && (exactPermTimes[counter] <= exactLongPermTimes[counter])){
        if(exactPermTimes[counter] <= approxPermTimes[counter]){
            maxFastPermErr.push_back(maxExactPermErr[counter]);
            fastestPermTimes.push_back(exactPermTimes[counter]);
        }else{
            maxFastPermErr.push_back(maxApproxPermErr[counter]);
            fastestPermTimes.push_back(approxPermTimes[counter]);            
        }
        
        if(verbose){
            std::cout<<"\nAssignment Probs:\n";
            for(size_t l = 0; l < pExactProbs[0].size(); l++){
                for(size_t m = 0; m < pExactProbs.size(); m++){
                    std::cout<<assignProbs[m][l]<<" ";
                    // std::cout<<<<" ";
                }
                std::cout<<std::endl;
            }

            std::cout<<"\nExact Perm Probs:\n";
            for(size_t l = 0; l < pExactProbs[0].size(); l++){
                for(size_t m = 0; m < pExactProbs.size(); m++){
                    std::cout<<pExactProbs[m][l]<<" ";
                    // std::cout<<<<" ";
                }
                std::cout<<std::endl;
            }

            std::cout<<"\nAssignment/Exact absdiff:\n";
            for(size_t l = 0; l < pExactProbs[0].size(); l++){
                for(size_t m = 0; m < pExactProbs.size(); m++){
                    std::cout<<std::abs(assignProbs[m][l]-pExactProbs[m][l])<<" ";
                    // std::cout<<<<" ";
                }
                std::cout<<std::endl;
            }
            std::cout<<"\nassignment error: "<<maxAssignmentErr[counter]<<std::endl;

        }
        if(maxAssignmentErr[counter] > 0.1){
            std::cout<<"\nBig error!! Frame: "<<frame<<" !!\n";
            std::cout<<"\nBig error!! Frame: "<<frame<<" !!\n";
            std::cout<<"\nBig error!! Frame: "<<frame<<" !!\n";
            return 1;
        }
    }
    bar.done();
    for(size_t frame = 0; frame < maxAssignmentErr.size(); frame++){
        if(maxAssignmentErr[frame] > 1e-8){
            std::cout<<"\nError in frame "<<frameCount[frame]<<" : "<<maxAssignmentErr[frame]<<" from size ( "<<rows[frame]<<" , "<<cols[frame]<<" ) \n";
        }
    }


    if(!verbose){
        timePlots(assignmentTimes,approxPermTimes,exactPermTimes,fastestPermTimes,rows,cols);
        errPlots(k,maxAssignmentErr,k20Errs,maxApproxPermErr,maxExactPermErr,maxFastPermErr,rows,cols);
        sweepKPlots(k1Times,k20Times,k1000Times,k100Times,k200Times,k1Errs,k20Errs,k1000Errs,k100Errs,k200Errs);
    }

    return 1;
}

void timePlots(const std::vector<double>& assignmentTimes,const std::vector<double>& approxPermTimes,
    const std::vector<double>& exactPermTimes,
    const std::vector<double>& fastestPermTimes,const std::vector<size_t>& rows,
    const std::vector<size_t>& cols){

    using namespace matplot;


    size_t N = assignmentTimes.size();
    std::vector<int> sizeForColor(N);
    std::vector<double> assnLogTimes(N);
    std::vector<double> aprxLogTimes(N);
    std::vector<double> exctLogTimes(N);
    // std::vector<double> longLogTimes(N);
    std::vector<double> fastLogTimes(N);


    double minTime = 1;
    double maxApxTime = 0;
    double maxExactTime = 0;
    // double maxExactLongTime = 0;
    double maxFastTime = 0;

    for(size_t n = 0;  n < N; n++){
        sizeForColor[n] = std::max(rows[n],cols[n]);
        
        assnLogTimes[n] = std::log10(assignmentTimes[n]);
        aprxLogTimes[n] = std::log10(approxPermTimes[n]);
        exctLogTimes[n] = std::log10(exactPermTimes[n]);
        // longLogTimes[n] = std::log10(exactLongPermTimes[n]);
        fastLogTimes[n] = std::log10(fastestPermTimes[n]);


        maxApxTime = std::max(maxApxTime,aprxLogTimes[n]);
        maxFastTime = std::max(maxFastTime,fastLogTimes[n]);

        if(sizeForColor[n] <= 32){
            maxExactTime = std::max(maxExactTime,exctLogTimes[n]);
            // maxExactLongTime = std::max(maxExactLongTime,longLogTimes[n]);
        }

        minTime = std::min(minTime,assnLogTimes[n]);
        minTime = std::min(minTime,fastLogTimes[n]);
    }

    auto f4 = figure();
    auto ax = f4->current_axes();
    auto l = scatter(assnLogTimes,aprxLogTimes,6,sizeForColor);
    ax->xlabel("Log of Assignment Times [log10(ms)]");
    ax->ylabel("Log of Approximate Permanent Times [log10(ms)]");
    l->marker_face(true);
    colorbar();
    axis({minTime, maxApxTime, minTime, maxApxTime});
    hold(on);
    line(minTime,minTime,maxApxTime,maxApxTime);
    grid(on);
    // title("Assignment vs Approximate Permanent");
    f4->draw();
    
    auto f5 = figure();
    ax = f5->current_axes();
    l = scatter(assnLogTimes,exctLogTimes,6,sizeForColor);
    ax->xlabel("Log of Assignment Times [log10(ms)]");
    ax->ylabel("Log of Exact Permanent Times [log10(ms)]");
    l->marker_face(true);
    colorbar();
    axis({minTime, maxExactTime, minTime, maxExactTime});
    hold(on);
    line(minTime,minTime,maxExactTime,maxExactTime);
    grid(on);
    // title("Assignment vs Exact Permanent");
    f5->draw();

    // auto f6 = figure();
    // ax = f6->current_axes();
    // l = scatter(assnLogTimes,longLogTimes,6,sizeForColor);
    // ax->xlabel("Log of Assignment Times [log10(ms)]");
    // ax->ylabel("Log of Exact (long double) Permanent Times [log10(ms)]");
    // l->marker_face(true);
    // colorbar();
    // axis({minTime, maxExactLongTime, minTime, maxExactLongTime});
    // hold(on);
    // line(minTime,minTime,maxExactLongTime,maxExactLongTime);
    // grid(on);
    // title("Assignment vs Exact (long double) Permanent");
    // f6->draw();
    
    auto f7 = figure();
    ax = f7->current_axes();
    l = scatter(assnLogTimes,fastLogTimes,6,sizeForColor);
    ax->xlabel("Log of Assignment Times [log10(ms)]");
    ax->ylabel("Log of Fastest Permanent Times [log10(ms)]");
    l->marker_face(true);
    colorbar();
    axis({minTime, maxFastTime, minTime, maxFastTime});
    hold(on);
    line(minTime,minTime,maxFastTime,maxFastTime);
    grid(on);
    // title("Assignment vs Fastest Permanent");
    f7->draw();
    // show();
    // return 0;
    // auto a = histogram::binning_algorithm::fd;

    // auto f1 = figure();
    // auto ax = f1->current_axes();    
    // // subplot(1, 3, 0);
    // auto h1 = hist(ax,assignLogTimes,50);
    // hold(on);
    // auto h2 = hist(ax,fastestLogTimes,50);
    // auto leg = legend("Ranked Assignments" , "Fastest Permanent");
    // // hist(ax,assignLogTimes, a, histogram::normalization::count);
    // grid(on);
    // title("Time to Compute");
    // f1->draw();

    // auto f2 = figure();
    // ax = f2->current_axes();    
    // // subplot(1, 3, 1);
    // hist(ax,approxLogTimes, 50);
    // // hist(ax,approxLogTimes, a, histogram::normalization::count);
    // grid(on);// 
    // title("Time to Compute using Approximate Permanent");

    // auto f3 = figure();
    // ax = f3->current_axes();  
    // // subplot(1, 3, 2);
    // hist(ax,exactLogTimes, 50);
    // // hist(ax,exactLogTimes, a, histogram::normalization::count);
    // grid(on);// 
    // title("Time to Compute using Exact Permanent");


    // auto h1 = hist(x);
    // hold(on);
    // auto h2 = hist(y);
    // h1->normalization(histogram::normalization::probability);
    // h1->bin_width(0.25);
    // h2->normalization(histogram::normalization::probability);
    // h2->bin_width(0.25);
    // show();

}


void errPlots(size_t k, const std::vector<double>& maxAssignmentErr,const std::vector<double>& k20Errs,
    const std::vector<double>& maxApproxPermErr,const std::vector<double>& maxExactPermErr,
    const std::vector<double>& maxFastPermErr,const std::vector<size_t>& rows,
    const std::vector<size_t>& cols){

    using namespace matplot;

    size_t N = maxAssignmentErr.size();
    std::vector<int> sizeForColor(N);
    double maxError = 0;
    std::vector<double> linPerc(N);

    for(size_t n = 0;  n < N; n++){
        // sizeForColor[n] = std::max(rows[n],cols[n]);
        
        linPerc[n] = static_cast<double>(1.0*n/N);

        // maxAssignmentErr[n] = std::max(maxAssignmentErr[n],1e-32); //no log of 0! 
        // maxApproxPermErr[n] = std::max(maxApproxPermErr[n],1e-32); //no log of 0! 
        // maxExactPermErr[n] = std::max(maxExactPermErr[n],1e-32); //no log of 0! 
        // maxExactLongPermErr[n] = std::max(maxExactLongPermErr[n],1e-32); //no log of 0! 
        // maxFastPermErr[n] = std::max(maxFastPermErr[n],1e-32); //no log of 0! 

        // maxError = std::max(maxError,assignLogErrs[n]);
        // maxError = std::max(maxError,approxLogErrs[n]);
    }


    // auto a = histogram::binning_algorithm::automatic;
    // auto a = histogram::binning_algorithm::fd;
    

    // auto f1 = figure();
    // auto ax = f1->current_axes();
    // auto l = scatter(assignLogErrs,fastLogErrs,6,sizeForColor);
    // ax->xlabel("Error from Assignment Approximation");
    // ax->ylabel("Error from Fastest Permanent");
    // // auto l = scatter(x, y, 6, c);
    // l->marker_face(true);
    // colorbar();
    // axis({-32, maxError, -32, maxError});
    // // hold(on);
    // line(-32,-32,maxError,maxError);
    // grid(on);
    // title("Assignment vs Fastest Permanent");

    // auto f2 = figure();
    // ax = f2->current_axes();
    // l = scatter(assignLogErrs,approxLogErrs,6,sizeForColor);
    // ax->xlabel("Error from Assignment Approximation");
    // ax->ylabel("Error from Approximate Permanent");
    // // auto l = scatter(x, y, 6, c);
    // l->marker_face(true);
    // colorbar();
    // axis({-32, maxError, -32, maxError});
    // // hold(on);
    // line(-32,-32,maxError,maxError);
    // grid(on);
    // title("Assignment vs Approximate Permanent");
    std::vector<double> k20ErrSorted = k20Errs;
    std::vector<double> assErrSorted = maxAssignmentErr;
    std::vector<double> apxErrSorted = maxApproxPermErr;
    std::vector<double> extErrSorted = maxExactPermErr;
    // std::vector<double> extLongErrSorted = maxExactLongPermErr;
    std::vector<double> fastErrSorted = maxFastPermErr;

    std::sort(k20ErrSorted.begin(),k20ErrSorted.end());
    std::sort(assErrSorted.begin(),assErrSorted.end());
    std::sort(apxErrSorted.begin(),apxErrSorted.end());
    std::sort(extErrSorted.begin(),extErrSorted.end());
    // std::sort(extLongErrSorted.begin(),extLongErrSorted.end());    
    std::sort(fastErrSorted.begin(),fastErrSorted.end());

    auto f3 = figure();
    auto ax = f3->current_axes();
    auto p = semilogy(linPerc,assErrSorted,linPerc,k20ErrSorted,"--",
        linPerc,extErrSorted,"-.",linPerc,fastErrSorted,":",linPerc,apxErrSorted,"-*");
    p[0]->line_width(3);
    p[1]->line_width(3);
    p[2]->line_width(3);
    p[3]->line_width(3);
    p[4]->line_width(3);
    std::vector<size_t> markerIdx(20);
    for(size_t i = 0; i < markerIdx.size(); i++){
        markerIdx[i] = static_cast<size_t>(i*N/markerIdx.size());
    }
    p[4]->marker_indices(markerIdx);
    // p[1]->marker(line_spec::marker_style::asterisk);
    // plot(linPerc,assErrSorted);
    // hold(on);
    // plot(linPerc,apxErrSorted);
    // hist(assErrSorted, a, histogram::normalization::cummulative_count);
    grid(on);
    // title("Error Order Statistics");
    ax->xlabel("Percentage of Problems");
    ax->ylabel("Max Error");
    auto leg = legend("Ranked Assignments (k="+std::to_string(k) + ")" , "Ranked Assignments (k=20)"  , "Exact Permanent" , 
        "Fastest Permanent","Approximate Permanent");
    leg->location(legend::general_alignment::bottomright);
    xlim({0,1});
    // auto f3 = figure();
    // ax = f3->current_axes();
    // // hist(apxErrSorted, a, histogram::normalization::cummulative_count);
    // grid(on);
    // title("Approximate Permanent Order Statistics");
    // ax->xlabel("Percentage of Problems");
    // ax->ylabel("Difference from Exact Permanent");

    show();
}


void sweepKPlots(std::vector<double>& k1Times,std::vector<double>& k20Times,
    std::vector<double>& k1000Times,std::vector<double>& k100Times,
    std::vector<double>& k200Times,std::vector<double>& k1Errs,
    std::vector<double>& k20Errs,std::vector<double>& k1000Errs,
    std::vector<double>& k100Errs,std::vector<double>& k200Errs){

    using namespace matplot;

    std::sort(k1Times.begin(),k1Times.end());
    std::sort(k20Times.begin(),k20Times.end());
    std::sort(k1000Times.begin(),k1000Times.end());
    std::sort(k100Times.begin(),k100Times.end());
    std::sort(k200Times.begin(),k200Times.end());

    std::sort(k1Errs.begin(),k1Errs.end());
    std::sort(k20Errs.begin(),k20Errs.end());
    std::sort(k1000Errs.begin(),k1000Errs.end());
    std::sort(k100Errs.begin(),k100Errs.end());
    std::sort(k200Errs.begin(),k200Errs.end());

    size_t N = k1Times.size();

    std::vector<double> linPerc(N);

    for(size_t n = 0;  n < N; n++){
        // sizeForColor[n] = std::max(rows[n],cols[n]);
        linPerc[n] = static_cast<double>(1.0*n/N);
    }

    auto f1 = figure();
    auto ax = f1->current_axes();
    auto p = semilogy(linPerc,k1Times,linPerc,k20Times,
        linPerc,k100Times,linPerc,k200Times,linPerc,k1000Times);
    p[0]->line_width(3);
    p[1]->line_width(3);
    p[2]->line_width(3);
    p[3]->line_width(3);
    p[4]->line_width(3);
    grid(on);
    ax->xlabel("Percentage of Problems");
    ax->ylabel("Time to Compute [ms]");
    auto leg = legend("k = 1" , "k = 20" , "k = 100" , 
        "k = 200" , "k = 1000");
    leg->location(legend::general_alignment::topleft);
    xlim({0,1});

    auto f2 = figure();
    ax = f2->current_axes();
    p = semilogy(linPerc,k1Errs,linPerc,k20Errs,
        linPerc,k100Errs,linPerc,k200Errs,linPerc,k1000Errs);
    p[0]->line_width(3);
    p[1]->line_width(3);
    p[2]->line_width(3);
    p[3]->line_width(3);
    p[4]->line_width(3);
    grid(on);
    ax->xlabel("Percentage of Problems");
    ax->ylabel("Maximum Error");
    leg = legend("k = 1" , "k = 20" , "k = 100" , 
        "k = 200" , "k = 1000");
    leg->location(legend::general_alignment::topleft);
    xlim({0,1});
    // auto f3 = figure();
    // ax = f3->current_axes();
    // // hist(apxErrSorted, a, histogram::normalization::cummulative_count);
    // grid(on);
    // title("Approximate Permanent Order Statistics");
    // ax->xlabel("Percentage of Problems");
    // ax->ylabel("Difference from Exact Permanent");

    show();
}