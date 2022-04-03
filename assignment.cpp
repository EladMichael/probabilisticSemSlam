#include "assignment.h"

#include "shortestPathCPP.hpp"
#include "nwPerm.h"

#include <fstream>
#include <cmath>

const static size_t cutoff = 42;
const static size_t apprxIter = 300;
const static double tau = 6.2831853071;

// this is a functor
struct negExpNorm {
  negExpNorm(double val) : x(val) {}  // Constructor
  // after -34, we're past sixteen digits down, so it's basically nonsense
  // buuuut if we're multiplying by a thousand?... still seems like nonsense
  double operator()(double y) const { return ((x+cutoff > y)? std::exp(x-y) : 0); }

private:
  double x;
};


//this is actually an upper bound approximation ON the upperbound approximation
//but this one doesn't require a factorial, then a power, which could be numerically 
//dangerous... this can be computed, for essentially any number
double mincConstant(size_t ni,size_t mi){
    double n = ni;
    double m = mi;
    return std::pow(tau,(m-n)/(2*n))*std::pow(n/m,m)*std::exp( m/(12*n*n) - 1/(12*m+1) );
}

inline double mincFactor(size_t n){
    return std::pow(tau*n,1.0/(2.0*n))*n*std::exp(-1+1.0/(12*n*n));
}
//##################################################################################
std::vector< std::vector<double> > getAssignmentProbs(const std::vector< gtsam_quadrics::ConstrainedDualQuadric >& meas,
                                            const std::vector< gtsam_quadrics::ConstrainedDualQuadric >& land,
                                            const semConsts& runConsts){

    std::vector< Eigen::Matrix<double,3,1> > measMeans = getMeans(meas);
    std::vector< Eigen::Matrix<double,3,3> > measCovs = getCovs(meas);
    std::vector< Eigen::Matrix<double,3,1> > landMeans = getMeans(land);
    std::vector< Eigen::Matrix<double,3,3> > landCovs = getCovs(land);
    size_t nL = land.size();
    size_t nM = meas.size();

    if(nM == 0){
        return std::vector< std::vector< double > >();
    }else if(nL == 0){
        return std::vector< std::vector< double > >(nM,std::vector<double>{1});        
    }

    std::vector<double> costMatrix = computeQuadricCostMatrix(landMeans,landCovs,measMeans,measCovs,runConsts);

    std::vector<ptrdiff_t> rowIdx;
    std::vector<double> conditionedCosts = conditionCosts(costMatrix,nL+nM,nM,rowIdx);

    size_t condL = (conditionedCosts.size()/nM)-nM;

    std::vector< std::vector<double> > conditionedProbs;
    if(runConsts.usePerm){
        conditionedProbs = permanentProb(conditionedCosts,condL,nM,1);
    }else{
        conditionedProbs = assignmentProb(conditionedCosts,condL,nM,runConsts.k);
    }
    std::vector< std::vector< double > > probs(nM, std::vector<double>(nL+1,0));
    for(size_t m = 0; m < nM; m++){
        for(size_t l = 0; l < condL; l++){
            probs[m][rowIdx[l]] = conditionedProbs[m][l];
        }
        probs[m][nL] = conditionedProbs[m][condL];
    }

    bool verbose = false;
    if(verbose){
        std::cout<<"===========================================\n";
        std::cout<<"Using permanent for this: "<<runConsts.usePerm<<std::endl;
        std::cout<<"rowIdx of size "<<rowIdx.size()<<" w/ condL "<<condL<<" :\n";
        // for(size_t l = 0; l < condL+1; l++){
        //     std::cout<<rowIdx[l]<<" ";
        // }
        // std::cout<<"\nCond Prob Matrix From getAssignmentProbs: \n";
        // for(size_t l = 0; l < condL+1; l++){
        //     for(size_t m = 0; m < nM; m++){
        //         std::cout<<conditionedProbs[m][l]<<" ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<"Prob Matrix From getAssignmentProbs: \n";
        // for(size_t l = 0; l < nL+1; l++){
        //     for(size_t m = 0; m < nM; m++){
        //         std::cout<<probs[m][l]<<" ";
        //     }
        //     std::cout<<std::endl;
        // }
        std::vector< std::vector<double> > assProbs = assignmentProb(costMatrix,nL,nM,runConsts.k);
        // std::cout<<"Prob Matrix From Assignments w/o conditioning: \n";
        // for(size_t l = 0; l < nL+1; l++){
        //     for(size_t m = 0; m < nM; m++){
        //         std::cout<<assProbs[m][l]<<" ";
        //     }
        //     std::cout<<std::endl;
        // }
        std::vector< std::vector<double> > permProbs = permanentProb(costMatrix,nL,nM,1);
        // std::cout<<"Prob Matrix From Permanent w/o preconditioning: \n";
        // for(size_t l = 0; l < nL+1; l++){
        //     for(size_t m = 0; m < nM; m++){
        //         std::cout<<permProbs[m][l]<<" ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<"Abs Diff between Assignments and Perm: \n";
        // for(size_t l = 0; l < nL+1; l++){
        //     for(size_t m = 0; m < nM; m++){
        //         std::cout<<std::abs(assProbs[m][l]-permProbs[m][l])<<" ";
        //     }
        //     std::cout<<std::endl;
        // }
        double maxErrAssThis = 0;
        double maxErrPermThis = 0;
        double maxErrAssPerm = 0;
        for(size_t l = 0; l < nL+1; l++){
            for(size_t m = 0; m < nM; m++){
                maxErrAssThis = std::max(maxErrAssThis,std::abs(probs[m][l]-assProbs[m][l]));
                maxErrPermThis = std::max(maxErrPermThis,std::abs(probs[m][l]-permProbs[m][l]));
                maxErrAssPerm = std::max(maxErrAssPerm,std::abs(assProbs[m][l]-permProbs[m][l]));
            }
        }
        std::cout<<"Max diff between this and assignment:      "<<maxErrAssThis<<std::endl;
        std::cout<<"Max diff between this and permanent:       "<<maxErrPermThis<<std::endl;
        std::cout<<"Max diff between assignment and permanent: "<<maxErrAssPerm<<std::endl;
        std::cout<<"===========================================\n";
        std::cin.get();
    }

    return probs;
}

//##################################################################################
// std::vector< std::vector<double> > permanentProb(const std::vector< gtsam_quadrics::ConstrainedDualQuadric >& meas,
//                                             const std::vector< gtsam_quadrics::ConstrainedDualQuadric >& land,
//                                             const semConsts& runConsts){
std::vector< std::vector<double> > permanentProb(std::vector< double > costMatrix, size_t nL, size_t nM,int permOpt){

    bool verbose = false;
    size_t nRows = nL+nM;
    size_t nCols = nM;


    if(verbose){
        std::cout<<"\n\n+++++++++++++++++++++++++++++++ BEGIN PERMANENT +++++++++++++++++++++++++++++++\n\n";
        std::cout<<"\n\n+++++++++++++++++++++++++++++++ BEGIN PERMANENT +++++++++++++++++++++++++++++++\n\n";
        std::cout<<"\nCost Matrix:\n";
        for(size_t r = 0; r < nRows; r++){
            for(size_t c = 0; c < nCols; c++){
                std::cout<<costMatrix[c*nRows+r]<<" ";
            }
            std::cout<<std::endl;
        }
    }

    toProbs(costMatrix);
    
    if(nM == 1){
        //only one column. Normalize by sum of the column, and return it!
        double norm = 1.0/std::reduce(costMatrix.begin(),costMatrix.end());
        std::transform(costMatrix.begin(), costMatrix.end(), costMatrix.begin(), [norm](double &c){ return c*norm; });
        return std::vector< std::vector<double> >{costMatrix};
    }

    if(verbose){
        std::cout<<"Element-wise Prob Matrix:\n";
        for(size_t r = 0; r < nRows; r++){
            for(size_t c = 0; c < nCols; c++){
                std::cout<<costMatrix[c*nRows+r]<<" ";
            }
            std::cout<<std::endl;
        }
    }

    Eigen::MatrixXd fullProbs(nRows,nCols);
    for(size_t col = 0; col < nCols; col++){
        for(size_t row = 0; row < nRows; row++){
            fullProbs(row,col) = costMatrix[col*nRows+row];
        }
    }

    std::vector< std::vector< double > > probs(nM,std::vector<double>(nL+1,0));

    // double normConstant = 1.0/permanentFastest(fullProbs);


    //submatrix of element-wise probabilities.
    Eigen::MatrixXd subProbs(nRows-1,nCols-1);

    //each column represents 1 measurement. We 
    //"force" the assignment of a measurement to a column
    //by extracting the remaining submatrix after the removal of the
    //col/row, then find the permanent, then multiply the perm by the
    //element we removed to find the prob of all assignments including 
    //this element.
    std::vector<int> colIdx(nCols-1); //looks like 1,2,3,4,5,6,7...,ncols
    // after first iteration will be  0,2,3,4,5,6,7,..,ncols
    // after next iteration will be   0,1,3,4,5,6,7,..,ncols
    for(int i = 1; i < nCols; i++){
        colIdx[i-1] = i;
    }

    double fullPerm = 0;

    for(size_t m = 0; m < nM; m++){
        double fullColPerm = 0;
        setupAssgnMatrix(subProbs,fullProbs,m);
        for(size_t l = 0; l < nL; l++){
            if(fullProbs(l,m) != 0){
                // if(verbose){
                //     std::cout<<"-----------------------------\n";
                // }
                double lmPerm = fullProbs(l,m)*conditionedPermanent(subProbs,permOpt);
                // double lmPerm = fullProbs(l,m)*permWAssignments(subProbs);
                
                fullColPerm += std::abs(lmPerm);
                probs[m][l] = std::abs(lmPerm);
                if(verbose){
                    std::cout<<"( "<<l<<" , "<<m<<" ) subProb is "<<fullProbs(l,m)<<" times perm of: \n"<<subProbs<<std::endl;
                    std::cout<<"    which has value: "<<probs[m][l]<<std::endl;
                    std::cout<<"brute force: "<<fullProbs(l,m)*permWAssignments(subProbs)<<std::endl;
                    std::cout<<"Percentage difference from the brute force: "<<std::abs(probs[m][l] - fullProbs(l,m)*permWAssignments(subProbs))/probs[m][l]<<std::endl;
                }
            }
            subProbs(l,Eigen::all) = fullProbs(l,colIdx);
        }
        //if col == 0, then this is already the non-assignment matrix, else we have to 
        // swap an element in...
        if(m != 0){
            subProbs(nL-1+m,0) = fullProbs(nL,0); //this effectively places that overwritten row in the empty spot.
        }

        //non-assignment probability
        double nonPerm = fullProbs(nL+m,m)*conditionedPermanent(subProbs,permOpt);
        // double nonPerm = fullProbs(nL+m,m)*permWAssignments(subProbs);
        fullColPerm += std::abs(nonPerm);
        probs[m][nL] = std::abs(nonPerm);

        if(verbose){
            std::cout<<" nonAssign subProbs is "<<fullProbs(nL+m,m)<<" times perm of : \n"<<subProbs<<std::endl;
            std::cout<<"      which has value: "<<probs[m][nL]<<std::endl;
            std::cout<<"brute force method: "<<fullProbs(nL+m,m)*permWAssignments(subProbs)<<std::endl;
            std::cout<<"Percentage differece from brute force method: "<<std::abs(probs[m][nL]-fullProbs(nL+m,m)*permWAssignments(subProbs))/probs[m][nL]<<std::endl;
            std::cout<<"---- Permanent Estimate from this column is: "<<fullColPerm<<std::endl;
        }
        
        fullPerm = std::max(fullPerm,fullColPerm);
        if(m < nM-1){
            //this doesn't work on the last iteration
            colIdx[m] = m;        
        }
    }
    if(verbose){
        std::cout<<"Estimated total permanent is: "<<fullPerm<<std::endl;
        std::cout<<"computed from the full: "<<conditionedPermanent(fullProbs,permOpt)<<std::endl;
    }

    double normConstant = 1.0/fullPerm;
    for(size_t m = 0; m < nM; m++){
        for(size_t l = 0; l < nL+1; l++){
            probs[m][l] *= normConstant;
        }
    }


    if(verbose){
        std::cout<<"Prob Matrix From Perm: \n";
        for(size_t l = 0; l < nL+1; l++){
            for(size_t m = 0; m < nM; m++){
                std::cout<<probs[m][l]<<" ";
            }
            std::cout<<std::endl;
        }

        std::cout<<"\n\n+++++++++++++++++++++++++++++++ END PERMANENT +++++++++++++++++++++++++++++++\n\n";
        std::cout<<"\n\n+++++++++++++++++++++++++++++++ END PERMANENT +++++++++++++++++++++++++++++++\n\n";

    }


    return probs;
}
//##################################################################################
void setupAssgnMatrix(Eigen::MatrixXd& subProbs, const Eigen::MatrixXd& fullProbs, size_t col){
//     subProbs.block(0,0,subProbs.rows(),col) = fullProbs.block(1,0,subProbs.rows(),col);
//     subProbs.block(0,col,subProbs.rows(),subProbs.cols()-col) = fullProbs.block(1,col+1,subProbs.rows(),subProbs.cols()-col);
// }

    //setup the sub matrix for permanent computation 

    const bool verbose = false;
    size_t nRows = fullProbs.rows();
    size_t nCols = fullProbs.cols();
    size_t nM = fullProbs.cols();
    size_t nL = nRows - nM;
    // const size_t row = 0;

    if(col>0){
        //if col == 0, nothing to the left
        // if(verbose){
        //     std::cout<<"Thing 1 full: \n"<<fullProbs.block(1,0,nRows-1,col)<<std::endl;
        //     std::cout<<"Thing 1 sub: \n"<<subProbs.block(0,0,nRows-1,col)<<std::endl;
        // }
        subProbs.block(0,0,nRows-1,col) = fullProbs.block(1,0,nRows-1,col);
    }

    if(col < nM-1){
        // if(verbose){
        //     std::cout<<"Thing 2 full: \n"<<fullProbs.block(1,col+1,nRows-1,nCols-1-col)<<std::endl;
        //     std::cout<<"Thing 2 sub: \n"<<subProbs.block(0,col,nRows-1,nCols-1-col)<<std::endl;
        // }
        subProbs.block(0,col,nRows-1,nCols-1-col) = fullProbs.block(1,col+1,nRows-1,nCols-1-col);
    }

}
//##################################################################################
double conditionedPermanent(const Eigen::MatrixXd& A,int permOpt){
    // to condition the matrix, we remove all totally zero rows/cols, scale by the max col values, and transpose it
    bool verbose = false;
//     if(!useExact){
//         return permanentApproximation(A.transpose(),apprxIter);
//     }else{
//         return permanentExact(A.transpose());
//     }
// }
    if(verbose){
        std::cout<<"==========================================================\n";
        auto t1 = tic();
        if(permOpt == 0){
            std::cout<<"trasposed original permanent: "<<permanentApproximation(A.transpose(),apprxIter) <<std::endl;
        }else if(permOpt == 1){
            std::cout<<"trasposed original permanent: "<<permanentExact(A.transpose()) <<std::endl;
        }else if(permOpt ==2){
            std::cout<<"trasposed original permanent: "<<permanentExactLong(A.transpose()) <<std::endl;
        }else{
            throw std::runtime_error("Unknown permanent option passed!");
        }
        std::cout<<"in: "<<1000*toc(t1)<<"ms\n";  
        std::cout<<"Perm from brute force method: "<<permWAssignments(A)<<std::endl;
    }
    auto setupT = tic();
    auto t1 = tic();

    size_t nRows = A.rows();
    size_t nCols = A.cols();

    Eigen::VectorXd maxColVals = A.colwise().maxCoeff();
    Eigen::VectorXd minColVals(nCols);
    std::vector<int> colsIdx;
    colsIdx.reserve(nCols);
    for(size_t c = 0; c < nCols; c++){
        double minVal = 1;
        for(size_t r = 0; r < nRows; r++){
            if(A(r,c) > 0 && (A(r,c) < minVal)){
                minVal = A(r,c);
            }
        }
        minColVals(c) = minVal;
        if(maxColVals[c] > 0){
            colsIdx.push_back(c);
        }
    }

    Eigen::VectorXd maxRowVals = A.rowwise().maxCoeff();
    std::vector<int> rowsIdx;
    rowsIdx.reserve(nRows);
    for(size_t r = 0; r < nRows; r++){
        if(maxRowVals[r] > 0){
            rowsIdx.push_back(r);
        }
    }

    Eigen::MatrixXd Ascaled(rowsIdx.size(),colsIdx.size());
    double scaleFactor = 1;
    size_t goodCol = 0;
    for(size_t c = 0; c < nCols; c++){
        if(c == colsIdx[goodCol]){
            goodCol++;
            double colScale = 1.0/std::pow(maxColVals[c]*minColVals[c],0.5);
            // double colScale = 1.0/maxColVals[c];
            scaleFactor *= colScale;
            //add this col, but chosen rows, and scaled
            Ascaled(Eigen::all,c) = colScale*A(rowsIdx,c);
        }
    }
    double result;
    auto setupToc = toc(setupT);

    size_t minDim = std::min(colsIdx.size(),rowsIdx.size());

    if(permOpt == 0){
        result = permanentApproximation(Ascaled.transpose(),apprxIter) / scaleFactor;
    }else if(permOpt==1){
        result = permanentExact(Ascaled.transpose()) / scaleFactor;
    }else if(permOpt ==2){
        result = permanentExactLong(Ascaled.transpose()) / scaleFactor;
    }else{
        throw std::runtime_error("Unknown perm option in conditioned permanent!");
    }
    
    if(result < 0){
        if(permOpt == 0){
            result = permanentApproximation(Ascaled,apprxIter) / scaleFactor;
        }else if(permOpt==1){
            result = permanentExact(Ascaled) / scaleFactor;
        }else if(permOpt ==2){
            result = permanentExactLong(Ascaled) / scaleFactor;
        }else{
            throw std::runtime_error("Unknown perm option in conditioned permanent!");
        }
    }

    auto t2 = toc(t1);
    if(verbose){
        std::cout<<"scaled permanent: "<<result<<std::endl;
        std::cout<<"in: "<<1000*t2<<"ms\n";
        std::cout<<"setup time: "<<1000*setupToc<<"ms\n";
        std::cout<<"==========================================================\n";
        std::cout<<"Original matrix: \n"<<A<<std::endl;
        std::cout<<"Scaled matrix: \n"<<Ascaled<<std::endl;
        std::cout<<"Scale factor: \n"<<scaleFactor<<std::endl;
        std::cout<<"Max vals: "<<maxColVals.transpose()<<"\n min vals: "<<minColVals.transpose()<<std::endl;
        std::cin.get();
    }

    return result;
}

//##################################################################################
//##################################################################################
std::vector<double> conditionCosts(const std::vector<double>& costs, size_t nRows, size_t nCols, std::vector<ptrdiff_t>& rowIdxOut){
    // to condition the matrix, we remove all totally zero rows/cols, scale by the max col values
    // anybody who is 35 worse than the best cost is essentially zero in probabiltiy space (16 decimal places)
    // but in log space we have more precision, so let it ride until 42 (allows for about 1000 assignments at that stupid precision)
    bool verbose = false;

    auto t1 = tic();
    //there can't be bad columns (will always be at least the null assignments for each meas),
    //so first we will check the min cost in each col

    // std::vector<double> rowMins(nRows,inf);
    std::vector<double> colMins(nCols,inf_d);
    for(size_t col = 0; col < nCols; col++){
        for(size_t row = 0; row < nRows; row++){
            size_t idx = col*nRows+row;
            if(costs[idx] < colMins[col]){
                colMins[col] = costs[idx];
            }
        }
    }

    //we now go through and see if any 
    //row is totally wiped out by the respective column mins
    std::vector<bool> goodRow(nRows);
    size_t goodRows = 0;
    for(size_t row = 0; row < nRows; row++){
        goodRow[row] = false;
        for(size_t col = 0; col < nCols; col++){
            size_t idx = col*nRows+row;
            if(costs[idx] <= colMins[col]+cutoff){
                goodRow[row] = true;
                goodRows++;
                break;
            }
        }
    }

    size_t offset = 0;
    std::vector<double> conCosts(nCols*goodRows,inf_d);
    std::vector<ptrdiff_t> rowIdx(goodRows);
    std::vector<size_t> rowCard(goodRows,0);
    // rowIdx.reserve(goodRows);
    for(size_t row = 0; row < nRows; row++){
        if(!goodRow[row]){
            offset++;
            continue;
        }
        rowIdx[row-offset] = row;
        for(size_t col = 0; col < nCols; col++){
            size_t oIdx = col*nRows+row;
            size_t nIdx = col*goodRows+(row-offset);
            if(costs[oIdx] <= colMins[col]+cutoff){
                conCosts[nIdx] = costs[oIdx]-colMins[col];
            }else{
                conCosts[nIdx] = inf_d;
            }
        }
    }    

    auto t2 = toc(t1);
    if(verbose){
        size_t arcsO = 0;
        size_t arcsN = 0;
        std::cout<<"\nOriginal Cost Matrix:\n";
        for(size_t r = 0; r < nRows; r++){
            for(size_t c = 0; c < nCols; c++){
                if(costs[c*nRows+r]<inf_d){arcsO++;}
                std::cout<<costs[c*nRows+r]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<"\nConditioned Cost Matrix:\n";
        for(size_t r = 0; r < goodRows; r++){
            for(size_t c = 0; c < nCols; c++){
                if(conCosts[c*goodRows+r]<inf_d){arcsN++;}
                std::cout<<conCosts[c*goodRows+r]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<"setup time: "<<1000*t2<<"ms\n";
        std::cout<<"Went from "<<nCols*nRows<<" elements to "<<nCols*goodRows<<std::endl;
        std::cout<<"Also from "<<arcsO<<" non-inf_d arcs to "<<arcsN<<std::endl;
    }

    rowIdxOut.swap(rowIdx);
    return conCosts;
}
//###############################################################################################
void toProbs(std::vector<double>& costMatrix){
    // find lowest cost assignment (highest probability),
    // which will be used to shift the whole matrix to
    // improve the range of the exp function 
    double minCost = *std::min_element(costMatrix.begin(),costMatrix.end());
    // apply exp(minCost - costMatrix[i]) for all elements of cost matrix
    // std::transform(costMatrix.begin(), costMatrix.end(), costMatrix.begin(), negExpNorm(minCost));
    //this ^ line does the following: 
    for(size_t i = 0; i < costMatrix.size(); i++){
        if(minCost+cutoff > costMatrix[i]){
            costMatrix[i] = std::exp(minCost-costMatrix[i]);
        }else{
            costMatrix[i] = 0;
        }
    }
}
//##################################################################################
// std::vector< std::vector<double> > assignmentProb(const std::vector< gtsam_quadrics::ConstrainedDualQuadric >& meas,
//                                             const std::vector< gtsam_quadrics::ConstrainedDualQuadric >& land,
//                                             const semConsts& runConsts,bool condition){
std::vector< std::vector<double> > assignmentProb(const std::vector< double >& costMatrix, size_t nL, size_t nM, size_t k){
                                            // const std::vector< gtsam_quadrics::ConstrainedDualQuadric >& land,
                                            // const semConsts& runConsts,bool condition){
    
    size_t nRows = nL+nM;
    size_t nCols = nM;

    if(nM == 1){
        //only one column. Normalize by sum of the column, and return it!
        // std::vector<double> costs = costMatrix;
        std::vector< std::vector<double> > probs(1,std::vector<double>(costMatrix.size(),0));
        double norm = 0;
        for(size_t i = 0; i <= nL; i++){
            if(costMatrix[i]<cutoff){
                probs[0][i] = std::exp(-costMatrix[i]);
                norm += probs[0][i];
            }
        }
        // probs[0] = costMatrix;
        // toProbs(costs);
        norm = 1.0/norm;
        std::transform(probs[0].begin(), probs[0].end(), probs[0].begin(), [norm](double &c){ return c*norm; });
        return probs;
    }
    // size_t nRowsOrig = nRows;

    // std::vector<ptrdiff_t> rowIdx;
    // if(condition){
    //     costMatrix = conditionCosts(costMatrix,nRows,nCols,rowIdx);
    //     //nCols will not change
    //     nRows = costMatrix.size()/nCols;
    // }

    // max number of enumerated assignments
    // TODO: add cost delta cutoff
    // int k = 100;
    ScratchSpace workMem;//Scratch space needed for the assignment algorithm.
    //Allocate scratch space, numRow numRow is on purpose! Row major.
    workMem.init(nRows,nRows);
    ptrdiff_t rowAssignments[nRows*k];
    ptrdiff_t colAssignments[nCols*k];
    double assignmentCosts[k];

    /*The assignment algorithm returns a nonzero value if no valid
     * solutions exist.*/
    bool maximize = false; //Maximize or minimize assignment
    // int numFound = kBest2D(k,nRows,nCols,maximize,costMatrix.data(),workMem,rowAssignments,colAssignments,assignmentCosts);
    int numFound = kBest2DCutoff(k,nRows,nCols,maximize,costMatrix.data(),workMem,rowAssignments,colAssignments,assignmentCosts,cutoff);

    bool verbose = false;
    if(verbose){
        std::cout<<"Cost Matrix In assignment:\n";
        for(size_t r = 0; r < nRows; r++){
            for(size_t c = 0; c < nCols; c++){
                std::cout<<costMatrix[c*nRows+r]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<"Assignments: \n";
        for(size_t i = 0; i < numFound; i++){
            for(size_t c = i*nCols; c < (i+1)*nCols; c++){
                std::cout<<colAssignments[c]<<" ";
            }
            std::cout<<" with cost "<<assignmentCosts[i]<<std::endl;
        }
    }

    // compute assignment probabilities
    // each measurement gets a vector of probabilities, with a prob for each landmark +1 for nonassignment
    std::vector< std::vector< double > > probs(nM,std::vector<double>(nL+1,0));
    double bestCost = assignmentCosts[0]; //used as a normalization constant, for numerical overflow. 
    double total = 0;

    for(int sol = 0; sol < numFound; sol++){
        double assgnProb;
        if(bestCost+cutoff > assignmentCosts[sol]){
            assgnProb = std::exp(bestCost - assignmentCosts[sol]); 
        }else{
            continue;
        }
        total += assgnProb;
        for(size_t col = 0; col < nCols; col++){
            ptrdiff_t assigned_to = colAssignments[sol*nCols+col];
            // if(condition){
            //     assigned_to = rowIdx[assigned_to];
            // }
            if(assigned_to >= nL){
                // non assignment
                probs[col][nL] += assgnProb;
            }else{
                probs[col][assigned_to] += assgnProb;
            }
        }
    }

    //normalize by the total to obtain probabilities
    double norm = 1.0/total;
    for( size_t m = 0; m < nM; m++){
        for( size_t l = 0; l < nL+1; l++){
            probs[m][l] *= norm; 
        }
    }

    // if(condition){
    //     auto t2 = toc(t1);
    //     auto t1Non = tic();
    //     std::vector< std::vector< double > > probsNonCond = assignmentProb(meas,land,runConsts,false);
    //     auto t2Non = toc(t1Non);
    if(verbose){
        std::cout<<"Probs from Assignment: \n ------ \n";
        double err;
        for(size_t l = 0; l < nL+1; l++){
            for(size_t m = 0; m < nM; m++){
                std::cout<<probs[m][l]<<" ";
            }
            std::cout<<std::endl;
        }
    }
    //     // std::cout<<" ------ \n";
    //     // std::cout<<"Probs with conditioning: \n ------ \n";
    //     // for(size_t l = 0; l < nL+1; l++){
    //     //     for(size_t m = 0; m < nM; m++){
    //     //         std::cout<<probs[m][l]<<" ";
    //     //     }
    //     //     std::cout<<std::endl;
    //     // }
    //     // std::cout<<" ------ \n";
    //     std::cout<<"Orig Dimensions: "<<nRowsOrig<<" , "<<nCols<<std::endl;
    //     std::cout<<"Cond Dimensions: "<<nRows<<" , "<<nCols<<std::endl;
    //     std::cout<<"Biggest error: "<<err<<std::endl;
    //     std::cout<<"time w/o conditioning: "<<1000*t2Non<<"ms.\n";
    //     std::cout<<"time w/ conditioning: "<<1000*t2<<"ms.\n";
    //     std::cin.get();
    // }
    
    return probs;
}
//##################################################################################
std::vector< Eigen::Matrix<double,3,1> > getMeans(const std::vector<gtsam_quadrics::ConstrainedDualQuadric>& quads){
    std::vector< Eigen::Matrix<double,3,1> > means(quads.size());
    for(size_t q = 0; q < quads.size(); q++){
        //in case matrix is not normalised (it should be though, in general)
        means[q] = quads[q].centroid();
    }
    return means;
}
//##################################################################################
std::vector< Eigen::Matrix<double,3,3> > getCovs(const std::vector<gtsam_quadrics::ConstrainedDualQuadric>& quads){
    std::vector< Eigen::Matrix<double,3,3> > covs(quads.size());
    for(size_t q = 0; q < quads.size(); q++){
        Eigen::Matrix<double,4,4> Q = quads[q].matrix();
        covs[q] << Q(0,0)+std::pow(Q(0,3),2), Q(0,1)+Q(0,3)*Q(1,3), Q(0,2)+Q(0,3)*Q(2,3),
               Q(0,1)+Q(0,3)*Q(1,3), Q(1,1)+pow(Q(1,3),2), Q(1,2)+Q(1,3)*Q(2,3),
               Q(0,2)+Q(0,3)*Q(2,3), Q(1,2)+Q(1,3)*Q(2,3), Q(2,2)+pow(Q(2,3),2);
    }
    return covs;
}
//##################################################################################
std::vector<double> computeQuadricCostMatrix(const std::vector< Eigen::Matrix<double,3,1> >& m1, const std::vector< Eigen::Matrix<double,3,3> >& cov1,
    const std::vector<Eigen::Vector3d>& m2, const std::vector< Eigen::Matrix<double,3,3> >& cov2, const semConsts& runConsts){

    size_t nRows = m1.size()+m2.size(); //rows will be landmarks + non-assignment dummies
    size_t nCols = m2.size(); //columns will be measurements
    std::vector<double> costs(nRows*nCols,inf_d); //initialize all to inf, non-dummies 
    // will be overwritten with true distance, and only the dummy landmark for each column
    // is ovewritten with the GATE cost
    for(size_t col = 0; col < nCols; col++){
        for(size_t row = 0; row<m1.size(); row++){
            // squared mahalanobis distance w/ actual landmarks
            Eigen::Matrix<double,3,1> d = m1[row]-m2[col];
            costs[col*nRows+row] = (d.transpose())*((cov1[row]+cov2[col]).ldlt().solve(d));
        }
        costs[col*nRows+m1.size()+col] = runConsts.NONASSIGN_QUADRIC;
    }
    return costs;
}
//##################################################################################
std::vector< int > asgnBB(const std::vector< boundBox >& bbL,const std::vector< boundBox >& bbR,const semConsts& runConsts)
{
    // assignment method wants more rows than columns, let left boxes be the cols
    // and the right boxes be the rows
    size_t nL = bbL.size();
    size_t nR = bbR.size();
    if((nL==0) || (nR==0)){
        return std::vector<int>(nL,-1);
    }

    std::vector<double> costMatrix = computeBBCostMatrix(bbL,bbR,runConsts);
    size_t nRows = nR+nL;
    size_t nCols = nL;
    // max number of enumerated assignments
    // TODO: add cost delta cutoff
    int k = 1;
    ScratchSpace workMem;//Scratch space needed for the assignment algorithm.
    //Allocate scratch space, numRow numRow is on purpose! Row major.
    workMem.init(nRows,nRows);
    ptrdiff_t rowAssignments[nRows*k];
    ptrdiff_t colAssignments[nCols*k];
    double assignmentCosts[k];

    /*The assignment algorithm returns a nonzero value if no valid
     * solutions exist.*/
    bool maximize = true; //Maximize or minimize assignment
    int numFound = kBest2D(k,nRows,nCols,maximize,costMatrix.data(),workMem,rowAssignments,colAssignments,assignmentCosts);

    //visualize, if desired
    bool verbose = false;
    if(verbose){
        std::cout<<"Cost Matrix:\n";
        for(size_t r = 0; r < nRows; r++){
            for(size_t c = 0; c < nCols; c++){
                std::cout<<costMatrix[c*nRows+r]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<"Col Assignment: ";
        for(size_t c = 0; c < nCols; c++){
            std::cout<<colAssignments[c]<<" ";
        }
        std::cout<<"\n with cost: "<<assignmentCosts[0]<<std::endl;
    }
    std::vector<int> assignment(bbL.size(),-1);
    for(size_t c = 0; c < nCols; c++){
        if(colAssignments[c] < bbR.size()){
            assignment[c] = colAssignments[c];
        }
    }
    return assignment;
}
// ##################################################################################################3
std::vector<double> computeBBCostMatrix(const std::vector< boundBox >& bbL, const std::vector< boundBox >& bbR, const semConsts& runConsts)
{
    size_t nRows = bbR.size()+bbL.size(); //rows will be bbR + non-assignment dummies
    size_t nCols = bbL.size(); //columns will be bb2
    std::vector<double> costs(nRows*nCols,-inf_d); //initialize all to inf, non-dummies 
    // will be overwritten with true distance, and only the dummy landmark for each column
    // is ovewritten with the GATE cost
    for(size_t col = 0; col < nCols; col++){
        for(size_t row = 0; row<bbR.size(); row++){
            // squared mahalanobis distance w/ actual landmarks
            double iou1 = bbR[row].IoU(bbL[col]);
            double iou2 = bbL[col].IoU(bbR[row]);
            costs[col*nRows+row] = std::min(iou1,iou2); 
        }
        //col*nRows brings you to the correct column
        // +bbR.size() brings you past the rectangle (bbR.size(),bbL.size()) of actual costs
        // +col brings you to the correct row of the dummy costs
        costs[col*nRows+bbR.size()+col] = runConsts.NONASSIGN_BOUNDBOX;
    }
    return costs;
}
//########################################################################################
void saveAssignmentProb(const std::vector< gtsam_quadrics::ConstrainedDualQuadric >& meas,
                        const std::vector< gtsam_quadrics::ConstrainedDualQuadric >& land,
                        const semConsts& runConsts,std::string savePath){

    std::vector< Eigen::Matrix<double,3,1> > measMeans = getMeans(meas);
    std::vector< Eigen::Matrix<double,3,3> > measCovs = getCovs(meas);
    std::vector< Eigen::Matrix<double,3,1> > landMeans = getMeans(land);
    std::vector< Eigen::Matrix<double,3,3> > landCovs = getCovs(land);

    // assignment method wants more rows than columns, landmarks will always be bigger
    // thanks to fake non-assignment landmarks introduced
    size_t nL = land.size();
    size_t nM = meas.size();

    if((nM == 0) || (nL == 0)){
        return;
    }

    std::vector<double> costMatrix = computeQuadricCostMatrix(landMeans,landCovs,measMeans,measCovs,runConsts);
    size_t nRows = nL+nM;
    size_t nCols = nM;

    std::ofstream myfile(savePath);
    for(size_t r = 0; r < nRows; r++){
        for(size_t c = 0; c < nCols; c++){
            myfile << std::to_string(costMatrix[c*nRows+r]);
            if(c < nCols-1){
                myfile << ",";
            }
        }
        myfile<<std::endl;
    }
    myfile.close();
    return;
}
//##################################################################################
std::vector< std::vector<double> > bruteForceProb(const std::vector< double >& costMatrix, size_t nL, size_t nM){
    
    size_t nRows = nL+nM;
    size_t nCols = nM;

    if(nM == 1){
        //only one column. Normalize by sum of the column, and return it!
        // std::vector<double> costs = costMatrix;
        std::vector< std::vector<double> > probs(1,std::vector<double>(costMatrix.size(),0));
        double norm = 0;
        for(size_t i = 0; i <= nL; i++){
            if(costMatrix[i]<cutoff){
                probs[0][i] = std::exp(-costMatrix[i]);
                norm += probs[0][i];
            }
        }
        // probs[0] = costMatrix;
        // toProbs(costs);
        norm = 1.0/norm;
        std::transform(probs[0].begin(), probs[0].end(), probs[0].begin(), [norm](double &c){ return c*norm; });
        return probs;
    }

    double mincBound = mincConstant(nRows,nCols);
    for(size_t r = 0; r < nRows; r++){
        size_t rowCard = 1;
        for(size_t c = 0; c < nCols; c++){
            if(costMatrix[c*nRows+r] < inf_d){
                rowCard++;
            }
        }
        mincBound *= mincFactor(rowCard);
    }
    size_t upperK = std::min(static_cast<size_t>(mincBound)+1,static_cast<size_t>(20000)); //+1 if it rounds down, +1 to be greater than that
    
    ScratchSpace workMem;//Scratch space needed for the assignment algorithm.
    //Allocate scratch space, numRow numRow is on purpose! Row major.
    workMem.init(nRows,nRows);
    ptrdiff_t rowAssignments[nRows*upperK];
    ptrdiff_t colAssignments[nCols*upperK];
    double assignmentCosts[upperK];

    /*The assignment algorithm returns a nonzero value if no valid
     * solutions exist.*/
    bool maximize = false; //Maximize or minimize assignment
    int numFound = kBest2D(upperK,nRows,nCols,maximize,costMatrix.data(),workMem,rowAssignments,colAssignments,assignmentCosts);
   
    bool verbose = false;

    // if(numFound == upperK){
    //     std::cout<<"\n\n Found the upper limit of assignments possible?! Seems not right....\n\n";
    //     std::cout<<"Minc Bound: "<<mincBound<<std::endl;
    //     std::cout<<"Upper bound: "<<upperK<<std::endl;
    //     std::cout<<"Found: "<<numFound<<std::endl;
    //     // verbose = false;
    // }

    if(verbose){
        std::cout<<"Cost Matrix In assignment:\n";
        for(size_t r = 0; r < nRows; r++){
            for(size_t c = 0; c < nCols; c++){
                std::cout<<costMatrix[c*nRows+r]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<"Assignments: \n";
        for(size_t i = 0; i < numFound; i++){
            for(size_t c = i*nCols; c < (i+1)*nCols; c++){
                std::cout<<colAssignments[c]<<" ";
            }
            std::cout<<" with cost "<<assignmentCosts[i]<<std::endl;
        }
        std::cin.get();
    }

    // compute assignment probabilities
    // each measurement gets a vector of probabilities, with a prob for each landmark +1 for nonassignment
    std::vector< std::vector< double > > probs(nM,std::vector<double>(nL+1,0));
    double bestCost = assignmentCosts[0]; //used as a normalization constant, for numerical overflow. 
    double total = 0;

    for(int sol = 0; sol < numFound; sol++){
        double assgnProb;
        //if(bestCost+cutoff > assignmentCosts[sol]){
        //this is probably nonsense, but I'll leave it, as a nod towards.... digits
        assgnProb = std::exp(bestCost - assignmentCosts[sol]); 
        //}else{
        //   continue;
        //}
        total += assgnProb;
        for(size_t col = 0; col < nCols; col++){
            ptrdiff_t assigned_to = colAssignments[sol*nCols+col];
            // if(condition){
            //     assigned_to = rowIdx[assigned_to];
            // }
            if(assigned_to >= nL){
                // non assignment
                probs[col][nL] += assgnProb;
            }else{
                probs[col][assigned_to] += assgnProb;
            }
        }
    }

    //normalize by the total to obtain probabilities
    double norm = 1.0/total;
    for( size_t m = 0; m < nM; m++){
        for( size_t l = 0; l < nL+1; l++){
            probs[m][l] *= norm; 
        }
    }

    // if(condition){
    //     auto t2 = toc(t1);
    //     auto t1Non = tic();
    //     std::vector< std::vector< double > > probsNonCond = assignmentProb(meas,land,runConsts,false);
    //     auto t2Non = toc(t1Non);
    if(verbose){
        std::cout<<"Probs from Assignment: \n ------ \n";
        double err;
        for(size_t l = 0; l < nL+1; l++){
            for(size_t m = 0; m < nM; m++){
                std::cout<<probs[m][l]<<" ";
            }
            std::cout<<std::endl;
        }
    }
    
    return probs;
}