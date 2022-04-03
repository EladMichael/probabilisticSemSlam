#ifndef sensSLAM_assignment
#define sensSLAM_assignment

#include "boundBox.h"
#include "constsUtils.h"

#include <vector>
#include <Eigen/Dense>
#include <gtsam_quadrics/geometry/ConstrainedDualQuadric.h>

std::vector< std::vector<double> > assignmentProb(const std::vector< double >& costMatrix, size_t nL, size_t nM,size_t k);

std::vector< std::vector<double> > permanentProb(std::vector< double > costMatrix, size_t nL, size_t nM, int permOpt);

void setupAssgnMatrix(Eigen::MatrixXd& subProbs, const Eigen::MatrixXd& elProbs, size_t col);

double conditionedPermanent(const Eigen::MatrixXd& A,int permOpt);

void toProbs(std::vector<double>& costMatrix);

std::vector< int > asgnBB(const std::vector< boundBox >& bbL,const std::vector< boundBox >& bbR, const semConsts& runConsts);

std::vector< Eigen::Matrix<double,3,1> > getMeans(const std::vector<gtsam_quadrics::ConstrainedDualQuadric>& quads);
std::vector< Eigen::Matrix<double,3,3> > getCovs(const std::vector<gtsam_quadrics::ConstrainedDualQuadric>& quads);

std::vector<double> conditionCosts(const std::vector<double>& costs, size_t nRows, size_t nCols, std::vector<ptrdiff_t>& rowIdxOut);

std::vector<double> computeQuadricCostMatrix(const std::vector<Eigen::Vector3d>& m1, const std::vector< Eigen::Matrix<double,3,3> >& cov1,
    const std::vector<Eigen::Vector3d>& m2, const std::vector< Eigen::Matrix<double,3,3> >& cov2, const semConsts& runConsts);

std::vector<double> computeBBCostMatrix(const std::vector< boundBox >& bbL, const std::vector< boundBox >& bbR, const semConsts& runConsts);

void saveAssignmentProb(const std::vector< gtsam_quadrics::ConstrainedDualQuadric >& meas,
                        const std::vector< gtsam_quadrics::ConstrainedDualQuadric >& land,
                        const semConsts& runConsts,std::string savePath);

std::vector< std::vector<double> > getAssignmentProbs(const std::vector< gtsam_quadrics::ConstrainedDualQuadric >& meas,
                                            const std::vector< gtsam_quadrics::ConstrainedDualQuadric >& land,
                                            const semConsts& runConsts);

double permWAssignments(const Eigen::MatrixXd& A);

std::vector< std::vector<double> > bruteForceProb(const std::vector< double >& costMatrix, size_t nL, size_t nM);

#endif