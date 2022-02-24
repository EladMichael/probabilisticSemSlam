#ifndef sensSLAM_perm
#define sensSLAM_perm

#include <eigen3/Eigen/Core>

/**
 * Functions for computing the permanent of nonnegative matrices.
 * 
 * Approximate algorithm and original matlab code by Mark Huber, www.math.duke.edu/~mhuber
 * Applies only to nonnegative matrices! Translated to C++ by Sean Bowman
 * 
 *
 * Exact algorithm due to Nijenhuis and Wilf and implemented in C++ by Nikolay Atanasov and Brian Butler
 */


// Approximate algorithm
double permanentApproximation(const Eigen::MatrixXd& A, size_t iterations);
double permanentApproximationSquare(const Eigen::MatrixXd& A, size_t iterations);

// Two different implementations of the exact algorithm
double permanentExact(const Eigen::MatrixXd& A);
long double permanentExactLong(const Eigen::MatrixXd& A);
double permanentExactSquare(const Eigen::MatrixXd& A);
long double permanentExactSquareLong(const Eigen::MatrixXd& A);

double permanentExactNx(const Eigen::MatrixXd& A);
double permanentExactNxSquare(const Eigen::MatrixXd& A);

// Automatic selection between the approximate and exact algorithms depending on the matrix size
double permanentFastest(const Eigen::MatrixXd& A);

// Helper functions for the approximate algorithm
Eigen::MatrixXd sinkhorn(const Eigen::MatrixXd& A, double epsilon, double * prodx, double * prody);
Eigen::VectorXd hl_factor(const Eigen::VectorXd& x);


#endif

