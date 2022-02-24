
#include <random>
#include <iostream>

#include "nwPerm.h"

/**
 * Functions for computing the permanent of nonnegative matrices.
 * 
 * Approximate algorithm and original matlab code by Mark Huber, www.math.duke.edu/~mhuber
 * Applies only to nonnegative matrices! Translated to C++ by Sean Bowman
 * 
 *
 * Exact algorithm due to Nijenhuis and Wilf and implemented in C++ by Nikolay Atanasov and Brian Butler
 */



double permanentFastest(const Eigen::MatrixXd& A) {
	// for small matrices the exact method (Nijenhuis and Wilf) is faster
	// for large matrices an approximation (Huber) is faster
	// point where they switch seems to be about matrices of dimension 20-ish...
	int switch_dimension = 20;
	int this_dim = std::max(A.rows(), A.cols());

	// std::cout << "COMPUTING PERMANENT DIMENSION = " << this_dim << std::endl;

	if (this_dim <= switch_dimension) {
		return permanentExact(A);
	} else {
		return permanentApproximation(A, 300);
	}
}


Eigen::MatrixXd sinkhorn(const Eigen::MatrixXd& A, double epsilon, double * prodx, double * prody) {
	size_t n = A.cols();
	Eigen::VectorXd x = Eigen::VectorXd::Ones(n);
	Eigen::VectorXd y = x;

	Eigen::MatrixXd B = A;

	// Eigen::VectorXd row_sum = B.rowwise().sum();
	// Eigen::VectorXd col_sum;
	// while ( (row_sum.array() - 1).abs().matrix().maxCoeff() > epsilon ) {
	// 	x.array() *= row_sum.array().inverse();
	// 	B = row_sum.array().inverse().matrix().asDiagonal() * B;
	// 	col_sum = B.colwise().sum();
	// 	y.array() *= col_sum.array().inverse();
	// 	B = B * col_sum.array().inverse().matrix().asDiagonal();
	// 	row_sum = B.rowwise().sum();
	// }

	Eigen::VectorXd c = 1.0 / B.colwise().sum().array();
	Eigen::VectorXd r = 1.0 / (B * c).array();

	int max_iter = 100000;
	int iter = 0;
	while (iter < max_iter) {
		iter++;

		Eigen::VectorXd cinv = r.transpose() * B;
		if ( (cinv.array() * c.array() - 1).abs().maxCoeff() <= epsilon ) {
			break;
		}

		c = 1.0 / cinv.array();
		r = 1.0 / (B * c).array();
	}

	B = B.array() * (r * c.transpose()).array();

	*prodx = r.prod();
	*prody = c.prod();

	return B;
}


Eigen::VectorXd hl_factor(const Eigen::VectorXd& x) {
	// Eigen::VectorXd y = (x.array() == 0).select(0.5, x); // so not taking log of 0

	// Eigen::VectorXd hl = (y.array() > 1).select(y.array() + 0.5*y.array().log() + std::exp(1) - 1, 
	// 											1 + (std::exp(1) - 1)*y.array());

	Eigen::VectorXd hl(x.size());

	for (int i = 0; i < x.size(); ++i) {
		if (x(i) > 1) {
			hl(i) = x(i) + 0.5*std::log(x(i)) + 2.71828182846 - 1;
		} else {
			hl(i) = 1 + (2.71828182846-1)*x(i);
		}
	}

	return hl;
}


//static std::random_device random_generator_permanent;
//static std::mt19937 random_permanent_mt(random_generator_permanent());

double rand01() {
	//std::uniform_real_distribution<> distribution(0.0, 1.0);
	//return distribution(random_permanent_mt);
	return (double)rand() / (double)RAND_MAX;
}

size_t pickRowFromProbs(const Eigen::VectorXd& probs) {
	double unifrnd = rand01();

	double sum = 0;
	int i;
	for (i = 0; i < probs.size(); ++i) {
		sum += probs(i);
		if (sum >= unifrnd) break;
	}

	return i;
}


/*
 * Applies only to nonnegative matrices!
 */
double permanentApproximation(const Eigen::MatrixXd& A, size_t iterations) {
	if (A.cols() == A.rows()) return permanentApproximationSquare(A, iterations);

	int m = A.rows(); 
	int n = A.cols();

	double scale = std::tgamma( std::abs(m - n) + 1 );
	size_t dim = std::max(m, n);

	Eigen::MatrixXd A_pad = Eigen::MatrixXd::Ones(dim, dim);

	A_pad.block(0, 0, m, n) = A;

	return permanentApproximationSquare(A_pad, iterations) / scale;
}


/*
 * Applies only to nonnegative matrices!
 */
double permanentApproximationSquare(const Eigen::MatrixXd& A, size_t iterations) {
	int n = A.cols();

	double x,y;
	Eigen::MatrixXd B = sinkhorn(A, .0001, &x, &y);

	Eigen::VectorXd row_scale = B.rowwise().maxCoeff().array().inverse();
	Eigen::MatrixXd C = row_scale.asDiagonal() * B;

	Eigen::MatrixXd C_orig = C;

	size_t number_successes = 0;

	Eigen::VectorXd row_sums;

	static constexpr double EE = 2.71828182846;

	for (size_t i = 0; i < iterations; ++i) {
		int column = 0;
		C = C_orig;
		row_sums = C_orig.rowwise().sum();
		while (column < n) {
			// find hl upper bound on permanent of C
			Eigen::VectorXd h = hl_factor(row_sums);
			double hl = (h.array() / EE).matrix().prod();
			
			// same thing but remove column under consideration
			Eigen::VectorXd h2 = hl_factor(row_sums - C.col(column));
			double hl2 = (h2.array() / EE).matrix().prod();

			// find probability of each row being selected
			Eigen::VectorXd row_probs = (hl2/hl)*EE * (C.col(column).array() / h2.array());

			// select random row to fill out the permutation
			int row_pick = pickRowFromProbs(row_probs);
      
			if (row_pick >= n) {
				// failed to get a permutation
				column = n + 1;
			} else {
				// remove current column from future consideration
				row_sums = (row_sums.array() - C.col(column).array()).matrix();
				// remove row from future consideration
				C.row(row_pick).setZero();
				column++;
				row_sums(row_pick) = 0;
			}
		}

		if (column == n) { //success
			number_successes++;
		}
	}

  //std::cout << "Number of successes: " << number_successes << std::endl;

	C = C_orig;
	row_sums = C.rowwise().sum();
	double hl_C = (hl_factor(row_sums).array() / EE).matrix().prod();
	double per_estimate = hl_C * number_successes / iterations;
	per_estimate = per_estimate / row_scale.prod() / x / y;

	//std::cout << "Estimated permanent: " << per_estimate << std::endl;

	return per_estimate;
}





double permanentExact(const Eigen::MatrixXd& A) {
	if (A.cols() == A.rows()) return permanentExactSquare(A);

	int m = A.rows(); 
	int n = A.cols();

	double scale = std::tgamma( std::abs(m - n) + 1 );
	size_t dim = std::max(m, n);

	Eigen::MatrixXd A_pad = Eigen::MatrixXd::Ones(dim, dim);

	A_pad.block(0, 0, m, n) = A;

	return permanentExactSquare(A_pad) / scale;
}

double permanentExactNx(const Eigen::MatrixXd& A) {
	if (A.cols() == A.rows()) return permanentExactNxSquare(A);

	int m = A.rows(); 
	int n = A.cols();

	double scale = std::tgamma( std::abs(m - n) + 1 );
	size_t dim = std::max(m, n);

	Eigen::MatrixXd A_pad = Eigen::MatrixXd::Ones(dim, dim);

	A_pad.block(0, 0, m, n) = A;

	return permanentExactNxSquare(A_pad) / scale;
}



double permanentExactSquare(const Eigen::MatrixXd& A) {
	double *pp;

	int m = A.rows();
	// int n = A.cols();

	double permanent = 0;
   
	pp = &permanent;

  if (m == 0)
	{
		*pp = 1.0;  // 1 by definition.
	}
  else if (m <= 32) // 1 <= m <= 32 (tested up to m=32, and fails after that)
	{
    const double *a;   // pointer to input matrix data
    double x[32];// temporary vector as used by Nijenhuis and Wilf
    double rs;   // row sum of matrix
    double s;    // +1 or -1
    double prod; // product of the elements in vector 'x'
    double p=1.0;  // many results accumulate here, MAY need extra precision
		double *xptr;
		const double *aptr; 
    int j, k;
    unsigned long int i, tn11 = (1UL<<(m-1))-1;  // tn11 = 2^(n-1)-1
    unsigned long int gray, prevgray=0, two_to_k;
              
    a = A.data();

		xptr = (double *)x;
		aptr = &a[(m-1)*m];
    for (j=0; j<m; j++)
    {
      rs = 0.0;
      for (k=0; k<m; k++)
        rs += a[j + k*m];    // sum of row j
      //x[j] = a[j + (n-1)*m] - rs/2;  // see Nijenhuis and Wilf
      *xptr = *aptr++ - rs/2;  // see Nijenhuis and Wilf
      //p *= x[j];   // product of the elements in vector 'x'            
      p *= *xptr++;   // product of the elements in vector 'x'
    }

    for (i=1; i<=tn11; i++)
    {
      gray=i^(i>>1); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
      //mexPrintf("i=%5lu GRAY: 0x%04lx 0x%04lx 0x%04lx  ",i,gray,prevgray,gray^prevgray);
      
      two_to_k=1;    // two_to_k = 2 raised to the k power (2^k)
      k=0;
      while (two_to_k < (gray^prevgray))
      {
          two_to_k<<=1;  // two_to_k is a bitmask to find location of 1
          k++;
      }
      s = (two_to_k & gray) ? +1.0 : -1.0;
      prevgray = gray;        
      //mexPrintf("k=%2d s=%5.1lf\n",k,s);
      
      prod = 1.0;
      xptr = (double *)x;
      aptr = &a[k*m];
      for (j=0; j<m; j++)
      {
        // Two equivalent versions: vector entry addressing or pointers (faster!)
        //x[j] += s * a[j + k*m];  // see Nijenhuis and Wilf
        *xptr += s * *aptr++;  // see Nijenhuis and Wilf
        //prod *= x[j];  // product of the elements in vector 'x'                
        prod *= *xptr++;  // product of the elements in vector 'x'
      }
      // Keep the summing below in the loop, moving it loses important resolution on x87
      p += ((i&1)? -1.0:1.0) * prod; 
    }
        
		*pp = (double)(4*(m&1)-2) * (double)p;        
	}
  else
  {
    throw std::runtime_error("Maximum matrix dimension limited to 32. Error inside permanentExactSquare().");
  }
  return permanent;
}




double permanentExactNxSquare(const Eigen::MatrixXd& A)
{
  int n = A.cols();
  const double *a = A.data(); 
  double p = -1.0;
  std::vector<double> x(n);
  for(int i = 0; i < n; ++i)
  {
    x[i] = a[i*n+n-1];
    for(int j = 0; j < n; ++j)
      x[i] -= a[i*n+j]/2.0;
    p = p * x[i];
  }

  double s = -1.0;
  std::vector<int> g(n,0);
  int two_n1 = (1<<(n-1));
  int j = 0;
  for(int k = 2; k <= two_n1; ++k)
  {
    if( k % 2 ){
      j = 1;
      for(int m = 0; m<n; ++m)
        if(g[m] == 0)
          j += 1;
        else
          break;
    }else
      j = 0;
    
    int z = 1-2*g[j];
    g[j] = 1-g[j];
    s = -s;
    double t = s;
    for(int i = 0; i < n; ++i)
    {
      x[i] += z*a[i*n+j];
      t = t*x[i];
    }
    p += t;
  }

  if( n % 2 )
    return -2*p;
  else
    return 2*p;
}


long double permanentExactLong(const Eigen::MatrixXd& A) {
    if (A.cols() == A.rows()) return permanentExactSquare(A);

    int m = A.rows(); 
    int n = A.cols();

    long double scale = std::tgamma( std::abs(m - n) + 1 );
    size_t dim = std::max(m, n);

    Eigen::MatrixXd A_pad = Eigen::MatrixXd::Ones(dim, dim);

    A_pad.block(0, 0, m, n) = A;

    return permanentExactSquare(A_pad) / scale;
}

long double permanentExactSquareLong(const Eigen::MatrixXd& A) {
    long double *pp;

    int m = A.rows();
    // int n = A.cols();

    long double permanent = 0;
   
    pp = &permanent;

  if (m == 0)
    {
        *pp = 1.0;  // 1 by definition.
    }
  else if (m <= 32) // 1 <= m <= 32 (tested up to m=32, and fails after that)
    {
    const double *a;   // pointer to input matrix data
    long double x[32];// temporary vector as used by Nijenhuis and Wilf
    long double rs;   // row sum of matrix
    long double s;    // +1 or -1
    long double prod; // product of the elements in vector 'x'
    long double p=1.0;  // many results accumulate here, MAY need extra precision
        long double *xptr;
        const double *aptr; 
    int j, k;
    unsigned long int i, tn11 = (1UL<<(m-1))-1;  // tn11 = 2^(n-1)-1
    unsigned long int gray, prevgray=0, two_to_k;
              
    a = A.data();

        xptr = (long double *)x;
        aptr = &a[(m-1)*m];
    for (j=0; j<m; j++)
    {
      rs = 0.0;
      for (k=0; k<m; k++)
        rs += a[j + k*m];    // sum of row j
      //x[j] = a[j + (n-1)*m] - rs/2;  // see Nijenhuis and Wilf
      *xptr = *aptr++ - rs/2;  // see Nijenhuis and Wilf
      //p *= x[j];   // product of the elements in vector 'x'            
      p *= *xptr++;   // product of the elements in vector 'x'
    }

    for (i=1; i<=tn11; i++)
    {
      gray=i^(i>>1); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
      //mexPrintf("i=%5lu GRAY: 0x%04lx 0x%04lx 0x%04lx  ",i,gray,prevgray,gray^prevgray);
      
      two_to_k=1;    // two_to_k = 2 raised to the k power (2^k)
      k=0;
      while (two_to_k < (gray^prevgray))
      {
          two_to_k<<=1;  // two_to_k is a bitmask to find location of 1
          k++;
      }
      s = (two_to_k & gray) ? +1.0 : -1.0;
      prevgray = gray;        
      //mexPrintf("k=%2d s=%5.1lf\n",k,s);
      
      prod = 1.0;
      xptr = (long double *)x;
      aptr = &a[k*m];
      for (j=0; j<m; j++)
      {
        // Two equivalent versions: vector entry addressing or pointers (faster!)
        //x[j] += s * a[j + k*m];  // see Nijenhuis and Wilf
        *xptr += s * *aptr++;  // see Nijenhuis and Wilf
        //prod *= x[j];  // product of the elements in vector 'x'                
        prod *= *xptr++;  // product of the elements in vector 'x'
      }
      // Keep the summing below in the loop, moving it loses important resolution on x87
      p += ((i&1)? -1.0:1.0) * prod; 
    }
        
        *pp = (long double)(4*(m&1)-2) * (long double)p;        
    }
  else
  {
    throw std::runtime_error("Maximum matrix dimension limited to 32. Error inside permanentExactSquare().");
  }
  return permanent;
}