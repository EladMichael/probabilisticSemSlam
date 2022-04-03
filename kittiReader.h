#ifndef poseGraph_kittiReader
#define poseGraph_kittiReader

#include <vector>
#include <string>
#include <gtsam/geometry/Pose3.h>
#include <eigen3/Eigen/Dense>
#include <gtsam/nonlinear/Values.h>
#include <matplot/matplot.h>


class kittiReader {
public:
	int seqN;
	std::string seqSpec;
	std::string pathToData;
	std::string pathToSeq;
	std::string pathToPose;
	std::string pathToResults;

	matplot::figure_handle f; //for live plotting
	matplot::axes_handle ax; //for live plotting
	// bool exists; //this sequence might not exist!
	// image calibration parameters
	// not going to store as matrix, 
	// to keep this as simple as possible
	bool color;
	double fx,fy,s,u0,v0,b;
	Eigen::Matrix<double,3,3> K;

	std::vector<double> gtx;
	std::vector<double> gtz;

	std::vector<double> trajx;
	std::vector<double> trajz;

	kittiReader(int seq,bool Color=true);

	void loadCalib();

	double initL,initR;

	void writePoses(const std::vector<gtsam::Pose3>& x,std::string ID);

	void writeMetrics(const std::vector<double>& time4BBs,const std::vector<double>& time4Opt,
		const std::vector<double>& time4ProbWin,const std::vector<double>& slidingFPS,
		const std::vector<int>& numMeas,const std::vector<int>& numLand,std::string ID);
	
	void showMetric(const std::vector<double>& metric, const std::string& name);

	void updateTrajectory(const gtsam::Values& estimate);

	gtsam::Pose3 groundTruth(int poseN);

	std::vector< gtsam::Pose3 > groundTruthSet();
};

#endif