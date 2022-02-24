#ifndef semSLAM_constsUtils
#define semSLAM_constsUtils

#include <string>
#include <chrono>
#include <sys/stat.h>
#include <fstream>
#include <gtsam/linear/NoiseModel.h>

#define inf_d std::numeric_limits<double>::infinity()
#define inf_i std::numeric_limits<int>::infinity()

inline bool file_exists(const std::string& name) {
	struct stat buffer;   
	return (stat (name.c_str(), &buffer) == 0); 
}

inline std::chrono::high_resolution_clock::time_point tic()
{ return std::chrono::high_resolution_clock::now(); }
inline double toc( const std::chrono::high_resolution_clock::time_point& t2)
{ return std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t2).count(); }


struct semConsts{

	double ODOM_ROT_SD_GEN;
	double ODOM_T_SD_GEN;
	double ODOM_ROT_SD_SENSE;
	double ODOM_T_SD_SENSE;
	double LAND_ROT_SD;
	double LAND_RAD_SD;
	double LAND_T_SD;
	double BOX_SD;
	double STEREO_SD; 
	//not much wiggle in the stereo factors please
	//this is punishment for not having a stereo factor

	// do not initialise a new landmark unless an observation is more than 1/3 assigned to it
	double NEW_LANDMARK_THRESH;
	// do not initialise a new bounding box factor unless it is assigned by more than 5%
	double NEW_FACTOR_PROB_THRESH;

	//gate cost (non-assignment cost) for estimated to to landmark quadrics (mahalanobis distance)
	double NONASSIGN_QUADRIC; //-ln(-1e+09)
	//gate profit (non-assignment profit) for overlapping bounding boxes (intersection over union)
	double NONASSIGN_BOUNDBOX;

	size_t optWin;
	size_t probWin;
	size_t k;
	bool usePerm;
	size_t netChoice;
	int landmark_age_thresh;

	boost::shared_ptr<gtsam::noiseModel::Diagonal> odomNoise;
	boost::shared_ptr<gtsam::noiseModel::Diagonal> stereoNoise;
	boost::shared_ptr<gtsam::noiseModel::Diagonal> landmarkPriorNoise;
	//return a scaled noise vector
	inline gtsam::noiseModel::Diagonal::shared_ptr bNoise(double prob=1)const {
		return gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector4::Ones()*BOX_SD/prob);
	}

	semConsts(std::string fileName){

		if(!file_exists(fileName)){
				std::string errMsg("Settings file could not be found: \n");
				errMsg += fileName;
				throw std::runtime_error(errMsg);
		}

		//default values, in case none are specified
		ODOM_ROT_SD_SENSE = 0.01;
		ODOM_T_SD_SENSE = 2.0;
		ODOM_ROT_SD_GEN = 0.001;
		ODOM_T_SD_GEN = 1.0;
		LAND_ROT_SD = 0.1;
		LAND_T_SD = 1;
		LAND_RAD_SD = 2;
		BOX_SD = 3;
		STEREO_SD = 0.0001;
		NEW_LANDMARK_THRESH = 0.33;
		NEW_FACTOR_PROB_THRESH = 0.05;
		NONASSIGN_QUADRIC = 55.0;
		NONASSIGN_BOUNDBOX = 0.20;
		optWin = 15;
		probWin = 15;
		k = 100;
		usePerm = 0;
		netChoice = 1;
		landmark_age_thresh = 2;

		std::ifstream myFile(fileName);
		std::string line;
		std::string valString;
		double val;

		while(std::getline(myFile,line,' ')){
			
			std::getline(myFile,valString);
			val = std::stod(valString);
			//unfortunately, can't use swithc statmenet with strings! Could
			// probably use hash values. Or you could suck it up.
			if(line == "ODOM_ROT_SD_SENSE"){
				ODOM_ROT_SD_SENSE = val;
			}else if(line == "ODOM_T_SD_SENSE"){
				ODOM_T_SD_SENSE = val;
			}else if(line == "ODOM_T_SD_GEN"){
				ODOM_T_SD_GEN = val;
			}else if(line == "ODOM_ROT_SD_GEN"){
				ODOM_ROT_SD_GEN = val;
			}else if(line == "BOX_SD"){
				BOX_SD = val;
			}else if(line == "STEREO_SD"){
				STEREO_SD  = val;
			}else if(line == "NEW_LANDMARK_THRESH"){
				NEW_LANDMARK_THRESH = val;
			}else if(line == "NEW_FACTOR_PROB_THRESH"){
				NEW_FACTOR_PROB_THRESH = val;
			}else if(line == "NONASSIGN_QUADRIC"){
				NONASSIGN_QUADRIC = val;
			}else if(line == "NONASSIGN_BOUNDBOX"){
				NONASSIGN_BOUNDBOX = val;
			}else if(line == "optWin"){
				optWin = val;
			}else if(line == "probWin"){
				probWin = val;
			}else if(line == "k"){
				k = val;
			}else if(line == "usePerm"){
				usePerm = val>0;
			}else if(line == "netChoice"){
				netChoice = val;
			}else if(line == "landmark_age_thresh"){
				landmark_age_thresh = val;
			}else if(line == "LAND_ROT_SD"){
				LAND_ROT_SD = val;
			}else if(line == "LAND_RAD_SD"){
				LAND_RAD_SD = val;
			}else if(line == "LAND_T_SD"){
				LAND_T_SD = val;
			}else{
				throw std::runtime_error(std::string("Unknown setting: ")+line);
			}
		}
		
		odomNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6(ODOM_ROT_SD_SENSE,
						ODOM_ROT_SD_SENSE,ODOM_ROT_SD_SENSE,ODOM_T_SD_SENSE,ODOM_T_SD_SENSE,ODOM_T_SD_SENSE));

		stereoNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Ones()*STEREO_SD);
		
		landmarkPriorNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector9(LAND_ROT_SD,
			LAND_ROT_SD,LAND_ROT_SD,LAND_T_SD,LAND_T_SD,LAND_T_SD,LAND_RAD_SD,LAND_RAD_SD,LAND_RAD_SD));
	}
};



#endif