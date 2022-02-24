#include "bbNet.h"

#include <fstream>

// #####################################################################
void bbNet::detectSSD(const cv::Mat& imgL, std::vector<boundBox>& bbL,
  const cv::Mat& imgR, std::vector<boundBox>& bbR){

    cv::Mat blob = singleImgBlob(imgL);

    net.setInput(blob);

    cv::Mat output = net.forward();
    // The data in det is only pointed at 
    // if you do not empty this now, it'll be rewritten
    // when doing detection on the next image!
    cv::Mat det = cv::Mat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    // extract boxes with confidence > confThresh
    // and class id < 16 (which is the vaguely)
    // car/outdoor/traffic related ones
    for(int row = 0; row < det.rows; row++){
        if((det.at<float>(row,2) > confThresh) && (det.at<float>(row,1) < 16)){
            bbL.push_back(boundBox(row,imgL,det));
            bbL[bbL.size()-1].class_id += -1;
        }
    }

    blob = singleImgBlob(imgR);
    net.setInput(blob);

    output = net.forward();
    // The data in det is only pointed at 
    // if you do not empty this now, it'll be rewritten
    // when doing detection on the next image!
    det = cv::Mat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    // extract boxes with confidence > confThresh
    // and class id < 16 (which is the vaguely)
    // car/outdoor/traffic related ones
    for(int row = 0; row < det.rows; row++){
        if((det.at<float>(row,2) > confThresh) && (det.at<float>(row,1) < 16)){
            bbR.push_back(boundBox(row,imgR,det));
            bbR[bbR.size()-1].class_id += -1;
        }
    }
    return;
}

// #####################################################################
void bbNet::detectYOLO(const cv::Mat& imgL, std::vector<boundBox>& bbL,
  const cv::Mat& imgR, std::vector<boundBox>& bbR){
    
    cv::Mat blob = stackImgBlob(imgL,imgR);

    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    //the image was stacked, so it was col x col
    //then it is shrunk to 320 x 320 for detection
    float scale_factor = imgL.cols / 320.0;
    
    //I fucking hate this raw pointer shit, but this all works
    // and is all tied up in how the network spits out its answers
    // so I am leaving it! This is terribly unsafe. If anything changes
    // it'll just crash constantly. Don't do this! Change the resolution?
    // crash. Change the network to the new version? Crash. 
    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    //1/4 resolution of 25200 (it's a native 640 network, run on 320 imgs)
    const int rows = 6300; 
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        if (confidence >= confThresh) {

            float * classes_scores = data + 5;
            cv::Mat scores(1, labels.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > scoreThresh) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);
                
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * scale_factor);
                int top = int((y - 0.5 * h) * scale_factor);
                int width = int(w * scale_factor);
                int height = int(h * scale_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, scoreThresh, nmsThresh, nms_result);
    
    int rowsL = imgL.rows;
    int gap = imgL.cols - (rowsL+imgR.rows);
    int rOffset = rowsL+gap;

    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];

        float conf = confidences[idx];
        int x0 = boxes[idx].x;
        int y0 = boxes[idx].y;
        int x1 = boxes[idx].x + boxes[idx].width;
        int y1 = boxes[idx].y + boxes[idx].height;
        int class_ID = class_ids[idx];

        //int ID = i; //for associating bounding boxes
        
        if(y1 < rOffset){
            //if the bottom of the box (y1) is above the second image,
            // add this box to bbL but don't let the box extend into the gap
            y1 = std::min(y1,rowsL);
            bbL.push_back(boundBox(conf,x0,y0,x1,y1,class_ID,i));
        }else if(y0 > rowsL){
            //if the top of the box (y0) is below the first image,
            // add this box to bbR but don't let the box extend into the gap
            y0 = std::max(y0,rOffset);
            bbR.push_back(boundBox(conf,x0,y0-rOffset,x1,y1-rOffset,class_ID,i));
        }else{
            //this box extends into both images, which is a fictitious space!
            // no objects in fictitious spaces allowed, new rule, well known rule.
            continue;
        }
    }
}

// #####################################################################
cv::Mat bbNet::stackImgBlob(const cv::Mat& imgL,const cv::Mat& imgR) const {
    //put left image on top of right image, with any remaining space
    //from squarification in the middle
    if(imgR.cols != imgL.cols){
        throw std::runtime_error("Cannot stack images, different column numbers!");
    }

    int rowL = imgL.rows;
    int rowR = imgR.rows;
    int col = imgL.cols;
    int gap = col-(rowL+rowR);    

    if(gap <= 0){
        throw std::runtime_error("Cannot stack images, cols <= 2*rows (too tall for square stacking)");
    }

    cv::Mat stacked = cv::Mat::zeros(col, col, CV_8UC3);
    imgL.copyTo(stacked(cv::Rect(0, 0, col, rowL)));
    imgR.copyTo(stacked(cv::Rect(0, rowL+gap, col, rowR)));
    if(netChoice == 0){ 
        //SSD Net
        return cv::dnn::blobFromImage(stacked, 1.0, cv::Size(300, 300), cv::mean(imgL), true, false);
    }else{
        //YOLOv5 Net
        return cv::dnn::blobFromImage(stacked, 1./255., cv::Size(320,320), cv::Scalar(), true, false);  
    }
}
// #####################################################################
cv::Mat bbNet::batchImgBlob(const cv::Mat& imgL,const cv::Mat& imgR) const {
    return cv::Mat();
}
// #####################################################################  
cv::Mat bbNet::singleImgBlob(const cv::Mat& img) const {
    if(netChoice == 0){
        //SSD Net
        return cv::dnn::blobFromImage(img, 1.0, cv::Size(300, 300), cv::mean(img), true, false);
    }else{
        //Yolov5 Net
        int dimMax = std::max(img.rows,img.cols);

        cv::Mat output = cv::Mat::zeros(dimMax, dimMax, CV_8UC3);
        img.copyTo(output(cv::Rect(0, 0, img.rows, img.cols)));

        return cv::dnn::blobFromImage(output, 1./255., cv::Size(320,320), cv::Scalar(), true, false);  
    }
}
// #####################################################################
void bbNet::set_netChoice(int netChoice){
    std::string dir("/home/emextern/Desktop/codeStorage/semSLAM/nets/");
    this->netChoice = netChoice;

    switch(netChoice){
        case 99: name = std::string("The uN-Net"); return;
        case 0: net = cv::dnn::readNet(dir+"frozen_inference_graph.pb", 
                dir+"ssd_mobilenet_v2_coco_2018_03_29.txt","TensorFlow"); 
                name = std::string("SSD MobileNet v2.0");
                break;
        case 1: net = cv::dnn::readNet(dir+"yolov5n_320.onnx");
                name = std::string("YOLOv5 Nano");
                break;
        case 2: net = cv::dnn::readNet(dir+"yolov5s_320.onnx");
                name = std::string("YOLOv5 Small");
                break;
        case 3: net = cv::dnn::readNet(dir+"yolov5m_320.onnx");
                name = std::string("YOLOv5 Medium");
                break;
    } 

    std::ifstream ifs;
    if(netChoice == 0){
        ifs.open(dir+"ssdClasses.txt");
    }else{
        ifs.open(dir+"yoloClasses.txt");
    }

    std::string line;
    while (getline(ifs, line))
    {
        labels.push_back(line);
    }
    ifs.close();
}
// #####################################################################