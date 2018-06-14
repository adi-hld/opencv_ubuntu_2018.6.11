#ifndef __function_h
#define __function_h
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#define MAX_HIGH 90
#define MIN_HIGH 30

std::vector<cv::Point> target_Tennis(std::vector<cv::Point> &last_Tennis,std::vector<cv::Point> &now_Tennis);
std::vector<cv::Point> Select_Tennis(std::vector<cv::Point> &now_frame_Tennis,std::vector<cv::Vec4i> &count_vec_tennis,std::vector<cv::Point> &vec_Tennis);
cv::Point close_Tennis(std::vector<cv::Point> &tennis,cv::Point &_now_pose);
void display_fuse(cv::Mat& img_1,cv::Mat& img_2,bool addline,std::string winname);
int Init_SrcIMG_Para(const std::string& intrin_m,const std::string& extrin_m,cv::Mat& M1_,cv::Mat& M2_,cv::Mat& D1_,cv::Mat& D2_,cv::Mat& R_,cv::Mat& T_,cv::Mat& Rl_,cv::Mat& Rr_,cv::Mat& Pl_,cv::Mat& Pr_,cv::Mat& Q_);
void detectTennis_edge(cv::Mat& img,std::vector<cv::Vec3f>& tennis,int flag_filter);
std::vector<cv::Point3i> test_show_tennis_coordinate(cv::Mat &img,std::vector<cv::Vec3f> &cir,cv::Mat &disp_real,cv::Mat &_world2camera,cv::Mat &_Camera_coordinate,bool center_seek,
	const std::string &winname,cv::VideoWriter &_video,bool whether_video);
void draw_tennis(cv::Mat &img,std::vector<cv::Vec3f> &cir);
void init_sgbm(cv::StereoSGBM &SGBM_,int cn,int _SADWindowSize,int _minDisparity,int _numberOfDisparities,int _uniquenessRatio);

#endif
