#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include "function.h"

using namespace std;
using namespace cv;

void callback_mouse(int event,int x,int y,int flags,void *param);
void callback_sgbm(int ,void *);

StereoSGBM sgbm;
TickMeter timer;
int ___minDisparity=0,___SADWindowSize=5,____numberOfDisparities=3,__uniquenessRatio=10;
Mat img1_gray,img2_gray,disp_SGBM,disp8,disp_real_SGBM,coor_camera,camera2robot;
string match="match",src="src",rectified="rectified",_canny="canny",result="result";
vector<Vec3f> Tennis_hough,Tennis_high;
int main(int argc,char **argv)
{
	Mat img1(480,640,CV_8UC3),img2(480,640,CV_8UC3),img1_canny,img2_canny,M1,M2,D1,D2,R,T,Rl,Rr,Pl,Pr,Q,map1x,map1y,map2x,map2y;
	cv::VideoWriter video;

	/*Init:read img and get para from intrisics and extrinsics*/
	img1=imread("left1.jpg"),img2=imread("right1.jpg");
	if(img1.empty()||img2.empty()) {cout<<"read src image error!"<<endl;return 0;}
	display_fuse(img1,img2,false,src);
	Init_SrcIMG_Para((string)(argv[1]),(string)(argv[2]),M1,M2,D1,D2,R,T,Rl,Rr,Pl,Pr,Q);
	initUndistortRectifyMap(M1, D1, Rl, Pl, img1.size(), CV_16SC2, map1x, map1y);
	initUndistortRectifyMap(M2, D2, Rr, Pr, img1.size(), CV_16SC2, map2x, map2y);
	FileStorage fs("rotate.yml",CV_STORAGE_READ);
	if(!fs.isOpened())   {std::cout<<"fail to open rotate.yml!";return -1;}
	fs["world_camera"]>>camera2robot;
	remap(img1,img1,map1x,map1y,CV_INTER_LINEAR),remap(img2,img2,map2x,map2y,CV_INTER_LINEAR);
	display_fuse(img1,img2,true,rectified);

    /*Detect tennis*/
	detectTennis_edge(img1,Tennis_hough,2);
	Mat canvas_hough=img1.clone();
	draw_tennis(canvas_hough,Tennis_hough);imshow("hough",canvas_hough);
	init_sgbm(sgbm,img1.channels(),___SADWindowSize,___minDisparity,____numberOfDisparities*16,__uniquenessRatio);
	cvtColor(img1,img1_gray,CV_BGR2GRAY),cvtColor(img2,img2_gray,CV_BGR2GRAY);
    sgbm(img1_gray,img2_gray,disp_SGBM);disp_SGBM.convertTo(disp8, CV_8U, 255/(sgbm.numberOfDisparities*16.));
	disp_SGBM.convertTo(disp_real_SGBM,CV_32F,0.0625);
	reprojectImageTo3D(disp_real_SGBM,coor_camera,Q,true);//获得摄像机坐标系下坐标
    test_show_tennis_coordinate(img1,Tennis_hough,disp_real_SGBM,camera2robot,coor_camera,false,result,video,false);

    /*build debug UI for correspodence*/
    namedWindow(match,CV_WINDOW_NORMAL);
    imshow(match,disp8);
	createTrackbar("minDisparity",match,&___minDisparity,100,callback_sgbm);
	createTrackbar("SADWindowSize",match,&___SADWindowSize,20,callback_sgbm);
	createTrackbar("numberOfDisparities",match,&____numberOfDisparities,30,callback_sgbm);
	createTrackbar("uniquenessRatio",match,&__uniquenessRatio,50,callback_sgbm);

    setMouseCallback(match,callback_mouse,(void*)&disp8);
    waitKey();
}

void callback_mouse(int event,int x,int y,int flags,void *param)
{
	//Mat& image = *(Mat*) param;
	Mat coor_robot(4,1,CV_64FC1,cv::Scalar::all(1));
	Point center(x,y);
	if(event==EVENT_LBUTTONUP)//判断左键是否抬起，分别可以判断鼠标移动，左键按下，左键抬起 详见毛星云的书P78
	{
		cout<<"点击的是 "<<y<<"行，"<<x<<"列"<<endl;
		coor_robot.at<double>(0,0)=coor_camera.at<Vec3f>(y,x)[0];
		coor_robot.at<double>(1,0)=coor_camera.at<Vec3f>(y,x)[1];
		coor_robot.at<double>(2,0)=coor_camera.at<Vec3f>(y,x)[2];
		cout<<"此点的视差为:"<<disp_real_SGBM.at<float>(y,x)<<endl<<"三维坐标为"<<((cv::Mat)camera2robot.inv())*coor_robot<<endl;
	}
}

void callback_sgbm(int ,void *)
{
	timer.reset();timer.start();
	sgbm.minDisparity=___minDisparity;
	sgbm.SADWindowSize=___SADWindowSize;
	sgbm.numberOfDisparities=____numberOfDisparities*16;
	sgbm.uniquenessRatio=__uniquenessRatio;
	sgbm(img1_gray,img2_gray,disp_SGBM);disp_SGBM.convertTo(disp8, CV_8U, 255/(sgbm.numberOfDisparities*16.));
	disp_SGBM.convertTo(disp_real_SGBM,CV_32F,0.0625);
	imshow(match,disp8);
	timer.stop();
	cout<<"minDisparity="<<sgbm.minDisparity<<endl;
	cout<<"SADWindowSize="<<sgbm.SADWindowSize<<endl;
	cout<<"numberOfDisparities="<<sgbm.numberOfDisparities<<endl;
	cout<<"uniquenessRatio="<<sgbm.uniquenessRatio<<endl;
	cout<<"run time="<<timer.getTimeMilli()<<"ms"<<endl<<endl;
}
