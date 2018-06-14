#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include "function.h"

#define switch_hough_show 1 //whether show hough detection's result
#define switch_canny_show 1 //whether show src's img edges by canny

using namespace std;
using namespace cv;

StereoSGBM sgbm;
TickMeter timer;
int ___minDisparity=0,___SADWindowSize=3,____numberOfDisparities=3,__uniquenessRatio=10;
Mat img1_gray,img2_gray,disp_SGBM,disp8,disp_real_SGBM,coor_camera,camera2robot;
string match="match",src="src",rectified="rectified",_canny="canny",result="result";
vector<Vec3f> Tennis_hough,Tennis_high;

int main(int argc,char **argv)
{
	Mat img1(480,640,CV_8UC3),img2(480,640,CV_8UC3),M1,M2,D1,D2,R,T,Rl,Rr,Pl,Pr,Q,map1x,map1y,map2x,map2y;
	cv::VideoWriter video;
	VideoCapture camera_left,camera_right;
	Init_SrcIMG_Para((string)(argv[1]),(string)(argv[2]),M1,M2,D1,D2,R,T,Rl,Rr,Pl,Pr,Q);
	initUndistortRectifyMap(M1, D1, Rl, Pl, img1.size(), CV_16SC2, map1x, map1y);
	initUndistortRectifyMap(M2, D2, Rr, Pr, img1.size(), CV_16SC2, map2x, map2y);
	FileStorage fs("rotate.yml",CV_STORAGE_READ);
	if(!fs.isOpened())   {std::cout<<"fail to open rotate.yml!";return -1;}
	fs["world_camera"]>>camera2robot;

	camera_left.open(1);
	camera_right.open(2);
	if (!camera_left.isOpened())  {cout<<"camera_left is't open"<<endl;return -1;}
	if (!camera_right.isOpened()) {cout<<"camera_right is't open"<<endl;return -1;}

	while(1)
    {
      timer.reset();timer.start();
      camera_left>>img1,camera_right>>img2;
      remap(img1,img1,map1x,map1y,CV_INTER_LINEAR),remap(img2,img2,map2x,map2y,CV_INTER_LINEAR);
      detectTennis_edge(img1,Tennis_hough,2);
      #if switch_hough_show //whether show hough detection's result
      Mat canvas_hough=img1.clone();draw_tennis(canvas_hough,Tennis_hough);imshow("hough",canvas_hough);
      #endif
      init_sgbm(sgbm,img1.channels(),___SADWindowSize,___minDisparity,____numberOfDisparities*16,__uniquenessRatio);
      cvtColor(img1,img1_gray,CV_BGR2GRAY),cvtColor(img2,img2_gray,CV_BGR2GRAY);
      #if switch_canny_show //whether show src's img edges by canny
      Mat canvas_canny=img1_gray.clone();GaussianBlur(canvas_canny,canvas_canny,Size(5,5),0);
      Canny(canvas_canny,canvas_canny,80,160);
      imshow(_canny,canvas_canny);
      #endif
      sgbm(img1_gray,img2_gray,disp_SGBM);disp_SGBM.convertTo(disp8, CV_8U, 255/(sgbm.numberOfDisparities*16.));
      disp_SGBM.convertTo(disp_real_SGBM,CV_32F,0.0625);
      reprojectImageTo3D(disp_real_SGBM,coor_camera,Q,true);
      test_show_tennis_coordinate(img1,Tennis_hough,disp_real_SGBM,camera2robot,coor_camera,false,result,video,false);
      if(waitKey(1)>=0)   {waitKey(5000);}
      timer.stop();
      cout<<"run time="<<timer.getTimeMilli()<<"ms"<<endl<<endl;
    }
}
