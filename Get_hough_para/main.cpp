/*
此程序是用来测试hough圆变换函数,为下一步实验提供houghcircles()函数的最佳参数:使用的图片是经过校正的图片(因为实际识别时使用的都是校正后的图片)
2018.5.15:使用图片img1,获得最佳参数:canny=400,圆度=25,dp=1,半径范围7-24,使用高斯滤波=size(5,5)
2018.5.16:使用在网球场强光下拍摄的照片,img13,其他参数不变,圆度要减小到17(canny参数可不变,也可适当减小)
*/

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <cstdlib>
using namespace cv;
using namespace std;

void draw_tennis(cv::Mat &img,vector<Vec3f> &cir);

Mat src,gray,blu_gray,dst,canny_gray;
vector<Vec3f> circles;
void callback_Houghcircle(int,void *);
int minDist=15,th_canny=400,th_circle=25,min_R=7,max_R=24,size_core=2;
int bilateral=26;
int blur_way=0;
int dp_int=100;
double dp=1.f;
int main()
{
    src=imread("left1.jpg"),dst=src.clone();
    if(src.empty())  {cout<<"read src error!"<<endl;return 0;}
    cvtColor(src,gray,CV_BGR2GRAY);

    namedWindow("dst",CV_WINDOW_NORMAL);
	createTrackbar("size_core","dst",&size_core,30,callback_Houghcircle);
	createTrackbar("minDist","dst",&minDist,100,callback_Houghcircle);
	createTrackbar("th_canny","dst",&th_canny,500,callback_Houghcircle);
	createTrackbar("th_circle","dst",&th_circle,100,callback_Houghcircle);
	createTrackbar("min_R","dst",&min_R,10,callback_Houghcircle);
	createTrackbar("dp","dst",&dp_int,200,callback_Houghcircle);
	createTrackbar("max_R","dst",&max_R,150,callback_Houghcircle);
	createTrackbar("bilater","dst",&bilateral,80,callback_Houghcircle);
	HoughCircles(gray,circles,CV_HOUGH_GRADIENT,dp,minDist,th_canny,th_circle,min_R,max_R);

	draw_tennis(dst,circles);
	imshow("dst",dst);

	while(1)
	{
 		if(waitKey(500)>=0) /*一定要注意是在imshow的框里按键会被检测到，在cmd框里按键不会检测到，按空格最好*/
		{
			blur_way++;
			if(blur_way==6) {blur_way=0;}
			if(blur_way==0) {cout<<"not use blur"<<endl;}
			if(blur_way==1) {cout<<"use box blur"<<endl;}
			if(blur_way==2) {cout<<"use average blur"<<endl;}
			if(blur_way==3) {cout<<"use gauss blur"<<endl;}
			if(blur_way==4) {cout<<"use median blur"<<endl;}
			if(blur_way==5) {cout<<"use bilater blur"<<endl;}
		}
	}
}

void callback_Houghcircle(int,void *)
{
    cvtColor(src,gray,CV_BGR2GRAY);
	blu_gray=gray.clone(),dst=src.clone();
	switch(blur_way)
	{
	case 1: boxFilter(gray,blu_gray,-1,Size(size_core*2+1,size_core*2+1));cout<<"box filter"<<endl;break;
	case 2: blur(gray,blu_gray,Size(size_core*2+1,size_core*2+1));cout<<"average filter"<<endl;break;
	case 3: GaussianBlur(gray,blu_gray,Size(size_core*2+1,size_core*2+1),0,0);cout<<"gauss filter"<<endl;break;
	case 4: medianBlur(gray,blu_gray,size_core*2+1);cout<<"median filter"<<endl;break;
	case 5: bilateralFilter(gray,blu_gray,bilateral,bilateral*2,bilateral/2);cout<<"bilater filter"<<endl;break;
	default:cout<<"not use blur"<<endl;break;
	}
	Canny(blu_gray,canny_gray,th_canny/2,th_canny);
	imshow("canny_gray",canny_gray);
	imshow("blur",blu_gray);//那种滤波方法好用还不好说，我看不用滤波还挺好
	dp=(double)dp_int/100.f;
	cout<<dp<<endl;
	HoughCircles(blu_gray,circles,CV_HOUGH_GRADIENT,dp,minDist,th_canny,th_circle,min_R,max_R);
    draw_tennis(dst,circles);
	imshow("dst",dst);
}


/*画网球并打印网球信息*/
/*第一个参数：要进行绘图的画布 Mat & 因为是引用形式，所以直接在原图上画
  第二个参数<Vec3f>形的容器向量，第一个元素为X坐标，第二个元素为Y坐标，第三个元素为半径*/
void draw_tennis(cv::Mat &img,vector<Vec3f> &cir)
{
	for(size_t i=0;i<cir.size();i++)
	{
		Point center(cvRound(cir[i][0]),cvRound(cir[i][1]));
		int radius=cvRound(cir[i][2]);
		circle(img,center,3, Scalar(0,255,0), -1);//-1为实心圆 这步是画圆心
		circle(img,center,radius,Scalar(155,50,255),3);
		string num=to_string((i+1));
		putText(img,num,Point(cir[i][0],cir[i][1]),CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(255,0,0),1);
		cout<<"第"<<(i+1)<<"个圆: X="<<cir[i][0]<<" ,  Y="<<cir[i][1]<<",   R="<<cir[i][2]<<endl;
	}
}
