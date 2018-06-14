/*
1.此程序是与windows上的程序BM_adi_2018_1_13一样的,只不过在ubuntu复现了一下,windows的运行时间是2200ms,ubuntu是234ms,为什么
差这么多有待研究
2.与windows程序的变化,将int  Init_SrcIMG_Para(const string& intrin_m,const string& extrin_m,.....),前两个函数参数加了const修饰,
这么看来ubuntu比vs2010相比更加严格,因为本身这两个参数是用给类FileStorage的构造函数的,本身规定就是const string
3.经get_hough_para_2018.5.15获得了更佳的houghcircles()函数参数,dp=1,canny=400,circle=17,gauss滤波size(5,5)

此程序为2018.1.13-2018.1.23用来调试bm匹配算法，网球识别算法，使用的是双目的同时图片
1.网球识别：
先使用detectTennis_edge（）：houghCircle()识别圆轮廓（完全放弃了网球颜色，这个做法可能有点武断，但也是被各种光照条件逼的，后期若想进一步加强算法的鲁棒性，可使用颜色辅助）
再通过网球高度进行下一步判断vector<Point3i> test_show_tennis_coordinate（），返回的筛选后最终的网球
2.3D重建
使用BM算法完成了3D重建，注意行对齐的效果对匹配的影响特别大，而影响行对齐效果的就是立体标定的精度，注意标定函数返回的重投影误差，越接近0越好，
这次标定的重投影误差是目前的获得的最好值，只有0.1几
在3D重建后获取的在摄像机坐标系下的三维坐标后，将其转换到了世界坐标系下，进而为网球的筛选提供了高度信息
*/
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <math.h>

using namespace std;
using namespace cv;

void detectTennis(Mat& img,vector<Vec3f>& tennis,int flag_filter);
void draw_tennis(cv::Mat &img,vector<Vec3f> &cir);
void display_fuse(Mat& img_1,Mat& img_2,bool addline,string winname);
int    Init_SrcIMG_Para(const string& intrin_m,const string& extrin_m,Mat& M1_,Mat& M2_,Mat& D1_,Mat& D2_,Mat& R_,Mat& T_,Mat& Rl_,Mat& Rr_,Mat& Pl_,Mat& Pr_,Mat& Q_);
void filterTennis(Mat &img,vector<Vec3f>& tennis,int flag_filter,string winname );
void init_bm(StereoBM &BM_,int _SADWindowSize,int _minDisparity,int _numberOfDisparities,int _uniquenessRatio);
void callback_bm(int ,void *);
vector<Point3i> test_show_tennis_coordinate(cv::Mat &img,vector<Vec3f> &cir,cv::Mat &disp_real,cv::Mat &_world2camera,cv::Mat &_Camera_coordinate,bool center_seek);
void detectTennis_edge(Mat& img,vector<Vec3f>& tennis,int flag_filter=2);
void callback_mouse(int event,int x,int y,int flags,void *param);
template<typename T> void replace(cv::Mat &img,T specify_src,T specify_dst);

StereoBM bm;
TickMeter timer;
int ___minDisparity=0,___SADWindowSize=6,____numberOfDisparities=3,__uniquenessRatio=8,__textureThreshold = 10;
Mat img1_gray,img2_gray,img1p,img2p,disp,disp_BM,disp8,disparity_real,Camera_coordinate,Q,world2camera,world_3D(4,1,CV_64FC1,Scalar::all(1));
string match="match";
int main(int argc,char **argv)
{
	Mat img1(480,640,CV_8UC3),img2(480,640,CV_8UC3),img1_canny,img2_canny,M1,M2,D1,D2,R,T,Rl,Rr,Pl,Pr,map1x,map1y,map2x,map2y;
	vector<Vec3f> tennis_left,tennis_right;
	timer.reset(),timer.start();
	img1=imread("left1.jpg"),img2=imread("right1.jpg");
	if(img1.empty()||img2.empty()) {cout<<"read src image error!"<<endl;return 0;}
	Init_SrcIMG_Para((string)(argv[1]),(string)(argv[2]),M1,M2,D1,D2,R,T,Rl,Rr,Pl,Pr,Q);
	initUndistortRectifyMap(M1, D1, Rl, Pl, img1.size(), CV_16SC2, map1x, map1y);
	initUndistortRectifyMap(M2, D2, Rr, Pr, img1.size(), CV_16SC2, map2x, map2y);
	int flag=1;
	vector<Vec3f> Tennis;
	string src="src",rectified="rectified",cannyy="canny";
	namedWindow(match,CV_WINDOW_NORMAL);
	//init_bm(bm,17,0,((img1.cols/8) + 15) & -16,15);
	init_bm(bm,13,0,16*3,8);

	createTrackbar("minDisparity",match,&___minDisparity,100,callback_bm);
	createTrackbar("SADWindowSize",match,&___SADWindowSize,20,callback_bm);
	createTrackbar("numberOfDisparities",match,&____numberOfDisparities,30,callback_bm);
	createTrackbar("uniquenessRatio",match,&__uniquenessRatio,50,callback_bm);
	createTrackbar("textureThreshold",match,&__textureThreshold,50,callback_bm);

	display_fuse(img1,img2,true,src);
	remap(img1,img1,map1x,map1y,CV_INTER_LINEAR),remap(img2,img2,map2x,map2y,CV_INTER_LINEAR);
	imwrite("img.jpg",img1);
	display_fuse(img1,img2,true,rectified);
	detectTennis_edge(img1,Tennis,2);
	Mat canvas_hough=img1.clone();draw_tennis(canvas_hough,Tennis);
	imshow("canvas_hough",canvas_hough);//显示hough圆变换后检测网球的结果，可选

	cvtColor(img1,img1_gray,CV_BGR2GRAY),cvtColor(img2,img2_gray,CV_BGR2GRAY);
	Mat canny_gray;
	GaussianBlur(img1_gray,img1_gray,Size(5,5),0,0);
	Canny(img1_gray,canny_gray,100,200);
	imshow("canny",canny_gray);
	//display_rectified(img1_gray,img2_gray,true,"gray");

	/*左右边界延拓*/
	copyMakeBorder(img1_gray, img1p, 0, 0, bm.state->numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder(img2_gray, img2p, 0, 0, bm.state->numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	//display_rectified(img1p,img2p,true,"Add_Border");

	bm(img1p,img2p,disp_BM);
	disp=disp_BM.colRange(bm.state->numberOfDisparities,img1p.cols);
	disp.convertTo(disp8, CV_8U, 255/(bm.state->numberOfDisparities *16.));
	disp.convertTo(disparity_real,CV_32F,0.0625);

	/*BM的视差值CV_16S,是放大了16倍的，disp_real为真实视差矩阵,在disp_SGBM中，没匹配到的点视差为-16，在disp_real中，没匹配到的点视差为-1*/
    reprojectImageTo3D(disparity_real, Camera_coordinate, Q, true);

	FileStorage fs("rotate.yml",CV_STORAGE_READ);
    if(!fs.isOpened()) {std::cout<<"fail to open rotate.yml";return -1;}
	fs["world_camera"]>>world2camera;

	vector<Point3i> Real_Tennis=test_show_tennis_coordinate(img1,Tennis,disparity_real,world2camera,Camera_coordinate,true);
	timer.stop();
	cout<<"run time="<<timer.getTimeMilli()<<"ms"<<endl<<endl;
	imshow(match,disp8);
	//setMouseCallback(match,callback_mouse,(void*)&disp8);
   setMouseCallback("canvas_hough",callback_mouse,(void*)&disp8);

	waitKey();
}

void callback_mouse(int event,int x,int y,int flags,void *param)
{
	//Mat& image = *(Mat*) param;
	int high_support=715,Y_2D=0;//左摄像头光心到网球顶点的高度
	Point center(x,y);
	if(event==EVENT_LBUTTONUP)//判断左键是否抬起，分别可以判断鼠标移动，左键按下，左键抬起 详见毛星云的书P78
	{
		cout<<"点击的是 "<<y<<"行，"<<x<<"列"<<endl;
		cout<<"此点的视差为:"<<disparity_real.at<float>(y,x)<<endl<<"三维坐标为"<<Camera_coordinate.at<Vec3f>(y,x)<<endl;
		world_3D.at<double>(0,0)=Camera_coordinate.at<Vec3f>(y,x)[0],world_3D.at<double>(1,0)=Camera_coordinate.at<Vec3f>(y,x)[1],world_3D.at<double>(2,0)=Camera_coordinate.at<Vec3f>(y,x)[2];
		cout<<(Mat)world2camera.inv()*world_3D<<endl;
	}
}

void callback_bm(int ,void *)
{
	timer.reset();timer.start();
	bm.state->minDisparity=___minDisparity;
	bm.state->SADWindowSize=___SADWindowSize*2+1;
	bm.state->numberOfDisparities=____numberOfDisparities*16;
	bm.state->uniquenessRatio=__uniquenessRatio;
	bm.state->textureThreshold=__textureThreshold;

	copyMakeBorder(img1_gray, img1p, 0, 0, bm.state->numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder(img2_gray, img2p, 0, 0, bm.state->numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	bm(img1p,img2p,disp_BM);
	disp=disp_BM.colRange(bm.state->numberOfDisparities,img1p.cols);
	disp.convertTo(disp8, CV_8U, 255/(bm.state->numberOfDisparities *16.));
	disp.convertTo(disparity_real,CV_32F,0.0625);
	reprojectImageTo3D(disparity_real, Camera_coordinate,Q,true);
	imshow(match,disp8);
	timer.stop();
	//cout<<"minDisparity="<<bm.state->minDisparity<<endl;
	cout<<"SADWindowSize="<<bm.state->SADWindowSize<<endl;
	cout<<"numberOfDisparities="<<bm.state->numberOfDisparities<<endl;
	cout<<"uniquenessRatio="<<bm.state->uniquenessRatio<<endl;
	cout<<"run time="<<timer.getTimeMilli()<<"ms"<<endl<<endl;
}
void init_bm(StereoBM &BM_,int _SADWindowSize,int _minDisparity,int _numberOfDisparities,int _uniquenessRatio)
{
	BM_.state->preFilterCap = 31;
	BM_.state->SADWindowSize = _SADWindowSize;//SADWindowSize：SAD窗口大小，容许范围是[5,255]，一般应该在 5x5 至 21x21 之间，参数必须是奇数odd
	BM_.state->minDisparity = _minDisparity;
	BM_.state->numberOfDisparities = _numberOfDisparities;//视差窗口，即最大视差值与最小视差值之差, 窗口大小必须是 16 的整数倍
	BM_.state->textureThreshold = 10;//低纹理区域的判断阈值
	BM_.state->uniquenessRatio = _uniquenessRatio;//视差唯一性百分比
	BM_.state->speckleWindowSize = 100;//检查视差连通区域变化度的窗口大小, 值为 0 时取消 speckle 检查
	BM_.state->speckleRange =32;//视差变化阈值，当窗口内视差变化大于阈值时，该窗口内的视差清零
	BM_.state->disp12MaxDiff = -1;//左视差图（直接计算得出）和右视差图（通过cvValidateDisparity计算得出）之间的最大容许差异。超过该阈值的视差值将被清零。该参数默认为 -1，即不执行左右视差检查。
}

int Init_SrcIMG_Para(const string& intrin_m,const string& extrin_m,Mat& M1_,Mat& M2_,Mat& D1_,Mat& D2_,Mat& R_,Mat& T_,Mat& Rl_,Mat& Rr_,Mat& Pl_,Mat& Pr_,Mat& Q_)
{
	FileStorage fs(intrin_m,CV_STORAGE_READ);
	if(!fs.isOpened()) {std::cout<<"fail to open intrinsics.yml";return -1;}
	fs["M1"]>>M1_,fs["M2"]>>M2_,fs["D1"]>>D1_,fs["D2"]>>D2_;

	fs.open(extrin_m,CV_STORAGE_READ);
	if(!fs.isOpened()) {std::cout<<"fail to open intrinsics.yml";return -1;}
	fs["R"]>>R_,fs["T"]>>T_,fs["R1"]>>Rl_,fs["R2"]>>Rr_,fs["P1"]>>Pl_,fs["P2"]>>Pr_,fs["Q"]>>Q_;
	//fs["map1x"]>>map1x_,fs["map1y"]>>map1y_,fs["map2x"]>>map2x_,fs["map2y"]>>map2y_;

	std::cout<<"success to read in_matrix,ex_matrix"<<std::endl;

	return 0;
}

/*2010.1.10 add show signal channel */
void display_fuse(Mat& img_1,Mat& img_2,bool addline,string winname)
{
	int flag=0;
	Mat canvas;
	if(img_1.channels()==1)      {Mat temp(img_1.rows,2*img_1.cols,CV_8UC1);canvas=temp.clone();flag=1;}
	else if(img_1.channels()==3) {Mat temp(img_1.rows,2*img_1.cols,CV_8UC3);canvas=temp.clone();}
	else                         {std::cout<<"canvas's channel is error,can't show fuse image!"<<endl;return;}
	Mat canvas_part_L=canvas(Rect(0,0,img_1.cols,img_1.rows));
	resize(img_1,canvas_part_L,img_1.size(), 0, 0, CV_INTER_AREA);
	Mat canvas_part_R=canvas(Rect(img_1.cols,0,img_1.cols,img_1.rows));
	resize(img_2,canvas_part_R,img_1.size(), 0, 0, CV_INTER_AREA);
	if(addline)
	{
	if(flag==0)
		{for( int j = 0; j < canvas.rows; j += 16 ) {line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);}}
	else
		{for( int j = 0; j < canvas.rows; j += 16 ) {line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(255), 1, 8);}}
	}
	imshow(winname,canvas);
}

/*input a img(must be RGB), output vector<Vec3f> tennis tennis[0]=x,tennis[1]=y,tennis[2]=R*/
/*flag_filter=0:not use filter, flag_filter=1:use median filter ,flag_filter=2:use gauss filter*/
void detectTennis(Mat& img,vector<Vec3f>& tennis,int flag_filter)
{
	CV_Assert(img.channels()==3);
	Mat hsv,hsv_range,canny,dst;
	cvtColor(img,hsv,CV_BGR2HSV);/*Attention:Must be CV_BGR2HSV,not CV_RGB2HSV 2018.1.16--Adi */
	int min_H=28,max_H=42,min_S=0,max_S=110,min_V=240,max_V=255,value_filter=1,min_canny=50,max_canny=150;
	int min_radius=5,max_radius=43,minDist=30,th_canny=100,th_circle=13;
	inRange(hsv,Scalar(min_H,min_S,min_V),Scalar(max_H,max_S,max_V),hsv_range);//inrange（）输出的tem为二值图，在范围内为白，不在为黑
	imshow("hsv_range",hsv_range);
	switch(flag_filter)
	{
	case 0:cout<<"not use filter"<<endl;break;
	case 1:cout<<"use median filter"<<endl;medianBlur(hsv_range,hsv_range,value_filter*2+1);break;
	case 2:	cout<<"use gauss filter"<<endl;GaussianBlur(hsv_range,hsv_range,Size(value_filter*2+1,value_filter*2+1),0,0);break;
	}
	Canny(hsv_range,canny,min_canny,max_canny);
	//imshow("canny",canny);

	HoughCircles(canny,tennis,CV_HOUGH_GRADIENT,1,minDist,th_canny,th_circle,min_radius,max_radius);
}

/*完全使用轮廓检验网球，不用任何颜色信息*/
void detectTennis_edge(Mat& img,vector<Vec3f>& tennis,int flag_filter)
{
	Mat gray_;
	int value_filter=2,minDist=15,th_canny=200,th_circle=20,min_R=7,max_R=24;
	double dp=1.f;
	if(img.channels()==3) {cvtColor(img,gray_,CV_BGR2GRAY);}
	else                  {gray_=img.clone();}
	switch(flag_filter)
	{
	case 0:cout<<"not use filter"<<endl;break;
	case 1:cout<<"use median filter"<<endl;medianBlur(gray_,gray_,value_filter*2+1);break;
	case 2:cout<<"use gauss filter"<<endl;GaussianBlur(gray_,gray_,Size(value_filter*2+1,value_filter*2+1),0,0);break;
	}
	HoughCircles(gray_,tennis,CV_HOUGH_GRADIENT,dp,minDist,th_canny,th_circle,min_R,max_R);
}

/*input: a img(must be RGB), output:img that only tennis(binary image) */
void filterTennis(Mat &img,vector<Vec3f>& tennis,int flag_filter,string winname )
{
	CV_Assert(img.channels()==3);
	Mat hsv,hsv_range,canny,dst;
	cvtColor(img,hsv,CV_BGR2HSV);
	int min_H=40,max_H=83,min_S=0,max_S=255,min_V=239,max_V=255,value_filter=1,min_canny=50,max_canny=150;
	int min_radius=5,max_radius=43,minDist=30,th_canny=100,th_circle=25;
	inRange(hsv,Scalar(min_H,min_S,min_V),Scalar(max_H,max_S,max_V),hsv_range);//inrange（）输出的tem为二值图，在范围内为白，不在为黑
	switch(flag_filter)
	{
	case 0:cout<<"not use filter"<<endl;break;
	case 1:cout<<"use median filter"<<endl;medianBlur(hsv_range,hsv_range,value_filter*2+1);break;
	case 2:cout<<"use gauss filter"<<endl;GaussianBlur(hsv_range,hsv_range,Size(value_filter*2+1,value_filter*2+1),0,0);break;
	}
	Canny(hsv_range,canny,min_canny,max_canny);
	imshow("canny",canny);
	HoughCircles(canny,tennis,CV_HOUGH_GRADIENT,1,minDist,th_canny,th_circle,min_radius,max_radius);
	dst=img.clone();
	draw_tennis(dst,tennis);
	imshow(winname,dst);
	img=canny;
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

/*test the houghCircle's result by Z_world,show the real tennis and tennis's 3D coordinate
bool center_seek:是否开启此功能：如果网球的中心没有被匹配到，就在网球的中心点寻找距离最近的匹配到的点，将此点的三维坐标视为中心点的三维坐标*/
vector<Point3i> test_show_tennis_coordinate(cv::Mat &img,vector<Vec3f> &cir,cv::Mat &disp_real,cv::Mat &_world2camera,cv::Mat &_Camera_coordinate,bool center_seek)
{
	cout<<"houghCircle检测到了"<<cir.size()<<"个网球"<<endl;
	cv::Mat canvas=img.clone(),world_3D(4,1,CV_64FC1,Scalar::all(1));
	vector<Point3i>  real_tennis;
	int X_world=0,Y_world=0,Z_world=0;
	int flag_center=0;//网球中心是否被匹配到的标志位
	for(int i=0;i<cir.size();i++)
	{
		Point center(cvRound(cir[i][0]),cvRound(cir[i][1]));
		int radius=cvRound(cir[i][2]);
		flag_center=0;
		if(center_seek)
		{
			if(disp_real.at<float>(center)==-1.f)
			{
				int j=1;
				std::cout<<"第"<<(i+1)<<"个网球中心没有被匹配到"<<std::endl;
				for (j=1;j<radius;j++)//上下左右4个方向偏移
				{
					if(disp_real.at<float>(center.y+j,center.x)!=-1.f) {center.y=center.y+j;break;}
					if(disp_real.at<float>(center.y-j,center.x)!=-1.f) {center.y=center.y-j;break;}
					if(disp_real.at<float>(center.y,center.x+j)!=-1.f) {center.x=center.x+j;break;}
					if(disp_real.at<float>(center.y,center.x-j)!=-1.f) {center.x=center.x-j;break;}
				}
				if(j<radius)  {flag_center=1;std::cout<<"第"<<(i+1)<<"个球中心偏移了"<<j<<"个像素"<<std::endl;}
				if(j==radius) {std::cout<<"第"<<(i+1)<<"个网球中心在此球的半径范围内没有被匹配到"<<std::endl;}//在网球半径范围内也没有匹配到的点，此时的三维数据为错的，视为错误的网球点
			}
		}

		world_3D.at<double>(0,0)=_Camera_coordinate.at<Vec3f>(center)[0],world_3D.at<double>(1,0)=_Camera_coordinate.at<Vec3f>(center)[1],world_3D.at<double>(2,0)=_Camera_coordinate.at<Vec3f>(center)[2];
		world_3D=(Mat)_world2camera.inv()*world_3D;
		X_world=world_3D.at<double>(0,0);
		Y_world=world_3D.at<double>(1,0);
		Z_world=world_3D.at<double>(2,0);
		if((Z_world>=120)||(Z_world<=20))  //粗略使用网球的高度进行筛选，认为(Z_world>=110)||(Z_world<=25)的都不是网球，网球中心没被匹配到的点也会被筛掉
		{
			circle(canvas,center,5, Scalar(0,0,0), -1);//将认为不符合网球高度信息的网球中心画出来，纯黑色
			std::cout<<"第"<<(i+1)<<"个球中心高度为"<<Z_world<<"mm,不在设定范围内"<<std::endl;
			continue;
		}
		real_tennis.push_back(Point3i(X_world,Y_world,Z_world));//将认为是网球的存入real_tennis

		/*显示经过houghCircle检测得到的圆*/
		if(flag_center==0)//检测到的网球中心正好被匹配到,圆心用绿色
		{circle(canvas,center,3, Scalar(0,255,0), -1);circle(canvas,center,radius,Scalar(155,50,255),2);}//-1为实心圆 这步是画圆心
		else//网球中心没有被匹配到，但在网球半径范围内被匹配到，圆心用蓝色
		{circle(canvas,center,3, Scalar(255,0,0), -1);circle(canvas,center,radius,Scalar(155,50,255),2);}//圆心会偏移

		int num_tennis=real_tennis.size();
		string num=to_string(num_tennis),coor_x=to_string(X_world),coor_y=to_string(Y_world),coor_z=to_string(Z_world);
		string coor="("+coor_x+","+coor_y+","+coor_z+")";
		putText(canvas,num,center,CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(255,0,0),1);
		putText(canvas,coor,Point(cir[i][0]+20,cir[i][1]),CV_FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,0,255),1);
		cout<<"第"<<(num_tennis)<<"个圆的像素坐标为: X="<<cir[i][0]<<" ,  Y="<<cir[i][1]<<",半径R="<<cir[i][2]<<endl;
		//cout<<"第"<<(num_tennis)<<"个圆摄像机三维坐标为: X="<<_Camera_coordinate.at<Vec3f>(center)[0]<<" ,Y="<<_Camera_coordinate.at<Vec3f>(center)[1]<<",Z="<<_Camera_coordinate.at<Vec3f>(center)[2]<<endl;
		cout<<"第"<<(num_tennis)<<"个圆世界三维坐标为: X="<<X_world<<" ,Y="<<Y_world<<",Z="<<Z_world<<endl<<endl;
	}
	std::cout<<"最后检测到的网球个数为"<<real_tennis.size()<<endl;
	imshow("tennis",canvas);

	return real_tennis;
}

/*用一个特定值取代矩阵中(must be signal channel)的某一特定值,用于将视差图中所有未匹配的点（视差=-1）设为某一特定值*/
template<typename T>
void replace(cv::Mat &img,T specify_src,T specify_dst)
{
	for(int i=0;i<img.rows;i++)
		for(int j=0;j<img.cols;j++)
		{
			if(img.at<T>(i,j)==specify_src) {img.at<T>(i,j)=specify_dst;}
		}
}
