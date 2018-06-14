#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <math.h>
#include "function.h"

/*将当前帧内识别到的网球经过筛选,认为符合条件的放入网球容器内*/
std::vector<cv::Point> target_Tennis(std::vector<cv::Point> &last_Tennis,std::vector<cv::Point> &now_Tennis)
{
	if(last_Tennis.empty()) {last_Tennis=now_Tennis;return last_Tennis;}
	std::vector<cv::Point> temp=last_Tennis;
	unsigned int flag=0;
	for (unsigned int i=0;i<now_Tennis.size();i++)
	{
		flag=0;
		for (unsigned int j=0;j<last_Tennis.size();j++)
		{
			cv::Rect rect_T(last_Tennis[j].x-200,last_Tennis[j].y-200,400,400);
			if (!rect_T.contains(now_Tennis[i]))
			{flag++;}
		}
		if(flag==last_Tennis.size()) {temp.push_back(now_Tennis[i]);}//如果flag==last_Tennis.size()，说明这个网球点不在之前所有所存在的网球构成的矩形中，视为新的网球点
	}

	last_Tennis=temp;
	return last_Tennis;
}

/*筛选网球,now_frame_Tennis是存储当前帧检测到的网球的容器,
count_vec_tennis是包含所有检测到的网球坐标,每个元素类型为cv::Vec4i,前两个量为网球坐标,第三个量为此坐标的网球对应的计数值(被检测到的次数),第4个量为是否存入过最终网球容器的标志位(0没存入过,1为存入过)
vec_Tennis是输出的最终网球容器
*/
std::vector<cv::Point> Select_Tennis(std::vector<cv::Point> &now_frame_Tennis,std::vector<cv::Vec4i> &count_vec_tennis,std::vector<cv::Point> &vec_Tennis)
{
    #define RECT_LENGTH  500
    #define THRESHHOLD_COUNT  4
    cv::Vec4i temp_count;
    cv::Point temp_point;
    std::vector<cv::Vec4i>  temp_count_store;
    if(now_frame_Tennis.empty())  {return vec_Tennis;}
    for(auto it_now_frame=now_frame_Tennis.begin();it_now_frame!=now_frame_Tennis.end();++it_now_frame)
    {
        if(!count_vec_tennis.empty())//非空
        {
                unsigned int flag_contain=0,num_count=count_vec_tennis.size();

                for(auto it_count=count_vec_tennis.begin();it_count!=count_vec_tennis.end();++it_count)
                {
                    cv::Rect rect_count((*it_count)[0]-RECT_LENGTH/2,(*it_count)[1]-RECT_LENGTH/2,RECT_LENGTH,RECT_LENGTH);//以此点为中心,构建一个指定边长的正方形
                    //std::cout<<rect_count<<std::endl;
                    //std::cout<<*it_now_frame<<std::endl;
                    if(rect_count.contains(*it_now_frame))  {(*it_count)[2]++;}//原有点包含此点,计数值加1
                    else{flag_contain++;}//不包含此点,将此点加入计数容器内,在这步将此点直接加入count_vec_tennis会导致内存破坏,所以用了个temp_count_store暂存一下,之后再将temp_count_stor加入count_vec_tennis
                }
                if(flag_contain==num_count)
                    {temp_count[0]=(*it_now_frame).x;temp_count[1]=(*it_now_frame).y;temp_count[2]=1;temp_count[3]=0;temp_count_store.push_back(temp_count);}
        }
        else//空
        {
                std::cout<<(*it_now_frame)<<std::endl;
                temp_count[0]=(*it_now_frame).x;temp_count[1]=(*it_now_frame).y;temp_count[2]=1;temp_count[3]=0;count_vec_tennis.push_back(temp_count);
        }
    }

    if(!temp_count_store.empty())
    {
        for(auto it_temp_count_store=temp_count_store.begin();it_temp_count_store!=temp_count_store.end();it_temp_count_store++)
        {
            count_vec_tennis.push_back(*it_temp_count_store);
        }
    }
    for(auto it=count_vec_tennis.begin();it!=count_vec_tennis.end();it++)
    {std::cout<<*it<<std::endl;}

    for(auto it_count2=count_vec_tennis.begin();it_count2!=count_vec_tennis.end();++it_count2)
    {
        if( ((*it_count2)[2]>=THRESHHOLD_COUNT) && ((*it_count2)[3]==0) )
        {temp_point.x=(*it_count2)[0];temp_point.y=(*it_count2)[1];(*it_count2)[3]=1;vec_Tennis.push_back(temp_point);}
    }
    if(vec_Tennis.empty())  {std::cout<<"vec_tennis is empty!"<<std::endl;}
    else{std::cout<<"vec_tennis's size is "<<vec_Tennis.size()<<std::endl;}
    return vec_Tennis;
}

/*找出tennis中距离当前位置最近的点，并将其与tennis[0]互换位置*/
cv::Point close_Tennis(std::vector<cv::Point> &tennis,cv::Point &_now_pose)
{
	cv::Point temp_1=tennis[0],temp_2;
	int distance=(tennis[0].x-_now_pose.x)*(tennis[0].x-_now_pose.x)+(tennis[0].y-_now_pose.y)*(tennis[0].y-_now_pose.y),distance_temp;
	for (unsigned int i=1;i<tennis.size();i++)
	{
		distance_temp=(tennis[i].x-_now_pose.x)*(tennis[i].x-_now_pose.x)+(tennis[i].y-_now_pose.y)*(tennis[i].y-_now_pose.y);
		if(distance_temp<distance)
		{
			distance=distance_temp;
			temp_1=tennis[i];
			temp_2=tennis[0];
			tennis[0]=tennis[i];
			tennis[i]=temp_2;
		}
	}
	return temp_1;
}

void display_fuse(cv::Mat& img_1,cv::Mat& img_2,bool addline,std::string winname)
{
	int flag=0;
	cv::Mat canvas;
	if(img_1.channels()==1)         {cv::Mat temp(img_1.rows,2*img_1.cols,CV_8UC1);canvas=temp.clone();flag=1;}
	else if(img_1.channels()==3) {cv::Mat temp(img_1.rows,2*img_1.cols,CV_8UC3);canvas=temp.clone();}
	else                         {std::cout<<"canvas's channel is error,can't show fuse image!"<<std::endl;return;}
	cv::Mat canvas_part_L=canvas(cv::Rect(0,0,img_1.cols,img_1.rows));
	cv::resize(img_1,canvas_part_L,img_1.size(), 0, 0, CV_INTER_AREA);
	cv::Mat canvas_part_R=canvas(cv::Rect(img_1.cols,0,img_1.cols,img_1.rows));
	cv::resize(img_2,canvas_part_R,img_1.size(), 0, 0, CV_INTER_AREA);
	if(addline)
	{
		if(flag==0)
		{for( int j = 0; j < canvas.rows; j += 16 ) {cv::line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j), cv::Scalar(0, 255, 0), 1, 8);}}
		else
		{for( int j = 0; j < canvas.rows; j += 16 ) {cv::line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j), cv::Scalar(255), 1, 8);}}
	}
	cv::imshow(winname,canvas);
}

int Init_SrcIMG_Para(const std::string& intrin_m,const std::string& extrin_m,cv::Mat& M1_,cv::Mat& M2_,cv::Mat& D1_,cv::Mat& D2_,cv::Mat& R_,cv::Mat& T_,cv::Mat& Rl_,cv::Mat& Rr_,cv::Mat& Pl_,cv::Mat& Pr_,cv::Mat& Q_)
{
	cv::FileStorage fs(intrin_m,CV_STORAGE_READ);
	if(!fs.isOpened()) {std::cout<<"fail to open intrinsics.yml";return -1;}
	fs["M1"]>>M1_,fs["M2"]>>M2_,fs["D1"]>>D1_,fs["D2"]>>D2_;

	fs.open(extrin_m,CV_STORAGE_READ);
	if(!fs.isOpened()) {std::cout<<"fail to open intrinsics.yml";return -1;}
	fs["R"]>>R_,fs["T"]>>T_,fs["R1"]>>Rl_,fs["R2"]>>Rr_,fs["P1"]>>Pl_,fs["P2"]>>Pr_,fs["Q"]>>Q_;
	//fs["map1x"]>>map1x_,fs["map1y"]>>map1y_,fs["map2x"]>>map2x_,fs["map2y"]>>map2y_;

	std::cout<<"success to read in_matrix,ex_matrix"<<std::endl;

	return 0;
}

/*完全使用轮廓检验网球，不用任何颜色信息*/
void detectTennis_edge(cv::Mat& img,std::vector<cv::Vec3f>& tennis,int flag_filter)
{
	cv::Mat gray_;
	int value_filter=2,minDist=15,th_canny=160,th_circle=23,min_R=7,max_R=24;//240,25
	double dp=1.f;
	if(img.channels()==3) {cv::cvtColor(img,gray_,CV_BGR2GRAY);}
	else                  {gray_=img.clone();}
	switch(flag_filter)
	{
	case 0:std::cout<<"not use filter"<<std::endl;break;
	case 1:std::cout<<"use median filter"<<std::endl;cv::medianBlur(gray_,gray_,value_filter*2+1);break;
	case 2:std::cout<<"use gauss filter"<<std::endl;cv::GaussianBlur(gray_,gray_,cv::Size(value_filter*2+1,value_filter*2+1),0,0);break;
	}
	cv::HoughCircles(gray_,tennis,CV_HOUGH_GRADIENT,dp,minDist,th_canny,th_circle,min_R,max_R);
}

/*test the houghCircle's result by Z_world,show the real tennis and tennis's 3D coordinate
bool center_seek:是否开启此功能：如果网球的中心没有被匹配到，就在网球的中心点寻找距离最近的匹配到的点，将此点的三维坐标视为中心点的三维坐标*/
std::vector<cv::Point3i> test_show_tennis_coordinate(cv::Mat &img,std::vector<cv::Vec3f> &cir,cv::Mat &disp_real,cv::Mat &_world2camera,cv::Mat &_Camera_coordinate,bool center_seek,
	const std::string &winname,cv::VideoWriter &_video,bool whether_video)
{
	std::cout<<"houghCircle检测到了"<<cir.size()<<"个网球"<<std::endl;
	//cv::Mat canvas=img.clone(),_world_3D(4,1,CV_64FC1,cv::Scalar::all(1));
	cv::Mat canvas=img,_world_3D(4,1,CV_64FC1,cv::Scalar::all(1));
	std::vector<cv::Point3i>  real_tennis;
	int X_world=0,Y_world=0,Z_world=0;
	int flag_center=0,flag_match=0;//网球中心是否被匹配到的标志位
	for( unsigned int i=0;i<cir.size();i++)
	{
		cv::Point center(cvRound(cir[i][0]),cvRound(cir[i][1]));
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
                    if( ((center.y+j)>img.rows) || ((center.y-j)<0) || ((center.x+j)>img.cols) || ((center.x-j)<0)   )  {flag_match=1;break;}//防越界,当网球在图像边缘位置时在附近搜索匹配可能会越界
					if(disp_real.at<float>(center.y+j,center.x)!=-1.f) {center.y=center.y+j;break;}
					if(disp_real.at<float>(center.y-j,center.x)!=-1.f) {center.y=center.y-j;break;}
					if(disp_real.at<float>(center.y,center.x+j)!=-1.f) {center.x=center.x+j;break;}
					if(disp_real.at<float>(center.y,center.x-j)!=-1.f) {center.x=center.x-j;break;}
				}

				if(1==flag_match)   {flag_match=0;std::cout<<"第"<<(i+1)<<"个网球在图像范围内没有被匹配到"<<std::endl;continue;}
				if(j<radius)  {flag_center=1;std::cout<<"第"<<(i+1)<<"个球中心偏移了"<<j<<"个像素"<<std::endl;}
				if(j==radius) {std::cout<<"第"<<(i+1)<<"个网球中心在此球的半径范围内没有被匹配到"<<std::endl;continue;}//在网球半径范围内也没有匹配到的点，此时的三维数据为错的，视为错误的网球点
			}
		}

		_world_3D.at<double>(0,0)=(int)_Camera_coordinate.at<cv::Vec3f>(center)[0],_world_3D.at<double>(1,0)=(int)_Camera_coordinate.at<cv::Vec3f>(center)[1],_world_3D.at<double>(2,0)=(int)_Camera_coordinate.at<cv::Vec3f>(center)[2];
		_world_3D=((cv::Mat)_world2camera.inv())*_world_3D;
		X_world=cvRound(_world_3D.at<double>(0,0));
		Y_world=cvRound(_world_3D.at<double>(1,0));
		Z_world=cvRound(_world_3D.at<double>(2,0));
		if((Z_world>=MAX_HIGH)||(Z_world<=MIN_HIGH))  //粗略使用网球的高度进行筛选，认为(Z_world>=110)||(Z_world<=25)的都不是网球，网球中心没被匹配到的点也会被筛掉
		{
			circle(canvas,center,5, cv::Scalar(0,0,0), -1);//将认为不符合网球高度信息的网球中心画出来，纯黑色
			std::cout<<"第"<<(i+1)<<"个球中心高度为"<<Z_world<<"mm,不在设定范围内"<<std::endl;
//			std::cout<<_Camera_coordinate.at<Vec3f>(center)<<std::endl;
//			std::cout<<_world_3D<<std::endl;
			continue;
		}
		real_tennis.push_back(cv::Point3i(X_world,Y_world,Z_world));//将认为是网球的存入real_tennis

		/*显示经过houghCircle检测得到的圆*/
		if(flag_center==0)//检测到的网球中心正好被匹配到,圆心用绿色
		{circle(canvas,center,3, cv::Scalar(0,255,0), -1);circle(canvas,center,radius,cv::Scalar(155,50,255),2);}//-1为实心圆 这步是画圆心
		else//网球中心没有被匹配到，但在网球半径范围内被匹配到，圆心用蓝色
		{circle(canvas,center,3, cv::Scalar(255,0,0), -1);circle(canvas,center,radius,cv::Scalar(155,50,255),2);}//圆心会偏移
        #if 1
		int num_tennis=real_tennis.size();
		std::string num=std::to_string(num_tennis),coor_x=std::to_string(X_world),coor_y=std::to_string(Y_world),coor_z=std::to_string(Z_world);
		std::string coor="("+coor_x+","+coor_y+","+coor_z+")";
		putText(canvas,num,center,CV_FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,0,0),1);
		putText(canvas,coor,cv::Point(cir[i][0]+20,cir[i][1]),CV_FONT_HERSHEY_SIMPLEX,0.8,cv::Scalar(0,0,255),1);
		//std::cout<<"第"<<(num_tennis)<<"个圆的像素坐标为: X="<<cir[i][0]<<" ,  Y="<<cir[i][1]<<",半径R="<<cir[i][2]<<std::endl;
		//cout<<"第"<<(num_tennis)<<"个圆摄像机三维坐标为: X="<<_Camera_coordinate.at<Vec3f>(center)[0]<<" ,Y="<<_Camera_coordinate.at<Vec3f>(center)[1]<<",Z="<<_Camera_coordinate.at<Vec3f>(center)[2]<<endl;
		std::cout<<"第"<<(num_tennis)<<"个圆世界三维坐标为: X="<<X_world<<" ,Y="<<Y_world<<",Z="<<Z_world<<std::endl<<std::endl;
		#endif
	}
	std::cout<<"最后检测到的网球个数为"<<real_tennis.size()<<std::endl;
	cv::imshow(winname,canvas);
	if(whether_video)//是否开启录像功能
	{
		_video<<canvas;
	}

	return real_tennis;
}

/*画网球并打印网球信息*/
/*第一个参数：要进行绘图的画布 Mat & 因为是引用形式，所以直接在原图上画
  第二个参数<Vec3f>形的容器向量，第一个元素为X坐标，第二个元素为Y坐标，第三个元素为半径*/
void draw_tennis(cv::Mat &img,std::vector<cv::Vec3f> &cir)
{
	for(size_t i=0;i<cir.size();i++)
	{
		cv::Point center(cvRound(cir[i][0]),cvRound(cir[i][1]));
		int radius=cvRound(cir[i][2]);
		circle(img,center,3, cv::Scalar(0,255,0), -1);//-1为实心圆 这步是画圆心
		circle(img,center,radius,cv::Scalar(155,50,255),3);
		std::string num=std::to_string(i+1);
		putText(img,num,cv::Point(cir[i][0],cir[i][1]),CV_FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,0,0),1);
		std::cout<<"第"<<(i+1)<<"个圆: X="<<cir[i][0]<<" ,  Y="<<cir[i][1]<<",   R="<<cir[i][2]<<std::endl;
	}
}

void init_sgbm(cv::StereoSGBM &SGBM_,int cn,int _SADWindowSize,int _minDisparity,int _numberOfDisparities,int _uniquenessRatio)
{
	SGBM_.preFilterCap = 63;
	SGBM_.SADWindowSize = _SADWindowSize;//SADWindowSize：SAD窗口大小，容许范围是[1,11]，一般应该在 3x3 至 11x11 之间，参数是奇数偶数都可以
	/*SGBM算法的状态参数大部分与BM算法的一致，下面只解释不同的部分：*/
	SGBM_.P1 = 8*cn*SGBM_.SADWindowSize*SGBM_.SADWindowSize;//P1, P2：控制视差变化平滑性的参数。P1、P2的值越大，视差越平滑。
	SGBM_.P2 = 32*cn*SGBM_.SADWindowSize*SGBM_.SADWindowSize;//P1是相邻像素点视差增/减 1 时的惩罚系数；P2是相邻像素点视差变化值大于1时的惩罚系数。P2必须大于P1。
	SGBM_.minDisparity = _minDisparity;
	SGBM_.numberOfDisparities = _numberOfDisparities;//视差窗口，即最大视差值与最小视差值之差, 窗口大小必须是 16 的整数倍
	SGBM_.uniquenessRatio = _uniquenessRatio;
	SGBM_.speckleWindowSize = 100;
	SGBM_.speckleRange =32;
	SGBM_.disp12MaxDiff = 1;
	SGBM_.fullDP = true;//fullDP：布尔值，当设置为 TRUE 时，运行双通道动态编程算法（full-scale 2-pass dynamic programming algorithm），会占用O(W*H*numDisparities)个字节，对于高分辨率图像将占用较大的内存空间。一般设置为 FALSE。但sample设置的为true
}
