#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <cstdlib>
using namespace cv;
using namespace std;

/*为要标定的双目摄像头获取标定图片*/
void display_fuse(Mat& img_1,Mat& img_2,bool addline,string winname);
int main()
{
	VideoCapture camera_left,camera_right;
	Mat frame,frame_left,frame_right;
	string photoName;
	string tem;
	int i_left=0,i_right=0;

	camera_left.open(1);
	camera_right.open(2);
	if (!camera_left.isOpened())
	{
		cout<<"camera_left is't open"<<endl;
	}

	if (!camera_right.isOpened())
	{
		cout<<"camera_right is't open"<<endl;
	}
	while(1)
	{
		camera_left>>frame_left,camera_right>>frame_right;

		if(!frame_left.empty())
		{
			imshow("camera_left",frame_left);//判断是否为空，这步十分重要
			if(i_left==0) {i_left=1;std::cout<<"left_rows="<<frame_left.rows<<",  "<<"left_cols="<<frame_left.cols<<std::endl;}
		}
		if(!frame_right.empty())
		{
			imshow("camera_right",frame_right);//判断是否为空，这步十分重要
			if(i_right==0) {i_right=1;std::cout<<"right_rows="<<frame_right.rows<<",  "<<"right_cols="<<frame_right.cols<<std::endl;}
		}
		//display_fuse(frame_left,frame_right,true,"fuse");
		if(waitKey(30)>=0) //30ms中按任意键进入此if块 30ms一幅图像
		{
		    tem=std::to_string(i_left);
			photoName.clear();
			photoName="left"+tem+".jpg";
			imwrite(photoName,frame_left);
			std::cout<<photoName<<endl;
			photoName.clear();
			photoName="right"+tem+".jpg";
            imwrite(photoName,frame_right);
			std::cout<<photoName<<endl;
            i_left++;i_right++;
		}

		if((char)(waitKey(1))=='e')  {return 0;}//按下e键退出
	}
}

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
