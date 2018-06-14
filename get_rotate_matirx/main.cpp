/*
2018.5.16:使用了新的求旋转矩阵的图片img2,6*9棋盘格,30mm,高度600,第一行棋盘格距离设定机器人坐标系原点425mm,这次角点从右下角开始,
右下角第一个检测到的角点在机器人坐标系下的坐标为(120,455,600)
2018.5.16:使用了新的求旋转矩阵的图片left1.jpg,6*9棋盘格,30mm,放在地面上,第一行棋盘格距离设定机器人坐标系原点877mm,这次从左上角开始,
左上角第一个角点坐标为(-120,877+180,4)
2018.5.16:使用了新的求旋转矩阵的图片left1.jpg,6*9棋盘格,30mm,放在地面上,第一行棋盘格距离设定机器人坐标系原点710mm,这次从右上角开始,
右下角第一个检测到的角点在机器人坐标系下的坐标为(120,740,4)
2018.5.17:使用了新的求旋转矩阵的图片left1.jpg,6*9棋盘格,30mm,放在地面上,第一行棋盘格距离设定机器人坐标系原点647mm,这次从右上角开始,
右下角第一个检测到的角点在机器人坐标系下的坐标为(120,677,4)
2018.5.22:将机器人坐标系中心由两驱动轮中心变为惯导中心,惯导中心与驱动轮中心相距118,
则右下角第一个检测到的角点在机器人坐标系下的坐标由(120,677,4)变为(120,677+118,4)=(120,795,4)
2018.6.13:左上角开始：第一行距离565ｍｍ，(-150,565+150+118,4)=(-150,833,4)
*/
/*获得摄像机坐标系到机器人坐标系的旋转矩阵*/
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/contrib/contrib.hpp>
#include<opencv2/calib3d/calib3d.hpp>
using namespace cv;
using namespace std;

int main()
{
	Mat M,D;
	FileStorage fs("intrinsics.yml",CV_STORAGE_READ);
	if(!fs.isOpened()) {std::cout<<"fail to open intrinsics.yml";return -1;}
	fs["M1"]>>M,fs["D1"]>>D;
	std::cout<<"success to read intrinsic paramater!"<<endl;

	vector<Point2f> corners;
	vector<Point3f> world_corners;
	bool found;
	Size boardsize(9,6);
	float squareSize=30.f;
	Mat src=imread("left1.jpg"),canvas,vec_rotate,matrix_rotate,vec_tran,gray_img;
	if(src.empty()) {std::cout<<"read src error!"<<endl;}
	found = findChessboardCorners(src, boardsize, corners,CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
	/*
	这里好像应该加一步提取亚像素精度角点更好--2018.3.20 做毕业论文时
	*/

	if(found)
	{
		canvas=src.clone();
		drawChessboardCorners(canvas,boardsize,corners,found);
		imshow("corners",canvas);
		waitKey(500);
	}
	else {cout<<"can't find corners = boardsize"<<endl;return 0;}
	/*寻找亚像素角点*/
//	cvtColor(src,gray_img,CV_BGR2GRAY);
//	cornerSubPix(gray_img, corners, Size(11,11), Size(-1,-1),TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,30, 0.01));
	cout<<corners<<endl;

	/*填充棋盘格的世界坐标
	这里要注意第一个角点是左上角出现的还是右下角出现的，
	根据对应关系填充坐标*/
	//cout<<boardsize.width<<"   "<<boardsize.height<<endl;
	for(int i=0;i<boardsize.height;i++)//6
	{
		for(int j=0;j<boardsize.width;j++)//9
		{
			world_corners.push_back(Point3f(-150+j*squareSize,833-i*squareSize,4));//2018.5.16
		}
	}

	solvePnP(world_corners,corners,M,D,vec_rotate,vec_tran);//=opencv1.0 cvFindExtrinsicCameraParams2
	Mat world_camera=Mat::eye(4,4,CV_64FC1);
	Rodrigues(vec_rotate,matrix_rotate);
	for(int i=0;i<matrix_rotate.rows;i++)
		for(int j=0;j<matrix_rotate.cols;j++)
		{
			world_camera.at<double>(i,j)=matrix_rotate.at<double>(i,j);
		}
	world_camera.at<double>(0,3)=vec_tran.at<double>(0,0);
	world_camera.at<double>(1,3)=vec_tran.at<double>(1,0);
	world_camera.at<double>(2,3)=vec_tran.at<double>(2,0);
	std::cout<<vec_rotate<<endl;
	std::cout<<vec_tran<<endl;
	std::cout<<matrix_rotate<<endl;
	std::cout<<world_camera<<endl;
	//std::cout<<corners<<endl;

	fs.open("rotate.yml",CV_STORAGE_WRITE);
	if(fs.isOpened())
	{
		fs<<"rotate_M"<<matrix_rotate,fs<<"transform_V"<<vec_tran,fs<<"world_camera"<<world_camera;
		fs.release();
	}

	waitKey();

}
