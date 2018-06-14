/*
2018.5.14:使用了新画的20mm边长的6*10个角点的棋盘格,boardSize与squareSize要更改
2018.5.15:重新制作了9*6的30mm边长的标定棋盘,boardSize与squareSize随之改变
*/
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

using namespace std;
using namespace cv;

static bool readStringList(const string& filename,vector<string>& list);
void display_fuse(Mat& img_1,Mat& img_2,bool addlines);
void stereoCalibRectify(vector<string>& imglist, Size boardsize ,bool storeRectified, bool display_corner);

int main()
{
	Size boardSize(9,6);//board's size
	string imgList_file="stereo_calib.xml";
	vector<string> imgList;
	readStringList(imgList_file,imgList);
	stereoCalibRectify(imgList,boardSize,true,true);
}

void stereoCalibRectify(vector<string>& imglist, Size boardsize ,bool storeRectified, bool display_corner)
{
 	CV_Assert(imglist.size() % 2 == 0);//image list contains odd (non-even) number of elements;
	vector<vector<Point2f>> imagePoints[2];//pixel coordinate
    vector<vector<Point3f>> objectPoints;  // world coordinate
	vector<string> goodImageList;// find all corners is goodImage,store in this
	int num_imagePair=(int)imglist.size()/2,i,j,k;//
	imagePoints[0].resize(num_imagePair),imagePoints[1].resize(num_imagePair);//先给容器的容量赋值
	float squareSize=30.f;
	Size imageSize;

	for(i=0,j=0;i<num_imagePair;i++)
	{
		for(k=0;k<2;k++)
		{
			string &filename=imglist[i*2+k];
			Mat img=imread(filename);
			if(img.empty())  {cout<<"read the"<<filename<<"fail";break;}
			if(imageSize == Size() )   {imageSize = img.size();} //如果imageSize为空，就给imageSize赋初值为第一张图片的大小，只赋一次值
			else if( img.size() != imageSize )
			{
				cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";break;
			}
			bool found = false;
			vector<Point2f>& corners = imagePoints[k][j];//将角点的存储位置设为imagePoints的引用，一旦找到角点直接存入imagePoints
			found = findChessboardCorners(img, boardsize, corners,CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);

			/*display the corners*/
			if(found)
			{
				if(display_corner)
				{
					cout << filename << endl;
					if(img.channels()==1) {cvtColor(img, img, COLOR_GRAY2BGR);}//如果原图是灰度图，就转化为彩色图，否则画出的角点连线是灰色的
					Mat img_display=img.clone();
					drawChessboardCorners(img_display, boardsize, corners, found);
					imshow("corners",img_display);
					waitKey(0);
				}

			}
			else   {cout<<filename<<"can't find corners = boardsize"<<endl;break;}

			/*if found=true,find SubPix*/
			Mat gray_img;
			cvtColor(img,gray_img,CV_BGR2GRAY);//寻找亚像素角点必须用灰度图
			cornerSubPix(gray_img, corners, Size(11,11), Size(-1,-1),TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,30, 0.01));
		}
		if( k == 2 )/*k==2意味着左右这组图片都找到了全部的角点，可以作为接下来立体标定的材料存储起来*/
		{
			goodImageList.push_back(imglist[i*2]);
			goodImageList.push_back(imglist[i*2+1]);
			j++;
		}
	}
	cout << j << " pairs have been successfully detected.\n";

	num_imagePair=j;
	imagePoints[0].resize(num_imagePair);
	imagePoints[1].resize(num_imagePair);
	objectPoints.resize(num_imagePair);
	/*填充棋盘格的世界坐标*/
	for(i=0;i<num_imagePair;i++)
	{
		for(j=0;j<boardsize.height;j++)
		{
			for(k=0;k<boardsize.width;k++)
			{
				objectPoints[i].push_back(Point3f(j*squareSize, k*squareSize,0));
			}
		}
	}

	/*left camera calibrate:*/
	cout<<"start left camera calibrate"<<endl;
	Mat M1(3,3,CV_32FC1,Scalar::all(0)),M2(3,3,CV_32FC1,Scalar::all(0)),D1(1,5,CV_32FC1,Scalar::all(0)),D2(1,5,CV_32FC1,Scalar::all(0)),R,T,E,F,R1,R2,P1,P2,Q,map1x,map1y,map2x,map2y;
	vector<Mat> rotate_vector_L,rotate_vector_R;//每幅图像的旋转向量
	vector<Mat> translate_vector_L,translate_vector_R;//每幅图像的平移向量
	double rms_L=calibrateCamera(objectPoints,imagePoints[0],imageSize,M1,D1,rotate_vector_L,translate_vector_L,CV_CALIB_FIX_K3);
	cout<<"rms_L="<<rms_L<<endl;
	cout<<"finish left camera calibrate,display the calibration"<<endl;
	for(i=0;i<imagePoints[0].size();i++)
	{
		cv::Mat src=cv::imread(goodImageList[i*2]),dst;
		cv::undistort(src,dst,M1,D1);
		display_fuse(src,dst,false);
		cv::waitKey();
	}

	/*right camera calibrate:*/
	cout<<"start right camera calibrate"<<endl;
	double rms_R=calibrateCamera(objectPoints,imagePoints[1],imageSize,
		M2,D2,rotate_vector_R,translate_vector_R,CV_CALIB_FIX_K3);
	cout<<"rms_R="<<rms_R<<endl;
	cout<<"finish right camera calibrate,display the calibration"<<endl;
	for(i=0;i<imagePoints[1].size();i++)
	{
		cv::Mat src=cv::imread(goodImageList[i*2+1]),dst;
		cv::undistort(src,dst,M1,D1);
		display_fuse(src,dst,false);
		cv::waitKey();
	}

	/*start stereo calibration:*/
	cout<<"start stereo calibration..."<<endl;
	double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],M1,D1,M2,D2,imageSize, R, T, E, F,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),CV_CALIB_FIX_INTRINSIC );
	/*参考邹宇华17，书P428,P466,CV_CALIB_FIX_INTRINSIC是指不改变M1,D1,M2,D2的值，立体标定完不变
	CV_CALIB_USE_INTRINSIC_GUESS是指将输入的M1,D1,M2,D2的值作为初始值，在立体标定的过程中进行优化，输出的M1,D1,M2,D2是改变了的
	详见opencv官网对参数的解释*/
	cout << "done with RMS error=" << rms << endl;

	/*CALIBRATION QUALITY CHECK*/
	/* because the output fundamental matrix implicitly,includes all the output information,
	we can check the quality of calibration using the epipolar geometry constraint: m2^t*F*m1=0*/
	cout<<"check stereo calibration quality"<<endl;
	double err = 0;
	int npoints = 0;
	vector<Vec3f> lines[2];
	for( i = 0; i < num_imagePair; i++ )
	{
		int npt = (int)imagePoints[0][i].size();
		Mat imgpt[2];
		Mat cameraMatrix[2]={M1,M2},distCoeffs[2]={D1,D2};
		for( k = 0; k < 2; k++ )
		{
			imgpt[k] = Mat(imagePoints[k][i]);
			undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
			computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
		}
		for( j = 0; j < npt; j++ )
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] +
				imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	cout << "average reprojection err = " <<  err/npoints << endl;

	/*start rectify */
	cout<<"start rectify..."<<endl;
	stereoRectify(M1,D1,M2,D2,imageSize,R,T,R1,R2,P1,P2,Q,CALIB_ZERO_DISPARITY, 0, imageSize);
	/*Operation flags that may be zero or docs.opencv.org/2.4.9 .
	If the flag is set, the function makes the principal points of each camera have the same pixel coordinates in the rectified views
	---docs.opencv.org/2.4.9
	也就是flag如果设置为CALIB_ZERO_DISPARITY，则校正后的两个相机具有相同的主点坐标*/
	initUndistortRectifyMap(M1,D1, R1, P1, imageSize, CV_16SC2, map1x, map1y);
	initUndistortRectifyMap(M2,D2, R2, P2, imageSize, CV_16SC2, map2x, map2y);

	/*save param*/
	cout<<"start to store params"<<endl;
	FileStorage fs("intrinsics.yml",CV_STORAGE_WRITE);
	if(fs.isOpened())
	{
		fs<<"M1"<<M1<<"D1"<<D1<<"M2"<<M2<<"D2"<<D2;
		cout<<"success to save the intrinsics paramaters\n";
		fs.release();
	}
	else{cout<<"Error:can not save the intrinsics paramaters\n";}

	fs.open("extrinsics.yml",CV_STORAGE_WRITE);
	if(fs.isOpened())
	{
		fs<<"R"<<R<<"T"<<T<<"R1"<<R1<<"R2"<<R2<< "P1" << P1 << "P2" << P2 << "Q" << Q<<"map1x"<<map1x<<"map1y"<<map1y<<"map2x"<<map2x<<"map2y"<<map2y;
		cout<<"success to save the exteinsics paramaters\n";
		fs.release();
	}
	else{cout<<"Error:can not save the exteinsics paramaters\n";}

	/*display rectify*/
	cout<<"finish rectify,diaplay rectify"<<endl;
	for(i=0;i<num_imagePair;i++)
	{
		Mat img1=imread(goodImageList[2*i]),img2=imread(goodImageList[2*i+1]),r_img1,r_img2;
		remap(img1,r_img1,map1x,map1y,CV_INTER_LINEAR),remap(img2,r_img2,map2x,map2y,CV_INTER_LINEAR);
		if(storeRectified)//whether to store rectified imgs
		{
            string store_filename="rectified_"+goodImageList[2*i];
			imwrite(store_filename,r_img1);
			store_filename="rectified"+goodImageList[2*i+1];
			imwrite(store_filename,r_img2);
		}
		if(img1.channels()==1) {cvtColor(r_img1,r_img1,CV_GRAY2BGR),cvtColor(r_img2,r_img2,CV_GRAY2BGR);}
		display_fuse(r_img1,r_img2,true);
		waitKey();
	}

	cout<<"Congratulations!!! finish stereo calibrate and rectified!!!";
}

static bool readStringList(const string& filename,vector<string>& list)
{
    list.resize(0);
    FileStorage fs(filename,FileStorage::READ);
    if( !fs.isOpened() )  {return false;}
    FileNode n=fs.getFirstTopLevelNode();
    if(n.type()!=FileNode::SEQ)  {return false;}
    FileNodeIterator it=n.begin(),it_end=n.end();
    for(;it!=it_end;++it)
    {list.push_back((string)*it);}
    return true;
}

/*input two image,display fuse image*/
void display_fuse(Mat& img_1,Mat& img_2,bool addlines)
{
	Mat canvas(img_1.rows,2*img_1.cols,CV_8UC3);
	Mat canvas_part_L=canvas(Rect(0,0,img_1.cols,img_1.rows));
	resize(img_1,canvas_part_L,img_1.size(), 0, 0, CV_INTER_AREA);
	Mat canvas_part_R=canvas(Rect(img_1.cols,0,img_1.cols,img_1.rows));
	resize(img_2,canvas_part_R,img_1.size(), 0, 0, CV_INTER_AREA);
	if(addlines)
	{
    for( int j = 0; j < canvas.rows; j += 16 ) {line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);}
	}
	imshow("fuse",canvas);
}
