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
    Mat src=imread("002.png");
    imshow("src",src);
    GaussianBlur(src,src,Size(5,5),0,0);
    imshow("dst",src);
    waitKey();
    return 0;
}
