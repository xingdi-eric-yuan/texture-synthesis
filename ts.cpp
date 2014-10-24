// Texture Synthesis
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#define ATD at<double>
#define elif else if

using namespace cv;
using namespace std;

typedef struct _unfilled_point{
    Point2i pt;
    int fNeighbors;

}ufp;

typedef struct _Matches{
    Point2i pt;
    double err;
}matches;

bool SortUFP(const ufp &v1, const ufp &v2){  
    return v1.fNeighbors > v2.fNeighbors;
}  

bool allFilled(const Mat &mat){
    Mat tmp = (mat != 0);
    tmp.convertTo(tmp, CV_64FC1, 1.0 / 255.0);
    return sum(tmp)[0] == (tmp.cols * tmp.rows);
}

bool isinImage(int y, int x, const Mat &img){
    return (x >= 0 && x < img.cols && y >= 0 && y < img.rows);
}

vector<Point2i> GetUnfilledNeighbors(const Mat& map){
    vector <ufp> data;
    for(int i = 0; i < map.rows; i++){
        for(int j = 0; j < map.cols; j++){
            if(map.ATD(i, j) != 0) continue;
            int temp = 0;
            if(isinImage(i - 1, j - 1, map)) temp += map.ATD(i - 1,  j - 1);
            if(isinImage(i - 1, j, map))     temp += map.ATD(i - 1,  j);
            if(isinImage(i - 1, j + 1, map)) temp += map.ATD(i - 1,  j + 1);
            if(isinImage(i, j - 1, map))     temp += map.ATD(i,      j - 1);
            if(isinImage(i, j + 1, map))     temp += map.ATD(i,      j + 1);
            if(isinImage(i + 1, j - 1, map)) temp += map.ATD(i + 1,  j - 1);
            if(isinImage(i + 1, j, map))     temp += map.ATD(i + 1,  j);
            if(isinImage(i + 1, j + 1, map)) temp += map.ATD(i + 1,  j + 1);
            ufp tufp;
            tufp.pt = Point2i(j, i);
            tufp.fNeighbors = temp;
            data.push_back(tufp);
        }
    }
    random_shuffle(data.begin(), data.end());
    sort(data.begin(), data.end(), SortUFP);
    vector <Point2i> result;    
    for(int i = 0; i < data.size(); i++){
        if(data[i].fNeighbors > 0){
            result.push_back(data[i].pt);
        }
    }
    data.clear();
    return result;
}

Mat getNeigborhoodWindow(const Mat &img, Point2i pt, int windowSize){
    Mat result = Mat(windowSize * 2 + 1, windowSize * 2 + 1, CV_64FC1);
    for(int i = 0; i < result.rows; i++){
        for(int j = 0; j < result.cols; j++){
            if(isinImage(pt.y - windowSize + i, pt.x - windowSize + j, img)){
                result.ATD(i, j) = img.ATD(pt.y - windowSize + i, pt.x - windowSize + j);
            }else result.ATD(i, j) = -1;
        }
    }
    return result;
}

vector<matches> FindMatches(Mat Template, Mat SampleImage, Mat img, Point2i templateCenter, int windowSize, Mat Map){
    Mat SampleF, TemplateF;   
    int topleftx = templateCenter.x - windowSize;
    int toplefty = templateCenter.y - windowSize;
    // ValidMask
    Mat ValidMask = Mat::zeros(Template.size(), CV_64FC1);
    for(int i = 0; i < ValidMask.rows; i++){
        for(int j = 0; j < ValidMask.cols; j++){
            if(isinImage(toplefty + i, topleftx + j, img)){
                ValidMask.ATD(i, j) = Map.ATD(i + toplefty, j + topleftx);
            }else ValidMask.ATD(i, j) = 0.0;
        }
    }
    // GaussMask
    double kernalsize = (double)windowSize / 6.0;
    kernalsize = sqrt(kernalsize);
    Mat tmpGaussian = getGaussianKernel(windowSize * 2 + 1, kernalsize); 
    Mat GaussianMask = tmpGaussian * tmpGaussian.t(); 
    // TotWeight
    double TotWeight = sum(ValidMask.mul(GaussianMask))[0];
    // Sum of Squared Differences.
    Mat SSD = Mat::zeros(Template.size(), CV_64FC1);
    int xPara = SampleImage.cols / 2 - windowSize;
    int yPara = SampleImage.rows / 2 - windowSize;
    double minSSD = (double)INT_MAX;

    for(int i = 0; i < Template.rows; i++){
        for(int j = 0; j < Template.cols; j++){
            if(!isinImage(toplefty + i, topleftx + j, img)){
                SSD.ATD(i, j) = -1.0;
                continue;
            }
            Mat dist = Mat::zeros(2 * windowSize + 1, 2 * windowSize + 1, CV_64FC1);
            Mat tp1 = Template;
            Mat tp2 = SampleImage(Rect(j + xPara - windowSize, i + yPara - windowSize, 2 * windowSize + 1, 2 * windowSize + 1));
            dist = tp1 - tp2;
            pow(dist, 2.0, dist);
            SSD.ATD(i, j) = sum(dist.mul(ValidMask.mul(GaussianMask)))[0] / TotWeight;
        }
    }

    for(int i = 0; i < Template.rows; i++){
        for(int j = 0; j < Template.cols; j++){
            if(SSD.ATD(i, j) == -1) continue;
            if(SSD.ATD(i, j) < minSSD) minSSD = SSD.ATD(i, j);
        }
    }
    double ErrThreshold = 0.1;
    vector<matches> PixelList;
    for(int i = 0; i < Template.rows; i++){
        for(int j = 0; j < Template.cols; j++){
            if(SSD.ATD(i, j) == -1) continue;
            if(SSD.ATD(i, j) <= minSSD * (1 + ErrThreshold)){
                matches tpmatch;
                tpmatch.pt = Point2i(j + xPara, i + yPara);
                tpmatch.err = fabs(minSSD - SSD.ATD(i, j));
                PixelList.push_back(tpmatch);
            }
        }
    }
    return PixelList;
}

Mat growImage(const Mat &SampleImage, Mat &Image, int windowSize, Mat &map){
    double MaxErrThreshold = 0.3;
    while(!allFilled(map)){
        bool progress = false;
        vector<Point2i> PixelList = GetUnfilledNeighbors(map);
        //cout<<"PixelList.size = "<<PixelList.size()<<endl;
        for(int i = 0; i < PixelList.size(); i++){
            vector<matches> BestMatches;
            Mat Template = getNeigborhoodWindow(Image, PixelList[i], windowSize);
            BestMatches = FindMatches(Template, SampleImage, Image, PixelList[i], windowSize, map);
            // randomly pick one
            int which = rand() % BestMatches.size();
            if(BestMatches[which].err < MaxErrThreshold){
                Image.ATD(PixelList[i].y, PixelList[i].x) = SampleImage.ATD(BestMatches[which].pt.y, BestMatches[which].pt.x);
                map.ATD(PixelList[i].y, PixelList[i].x) = 1.0;
                progress = true;
            }
        }         
        if(!progress){
            MaxErrThreshold *= 1.1;
        }   
    }
    cout<<"MaxErrThreshold = "<<MaxErrThreshold<<endl;
    return Image;
}

//expand gray image
Mat expandImage(const Mat &img, int x_expand, int y_expand, int windowSize){
    
    Mat newimg = Mat::zeros(img.rows + y_expand * 2, img.cols + x_expand * 2, CV_64FC1);
    Rect rect(x_expand, y_expand, img.cols, img.rows);
    img.copyTo(newimg(rect));
    int width = newimg.cols;
    int height = newimg.rows;
    Mat map = Mat::zeros(height, width, CV_64FC1);
    map(rect) = Mat::ones(img.rows, img.cols, CV_64FC1);
    //now build a map of which pixels are unfilled pixels
    if((img.cols >= windowSize * 4) && (img.rows >= windowSize * 4)){
        Mat result = growImage(img, newimg, windowSize, map);
        return result;
    }else{
        cout<<"Window Size too big, it must less than or equal to "<<(img.cols < img.rows ? img.cols : img.rows) / 4<<endl;
        return img;
    }
}

int main(){

    // Extend  gray image.
    Mat img = imread("D34.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    img.convertTo(img, CV_64FC1, 1.0/255.0, 0);
    Mat result = expandImage (img, 75, 75, 17);
    imshow("img", img);
    imshow("result", result);

    waitKey(0);
    return 0;
}