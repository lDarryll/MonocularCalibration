#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
// #include <opencv2\imgproc\types_c.h> 1.091680007048003
#include <opencv2/imgproc/types_c.h>
#include <iostream>
#include <fstream>
#include <float.h>
#include <vector>
using namespace cv;
using namespace std;

template<typename T>
void printVector(vector<T>& v) {

	for(int i=0; i<v.size(); i++)
    {
        cout << v[i] << " ";
    }
	cout << endl;
}
//计算以板子中心为原点的各个点的坐标
vector<cv::Point2f> get_world_point_base_board()
{
    vector<cv::Point2f> points2f;
    vector<double> pitches; //板子相邻点之间的距离（左右点距离，上下点距离）
    pitches.assign(2, 2);
    vector<int> board_size; // 板子的大小
    board_size.assign(2, 7);
    // 以板子中心为原点计算出左上角第一个点的坐标
    float xs = -0.5 * (board_size[1] - 1) * pitches[1];
    float ys = -0.5 * (board_size[0] - 1) * pitches[0];
    for (int yi = 0; yi < board_size[0]; ++yi, ys += pitches[0])
    {
        float xss = xs;
        for (int xi = 0; xi < board_size[1]; ++xi, xss += pitches[1])
        {
            points2f.emplace_back(cv::Point2f{xss, ys});
        }
    }
    return points2f;
}
bool loadMatDist(cv::FileStorage &fs, cv::Mat &mtx, cv::Mat &dist)
{
    if (fs["mtx"].isNone() || fs["dist"].isNone())
    {
        cout << "Nodes mtx and/or dist are not in the camera file" << endl;
        return false;
    }
    fs["mtx"] >> mtx;
    fs["dist"] >> dist;

    return true;
}
// 加载相机内参
void loadIntrinsics(string const &filename, cv::Mat &mtx, cv::Mat &dist)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "Failed to open camera file: " << filename << endl;
    }

    cout << "Read intrinsic from file: " << filename << endl;
    if (!loadMatDist(fs, mtx, dist))
    {
        cout << "Failed to load camera intrinsics from file: " << filename << endl;
    }
    cout << "Success to load camera intrinsics from file: " << filename << endl;
    fs.release();
}

// 加载图片
void loadImage(string const &filepath, cv::Mat &img)
{
    img = cv::imread(filepath);
    Size image_size;
    image_size.width = img.cols;
    image_size.height = img.rows;
    cout << "image_size.width = " << image_size.width << endl;
    cout << "image_size.height = " << image_size.height << endl;
}

//转为灰度图
void conver_BGR2GRAY(cv::Mat &input, cv::Mat &gray, string const &save_path)
{
    cv::cvtColor(input, gray, CV_RGB2GRAY);
    cout << "Success convert image to gray" << endl;
    cv::imwrite(save_path + "/gary.jpg", gray);
}

//寻找角点
void findCorners(cv::Mat &img, cv::Mat &view_gray, Size const &board_size,
                 vector<cv::Point2f> &image_2fpoints_buf, vector<cv::Point2f> &image_2fpoints_seq)
{
    /* 提取角点 */
    if (0 == findChessboardCorners(view_gray, board_size, image_2fpoints_buf))
    {
        cout << "can not find chessboard corners!\n"; //找不到角点
        // exit(1);
    }
    else
    {
        /* 亚像素精确化 */
        find4QuadCornerSubpix(view_gray, image_2fpoints_buf, Size(5, 5)); //对粗提取的角点进行精确化

        // image_2fpoints_seq.push_back(image_2fpoints_buf); //保存亚像素角点
        image_2fpoints_seq.assign(image_2fpoints_buf.begin(), image_2fpoints_buf.end());
        /* 在图像上显示角点位置 */
        drawChessboardCorners(img, board_size, image_2fpoints_buf, true); //用于在图片中标记角点
        string src = "/mnt/7c29025e-8959-4011-9c6c-607048c15cfc/ZS/job_works/calib/MonocularCalibration/temp_and_result/";
        imwrite(src + "/plotp.jpg", img);
        int npts = image_2fpoints_seq.size();
        if (board_size.width * board_size.height != npts)
        {
            cout << board_size.width * board_size.height << "  " << npts << endl;
            cout << "Failed to find circular grid";
        }
        cout << "Success find chessboard corners " << endl;
        cout << board_size.width * board_size.height << "  " << npts << endl;
    }
}




int main()
{
    // int i=0;
    // for (vector<cv::Point2f>::iterator it = world_point_base_board.begin(); it != world_point_base_board.end(); it++) {
    // 	cout << *it << " ";
    //     cout << endl;
    //     ++i;
    // }
    // cout << i <<endl;

    //计算以板子中心为原点的各个点的坐标
    vector<cv::Point2f> world_point_base_board = get_world_point_base_board();
    cv::Mat _cameraMatrix;
    cv::Mat _distCoeffs;
    cv::Mat _rvec;
    cv::Mat _tvec;
    cv::Mat _image;
    cv::Mat _view_gray;
    cv::Mat _img_thres;
    Size _board_size = Size(7, 7);
    vector<float> _board_center;
    _board_center.push_back(0);
    _board_center.push_back(-19.2);
    _board_center.push_back(187.3);

    vector<cv::Point2f> _image_2fpoints_buf;   /* 缓存每幅图像上检测到的角点 */
    vector<cv::Point2f> _image_2fpoints_seq;   /* 保存检测到的所有角点 */
    vector<cv::Point2f> _ideal_image_2fpoints; /* 保存检测到的所有角点 */
    
    vector<cv::Point3f> _world_pt;
    // _ideal_image_2fpoints.swap();
    string in_params_path = "/mnt/7c29025e-8959-4011-9c6c-607048c15cfc/ZS/job_works/calib/MonocularCalibration/temp_and_result/calib_Intrinsics.yaml";
    string calib_out_img_path = "/mnt/7c29025e-8959-4011-9c6c-607048c15cfc/ZS/job_works/calib/MonocularCalibration/calib_imgs/out_images/WIN_20220729_17_09_37_Pro.jpg";
    string save_path = "/mnt/7c29025e-8959-4011-9c6c-607048c15cfc/ZS/job_works/calib/MonocularCalibration/temp_and_result";
    loadIntrinsics(in_params_path, _cameraMatrix, _distCoeffs);
    loadImage(calib_out_img_path, _image);
    conver_BGR2GRAY(_image, _view_gray, save_path);
    // cv::adaptiveThreshold(_view_gray, _img_thres, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, );
    findCorners(_image, _view_gray, _board_size, _image_2fpoints_buf, _image_2fpoints_seq);

    // vector<cv::Point2f>(_image_2fpoints_seq.size()).swap(_ideal_image_2fpoints);
    cv::undistortPoints(_image_2fpoints_seq, _ideal_image_2fpoints, _cameraMatrix, _distCoeffs);
    for (int i = 0; i != 49; ++i)
    {
        cout << "(" << _image_2fpoints_seq[i].x << ", " << _image_2fpoints_seq[i].y << ") ==> ("
             << _ideal_image_2fpoints[i].x << ", " << _ideal_image_2fpoints[i].y << ")" << endl;
        cout << i << endl;
    }
    
    // printVector<float>(_board_center);
    
    float z=0;
    int nfeat = world_point_base_board.size();
    if (0 == nfeat) {
        cout << "WorldPointsEngine: no template points are given"<<endl;;
        
    }
    float x = _board_center[0];
    float y = _board_center[1];
    for (auto const &pt : world_point_base_board) {
        _world_pt.emplace_back(cv::Point3f{pt.x + x, y + pt.y, z});
    }
    // for (auto const &pt : _world_pt) {
    //     cout << "(" << pt.x << ", " << pt.y << ", " << pt.z << ")"<<endl;
    // }

    bool ret = cv::solvePnP(_world_pt, _image_2fpoints_seq, _cameraMatrix, _distCoeffs,
                            _rvec, _tvec, false);


    

    return 0;
}
