#include<iostream>
#include<opencv2opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

    int n_boards = 10;
    float image_sf = 0.5f;
    float delay = 1.f;
    int board_w = 3;
    int board_h = 3;
    int board_n = board_w * board_h;
    Size board_sz = Size(board_w, board_h);

    //开启摄像头
    VideoCapture capture(0);
    if (!capture.isOpened()) {
        cout << "nCouldn't open the cameran";
        return -1;
    }

    //分配存储空间
    vector<vector<Point2f>> image_points;
    vector<vector<Point3f>> object_points;

    double last_captured_timestamp = 0;
    Size image_size;

    //不断取图，直到取够n_boards张
    while (image_points.size() < (size_t)n_boards) {
        Mat image0, image;
        capture >> image0;
        image_size = image0.size();
        resize(image0, image, Size(), image_sf, INTER_LINEAR);

        //Find the board
        vector<Point2f> corners;
        bool found = findChessboardCorners(image, board_sz, corners);

        //Draw it 
        drawChessboardCorners(image, board_sz, corners, found);

        double timestamp = (double)clock() / CLOCKS_PER_SEC;

        if (found && timestamp - last_captured_timestamp > 1) {
            last_captured_timestamp = timestamp;
            image ^= Scalar::all(255);
            Mat mcorners(corners);
            mcorners *= (1. / image_sf);
            image_points.push_back(corners);
            object_points.push_back(vector<Point3f>());
            vector<Point3f>& opts = object_points.back();
            opts.resize(board_n);
            for (int j = 0; j < board_n; j++) {
                opts[j] = Point3f((float)(j / board_w), (float)(j % board_w), 0.f);
            }
            cout << "Collected our " << (int)image_points.size() << "of" << n_boards << "needed chessboard imagesn" << endl;
        }
        imshow("calibration", image);
        if ((waitKey(30) & 255) == 27)
            return -1;
    }
    destroyWindow("calibration");
    cout << "nn*** CALIBRATIING THE CAMERA... n" << endl;

    //Calibrate the camera
    Mat intrinsic_matrix, distortion_coeffs;
    double err = calibrateCamera(
        object_points,
        image_points,
        image_size,
        intrinsic_matrix,
        distortion_coeffs,
        noArray(), noArray(),
        CALIB_ZERO_TANGENT_DIST | CALIB_FIX_PRINCIPAL_POINT);

    //保存相机内参和畸变
    cout << "***DONE!nn Reprojection error is" << err << "nStoring Intrinsics.xml and Distortions.xml filesnn";
    FileStorage fs("intrinsics.xml", FileStorage::WRITE);
    fs << "image_width" << image_size.width << "image_height" << image_size.height
        << "camera_matrix" << intrinsic_matrix << "distortion_coeffs" << distortion_coeffs;
    fs.release();

    //加载这些参数
    fs.open("intrinsics.xml", FileStorage::READ);
    cout << "nimage width:" << (int)fs["image_width"];
    cout << "nimage height:" << (int)fs["image_height"];

    Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
    fs["camera_matrix"] >> intrinsic_matrix_loaded;
    fs["distortion_coeffs"] >> distortion_coeffs_loaded;
    cout << "nintrinsic matrix:" << intrinsic_matrix_loaded;
    cout << "ndistortion coefficients:" << distortion_coeffs_loaded << endl;

    //矫正映射
    Mat map1, map2;
    initUndistortRectifyMap(
        intrinsic_matrix_loaded,
        distortion_coeffs_loaded,
        Mat(),
        intrinsic_matrix_loaded,
        image_size,
        CV_16SC2,
        map1,
        map2
    );

    //传入图像,显示的是没有畸变的图像
    for (;;) {
        Mat image, image0;
        capture >> image0;
        if (image0.empty()) break;
        remap(
            image0,
            image,
            map1,
            map2,
            INTER_LINEAR,
            BORDER_CONSTANT, 
            Scalar()
        );
        imshow("Undistorted", image);
        if ((waitKey(30) & 255) == 27) break;
    }
    return 0;
}