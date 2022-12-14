#include <iostream>
#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <math.h>
#include<iostream>
#include <fstream>
#include<random>
#include<windows.h>

#include "HighSpeedProjector.h"
#include "ProjectorUtility.h"

// Include OpenCV
#include <opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/calib3d/calib3d.hpp>
// Include GLEW
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include<GL/glut.h>

using namespace cv;
using namespace std;

VideoCapture cap(0); // 使用する動画
const uint WIDTH = 1024, HEIGHT = 768; // 画像のサイズ
Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
Rect roi;
GLfloat r = 1, theta = 30, phi = 30;
const int imgNum = 1;

string writePath = "D:/IE浏览器下载/OpenGL_Test/AR_DATA";
float tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y;

void cameraplay() {
	VideoCapture capture(0);
	string name;
	namedWindow("hello", WINDOW_AUTOSIZE);
	int i = 0;
	while (1) {
		Mat frame;
		capture >> frame;
		flip(frame, frame, 1);
		if (32 == waitKey(20)) {			//空格拍照
			name = writePath + to_string(i) + ".jpg";
			cv::imwrite(name, frame);
			cout << name << endl;
			i++;
		}
		if (97 == waitKey(10)) {			//'a'退出
			break;
		}
		imshow("Capture", frame);
	}
}
void CalibrateCamera() {
	Mat image, img_gray;
	cv::Size patternsize(8, 5);//棋盘格每行每列角点个数
	vector<vector<Point3f>> objpoints_img;//保存棋盘格上角点的三维坐标
	vector<Point3f> obj_world_pts;//三维世界坐标
	vector<vector<Point2f>> images_points;//保存所有角点
	vector<Point2f> img_corner_points;//保存每张图检测到的角点
	vector<String> images_path;//创建容器存放读取图像路径
	string image_path = "D:/IE浏览器下载/OpenGL_Test/AR_DATA/Ca_Matrix/*.jpg";//待处理图路径
	glob(image_path, images_path);//读取指定文件夹下图像
		//转世界坐标系
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 8; j++) { obj_world_pts.push_back(Point3f(j, i, 0)); }
	}
	for (int i = 0; i < images_path.size(); i++)
	{
		image = imread(images_path[i]);
		cvtColor(image, img_gray, COLOR_BGR2GRAY);
		bool found_success = findChessboardCorners(img_gray, patternsize,//检测角点
			img_corner_points, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
		if (found_success) { cout << "..."; }
		else { cout << "error!"; }
		//显示角点
		if (found_success) {
			TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);//迭代终止条件
			cornerSubPix(img_gray, img_corner_points, Size(11, 11), Size(-1, -1), criteria);//进一步提取亚像素角点
			drawChessboardCorners(image, patternsize, img_corner_points, found_success);//绘制角点
			imshow("Test Corner", image);
			waitKey(400);
			destroyAllWindows();
			objpoints_img.push_back(obj_world_pts);//从世界坐标系到相机坐标系
			images_points.push_back(img_corner_points);
		}
	}
	Mat cameraMatrix, distCoeffs, R, T;//内参矩阵，畸变系数，旋转量，偏移量
	calibrateCamera(objpoints_img, images_points, img_gray.size(), cameraMatrix, distCoeffs, R, T);
	cout << " " << endl;
	cout << "--------------------------" << endl; cout << "cameraMatrix:" << endl; cout << cameraMatrix << endl;
	cout << "--------------------------" << endl; cout << "distCoeffs:" << endl; cout << distCoeffs << endl;
	cout << "--------------------------" << endl; cout << "Rotation vector:" << endl; cout << R << endl;
	cout << "--------------------------" << endl; cout << "Translation vector:" << endl; cout << T << endl;
	cout << "--------------------------" << endl;
}
static void init()
{
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glClearColor(0.0, 0.0, 0.0, 1.0);
}
class MarkerDetect
{
	Mat Image, ImageGray, ImageAdaptiveBinary; //分别是 原图像 灰度图像 自适应阈值化图像
	vector<vector<Point>> ImageContours; //图像所有边界信息
	vector<vector<Point2f>> ImageQuads, ImageMarkers; //图像所有四边形 与 验证成功的四边形
	vector<Point2f> FlatMarkerCorners; //正方形化标记时用到的信息
	Size FlatMarkerSize;
	//正方形化标记时用到的信息7x7黑白标记的颜色信息
	uchar CorrectMarker[7 * 7] = {
		0,0,0,0,0,0,0,
		0,255,255,255,255,255,0,
		0,255,0,255,0,255,0,
		0,255,255,255,255,255,0,
		0,255,0,255,0,255,0,
		0,255,255,255,255,255,0,
		0,0,0,0,0,0,0 };
	// 用于新一帧处理前的初始化
	void Clean() {
		ImageContours.clear();
		ImageQuads.clear();
		ImageMarkers.clear();
	}
	//转换图片颜色GRAY
	void ConvertColor() {
		cvtColor(Image, ImageGray, CV_BGR2GRAY);
		adaptiveThreshold(ImageGray, ImageAdaptiveBinary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 7);
	}
	//获取图片所有边界
	void GetContours(int ContourCountThreshold) {
		vector<vector<Point>> AllContours; // 所有边界信息
		findContours(ImageAdaptiveBinary, AllContours,
			CV_RETR_LIST, CV_CHAIN_APPROX_NONE); // 用自适应阈值化图像寻找边界
		for (size_t i = 0; i < AllContours.size(); ++i) { // 只储存低于阈值的边界
			int contourSize = AllContours[i].size();
			if (contourSize > ContourCountThreshold) { ImageContours.push_back(AllContours[i]); }
		}
	}
	//寻找所有四边形 TargetPoints[]
	void FindQuads(int ContourLengthThreshold)
	{
		vector<vector<Point2f>> PossibleQuads;
		for (int i = 0; i < ImageContours.size(); ++i)
		{
			vector<Point2f> InDetectPoly;
			approxPolyDP(ImageContours[i], InDetectPoly,
				ImageContours[i].size() * 0.05, true); // 对边界进行多边形拟合
			if (InDetectPoly.size() != 4) continue;// 只对四边形感兴趣
			if (!isContourConvex(InDetectPoly)) continue; // 只对凸四边形感兴趣
			float MinDistance = 1e10; // 寻找最短边
			for (int j = 0; j < 4; ++j)
			{
				Point2f Side = InDetectPoly[j] - InDetectPoly[(j + 1) % 4];
				float SquaredSideLength = Side.dot(Side);
				MinDistance = min(MinDistance, SquaredSideLength);
			}
			if (MinDistance < ContourLengthThreshold) continue; // 最短边必须大于阈值
			vector<Point2f> TargetPoints;
			for (int j = 0; j < 4; ++j) // 储存四个点
			{
				TargetPoints.push_back(Point2f(InDetectPoly[j].x, InDetectPoly[j].y));
			}
			Point2f Vector1 = TargetPoints[1] - TargetPoints[0]; // 获取一个边的向量
			Point2f Vector2 = TargetPoints[2] - TargetPoints[0]; // 获取一个斜边的向量
			if (Vector2.cross(Vector1) < 0.0) // 计算两向量的叉乘 判断点是否为逆时针储存
				swap(TargetPoints[1], TargetPoints[3]); // 如果大于0则为顺时针，需要交替
			PossibleQuads.push_back(TargetPoints); // 保存进可能的四边形，进行进一步判断
		}
		vector<pair<int, int>> TooNearQuads; // 准备删除几组靠太近的多边形
		for (int i = 0; i < PossibleQuads.size(); ++i) {
			vector<Point2f>& Quad1 = PossibleQuads[i]; // 第一个             
			for (int j = i + 1; j < PossibleQuads.size(); ++j) {
				vector<Point2f>& Quad2 = PossibleQuads[j]; // 第二个
				float distSquared = 0;
				float x1Sum = 0.0, x2Sum = 0.0, y1Sum = 0.0, y2Sum = 0.0, dx = 0.0, dy = 0.0;
				for (int c = 0; c < 4; ++c) {
					x1Sum += Quad1[c].x;
					x2Sum += Quad2[c].x;
					y1Sum += Quad1[c].y;
					y2Sum += Quad2[c].y;
				}
				x1Sum /= 4; x2Sum /= 4; y1Sum /= 4; y2Sum /= 4; // 计算平均值（中点）
				dx = x1Sum - x2Sum;
				dy = y1Sum - y2Sum;
				distSquared = sqrt(dx * dx + dy * dy); // 计算两多边形距离
				if (distSquared < 50) {
					TooNearQuads.push_back(pair<int, int>(i, j)); // 过近则准备剔除
				}
			}
		}
		vector<bool> RemovalMask(PossibleQuads.size(), false); // 移除标记列表
		for (int i = 0; i < TooNearQuads.size(); ++i)
		{
			float p1 = CalculatePerimeter(PossibleQuads[TooNearQuads[i].first]);  //求周长
			float p2 = CalculatePerimeter(PossibleQuads[TooNearQuads[i].second]);
			int removalIndex;  //移除周长小的多边形
			if (p1 > p2) removalIndex = TooNearQuads[i].second;
			else removalIndex = TooNearQuads[i].first;
			RemovalMask[removalIndex] = true;
		}
		for (size_t i = 0; i < PossibleQuads.size(); ++i)
		{
			// 只录入没被剔除的多边形
			if (!RemovalMask[i]) ImageQuads.push_back(PossibleQuads[i]);
		}
	}
	//变换为正方形并验证是否为标记
	void TransformVerifyQuads()
	{
		Mat FlatQuad;
		for (size_t i = 0; i < ImageQuads.size(); ++i)
		{
			vector<Point2f>& Quad = ImageQuads[i];
			Mat TransformMartix = getPerspectiveTransform(Quad, FlatMarkerCorners);
			warpPerspective(ImageGray, FlatQuad, TransformMartix, FlatMarkerSize);
			threshold(FlatQuad, FlatQuad, 0, 255, THRESH_OTSU); // 变为二值化图像
			if (MatchQuadWithMarker(FlatQuad)) {        // 与正确标记比对
				ImageMarkers.push_back(ImageQuads[i]); // 成功则记录
			}
			else { // 如果失败，则旋转，每次90度进行比对
				for (int j = 0; j < 3; ++j) {
					rotate(FlatQuad, FlatQuad, ROTATE_90_CLOCKWISE);
					if (MatchQuadWithMarker(FlatQuad)) {
						ImageMarkers.push_back(ImageQuads[i]); // 成功则记录
						break;
					}
				}
			}
		}
	} //变换为正方形并验证是否为标记
	//Draw the Marker Border
	void DrawMarkerBorder(Scalar Color) //绘制标记边界和点
	{
		for (vector<Point2f> Marker : ImageMarkers)
		{
			line(Image, Marker[0], Marker[1], Color, 2, LINE_AA);
			line(Image, Marker[1], Marker[2], Color, 2, LINE_AA);
			line(Image, Marker[2], Marker[3], Color, 2, LINE_AA);
			line(Image, Marker[3], Marker[0], Color, 2, LINE_AA);//LINE_AA是抗锯齿
			circle(Image, Marker[0], 5, (0, 255, 0), 5, LINE_8, 0);
			circle(Image, Marker[1], 5, (0, 255, 0), 5, LINE_8, 0);
			circle(Image, Marker[2], 5, (0, 255, 0), 5, LINE_8, 0);
			circle(Image, Marker[3], 5, (0, 255, 0), 5, LINE_8, 0);
			tl_x = (int)Marker[0].x;
			tl_y = (int)Marker[0].y;
			tr_x = (int)Marker[1].x;
			tr_y = (int)Marker[1].y;
			br_x = (int)Marker[2].x;
			br_y = (int)Marker[2].y;
			bl_x = (int)Marker[3].x;
			bl_y = (int)Marker[3].y;
		}
	}
	// 检验正方形是否为标记
	bool MatchQuadWithMarker(Mat& Quad) {
		int  Pos = 0;
		for (int r = 2; r < 33; r += 5) // 正方形图像大小为(35,35)
		{
			for (int c = 2; c < 33; c += 5)// 读取每块图像中心点
			{
				uchar V = Quad.at<uchar>(r, c);
				uchar K = CorrectMarker[Pos];
				if (K != V) // 与正确标记颜色信息比对
					return false;
				Pos++;
			}
		}
		return true;
	}
	// 计算周长
	float CalculatePerimeter(const vector<Point2f>& Points) {
		float sum = 0, dx, dy;
		for (size_t i = 0; i < Points.size(); ++i)
		{
			size_t i2 = (i + 1) % Points.size();
			dx = Points[i].x - Points[i2].x;
			dy = Points[i].y - Points[i2].y;
			sum += sqrt(dx * dx + dy * dy);
		}
		return sum;
	}

public:
	MarkerDetect() // 构造函数
	{
		FlatMarkerSize = Size(35, 35);
		FlatMarkerCorners = { Point2f(0,0),Point2f(FlatMarkerSize.width - 1,0),Point2f(FlatMarkerSize.width - 1,FlatMarkerSize.height - 1),Point2f(0,FlatMarkerSize.height - 1) };
	}
	Mat Process(Mat& Image)// 处理一帧图像
	{
		Clean(); // 新一帧初始化
		Image.copyTo(this->Image); // 复制原始图像到Image中
		ConvertColor(); // 转换颜色
		GetContours(50); // 获取边界
		FindQuads(100); // 寻找四边形
		TransformVerifyQuads(); // 变形并校验四边形
		DrawMarkerBorder(Scalar(255, 0, 0)); // 在得到的标记周围画边界Scalar(颜色）
		return this->Image; // 返回结果图案
	}
};

void Transform() {
	vector<Point2d> image_points;
	image_points.push_back(Point2d(320, 0));
	image_points.push_back(Point2d(640, 0));
	image_points.push_back(Point2d(640, 240));
	image_points.push_back(Point2d(320, 240));
	std::vector<Point3f> model_points;
	model_points.push_back(Point3f(0.0f, +1.0f, 0));
	model_points.push_back(Point3f(+1.0f, +1.0f, 0));
	model_points.push_back(Point3f(+1.0f, 0.0f, 0));
	model_points.push_back(Point3f(0.0f, 0.0f, 0));
	Mat camera_matrix = (Mat_<double>(3, 3) << 627.4745076274827, 0, 315.8798583648381, 0, 626.7883021452898, 250.3288591636727, 0, 0, 1);
	Mat dist_coeffs = (Mat_<double>(5, 1) << 0.0408234533262983, -0.4041857394382163, 0.008323029967130271, -0.003821790226485829, 0.814698774630893);
	Mat rotation_vector;
	Mat translation_vector;
	solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
		rotation_vector, translation_vector, 0, SOLVEPNP_EPNP);
	vector<Point3f> points3D_image;
	points3D_image.push_back(Point3f(0, 0, 0));

	vector<Point2f> points_reproj;
	projectPoints(points3D_image, rotation_vector, translation_vector, camera_matrix, dist_coeffs, points_reproj);
	//circle(temp, points_reproj[0], 3, Scalar(0, 0, 255), -1);
	//imshow("point", temp);
	Mat Rvec; Mat_<float> Tvec;
	rotation_vector.convertTo(Rvec, CV_32F); translation_vector.convertTo(Tvec, CV_32F);
	Mat_<float> rotMat(3, 3); Rodrigues(Rvec, rotMat); // transfer the vector to matrix
	//////////////////////////////////////////////
	//   OpenGL部分  //
	//////////////////////////////////////////////
	//读取四个顶点的xy坐标
	glEnable(GL_DEPTH_TEST);
	//gluLookAt(1 * sin(theta * 3.14159 / 180) * sin(phi * 3.14159 / 180), 1 * cos(theta * 3.14159 / 180), 1 * sin(theta * 3.14159 / 180) * cos(phi * 3.14159 / 180), 0, 0, 0, 0, 1, 0);
	gluLookAt(-0.034987133 * sin(theta * 3.14159 / 180) * sin(phi * 3.14159 / 180), -0.1393106 * cos(theta * 3.14159 / 180), 2.4853776 * sin(theta * 3.14159 / 180) * cos(phi * 3.14159 / 180), 0, 0, 0, 0, 1, 0);
	//draw the square
	glBegin(GL_QUADS);
	//转化到世界坐标，我这里是640*480分辨率，所以要按照实验室实际的分辨率进行修改
	tl_x = (tl_x - 320) / 320;
	tl_y = (-tl_y + 240) / 240;
	tr_x = (tr_x - 320) / 320;
	tr_y = (-tr_y + 240) / 240;
	br_x = (br_x - 320) / 320;
	br_y = (-br_y + 240) / 240;
	bl_x = (bl_x - 320) / 320;
	bl_y = (-bl_y + 240) / 240;
	glColor3f(0.0, 1.0, 1.0);
	glVertex3f(tl_x, tl_y, 0);
	glVertex3f(tr_x, tr_y, 0);
	glVertex3f(br_x, br_y, 0);
	glVertex3f(bl_x, bl_y, 0);
	glEnd();
	//glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1, 1, -1, 1, -1, 5);
	glFlush();
}
int main()
{
	bool loopFlag = true;
	int projFrame = 0;
	int glFrame = 0;
	int type = 0;
	cv::Mat output = (cv::Mat(HEIGHT, WIDTH, CV_8UC3, cv::Scalar::all(255)));
	std::thread thrGL([&] {
	Mat Frame, ProceedFrame;
	while (!cap.isOpened()); // 等待相机加载完成
	glfwInit();
	// open a window
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Dynamic Projection Mapping", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window.\n");
		getchar();
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	init();// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	    if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return -1;
	    }
	    glMatrixMode(GL_PROJECTION);
	    glLoadIdentity();
	    glOrtho(WIDTH / 2, WIDTH / 2, HEIGHT / 2, HEIGHT / 2, -1, 1);
		while (!glfwWindowShouldClose(window)&& loopFlag) {
			MarkerDetect Processor; // 构造一个AR处理类
			while (waitKey(2)) // 每次循环延迟5ms
			{
				glClear(GL_COLOR_BUFFER_BIT);
				cap >> Frame; // 读一帧
				ProceedFrame = Processor.Process(Frame); // 处理图像
				imshow("Marker", ProceedFrame);
				Transform();
				glfwSwapBuffers(window);
				glReadPixels(0, 0, WIDTH, HEIGHT, GL_BGR, GL_UNSIGNED_BYTE, output.data);
				flip(output, output, -1);
				glfwPollEvents();
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			}
			glfwDestroyWindow(window);
			glfwTerminate();
		}
	});

	std::thread thrProj([&] {
		try {
			HighSpeedProjector proj_V3;
			DYNAFLASH_PARAM param = getDefaultDynaParamRGB();
			param.dFrameRate = 500.0;
			param.nMirrorMode = 0;
			printDynaParam(param);

			proj_V3.connect(0);
			proj_V3.setParam(param);
			proj_V3.start();

			while (loopFlag) {
				projFrame += proj_V3.sendImage(output.data);
			}

			proj_V3.stop();
			proj_V3.disconnect();
		}
		catch (std::exception& e) {
			std::cout << "\033[41m ERROR \033[49m\033[31m thrProj : " << e.what() << "\033[39m" << std::endl;
		}
		});

	clock_t start = clock();
	while (loopFlag) {
		clock_t now = clock();
		const double time_ = static_cast<double>(now - start) / CLOCKS_PER_SEC;
		if (time_ > 1.0) {
			std::cout << "projection : " << projFrame / time_ << " fps" << std::endl;
			projFrame = 0;
			start = now;
		}

		if ((GetAsyncKeyState(VK_ESCAPE) & 0x80000000) != 0) {
			loopFlag = false;
		}
	}
	thrGL.join();
	thrProj.join();
	return 0;
}
