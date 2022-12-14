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

VideoCapture cap(0); // ʹ�ä���ӻ�
const uint WIDTH = 1024, HEIGHT = 768; // ����Υ�����
Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
Rect roi;
GLfloat r = 1, theta = 30, phi = 30;
const int imgNum = 1;

string writePath = "D:/IE���������/OpenGL_Test/AR_DATA";
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
		if (32 == waitKey(20)) {			//�ո�����
			name = writePath + to_string(i) + ".jpg";
			cv::imwrite(name, frame);
			cout << name << endl;
			i++;
		}
		if (97 == waitKey(10)) {			//'a'�˳�
			break;
		}
		imshow("Capture", frame);
	}
}
void CalibrateCamera() {
	Mat image, img_gray;
	cv::Size patternsize(8, 5);//���̸�ÿ��ÿ�нǵ����
	vector<vector<Point3f>> objpoints_img;//�������̸��Ͻǵ����ά����
	vector<Point3f> obj_world_pts;//��ά��������
	vector<vector<Point2f>> images_points;//�������нǵ�
	vector<Point2f> img_corner_points;//����ÿ��ͼ��⵽�Ľǵ�
	vector<String> images_path;//����������Ŷ�ȡͼ��·��
	string image_path = "D:/IE���������/OpenGL_Test/AR_DATA/Ca_Matrix/*.jpg";//������ͼ·��
	glob(image_path, images_path);//��ȡָ���ļ�����ͼ��
		//ת��������ϵ
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 8; j++) { obj_world_pts.push_back(Point3f(j, i, 0)); }
	}
	for (int i = 0; i < images_path.size(); i++)
	{
		image = imread(images_path[i]);
		cvtColor(image, img_gray, COLOR_BGR2GRAY);
		bool found_success = findChessboardCorners(img_gray, patternsize,//���ǵ�
			img_corner_points, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
		if (found_success) { cout << "..."; }
		else { cout << "error!"; }
		//��ʾ�ǵ�
		if (found_success) {
			TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);//������ֹ����
			cornerSubPix(img_gray, img_corner_points, Size(11, 11), Size(-1, -1), criteria);//��һ����ȡ�����ؽǵ�
			drawChessboardCorners(image, patternsize, img_corner_points, found_success);//���ƽǵ�
			imshow("Test Corner", image);
			waitKey(400);
			destroyAllWindows();
			objpoints_img.push_back(obj_world_pts);//����������ϵ���������ϵ
			images_points.push_back(img_corner_points);
		}
	}
	Mat cameraMatrix, distCoeffs, R, T;//�ڲξ��󣬻���ϵ������ת����ƫ����
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
	Mat Image, ImageGray, ImageAdaptiveBinary; //�ֱ��� ԭͼ�� �Ҷ�ͼ�� ����Ӧ��ֵ��ͼ��
	vector<vector<Point>> ImageContours; //ͼ�����б߽���Ϣ
	vector<vector<Point2f>> ImageQuads, ImageMarkers; //ͼ�������ı��� �� ��֤�ɹ����ı���
	vector<Point2f> FlatMarkerCorners; //�����λ����ʱ�õ�����Ϣ
	Size FlatMarkerSize;
	//�����λ����ʱ�õ�����Ϣ7x7�ڰױ�ǵ���ɫ��Ϣ
	uchar CorrectMarker[7 * 7] = {
		0,0,0,0,0,0,0,
		0,255,255,255,255,255,0,
		0,255,0,255,0,255,0,
		0,255,255,255,255,255,0,
		0,255,0,255,0,255,0,
		0,255,255,255,255,255,0,
		0,0,0,0,0,0,0 };
	// ������һ֡����ǰ�ĳ�ʼ��
	void Clean() {
		ImageContours.clear();
		ImageQuads.clear();
		ImageMarkers.clear();
	}
	//ת��ͼƬ��ɫGRAY
	void ConvertColor() {
		cvtColor(Image, ImageGray, CV_BGR2GRAY);
		adaptiveThreshold(ImageGray, ImageAdaptiveBinary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 7);
	}
	//��ȡͼƬ���б߽�
	void GetContours(int ContourCountThreshold) {
		vector<vector<Point>> AllContours; // ���б߽���Ϣ
		findContours(ImageAdaptiveBinary, AllContours,
			CV_RETR_LIST, CV_CHAIN_APPROX_NONE); // ������Ӧ��ֵ��ͼ��Ѱ�ұ߽�
		for (size_t i = 0; i < AllContours.size(); ++i) { // ֻ���������ֵ�ı߽�
			int contourSize = AllContours[i].size();
			if (contourSize > ContourCountThreshold) { ImageContours.push_back(AllContours[i]); }
		}
	}
	//Ѱ�������ı��� TargetPoints[]
	void FindQuads(int ContourLengthThreshold)
	{
		vector<vector<Point2f>> PossibleQuads;
		for (int i = 0; i < ImageContours.size(); ++i)
		{
			vector<Point2f> InDetectPoly;
			approxPolyDP(ImageContours[i], InDetectPoly,
				ImageContours[i].size() * 0.05, true); // �Ա߽���ж�������
			if (InDetectPoly.size() != 4) continue;// ֻ���ı��θ���Ȥ
			if (!isContourConvex(InDetectPoly)) continue; // ֻ��͹�ı��θ���Ȥ
			float MinDistance = 1e10; // Ѱ����̱�
			for (int j = 0; j < 4; ++j)
			{
				Point2f Side = InDetectPoly[j] - InDetectPoly[(j + 1) % 4];
				float SquaredSideLength = Side.dot(Side);
				MinDistance = min(MinDistance, SquaredSideLength);
			}
			if (MinDistance < ContourLengthThreshold) continue; // ��̱߱��������ֵ
			vector<Point2f> TargetPoints;
			for (int j = 0; j < 4; ++j) // �����ĸ���
			{
				TargetPoints.push_back(Point2f(InDetectPoly[j].x, InDetectPoly[j].y));
			}
			Point2f Vector1 = TargetPoints[1] - TargetPoints[0]; // ��ȡһ���ߵ�����
			Point2f Vector2 = TargetPoints[2] - TargetPoints[0]; // ��ȡһ��б�ߵ�����
			if (Vector2.cross(Vector1) < 0.0) // �����������Ĳ�� �жϵ��Ƿ�Ϊ��ʱ�봢��
				swap(TargetPoints[1], TargetPoints[3]); // �������0��Ϊ˳ʱ�룬��Ҫ����
			PossibleQuads.push_back(TargetPoints); // ��������ܵ��ı��Σ����н�һ���ж�
		}
		vector<pair<int, int>> TooNearQuads; // ׼��ɾ�����鿿̫���Ķ����
		for (int i = 0; i < PossibleQuads.size(); ++i) {
			vector<Point2f>& Quad1 = PossibleQuads[i]; // ��һ��             
			for (int j = i + 1; j < PossibleQuads.size(); ++j) {
				vector<Point2f>& Quad2 = PossibleQuads[j]; // �ڶ���
				float distSquared = 0;
				float x1Sum = 0.0, x2Sum = 0.0, y1Sum = 0.0, y2Sum = 0.0, dx = 0.0, dy = 0.0;
				for (int c = 0; c < 4; ++c) {
					x1Sum += Quad1[c].x;
					x2Sum += Quad2[c].x;
					y1Sum += Quad1[c].y;
					y2Sum += Quad2[c].y;
				}
				x1Sum /= 4; x2Sum /= 4; y1Sum /= 4; y2Sum /= 4; // ����ƽ��ֵ���е㣩
				dx = x1Sum - x2Sum;
				dy = y1Sum - y2Sum;
				distSquared = sqrt(dx * dx + dy * dy); // ����������ξ���
				if (distSquared < 50) {
					TooNearQuads.push_back(pair<int, int>(i, j)); // ������׼���޳�
				}
			}
		}
		vector<bool> RemovalMask(PossibleQuads.size(), false); // �Ƴ�����б�
		for (int i = 0; i < TooNearQuads.size(); ++i)
		{
			float p1 = CalculatePerimeter(PossibleQuads[TooNearQuads[i].first]);  //���ܳ�
			float p2 = CalculatePerimeter(PossibleQuads[TooNearQuads[i].second]);
			int removalIndex;  //�Ƴ��ܳ�С�Ķ����
			if (p1 > p2) removalIndex = TooNearQuads[i].second;
			else removalIndex = TooNearQuads[i].first;
			RemovalMask[removalIndex] = true;
		}
		for (size_t i = 0; i < PossibleQuads.size(); ++i)
		{
			// ֻ¼��û���޳��Ķ����
			if (!RemovalMask[i]) ImageQuads.push_back(PossibleQuads[i]);
		}
	}
	//�任Ϊ�����β���֤�Ƿ�Ϊ���
	void TransformVerifyQuads()
	{
		Mat FlatQuad;
		for (size_t i = 0; i < ImageQuads.size(); ++i)
		{
			vector<Point2f>& Quad = ImageQuads[i];
			Mat TransformMartix = getPerspectiveTransform(Quad, FlatMarkerCorners);
			warpPerspective(ImageGray, FlatQuad, TransformMartix, FlatMarkerSize);
			threshold(FlatQuad, FlatQuad, 0, 255, THRESH_OTSU); // ��Ϊ��ֵ��ͼ��
			if (MatchQuadWithMarker(FlatQuad)) {        // ����ȷ��Ǳȶ�
				ImageMarkers.push_back(ImageQuads[i]); // �ɹ����¼
			}
			else { // ���ʧ�ܣ�����ת��ÿ��90�Ƚ��бȶ�
				for (int j = 0; j < 3; ++j) {
					rotate(FlatQuad, FlatQuad, ROTATE_90_CLOCKWISE);
					if (MatchQuadWithMarker(FlatQuad)) {
						ImageMarkers.push_back(ImageQuads[i]); // �ɹ����¼
						break;
					}
				}
			}
		}
	} //�任Ϊ�����β���֤�Ƿ�Ϊ���
	//Draw the Marker Border
	void DrawMarkerBorder(Scalar Color) //���Ʊ�Ǳ߽�͵�
	{
		for (vector<Point2f> Marker : ImageMarkers)
		{
			line(Image, Marker[0], Marker[1], Color, 2, LINE_AA);
			line(Image, Marker[1], Marker[2], Color, 2, LINE_AA);
			line(Image, Marker[2], Marker[3], Color, 2, LINE_AA);
			line(Image, Marker[3], Marker[0], Color, 2, LINE_AA);//LINE_AA�ǿ����
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
	// �����������Ƿ�Ϊ���
	bool MatchQuadWithMarker(Mat& Quad) {
		int  Pos = 0;
		for (int r = 2; r < 33; r += 5) // ������ͼ���СΪ(35,35)
		{
			for (int c = 2; c < 33; c += 5)// ��ȡÿ��ͼ�����ĵ�
			{
				uchar V = Quad.at<uchar>(r, c);
				uchar K = CorrectMarker[Pos];
				if (K != V) // ����ȷ�����ɫ��Ϣ�ȶ�
					return false;
				Pos++;
			}
		}
		return true;
	}
	// �����ܳ�
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
	MarkerDetect() // ���캯��
	{
		FlatMarkerSize = Size(35, 35);
		FlatMarkerCorners = { Point2f(0,0),Point2f(FlatMarkerSize.width - 1,0),Point2f(FlatMarkerSize.width - 1,FlatMarkerSize.height - 1),Point2f(0,FlatMarkerSize.height - 1) };
	}
	Mat Process(Mat& Image)// ����һ֡ͼ��
	{
		Clean(); // ��һ֡��ʼ��
		Image.copyTo(this->Image); // ����ԭʼͼ��Image��
		ConvertColor(); // ת����ɫ
		GetContours(50); // ��ȡ�߽�
		FindQuads(100); // Ѱ���ı���
		TransformVerifyQuads(); // ���β�У���ı���
		DrawMarkerBorder(Scalar(255, 0, 0)); // �ڵõ��ı����Χ���߽�Scalar(��ɫ��
		return this->Image; // ���ؽ��ͼ��
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
	//   OpenGL����  //
	//////////////////////////////////////////////
	//��ȡ�ĸ������xy����
	glEnable(GL_DEPTH_TEST);
	//gluLookAt(1 * sin(theta * 3.14159 / 180) * sin(phi * 3.14159 / 180), 1 * cos(theta * 3.14159 / 180), 1 * sin(theta * 3.14159 / 180) * cos(phi * 3.14159 / 180), 0, 0, 0, 0, 1, 0);
	gluLookAt(-0.034987133 * sin(theta * 3.14159 / 180) * sin(phi * 3.14159 / 180), -0.1393106 * cos(theta * 3.14159 / 180), 2.4853776 * sin(theta * 3.14159 / 180) * cos(phi * 3.14159 / 180), 0, 0, 0, 0, 1, 0);
	//draw the square
	glBegin(GL_QUADS);
	//ת�����������꣬��������640*480�ֱ��ʣ�����Ҫ����ʵ����ʵ�ʵķֱ��ʽ����޸�
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
	while (!cap.isOpened()); // �ȴ�����������
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
			MarkerDetect Processor; // ����һ��AR������
			while (waitKey(2)) // ÿ��ѭ���ӳ�5ms
			{
				glClear(GL_COLOR_BUFFER_BIT);
				cap >> Frame; // ��һ֡
				ProceedFrame = Processor.Process(Frame); // ����ͼ��
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
