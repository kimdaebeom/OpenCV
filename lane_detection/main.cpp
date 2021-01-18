#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;

Mat histogram_stretching(Mat img) // 영상의 픽셀값 분포가 grayscale 전체영역에서 골고루 나타나게 하는 알고리즘
{
        double gmin, gmax;
        minMaxLoc(img, &gmin, &gmax);
        Mat dst = (img - gmin) * 255 / (gmax - gmin);
        return dst;
}


Mat bilateral_filter(Mat img) // 필터 : 노이즈제거 (GaussianBlur의 단점 : 에지까지 불명확화) vs median filter
{
	Mat dst1;
	GaussianBlur(img, dst1, Size(), 5);
	Mat dst2;
	bilateralFilter(img, dst2, -1, 10, 5);
	return dst2;
}

Mat binary_mean(Mat img) // 이진화 : 
{
	Mat dst;
	adaptiveThreshold(img,dst,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY_INV,5,13);
	return dst;
}

Mat canny(Mat img)
{
	Mat dst1;
	Canny(img, dst1, 50, 100);
	return dst1;
}

Mat region_of_interest(Mat img)
{
	int width = img.cols;
	int height = img.rows;
	Point points[1][4];
	
	points[0][0] = Point(width*0.1, height);
	points[0][1] = Point(width*0.42, height*0.7);
	points[0][2] = Point(width*0.65, height*0.7);
	points[0][3] = Point(width*0.95, height);

        Mat dst = Mat::zeros(height,width,CV_8UC1);
        const Point* ppt[1] = { points[0] };
        int npt[] = {4};
        fillPoly(dst, ppt, npt, 1, Scalar(255, 255, 255), 8);

	Point points_centre[1][4];
	
	points_centre[0][0] = Point(width*0.25, height);
	points_centre[0][1] = Point(width*0.51, height*0.7);
	points_centre[0][2] = Point(width*0.51, height*0.7);
	points_centre[0][3] = Point(width*0.75, height);

        const Point* ppt_centre[1] = { points_centre[0] };
        int npt_centre[] = {4};

        Mat dst_result;
        bitwise_and(img, dst, dst_result);
        fillPoly(dst_result, ppt_centre, npt_centre, 1, Scalar(0, 0, 0), 8);
        return dst_result;
}

float get_radian(int input)
{
	return input*CV_PI/180;
}

Point IntersectionPoint1(Point p1, Point p2, Point p3,  Point p4) 
{ //두 선의 교차점 출력함수
        Point ret;
        //교차되는 점의 x 좌표 계산결과
        ret.x = ((p1.x*p2.y - p1.y*p2.x)*(p3.x - p4.x) - (p1.x - p2.x)*(p3.x*p4.y - p3.y*p4.x)) / ((p1.x - p2.x)*(p3.y - p4.y) - (p1.y - p2.y)*(p3.x - p4.x)+0.000001);

        //교차되는 점의 y 좌표 계산결과
        ret.y = ((p1.x*p2.y - p1.y*p2.x)*(p3.y - p4.y) - (p1.y - p2.y)*(p3.x*p4.y - p3.y*p4.x)) / ((p1.x - p2.x)*(p3.y - p4.y) - (p1.y - p2.y)*(p3.x - p4.x)+0.000001);
        return ret;
}

Mat drivingArea(Mat image, Point pt1, Point pt2, Point pt3, Point pt4)
{
        Point pty=IntersectionPoint1(pt2, pt3, pt1, pt4) + Point(0, 50); //평균 직선의 중앙점에 약간 아래의 좌표를 추출
        Point a1 = pty + Point(200,0); //x축과 평행한 직선을 만들기 위해 값을 넣음
	Point a2 = pty + Point(-200,0);
        Point Pt23Fix = IntersectionPoint1(a1, a2, pt1, pt4);  //이전 평균 직선과 사다리꼴 형태를 만들기 위해 구한 직선의 교차점을 넣음
        Point Pt14Fix = IntersectionPoint1(pt2, pt3, a1, a2);
        Point points[4] = { pt3,pt4,Pt23Fix ,Pt14Fix }; // 각 점의 좌표를 함수에 전달하기위해 저장
        const Point* ppt[1] = { points }; 
        int npt[] = { 4 };
        Mat result = image;
        Mat img_mask (image.size(), CV_8UC3, cv::Scalar(255, 255, 255)); //하얀색바탕 생성
        fillPoly(img_mask, ppt, npt, 1, Scalar(0, 255, 0), LINE_8); //초록색으로 사다리꼴 색칠
        addWeighted(result, 0.8, img_mask, 0.2, 0, result); //  원래 그림과 사다리꼴 영역을 이용해 가중치를 두어 그림을 합침
        line(result, Pt14Fix, pt3, Scalar(0, 0, 255), LINE_8); // 사다리꼴의 옆부분에 빨간 선을 그어줌
        line(result, Pt23Fix, pt4, Scalar(0, 0, 255), LINE_8);
        return result; //나온 결과 그림 반환
}

#define ArraySize 100
Point* aPoint1 = new Point[ArraySize], *aPoint2 = new Point[ArraySize], *bPoint1 = new Point[ArraySize], *bPoint2 = new Point[ArraySize];
int aIndex = 0, bIndex = 0;
Point aP1S(0,0), aP2S(0, 0), bP1S(0, 0), bP2S(0, 0); //각 끝점의 평균값을 담기 위한 점

int main()
{
	VideoCapture cap("test_1.mp4");
	Mat org, frame;
	while (cap.isOpened()){
		cap >> frame;		
		if (frame.empty())
	        	break;
		cvtColor(frame, org, COLOR_BGR2GRAY);
		Mat bf_filter_img = bilateral_filter(org);

		Mat binary_img = binary_mean(bf_filter_img);
	
		Mat canny_img = canny(binary_img);
	
		Mat roi_img = region_of_interest(canny_img);



		std::vector<Vec2f> lines;
		HoughLines(roi_img, lines, 1, CV_PI/180, 30, 30, 100);
		Mat dst(roi_img.rows,roi_img.cols,CV_8U,Scalar(255));
		std::vector<Vec2f>::const_iterator it = lines.begin();
		while(it != lines.end())
		{
			float rho = (*it)[0];
			float theta = (*it)[1]; // 두 번째 요소는 델타 각도
			if (theta < get_radian(85) && theta >get_radian(35)) 
			{ // 각도 우측 범위 35 ~ 85 각도 선에서 인지
				Point pt1(rho / cos(theta), 0); // 첫 우측각에서 해당 선의 교차점   
			        aPoint1[(aIndex) % ArraySize] = pt1; //마치 Queue가 돌아가듯 사이즈 안에서만 값을 덧씌우며 채움
			        Point pt2((rho - dst.rows*sin(theta)) / cos(theta), dst.rows);// 마지막 우측각에서 해당 선의 교차점
			        aPoint2[(aIndex++) % ArraySize] = pt2; // 다음 인덱스 값으로 넘어가며 값을 넣음
			        //cv::line(roi_img, pt1, pt2, cv::Scalar(255), 2); // 모든 직선 그리기
			}

			else if(theta<get_radian(145)&&theta>get_radian(95))
			{ // 각도 좌측 범위 95 ~ 145 각도 선에서 인지
				Point pt1(0, rho / sin(theta)); // 첫 좌측각에서 해당 선의 교차점  
				bPoint1[(bIndex) % ArraySize] = pt1; //마치 Queue가 돌아가듯 사이즈 안에서만 값을 덧씌우며 채움
				Point pt2(dst.cols, (rho - dst.cols*cos(theta)) / sin(theta)); // 마지막 좌측각에서 해당 선의 교차점
				bPoint2[(bIndex++) % ArraySize] = pt2; // 다음 인덱스 값으로 넘어가며 값을 넣음
				//cv::line(roi_img, pt1, pt2, cv::Scalar(255), 2); // 모든 직선 그리기
			}
			++it; //반복자 증가
		}
		

		for (int i = 0; i < ArraySize; i++)
		{//배열에 저장된 모든 점의 좌표 합을 구함
			aP1S += aPoint1[i];
			aP2S += aPoint2[i];
			bP1S += bPoint1[i];
			bP2S += bPoint2[i];
		}
		aP1S /= ArraySize;//배열의 저장된 모든 점의 평균값을 구함
		aP2S /= ArraySize;
		bP1S /= ArraySize;
		bP2S /= ArraySize;
		//cv::line(roi_img, aP1S, aP2S, cv::Scalar(255,0,0), 2); // 추정된 평균 점으로 직선 그리기
		//cv::line(roi_img, bP1S, bP2S, cv::Scalar(255,0,0), 2); //추정된 평균 점으로 직선 그리기
		Mat result = drivingArea(frame, bP1S, aP1S, aP2S, bP2S);


		imshow("Result_1", result);

		if (waitKey(10)==27)
			break;
	}        
	destroyAllWindows();
}





