#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

//計算邊界 返回寬長
cv::Point2i FindBoundary(cv::Mat src, double angle)
{
    cv::Point2i vertices[4] = { cv::Point2i(0, 0),
        cv::Point2i(0, src.cols - 1),
        cv::Point2i(src.rows - 1, 0),
        cv::Point2i(src.rows - 1, src.cols - 1) };

    cv::Point2f newVertices[4];
    for (int i = 0; i < 4; i++)
    {
        newVertices[i].x = ( vertices[i].x*cos(angle) - vertices[i].y*sin(angle) );
        newVertices[i].y = ( vertices[i].x*sin(angle) + vertices[i].y*cos(angle) );
    }

    float min_x = std::min(newVertices[0].x, std::min(newVertices[1].x, std::min(newVertices[2].x, newVertices[3].x)));
    float max_x = std::max(newVertices[0].x, std::max(newVertices[1].x, std::max(newVertices[2].x, newVertices[3].x)));
    float min_y = std::min(newVertices[0].y, std::min(newVertices[1].y, std::min(newVertices[2].y, newVertices[3].y)));
    float max_y = std::max(newVertices[0].y, std::max(newVertices[1].y, std::max(newVertices[2].y, newVertices[3].y)));


    cv::Point2i boundary = cv::Point2i(max_x - min_x + 1, max_y - min_y + 1);

    std::cout << "new height : " << boundary.x << " new width : " << boundary.y << std::endl;
    return boundary;
}

void NearestNeighbor(cv::Mat src, double angle, cv::Point2i boundary, std::string name)
{
    int oldWidth = src.cols;
    int oldHeight = src.rows;
    int oldCenterRow = (oldHeight - 1) / 2;
    int oldCenterCol = (oldWidth - 1) / 2;

    int newWidth = boundary.y;
    int newHeight = boundary.x;
    int newCenterRow = (newHeight - 1) / 2;
    int newCenterCol = (newWidth - 1) / 2;

    cv::Mat output = cv::Mat(newHeight, newWidth, src.type());

    int paddRow = -newCenterRow * cos(angle) + newCenterCol * sin(angle) + oldCenterRow;
	int paddCol = -newCenterRow * sin(angle) - newCenterCol * cos(angle) + oldCenterCol;
    
    for (int row = 0; row < newHeight; row++)
    {
        for (int col = 0; col < newWidth; col++)
        {
            //先旋轉再偏移
            int oldRow = row * cos(angle) - col * sin(angle) + 0.5 + paddRow;
            int oldCol = row * cos(angle) + col * sin(angle) + 0.5 + paddCol;

            //防出界
            if (oldRow >= 0 && oldRow < oldHeight &&
                oldCol >= 0 && oldCol < oldWidth)
            {
                output.at<cv::Vec3b>(row, col) = src.at<cv::Vec3b>(oldRow, oldCol);
            }
            else
                output.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);

        }
    }

    cv::imwrite("NearestNeighbor_" + name + ".jpg", output);
}

void Bilinear(cv::Mat src, double angle, cv::Point2i boundary, std::string name)
{
    int oldWidth = src.cols;
    int oldHeight = src.rows;
    int oldCenterRow = (oldHeight - 1) / 2;
    int oldCenterCol = (oldWidth - 1) / 2;

    int newWidth = boundary.y;
    int newHeight = boundary.x;
    int newCenterRow = (newHeight - 1) / 2;
    int newCenterCol = (newWidth - 1) / 2;

    cv::Mat output = cv::Mat(newHeight, newWidth, src.type());

    int paddRow = -newCenterRow * cos(angle) + newCenterCol * sin(angle) + oldCenterRow;
	int paddCol = -newCenterRow * sin(angle) - newCenterCol * cos(angle) + oldCenterCol;

    for (int row = 0; row < newHeight; row++)
    {
        for (int col = 0; col < newWidth; col++)
        {
            //先旋轉再偏移
            double tempRow = row * cos(angle) - col * sin(angle) + paddRow + 0.5;
            double tempCol = row * cos(angle) + col * sin(angle) + paddCol + 0.5;

            //防出界
            if (tempRow < 0 || tempRow >= oldHeight || tempCol < 0 || tempCol >= oldWidth)
            {
                output.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);
                continue;
            }

            cv::Point a = cv::Point((int)tempRow, (int)tempCol);  //a離p點最近

                                                                  /***
                                                                  a     b
                                                                  p

                                                                  c     d

                                                                  ****/

            double a_to_row = tempRow - a.x;    //p點到a點的垂直距離
            double a_to_col = tempCol - a.y;    //p點到a點的水平距離
            //設定其他三個點
            cv::Point b = cv::Point(a.x, a.y + 1);
            cv::Point c = cv::Point(a.x + 1, a.y);
            cv::Point d = cv::Point(a.x + 1, a.y + 1);
            //防邊邊
            if (a.x == oldHeight - 1)
            {
                c = a;
                d = b;
            }

            if (a.y == oldWidth - 1)
            {
                b = a;
                d = c;
            }


            output.at<cv::Vec3b>(row, col) = (1 - a_to_row) * (1 - a_to_col) * src.at<cv::Vec3b>(a.x, a.y)
                + (1 - a_to_row)* a_to_col * src.at<cv::Vec3b>(b.x, b.y)
                + a_to_row * (1 - a_to_col) * src.at<cv::Vec3b>(c.x, c.y)
                + a_to_row * a_to_col * src.at<cv::Vec3b>(d.x, d.y);

        }
    }

    cv::imwrite("Bilinear_" + name + ".jpg", output);
}

double CalculateW(double x)
{
    double a = -0.1, value, X = std::abs(x);

    if (X == 0)
    {
        value = 1;
    }
    else if (X <= 1)
    {
        value = (a + 2) * std::pow(X, 3) - (a + 3) * std::pow(X, 2) + 1;
    }
    else if (X < 2)
    {
        value = a * std::pow(X, 3) - 5 * a * std::pow(X, 2) + 8 * a * X - 4 * a;
    }
    else {
        value = 0.0;
    }

    return value;
}

void Bicubic(cv::Mat src, double angle, cv::Point2i boundary, std::string name)
{
    int oldWidth = src.cols;
    int oldHeight = src.rows;
    int oldCenterRow = (oldHeight - 1) / 2;
    int oldCenterCol = (oldWidth - 1) / 2;

    int newWidth = boundary.y;
    int newHeight = boundary.x;
    int newCenterRow = (newHeight - 1) / 2;
    int newCenterCol = (newWidth - 1) / 2;

    cv::Mat output = cv::Mat(newHeight, newWidth, src.type());

    int paddRow = -newCenterRow * cos(angle) + newCenterCol * sin(angle) + oldCenterRow;
	int paddCol = -newCenterRow * sin(angle) - newCenterCol * cos(angle) + oldCenterCol;

    for (int row = 0; row < newHeight; row++)
    {
        for (int col = 0; col < newWidth; col++)
        {
            //先旋轉再偏移
            double tempRow = row * cos(angle) - col * sin(angle) + paddRow;
            double tempCol = row * cos(angle) + col * sin(angle) + paddCol;

            int tempX = tempRow + 0.5, tempY = tempCol + 0.5;

            int arrX[4] = { tempX - 1, tempX, tempX + 1, tempX + 2 };
            int arrY[4] = { tempY - 1, tempY, tempY + 1, tempY + 2 };



            //防出界
            if (arrX[0] >= 0 && arrX[3] < oldHeight && arrY[0] >= 0 && arrY[3] < oldWidth)
            {
                cv::Vec3b value = cv::Vec3b(0, 0, 0);
                //套公式
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        value += (src.at<cv::Vec3b>(arrX[i], arrY[j]) * CalculateW(tempRow - arrX[i]) * CalculateW(tempCol - arrY[j]));
                    }
                }

                output.at<cv::Vec3b>(row, col) = value;
            }
            else
                output.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);
        }
    }

    cv::imwrite("Bicubic_" + name + ".jpg", output);
}

int main()
{
    double degree = 30;		//degree of rotate
    double angle = degree * CV_PI / 180.;	//弧度

    cv::Mat lena = cv::imread("lena.png");
    if (!lena.data)
    {
        std::cout << "image null" << std::endl;
        cv::waitKey(0);
    }

    cv::imshow("original_lena", lena);

    cv::Point2i boundary = FindBoundary(lena, angle);

    NearestNeighbor(lena, angle, boundary, "lena");
    Bilinear(lena, angle, boundary, "lena");
    Bicubic(lena, angle, boundary, "lena");

    cv::waitKey(0);
}
