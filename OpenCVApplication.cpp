// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <tuple>
#include <vector>

typedef std::tuple <float, float, float> LineEq;
typedef std::mt19937 RNGeng;
struct LocalPeak 
{
    int theta, ro, hval;
    bool operator < (LocalPeak const& other) const
    {
        return hval > other.hval;
    }
};
RNGeng engine;
std::string INPUT_PREFIX = "c:\\Users\\logoeje\\source\\repos\\PRS\\Inputs\\";
std::string LAB_FOLDER = "lab3\\";
std::string FILE_NAME = "edge_";
std::string TEST_NAME[] = {"simple", "complex"};
//int NO_TESTS = 6;
int NO_TESTS = 2;

std::pair <float, float> calculateParameters(int noPoints, std::vector<Point2f> const &points)
{
    float cnt = noPoints;
    float sumXY = 0;
    float sumX = 0;
    float sumX2 = 0;
    float sumY = 0;
    float t1, t0;

    for (auto p : points)
    {
        sumXY += p.x * p.y;
        sumX += p.x;
        sumY += p.y;
        sumX2 += p.x * p.x;
    }

    t1 = (cnt * sumXY - sumX * sumY) / (cnt * sumX2 - sumX * sumX);
    t0 = (sumY - t1 * sumX) / cnt;

    return { t0, t1 };
}

std::pair <float, float> calculatePolarParameters(int noPoints, std::vector <Point2f> const &points)
{
    float cnt = noPoints;
    float sumXY = 0;
    float sumX = 0;
    float sumX2 = 0;
    float sumY = 0;
    float diffYX = 0;
    float a, beta, mag;

    for (auto p : points)
    {
        sumXY += p.x * p.y;
        sumX += p.x;
        sumY += p.y;
        sumX2 += p.x * p.x;
        diffYX += p.y * p.y - p.x * p.x;
    }

    a = atan2(2 * sumXY - (2 * sumX * sumY) / cnt,
        diffYX + (sumX * sumX) / cnt - (sumY * sumY) / cnt);
    beta = -a / 2;
    mag = (cos(beta) * sumX + sin(beta) * sumY) / cnt;

    return { beta, mag };
}

LineEq calculateLineEq(Point a, Point b)
{
    return { a.y - b.y, b.x - a.x, 1LL*a.x * b.y - 1LL*a.y * b.x };
}

float calculateLineDist(LineEq const &line, Point p)
{
    double a, b, c;
    std::tie(a, b, c) = line;
    return (abs(1LL*a * p.x + 1LL*b * p.y + c) / sqrt(1LL*a * a + 1LL*b * b));
}

void findLocalMax(Mat_<int> const &src, int i, int j, int l, std::vector<LocalPeak> &collector)
{
    int trueY, trueX;
    //std::cout << "Trying for " << i << ' ' << j << " -> " << (int)src(i, j) << '\n';
    for (int y = -l / 2; y <= l / 2; ++y)
        for (int x = -l / 2; x <= l / 2; ++x)
        {
            trueY = (src.rows + i + y) % src.rows;
            trueX = (src.cols + j + x) % src.cols;
            if (trueY == i && trueX == j)
                continue;
            //std::cout << trueY << ' ' << trueX << " -> " << (int) src(trueY, trueX) << '\n';
            if (src(i, j) < src(trueY, trueX))
                return;
        }

    collector.push_back({ j, i, (int) src(i, j) });
}

float rad(const int degree) { return degree * (PI / 180.0); }

int main()
{
    std::string fpath;
    Mat_<uchar> img;
    Mat_<Vec3b> displayImg;
    Mat_<int> H;
    Mat_<uchar> displayH;
    std::vector<LocalPeak> localMax;
    float ro;
    float diagonal;
    float maxFreq;
    float resolutionEPS = 0.1f;

    for (int testNo = 0; testNo < NO_TESTS; ++testNo)
    {
        // Initialize
        fpath = INPUT_PREFIX + LAB_FOLDER + FILE_NAME + TEST_NAME[testNo] + ".bmp";
        img = imread(fpath, CV_LOAD_IMAGE_GRAYSCALE);
        diagonal = sqrt(img.rows * img.rows + img.cols * img.cols);
        H = Mat_<int>((int) diagonal, 360);
        displayH = Mat_<uchar>((int) diagonal, 360);
        H.setTo(0);
        displayH.setTo(0);
        maxFreq = 0;
        localMax.clear();

        for (int i = 0; i < img.rows; ++i)
            for (int j = 0; j < img.cols; ++j)
                if (img(i, j))
                    for (float theta = 0; theta < 360; ++theta)
                    {
                        ro = j * cos(theta * PI / 180.0f) + i * sin(theta * PI / 180.0f);
                        if (0 < ro && ro < H.rows)
                            maxFreq = max(maxFreq, ++H((int)ro, (int)theta));
                    }

        for (int i = 0; i < H.rows; ++i)
            for (int j = 0; j < H.cols; ++j)
                displayH(i, j) = (255.0f * H(i, j)) / maxFreq;

        //imshow("Frequencies", displayH);
        //waitKey(0);

        for (int k = 3; k < 15; k += 4) {
            localMax.clear();
            for (int i = 0; i < displayH.rows; ++i)
                for (int j = 0; j < displayH.cols; ++j)
                {
                    findLocalMax(H, i, j, k, localMax);
                }
            displayImg = Mat_<Vec3b>(img.rows, img.cols);
            for (int i = 0; i < displayImg.rows; ++i)
                for (int j = 0; j < displayImg.cols; ++j)
                    displayImg(i, j) = { img(i, j), img(i, j), img(i, j) };
            std::sort(localMax.begin(), localMax.end());
            for (int lines = 0; lines < min(10, localMax.size()); ++lines)
            {

                Point pt1, pt2;
                double a = cos(rad(localMax.at(lines).theta)), b = sin(rad(localMax.at(lines).theta));
                double x0 = a * localMax.at(lines).ro, y0 = b * localMax.at(lines).ro;
                pt1.x = cvRound(x0 + 1000 * (-b));
                pt1.y = cvRound(y0 + 1000 * (a));
                pt2.x = cvRound(x0 - 1000 * (-b));
                pt2.y = cvRound(y0 - 1000 * (a));
                line(displayImg, pt1, pt2, { 0, 0, 255 });
            }
            imshow("Result", displayImg);
            waitKey(0);
        }

    }

	return 0;
}