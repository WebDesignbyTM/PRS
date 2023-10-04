// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <fstream>
#include <iostream>
#include <vector>

std::string INPUT_PREFIX = "c:\\Users\\logoeje\\source\\repos\\PRS\\Inputs\\";
std::string LAB_FOLDER = "lab1\\";
std::string FILE_NAME = "points";
int NO_TESTS = 6;
//int NO_TESTS = 1;

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

int main()
{
    int noPoints;
    std::vector<Point2f> points;
    std::string fpath;
    std::ifstream fi;
    float x, y;
    Mat_<Vec3b> img(500, 500);
    std::pair<float, float> theta, polar;
    Point refPoint0, refPoint1;
    float resolutionEPS = 0.1f;
    //float learningRate = 0.3f;

    for (int i = 0; i < NO_TESTS; ++i)
    {
        // Initialize
        points.clear();
        fpath = INPUT_PREFIX + LAB_FOLDER + FILE_NAME + std::to_string(i) + ".txt";
        fi = std::ifstream(fpath);
        img.setTo(255);

        // Read
        fi >> noPoints;
        for (int j = 0; j < noPoints; ++j)
        {
            fi >> x >> y;
            points.push_back({ x, y });
        }

        // Model 1
        //theta = calculateParameters(noPoints, points);
        //line(img, { 0, (int) theta.first }, { 499, (int) (theta.first + theta.second * 499) }, { 0, 0, 0 }, 1);

        // Model 2
        polar = calculatePolarParameters(noPoints, points);
        if (abs(sin(polar.first)) > resolutionEPS)
        {
            refPoint0 = { 0, (int)(polar.second / sin(polar.first)) };
            refPoint1 = { 499, (int)((polar.second - 499 * cos(polar.first)) / sin(polar.first)) };
        }
        else
        {
            refPoint0 = { (int)(polar.second / cos(polar.first)), 0 };
            refPoint1 = { (int)((polar.second - 499 * sin(polar.first)) / cos(polar.first)), 499 };
        }
        line(img, refPoint0, refPoint1, { 0, 0, 0 }, 1);

        // Print
        for (auto p : points) 
        {
            if (p.x < 0 || p.y < 0 || p.x >= 500 || p.y >= 500)
                continue;

            circle(img, p, 3, { 0, 0, 255 });
        }

        imshow("Result", img);
        waitKey(0);
    }

	return 0;
}