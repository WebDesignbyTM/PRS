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
RNGeng engine;
std::string INPUT_PREFIX = "c:\\Users\\logoeje\\source\\repos\\PRS\\Inputs\\";
std::string LAB_FOLDER = "lab2\\";
std::string FILE_NAME = "points";
int NO_TESTS = 6;
//int NO_TESTS = 2;

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

float calculateLineDist(LineEq line, Point p)
{
    double a, b, c;
    std::tie(a, b, c) = line;
    return (abs(1LL*a * p.x + 1LL*b * p.y + c) / sqrt(1LL*a * a + 1LL*b * b));
}

int main()
{
    // Far too many variables
    std::string fpath;
    std::vector<Point> points;
    LineEq bestParams;
    LineEq crtParams;
    Mat_<uchar> img;
    std::set<std::tuple<int, int, int, int> > usedSamples;
    Point p1, p2;
    float distThreshold = 10; // t
    float targetCertainty = 0.99; // p
    float expectedInliers = 0.8; // q
    int sampleSize = 2; // s
    int totalTrials; // N
    int consensusSize; // T
    int bestConsensus;
    int crtConsensus;
    uint32_t seed = std::time(nullptr);
    std::uniform_int_distribution<uint32_t> uDist;
    engine.seed(seed);

    for (int testNo = 1; testNo < NO_TESTS; ++testNo)
    {
        // Initialize
        fpath = INPUT_PREFIX + LAB_FOLDER + FILE_NAME + std::to_string(testNo) + ".bmp";
        img = imread(fpath, CV_LOAD_IMAGE_GRAYSCALE);
        usedSamples.clear();
        points.clear();
        bestConsensus = crtConsensus = 0;
        bestParams = bestParams = { 0, 0, 0 };
        expectedInliers = (testNo == 1) ? 0.3 : 0.8;

        for (int i = 0; i < img.rows; ++i)
            for (int j = 0; j < img.cols; ++j)
                if (!img(i, j))
                    points.push_back({ j, i });

        // Calculate metaparameters
        uDist = std::uniform_int_distribution<uint32_t>(0, points.size() - 1);
        totalTrials = (int)(log(1 - targetCertainty) / log(1 - pow(expectedInliers, sampleSize)));
        consensusSize = (int)(expectedInliers * points.size());

        //std::cout << "Targets: " << totalTrials << ' ' << consensusSize << '\n';

        // Test samples
        while (totalTrials-- && bestConsensus < consensusSize)
        {
            // Select sample -- hardcoded for 2 elements, cannot determine the line of more than 2 points unless they all are colinear
            do {
                p1 = points.at(uDist(engine));
                p2 = points.at(uDist(engine));
            } while (usedSamples.find({p1.x, p1.y, p2.x, p2.y}) != usedSamples.end() 
                || usedSamples.find({ p2.x, p2.y, p1.x, p1.y }) != usedSamples.end());
            usedSamples.insert({ p1.x, p1.y, p2.x, p2.y });

            crtConsensus = 0;
            crtParams = calculateLineEq(p1, p2);
            for (auto p : points)
                crtConsensus += (calculateLineDist(crtParams, p) < distThreshold);
            //std::cout << p1 << ' ' << p2 << " -> " << crtConsensus << '\n';

            if (crtConsensus > bestConsensus)
            {
                bestConsensus = crtConsensus;
                bestParams = crtParams;
            }
        }

        double a, b, c;
        std::tie(a, b, c) = bestParams;
        //if (b)
        //{
            p1 = { 0,  (int)(-c / b) };
            p2 = { img.cols - 1, (int)-(1LL*((img.cols - 1) * a + c) / b) };
            line(img, p1, p2, {0, 0, 0});
        //}
        //else
        //{
            p1 = { (int)(-c / a), 0};
            p2 = { (int)-((1LL*(img.rows - 1) * b + c) / a), img.rows - 1 };
            line(img, p1, p2, {0, 0, 0});
        //}
        //std::cout << "Line equation: " << a << ' ' << b << ' ' << c << '\n';
        //std::cout << "Finally: " << p1 << ' ' << p2 << '\n';

        imshow("Result", img);
        waitKey(0);
    }

	return 0;
}