// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <ctime>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <random>
#include <set>
#include <tuple>
#include <vector>
namespace fs = std::filesystem;

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
std::string LAB_FOLDER = "lab5\\";

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

bool isInside(Mat const& img, int i, int j)
{
    return 0 <= i && i < img.rows && 0 <= j && j < img.cols;
}

Mat_<uchar> computeDT(Mat_<uchar> const& orig)
{
    int di[] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
    int dj[] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    int weights[] = { 3, 2, 3, 2, 0, 2, 3, 2, 3 };
    int maxDist;
    Mat_<uchar> dist;

    orig.copyTo(dist);
    maxDist = 0;

    for (int i = 0; i < dist.rows; ++i)
        for (int j = 0; j < dist.cols; ++j)
            for (int k = 0; k < 5; ++k)
                if (isInside(dist, i + di[k], j + dj[k]))
                    dist(i, j) = min(dist(i, j), dist(i + di[k], j + dj[k]) + weights[k]);

    for (int i = dist.rows - 1; i >= 0; --i)
        for (int j = dist.cols - 1; j >= 0; --j)
        {
            for (int k = 8; k >= 4; --k)
                if (isInside(dist, i + di[k], j + dj[k]))
                    dist(i, j) = min(dist(i, j), dist(i + di[k], j + dj[k]) + weights[k]);
            maxDist = max(maxDist, dist(i, j));
        }

    return dist;
}

float computeMatchingScore(Mat_<uchar> const& dtMat, Mat_<uchar> const& sample)
{
    float total = 0;
    int count = 0;
    for (int i = 0; i < sample.rows; ++i)
        for (int j = 0; j < sample.cols; ++j)
            if (!sample(i, j))
                total += dtMat(i, j), ++count;

    return total / count;
}

Mat_<uchar> translateImage(Mat_<uchar> orig, int y, int x)
{
    Mat_<uchar> res = Mat_<uchar>(orig.rows, orig.cols, 255);

    for (int i = 0; i < orig.rows; ++i)
        for (int j = 0; j < orig.cols; ++j)
            if (!orig(i, j) && isInside(res, i + y, j + x))
                res(i + y, j + x) = orig(i, j);
                
    return res;
}

Point calculateCom(Mat_<uchar> orig)
{
    Point total = { 0, 0 };
    int count = 0;
    for (int i = 0; i < orig.rows; ++i)
        for (int j = 0; j < orig.cols; ++j)
            if (!orig(i, j))
            {
                total.y += i;
                total.x += j;
                ++count;
            }

    return { total.x / count, total.y / count };
}

int main()
{
    const int IMAGE_ROWS = 19;
    const int IMAGE_COLS = 19;
    const int TOTAL_IMAGES = 400;
    int totalFeatures = IMAGE_ROWS * IMAGE_COLS;
    std::ofstream means_file("means.csv");
    std::ofstream deviations_file("devations.csv");
    std::ofstream covariance_file("covariance.csv");
    std::ofstream correlation_file("correlation.csv");

    std::string fpath;
    Mat_<uchar> I = Mat(TOTAL_IMAGES, totalFeatures, CV_8UC1);
    Mat_<uchar> img;
    Mat_<uchar> chart = Mat(256, 256, CV_8UC1);;
    Mat_<double> covariance = Mat(totalFeatures, totalFeatures, CV_64FC1);
    Mat_<double> correlation = Mat(totalFeatures, totalFeatures, CV_64FC1);
    std::vector<double> means, stdDeviations;
    // 5 * 19 + 4 = 99, 5 * 19 + 14 = 109; 0.94
    // 10 * 19 + 3 = 193, 9 * 19 + 15 = 186; 0.84
    // 5 * 19 + 4 = 99, 18 * 19 + 0 = 342; 0.07
    std::vector<std::pair<int, int>> chartIndices({ {99, 109}, {193, 186}, {99, 342} });
    double mean, stdDeviation;
    int k = 0;
    int baseY, baseX;

    fpath = INPUT_PREFIX + LAB_FOLDER;
    covariance.setTo(0);
    correlation.setTo(0);

    for (const auto& entry : fs::directory_iterator(fpath))
    {
        img = imread(entry.path().string(), CV_LOAD_IMAGE_GRAYSCALE);
        mean = 0;
        stdDeviation = 0;

        for (int i = 0; i < img.rows; ++i)
            for (int j = 0; j < img.cols; ++j)
                I(k, i * IMAGE_COLS + j) = img(i, j);

        ++k;
    }

    for (int i = 0; i < totalFeatures; ++i)
    {
        mean = 0;
        stdDeviation = 0;

        for (k = 0; k < TOTAL_IMAGES; ++k)
            mean += I(k, i);
        mean /= TOTAL_IMAGES;

        for (k = 0; k < TOTAL_IMAGES; ++k)
            stdDeviation += pow(I(k, i) - mean, 2);

        stdDeviation = sqrt(stdDeviation / TOTAL_IMAGES);

        means.push_back(mean);
        stdDeviations.push_back(stdDeviation);

        means_file << mean << ",";
        deviations_file << stdDeviation << ",";
    }

    for (int i = 0; i < totalFeatures; ++i)
    {
        for (int j = 0; j < totalFeatures; ++j)
        {
            for (int k = 0; k < TOTAL_IMAGES; ++k)
                covariance(i, j) += (I(k, i) - means.at(i)) * (I(k, j) - means.at(j));
            
            covariance(i, j) /= TOTAL_IMAGES;
            correlation(i, j) = covariance(i, j) / (stdDeviations.at(i) * stdDeviations.at(j));

            covariance_file << covariance(i, j) << ",";
            correlation_file << correlation(i, j) << ",";
        }
        covariance_file << "\n";
        correlation_file << "\n";
    }

    for (auto p : chartIndices)
    {
        chart.setTo(255);
        for (k = 0; k < TOTAL_IMAGES; ++k)
            chart(I(k, p.second), I(k, p.first)) = 0;
        std::cout << "The correlation coefficient is " << correlation(p.first, p.second) << '\n';
        imshow("Correlation chart", chart);
        waitKey(0);
    }

	return 0;
}