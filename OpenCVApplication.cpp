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
#include <limits>
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
RNGeng engine(100);
std::string INPUT_PREFIX = "c:\\Users\\logoeje\\source\\repos\\PRS\\Inputs\\";
std::string LAB_FOLDER = "lab10\\";
std::string SUBFOLDERS[] = { "train\\", "test\\" };
constexpr int SUBFOLDERS_NO = 2; // 2
std::string CLASSES[] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };
constexpr int CLASSES_NO = 10; // 10
const int TOTAL_TESTS = 7;

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

template <typename T>
bool isEmptyPoint(T point, int imageType)
{
    if (!imageType)
        return point == 255;
    
    if (imageType == 1)
    {
        Vec3b s = Vec3b(point);
        return s[0] == 255 && s[1] == 255 && s[2] == 255;
    }
    
    return true;
}

bool isEmpty(Vec3b point)
{
    return point[0] == point[1] && point[1] == point[2] && point[2] == 255;
}

// imageType - 0 for grayscale image; 1 for 3 channel image
template <typename T>
Mat_<int> prepareData(Mat_<T> const& source, int d, int imageType)
{
    Mat_<int> res;
    std::vector<Point> points;
    Point aux;
    T matVal;

    if (!imageType && d == 3)
        throw std::invalid_argument("Cannot extract 3 dimensions from grayscale image");

    for (int i = 0; i < source.rows; ++i)
        for (int j = 0; j < source.cols; ++j)
            if (!isEmptyPoint<T>(source(i, j), imageType))
                points.push_back({j, i});

    res = Mat_<int>(points.size(), d, 0);

    for (int i = 0; i < points.size(); ++i)
    {
        aux = points.at(i);
        matVal = source(aux.y, aux.x);

        if (imageType && d == 1)
        {
            Vec3b s = matVal;
            res(i, 0) = 1.0f * (s[0] + s[1] + s[2]) / 3.0f;
            continue;
        }
        else if (d == 3)
        {
            Vec3b s = matVal;
            res(i, 0) = s[0], res(i, 1) = s[1], res(i, 2) = s[2];
        }

        switch (d)
        {
        case 1:
            res(i, 0) = matVal;
            break;
        case 2:
            res(i, 0) = aux.x, res(i, 1) = aux.y;
            break;
        default:
            break;
        }
    }

    return res;
}

std::pair<std::vector<int>, Mat_<double>> kmeans(Mat_<int> const& x, int k)
{
    std::uniform_int_distribution<int> distribution(0, x.rows - 1);
    std::vector<int> clusterSize(k);
    Mat_<double> clusterMeans(k, x.cols);
    Mat_<double> newClusters(k, x.cols);
    std::vector<int> memberships(x.rows);
    std::vector<int> oldMemberships(x.rows, -1);
    bool membershipsChanged = true;
    double bestDiff, currentDiff;
    int bestCluster;
    int idx;

    //std::cout << "Rows " << x.cols << '\n';
    for (int i = 0; i < clusterMeans.rows; ++i)
    { 
        idx = distribution(engine);
        for (int j = 0; j < x.cols; ++j)
            clusterMeans(i, j) = x(idx, j);
    }

    for (int iterationNo = 0; iterationNo < 10 && membershipsChanged; ++iterationNo)
    {
        membershipsChanged = false;

        // compute clusters
        for (int i = 0; i < x.rows; ++i)
        {
            bestDiff = INT_MAX;
            bestCluster = -1;
            for (int j = 0; j < clusterMeans.rows; ++j)
            {
                currentDiff = 0;
                for (int z = 0; z < x.cols; ++z)
                    currentDiff += (x(i, z) - clusterMeans(j, z)) * (x(i, z) - clusterMeans(j, z));
                currentDiff = sqrt(currentDiff);
                if (currentDiff < bestDiff)
                {
                    bestDiff = currentDiff;
                    bestCluster = j;
                }
            }
            memberships.at(i) = bestCluster;
        }

        // compute new means
        std::fill(clusterSize.begin(), clusterSize.end(), 0);
        newClusters.setTo(0);
        for (int i = 0; i < x.rows; ++i)
        {
            ++clusterSize.at(memberships.at(i));
            for (int j = 0; j < x.cols; ++j)
                newClusters(memberships.at(i), j) += x(i, j);
        }
        for (int i = 0; i < newClusters.rows; ++i)
            for (int j = 0; j < newClusters.cols; ++j)
                newClusters(i, j) /= clusterSize.at(i);
        newClusters.copyTo(clusterMeans);

        // check for membership changes
        for (int i = 0; i < memberships.size(); ++i)
            if (memberships.at(i) != oldMemberships.at(i))
            {
                oldMemberships.at(i) = memberships.at(i);
                membershipsChanged = true;
            }
    }

    return { memberships, clusterMeans };
}

std::vector<float> computeHistogram(Mat_<Vec3b> const& img, int noBins)
{
    std::vector<float> histogram(3 * noBins, 0);
    int bucketSize = 256 / noBins;

    for (int i = 0; i < img.rows; ++i)
        for (int j = 0; j < img.cols; ++j)
        {
            // bucketSize = 256 / noBins
            // a * bucketSize <= x < a bucketSize + bucketSize
            // a <= x / bucketSize < a + 1
            ++histogram.at(img(i, j)[2] / bucketSize); // red
            ++histogram.at(noBins + img(i, j)[1] / bucketSize); // green
            ++histogram.at(2 * noBins + img(i, j)[0] / bucketSize); // blue
        }

    for (auto& i : histogram)
        i /= 1.0f * img.rows * img.cols;

    return histogram;
}

float calculateHistogramDistance(std::vector<float> const& h1, std::vector<float> const& h2)
{
    float total = 0;

    if (h1.size() != h2.size())
        throw std::exception("Mismatched histograms: different sizes");

    for (int i = 0; i < h1.size(); ++i)
        total += abs(h1.at(i) - h2.at(i));

    return total;
}

Mat_<Vec3b> onlinePerception(Mat_<Vec3b> img, Mat_<float> const& X, Mat_<float> const& Y, int maxIter, int noPoints, float eta, float errorLimit)
{
    Point dp1, dp2;
    Mat_<Vec3b> displayImg;
    std::vector<float> weights({ 1, 1, -1 });
    float error, z;

    img.copyTo(displayImg);
    int iterNo;

    for (iterNo = 0; iterNo < maxIter; ++iterNo)
    {
        error = 0;
        for (int i = 0; i < noPoints; ++i)
        {
            z = weights.at(0) * X(i, 0) + weights.at(1) * X(i, 1) + weights.at(2) * X(i, 2);
            if (z * Y(i, 0) > 0)
                continue;

            // update parameters if wrong
            for (int j = 0; j < weights.size(); ++j)
                weights.at(j) += eta * X(i, j) * Y(i, 0);
            ++error;
        }

        img.copyTo(displayImg);
        if (weights.at(1) > 0.001f)
        {
            dp1.y = 0;
            dp1.x = -weights.at(0) / weights.at(1);
            dp2.y = img.rows - 1;
            dp2.x = (-weights.at(0) - dp2.y * weights.at(2)) / weights.at(1);
        }
        else
        {
            dp1.x = 0;
            dp1.y = -weights.at(0) / weights.at(2);
            dp2.x = img.cols - 1;
            dp2.y = (-weights.at(0) - dp2.x * weights.at(1)) / weights.at(2);
        }
        line(displayImg, dp1, dp2, { 0, 255, 0 });

        error /= noPoints;
        if (error < errorLimit)
            break;

        if (iterNo == maxIter - 1)
            std::cout << "Ran out of iterations\n";
    }

    std::cout << iterNo << ' ' << error << '\n';
    std::cout << "Determined online " << weights.at(0) << ' ' << weights.at(1) << ' ' << weights.at(2) << '\n';

    return displayImg;
}

Mat_<Vec3b> batchPerception(Mat_<Vec3b> img, Mat_<float> const& X, Mat_<float> const& Y, int maxIter, int noPoints, float eta, float errorLimit)
{
    Point dp1, dp2;
    Mat_<Vec3b> displayImg;
    std::vector<float> weights({ 1, 1, -1 });
    std::vector<float> losses;
    float error, z, loss;
    img.copyTo(displayImg);
    int iterNo;

    for (iterNo = 0; iterNo < maxIter; ++iterNo)
    {
        error = 0;
        loss = 0;
        losses = std::vector<float>({ 0, 0, 0 });

        for (int i = 0; i < noPoints; ++i)
        {
            z = weights.at(0) * X(i, 0) + weights.at(1) * X(i, 1) + weights.at(2) * X(i, 2);
            if (z * Y(i, 0) > 0)
                continue;

            // update parameters if wrong
            for (int j = 0; j < losses.size(); ++j)
                losses.at(j) -= X(i, j) * Y(i, 0);
            ++error;
            loss -= Y(i, 0) * z;
        }

        img.copyTo(displayImg);
        if (weights.at(1) > 0.001f)
        {
            dp1.y = 0;
            dp1.x = -weights.at(0) / weights.at(1);
            dp2.y = img.rows - 1;
            dp2.x = (-weights.at(0) - dp2.y * weights.at(2)) / weights.at(1);
        }
        else
        {
            dp1.x = 0;
            dp1.y = -weights.at(0) / weights.at(2);
            dp2.x = img.cols - 1;
            dp2.y = (-weights.at(0) - dp2.x * weights.at(1)) / weights.at(2);
        }
        line(displayImg, dp1, dp2, { 0, 255, 0 });

        error /= noPoints;
        loss /= noPoints;
        for (int i = 0; i < losses.size(); ++i)
            losses.at(i) /= noPoints;
        if (error < errorLimit)
            break;
        for (int i = 0; i < losses.size(); ++i)
            weights.at(i) -= ((!i) ? eta * 100 : eta) * losses.at(i);
    }

    std::cout << iterNo << ' ' << error << '\n';
    std::cout << "Determined batch " << weights.at(0) << ' ' << weights.at(1) << ' ' << weights.at(2) << '\n';

    return displayImg;
}

int main()
{
    Mat_<Vec3b> img;
    Mat_<Vec3b> displayImg1;
    Mat_<Vec3b> displayImg2;
    Mat_<float> X;
    Mat_<float> Y;
    std::vector<Point> points;
    std::vector<float> weights({1, 1, -1});
    int noPoints;
    Point dp1, dp2;
    constexpr float errorLimit = 0.00001f;
    constexpr float eta = 0.01f;
    constexpr int maxIter = 100000;
    float error;
    std::string fpath;
    
    for (int testNo = 0; testNo < TOTAL_TESTS; ++testNo)
    {
        points.clear();
        fpath = INPUT_PREFIX + LAB_FOLDER + "test0" + std::to_string(testNo) +".bmp";
        img = imread(fpath, CV_LOAD_IMAGE_COLOR);

        for (int i = 0; i < img.rows; ++i)
            for (int j = 0; j < img.cols; ++j)
                if (!isEmpty(img(i, j)))
                    points.push_back({j, i});

        noPoints = points.size();
        X = Mat_<float>(noPoints, 3);
        Y = Mat_<int>(noPoints, 1);

        for (int i = 0; i < noPoints; ++i)
        {
            Point p = points.at(i);
            X(i, 0) = 1;
            X(i, 1) = p.x;
            X(i, 2) = p.y;
            // blue point if B > R
            Y(i, 0) = (img(p.y, p.x)[0] > img(p.y, p.x)[2]) ? -1 : 1;
        }

        resize(onlinePerception(img, X, Y, maxIter, noPoints, eta, errorLimit), displayImg1, Size(), 2, 2);
        resize(batchPerception(img, X, Y, maxIter, noPoints, eta, errorLimit), displayImg2, Size(), 2, 2);

        imshow("Online perception", displayImg1);
        imshow("Batch perception", displayImg2);
        waitKey(0);
    }

	return 0;
}