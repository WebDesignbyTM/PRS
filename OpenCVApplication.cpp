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
std::string LAB_FOLDER = "lab8\\";
std::string SUBFOLDERS[] = { "train\\", "test\\" };
constexpr int SUBFOLDERS_NO = 2; // 2
std::string CLASSES[] = { "beach", "city", "desert", "forest", "landscape", "snow" };
constexpr int CLASSES_NO = 6; // 6
const int TOTAL_TESTS = 5;

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

int main()
{
    Mat_<Vec3b> img;
    Mat_<Vec3b> imgHsv;
    std::vector<float> hist;
    std::vector<float> aux;
    constexpr int histBins = 128;
    constexpr int totalTrainingSamples = 672;
    constexpr int k = 11;
    Mat_<float> X(totalTrainingSamples, 3 * histBins);
    std::vector<int> y(totalTrainingSamples);
    std::string fpath;
    int subfolder;
    // <distance, class>
    std::vector<std::pair<float, int>> distances;
    std::vector<int> votes(6, 0);
    Mat_<float> confusion(CLASSES_NO, CLASSES_NO);
    float accuracy = 0;
    float correct = 0;
    float total = 0;
    
    for (int crtBins = 8; crtBins <= 64; crtBins += 8)
    {
        for (int crtK = 5; crtK <= 35; crtK += 2)
        {
            std::cout << crtBins << " bins and " << crtK << " k\n";

            // reset all data
            confusion.setTo(0);
            hist.clear();
            aux.clear();
            X = Mat_<float>(totalTrainingSamples, 3 * crtBins);
            X.setTo(0);
            y.clear();
            for (int i = 0; i < totalTrainingSamples; ++i)
                y.push_back(0);
            accuracy = correct = total = 0;

            // train data - only take the train subfolder
            subfolder = 0;
            int fileIdx = 0;
            for (int category = 0; category < CLASSES_NO; ++category)
            {
                fpath = INPUT_PREFIX + LAB_FOLDER + SUBFOLDERS[subfolder] + CLASSES[category];
                for (auto const &entry : fs::directory_iterator(fpath))
                {
                    img = imread(entry.path().string(), CV_LOAD_IMAGE_COLOR);
                    cvtColor(img, imgHsv, CV_RGB2HSV);
                    hist = computeHistogram(imgHsv, crtBins);
                    for (int j = 0; j < X.cols; ++j)
                        X(fileIdx, j) = hist.at(j);
                    y.at(fileIdx) = category;
                    ++fileIdx;
                }
            }

            // test data - only take the test subfolder
            subfolder = 1;
            for (int category = 0; category < CLASSES_NO; ++category)
            {
                fpath = INPUT_PREFIX + LAB_FOLDER + SUBFOLDERS[subfolder] + CLASSES[category];
                for (auto const& entry : fs::directory_iterator(fpath))
                {
                    distances.clear();
                    for (int i = 0; i < CLASSES_NO; ++i)
                        votes.at(i) = 0;
                    img = imread(entry.path().string(), CV_LOAD_IMAGE_COLOR);
                    cvtColor(img, imgHsv, CV_RGB2HSV);
                    hist = computeHistogram(imgHsv, crtBins);

                    // calculate distances
                    for (int i = 0; i < X.rows; ++i)
                    {
                        X.row(i).copyTo(aux);
                        distances.push_back({ calculateHistogramDistance(hist, aux), y.at(i) });
                    }

                    // count votes
                    std::sort(distances.begin(), distances.end());
                    for (int i = 0; i < crtK; ++i)
                        ++votes.at(distances.at(i).second);

                    // find best class
                    int bestClass, bestVotes;
                    bestClass = bestVotes = 0;
                    for (int i = 0; i < CLASSES_NO; ++i)
                    {
                        //std::cout << votes.at(i) << ' ';
                        if (votes.at(i) > bestVotes)
                        {
                            bestVotes = votes.at(i);
                            bestClass = i;
                        }
                    }

                    //std::cout << CLASSES[bestClass] << '\n';
                    ++confusion(bestClass, category);
                    //imshow("Sample", img);
                    //waitKey(0);
                }
            }

            for (int i = 0; i < confusion.rows; ++i)
            {
                for (int j = 0; j < confusion.cols; ++j)
                {
                    //std::cout << confusion(i, j) << ' ';
                    if (i == j)
                        correct += confusion(i, j);
                    total += confusion(i, j);
                }
                //std::cout << '\n';
            }

            accuracy = correct / total;
            if (accuracy > 0.63)
                std::cout << "Accuracy " << accuracy << '\n';

        }
    }

	return 0;
}