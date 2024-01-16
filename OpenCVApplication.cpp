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
#include <map>
#include <set>

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
std::string INPUT_CSV = "C:\\Users\\logoeje\\source\\repos\\PRS\\project_input\\spotify_songs.csv";

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

struct WeakLearner
{
    int featureIdx;
    int threshold;
    int classLabel;
    float error;
    int classify(Mat_<int> X)
    {
        return (X.at<int>(featureIdx) < threshold) ? classLabel : -classLabel;
    }
};

WeakLearner findWeakLearner(Mat_<Vec3b> img, Mat_<int> X, std::vector<int> Y, std::vector<float> weights)
{
    WeakLearner bestH;
    float bestErr = LONG_MAX;
    float err;
    int featureLimit;
    int z;
    
    // select the inspected feature
    for (int featureIdx = 0; featureIdx < X.cols; ++featureIdx)
    {
        // select the threshold
        featureLimit = featureIdx ? img.rows : img.cols;
        for (int threshold = 0; threshold < featureLimit; ++threshold)
        {
            // select the class of interest
            for (int classLabel = -1; classLabel <= 1; classLabel += 2)
            {
                err = 0;
                // iterate through the points
                for (int i = 0; i < X.rows; ++i)
                {
                    z = (X(i, featureIdx) < threshold) ? classLabel : -classLabel;
                    if (z * Y.at(i) < 0)
                        err += weights.at(i);
                }
                if (err < bestErr)
                {
                    bestErr = err;
                    bestH = { featureIdx, threshold, classLabel, err };
                }
            }
        }
    }

    return bestH;
}

struct Track
{
    std::string track_id = "";
    std::string track_name = "";
    std::string track_artist = "";
    double track_popularity = 0;
    std::string track_album_id = "";
    std::string track_album_name = "";
    std::string track_album_release_date = "";
    std::string playlist_name = "";
    std::string playlist_id = "";
    std::string playlist_genre = "";
    std::string playlist_subgenre = "";
    double danceability = 0;
    double energy = 0;
    double key = 0;
    double loudness = 0;
    double mode = 0;
    double speechiness = 0;
    double acousticness = 0;
    double instrumentalness = 0;
    double liveness = 0;
    double valence = 0;
    double tempo = 0;
    double duration_ms = 0;

    void extract_from_stream(std::string line)
    {
        using namespace std;
        istringstream iss(line);
        string temp;
        getline(iss, temp, ',');
        track_id = temp;
        getline(iss, temp, ',');
        track_name = temp;
        getline(iss, temp, ',');
        track_artist = temp;
        getline(iss, temp, ',');
        istringstream tiss(temp);
        tiss >> track_popularity;
        getline(iss, temp, ',');
        track_album_id = temp;
        getline(iss, temp, ',');
        track_album_name = temp;
        getline(iss, temp, ',');
        track_album_release_date = temp;
        getline(iss, temp, ',');
        playlist_name = temp;
        getline(iss, temp, ',');
        playlist_id = temp;
        getline(iss, temp, ',');
        playlist_genre = temp;
        getline(iss, temp, ',');
        playlist_subgenre = temp;
        getline(iss, temp, ',');
        tiss = istringstream(temp);
        tiss >> danceability;
        getline(iss, temp, ',');
        tiss = istringstream(temp);
        tiss >> energy;
        getline(iss, temp, ',');
        tiss = istringstream(temp);
        tiss >> key;
        getline(iss, temp, ',');
        tiss = istringstream(temp);
        tiss >> loudness;
        getline(iss, temp, ',');
        tiss = istringstream(temp);
        tiss >> mode;
        getline(iss, temp, ',');
        tiss = istringstream(temp);
        tiss >> speechiness;
        getline(iss, temp, ',');
        tiss = istringstream(temp);
        tiss >> acousticness;
        getline(iss, temp, ',');
        tiss = istringstream(temp);
        tiss >> instrumentalness;
        getline(iss, temp, ',');
        tiss = istringstream(temp);
        tiss >> liveness;
        getline(iss, temp, ',');
        tiss = istringstream(temp);
        tiss >> valence;
        getline(iss, temp, ',');
        tiss = istringstream(temp);
        tiss >> tempo;
        iss >> duration_ms;
    }
};

int main()
{
    using namespace std;
    vector<Track> initial_tracks; // X
    vector<Track> tracks; // X
    set<string> malformed_genres;
    map<string, int> genre_popularity;
    Track aux;
    ifstream fi(INPUT_CSV);
    char delimiter = ',';
    string line, temp;

    // Get the field names
    getline(fi, line);
    while (getline(fi, line))
    {
        aux.extract_from_stream(line);
        initial_tracks.push_back(aux);
        //std::cout << aux.track_name << ' ' << aux.duration_ms << '\n';
    }

    // Explore the labels
    for (auto const& track : initial_tracks)
    {
        if (!genre_popularity.count(track.playlist_genre))
        {
            genre_popularity[track.playlist_genre] = 0;
        }
        ++genre_popularity[track.playlist_genre];
    }

    // Remove all genres with less than 100 tracks
    // This aims to remove low popularity genres that would negatively impact
    // the classifier, along with misread lines due to extra commas in the initial CSV
    for (auto const& key : genre_popularity)
        if (key.second < 200)
            malformed_genres.insert(key.first);

    for (auto const& genre : malformed_genres)
        genre_popularity.erase(genre);

    for (auto track : initial_tracks)
        if (!malformed_genres.count(track.playlist_genre))
            tracks.push_back(track);

    cout << "Reduced " << initial_tracks.size() << " to " << tracks.size() << " songs.\n";

    cout << "Genre popularities:\n";
    for (auto const& key : genre_popularity)
        cout << key.first << ": " << key.second << " songs" << '\n';
	return 0;
}