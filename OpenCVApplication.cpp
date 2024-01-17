// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <bitset>
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
#define MAX_TRACKS 33000

typedef std::mt19937 RNGeng;
RNGeng engine(100);
std::string INPUT_CSV = "C:\\Users\\logoeje\\source\\repos\\PRS\\project_input\\spotify_songs.csv";

struct WeakLearner
{
    int featureIdx;
    float threshold;
    int classLabel;
    float error;
    int classify(Mat_<int> X)
    {
        return (X.at<int>(featureIdx) < threshold) ? 1 : -1;
    }
};

struct FeatureDescription
{
    float min;
    float max;
    float increment;
};

struct ClassDescription
{
    std::string label;
    int samples;
};

struct LabelVote
{
    float confidence;
    int label;
    bool operator<(LabelVote const& other) const 
    {
        return confidence > other.confidence;
    }
};

std::vector<WeakLearner> findWeakLearners(Mat_<int> X, std::vector<int> Y, std::vector<float> weights, std::vector<FeatureDescription> feature_descriptions, int total_tracks)
{
    std::vector<WeakLearner> bestLearners;
    FeatureDescription current_description;
    float bestErr = LONG_MAX;
    float err;
    float featureLimit;
    int z;
    
    // Select the class of interest
    // i.e.: from 0 to 5, for each genre
    // Hardcoded instead of passed as parameter for conveience
    for (int classLabel = 0; classLabel < 6; ++classLabel)
    {
        WeakLearner bestH;
        bestErr = LONG_MAX;
        // Select the inspected feature
        for (int featureIdx = 0; featureIdx < X.cols; ++featureIdx)
        {
            // Select a threshold in the relevant interval 
            // i.e.: go over [-1.4324, 0.78234] in increments of 0.05
            current_description = feature_descriptions.at(featureIdx);
            featureLimit = current_description.max;
            //std::cout << "Looking at feature " << featureIdx << '\n';
            for (float threshold = current_description.min;
                threshold < featureLimit;
                threshold += current_description.increment)
            {
                err = 0;
                //std::cout << "Running through class " << classLabel << '\n';
                // iterate through the tracks
                for (int i = 0; i < total_tracks; ++i)
                {
                    z = (X(i, featureIdx) < threshold) ? 1 : -1;
                    // Update the error if the label does not match the predicted class
                    if (z * (Y.at(i) == classLabel ? 1 : -1) < 0)
                        err += weights.at(i);
                }
                if (err < bestErr)
                {
                    bestErr = err;
                    bestH = { featureIdx, threshold, classLabel, err };
                }
            }
        }
        bestLearners.push_back(bestH);
    }
    return bestLearners;
}

int compute_prediction(Mat_<float> X, std::vector<WeakLearner> learners)
{
    std::vector<LabelVote> votes;
    for (auto learner : learners)
    {
        int pred = learner.classify(X);
        votes.push_back({pred * (1 - learner.error), learner.classLabel});
    }
    std::sort(votes.begin(), votes.end());
    return votes.at(0).label;
}

struct Track
{
    std::string track_id = "";
    std::string track_name = "";
    std::string track_artist = "";
    double track_popularity = 0; // *
    std::string track_album_id = "";
    std::string track_album_name = "";
    std::string track_album_release_date = "";
    std::string playlist_name = "";
    std::string playlist_id = "";
    std::string playlist_genre = "";
    std::string playlist_subgenre = "";
    double danceability = 0;  // *
    double energy = 0; // *
    double key = 0;
    double loudness = 0; // *
    double mode = 0;
    double speechiness = 0; // *
    double acousticness = 0; // *
    double instrumentalness = 0; // *
    double liveness = 0; // *
    double valence = 0; // *
    double tempo = 0; // *
    double duration_ms = 0; // *

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

    // DATA PROCESSING
    ifstream fi(INPUT_CSV);
    char delimiter = ',';
    string line, temp;
    vector<Track> initial_tracks;
    map<string, int> genre_popularity;
    set<string> malformed_genres;
    vector<Track> tracks;
    Track aux;

    // CLASSIFIER CONSTANTS
    constexpr int RELEVANT_FEATURES = 11;
    constexpr int MAX_LEARNERS = 10;
    vector<string> relevant_feature_names = { 
        "track_popularity", "danceability", "energy", "loudness", "speechiness", "acousticness", 
        "instrumentalness", "liveness", "valence", "tempo", "duration_ms" };

    // CLASSIFIER COMPUTATION
    int train_samples = 0;
    Mat_<float> X = Mat_<float>(MAX_TRACKS, RELEVANT_FEATURES);
    vector<int> Y(MAX_TRACKS, 0);
    vector<float> weights;
    bitset<MAX_TRACKS> train_data;
    int total_classes = 0;
    vector<ClassDescription> class_descriptions;
    map<string, int> genre_mapping;
    vector<FeatureDescription> feature_descriptions;
    vector<vector<WeakLearner>> selected_learners;
    vector<float> alphas;

    // Get the fields
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

    cout << "Reduced " << initial_tracks.size() << " to " << tracks.size() << " songs.\n\n";

    cout << "Genre popularities:\n";
    for (auto const& entry : genre_popularity)
        cout << entry.first << ": " << entry.second << " tracks\n";
    cout << '\n';

    // Split up the data into training and test subsets
    train_data.reset();

    uniform_int_distribution<RNGeng::result_type> dist100(1, 100);
    for (int i = 0; i < tracks.size(); ++i)
        if (dist100(engine) < 85)
            train_data.set(i);

    // Map genres to numeric values and prepare class metrics
    for (auto const& entry : genre_popularity)
    {
        class_descriptions.push_back({entry.first, 0});
        genre_mapping[entry.first] = total_classes++;
    }

    // Set up working dataset
    if (tracks.size() > MAX_TRACKS)
        throw new exception("There are more tracks registered than should be possible");

    for (int i = 0; i < RELEVANT_FEATURES; ++i)
    {
        FeatureDescription mockDescription = { LONG_MAX, LONG_MIN, 0 }; // TODO - why is this necessary?
        feature_descriptions.push_back(mockDescription);
    }

    X.setTo(0);
    for (int i = 0; i < tracks.size(); ++i)
        if (train_data[i])
        {
            aux = tracks.at(i);
            X(train_samples, 0) = aux.track_popularity;
            X(train_samples, 1) = aux.danceability;
            X(train_samples, 2) = aux.energy;
            X(train_samples, 3) = aux.loudness;
            X(train_samples, 4) = aux.speechiness;
            X(train_samples, 5) = aux.acousticness;
            X(train_samples, 6) = aux.instrumentalness;
            X(train_samples, 7) = aux.liveness;
            X(train_samples, 8) = aux.valence;
            X(train_samples, 9) = aux.tempo;
            X(train_samples, 10) = aux.duration_ms;
            Y.at(train_samples) = genre_mapping[aux.playlist_genre];
            ++class_descriptions.at(Y.at(train_samples)).samples;

            for (int j = 0; j < RELEVANT_FEATURES; ++j)
            {
                FeatureDescription& fd = feature_descriptions.at(j);
                fd.min = min(fd.min, X(train_samples, j));
                fd.max = max(fd.max, X(train_samples, j));
            }

            ++train_samples;
        }

    // Since features differ in meaning and magnitute, each increment is different
    // The objective is to make each increment ~1% of the interval, in order to 
    // limit the training and testing time
    for (auto& description : feature_descriptions)
        description.increment = (description.max - description.min) / 100.0f;

    cout << "Training data metrics:\n";
    for (auto const& description : class_descriptions)
        std::cout << description.label << ": " << description.samples << " tracks\n";
    cout << '\n';

    cout << "Feature data:\n";
    for (int i = 0; i < RELEVANT_FEATURES; ++i)
        cout << relevant_feature_names.at(i) << ": " << feature_descriptions.at(i).min << " -> " 
            << feature_descriptions.at(i).max << " (inc: " << feature_descriptions.at(i).increment << ")\n";
    cout << '\n';

    // Train the model

    weights = vector<float>(train_samples, 1.0f / (1.0f * train_samples));
    for (int learnerIdx = 0; learnerIdx < MAX_LEARNERS; ++learnerIdx)
    {
        vector<WeakLearner> weak_learners = findWeakLearners(X, Y, weights, feature_descriptions, train_samples);
        float alpha;
        float err = 0;
        int predicted_class;

        cout << "Selected learners:\n";
        for (auto learner : weak_learners)
        {
            cout << "Target: " << learner.classLabel << '\n';
            cout << "Feature: " << learner.featureIdx << '\n';
            cout << "Threshold: " << learner.threshold << '\n';
            cout << "Error: " << learner.error << "\n\n";
        }

        for (int i = 0; i < train_samples; ++i)
        {
            predicted_class = compute_prediction(X.row(i), weak_learners);
            if (predicted_class != Y.at(i))
                ++err;
        }

        err /= train_samples;
        cout << "Initial error: " << err << '\n';

        alpha = 0.5f * log((1 - err) / err);

        float s = 0;
        for (int i = 0; i < train_samples; ++i)
        {
            predicted_class = compute_prediction(X.row(i), weak_learners);
            weights.at(i) = weights.at(i) * exp(-alpha * (Y.at(i) == predicted_class ? 1 : -1));
            s += weights.at(i);
        }

        for (int i = 0; i < train_samples; ++i)
            weights.at(i) /= s;

        alphas.push_back(alpha); 
        selected_learners.push_back(weak_learners);
    }



    // TODO
    // Create generic functions for training weak learners for a genre DONE
    // i.e.: deduce from the numeric fields of each track how likely it is to be rap
    // 
    // Split up each genre track set into training and test samples DONE
    // 
    // Perform training for each genre DONE
    // 
    // Clump together all test data, and run each track through all classifiers DONE
    // 
    // Assume for each track that its highest probability genre is the correct one and draw conclusions DONE


	return 0;
}