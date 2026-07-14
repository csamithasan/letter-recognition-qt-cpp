#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <QDebug>
#include <QFile>
#include <QTextStream>

#include "globalVariables.h"

struct LetterStructure;

// fixed sizes
#define INPUT_NEURONS   16
#define OUTPUT_NEURONS  26

#define MAX_H1_NEURONS  256
#define MAX_H2_NEURONS  256

enum HiddenActivation {
    HACT_SIGMOID = 0,
    HACT_TANH    = 1,
    HACT_RELU    = 2
};

class Backpropagation
{
public:
    Backpropagation();

    // GUI-driven config
    void setLayerSizes(int h1, int h2);
    void setHiddenActivation(HiddenActivation act);
    void setLearningRate(double lr);          // writes global LEARNING_RATE
    void setShuffleEnabled(bool on);
    void setSeed(unsigned int s);             // default 123

    HiddenActivation getHiddenActivation() const { return hiddenAct; }

    void assignRandomWeights(void);
    void feedForward(void);
    void backPropagate(void);
    double* testNetwork(LetterStructure testPattern);
    double getSSEForPattern(double *vec, int targetIndex);
    int action(double *vector);

    // dataset helpers
    double evaluateMSE_train();
    double evaluatePGC_train();
    double evaluateMSE_test();
    double evaluatePGC_test();

    // weight I/O 
    bool saveWeight(const QString &path);
    bool loadWeight(const QString &path);

    // GUI compatibility wrappers
    double trainNetwork();                    // trains ONE epoch, returns avg SSE
    void saveWeights(const QString &path);    // wrappers for GUI code
    void loadWeights(const QString &path);
    void initialise();                        // re-init weights

    // Trains up to maxEpochs, logs per-epoch CSV, early-stops at targetPGC (on TEST).
    // If bestWeightsPath is non-empty, saves the best test-PGC weights there.
    bool trainUntilAndLog(const QString &csvPath,
                          int maxEpochs,
                          double targetPGC,
                          const QString &bestWeightsPath = QString());

    // Writes 26x26 confusion matrix on TEST set to CSV.
    bool writeConfusionMatrixCSV(const QString &csvPath);

private:
    inline int getRand(int maxExclusive) { return (int)(rand() % maxExclusive); }

    // Hidden activations
    inline double sigmoid(double x) const { return 1.0 / (1.0 + std::exp(-x)); }
    inline double sigmoidDerivFromY(double y) const { return y * (1.0 - y); }
    inline double mytanh(double x) const { return std::tanh(x); }
    inline double tanhDerivFromY(double y) const { return 1.0 - (y*y); }
    inline double relu(double x) const { return (x > 0.0) ? x : 0.0; }
    inline double reluDerivFromY(double y) const { return (y > 0.0) ? 1.0 : 0.0; }

    inline double actHidden(double x) const {
        switch (hiddenAct) {
        case HACT_TANH:  return mytanh(x);
        case HACT_RELU:  return relu(x);
        default:         return sigmoid(x);
        }
    }
    inline double actHiddenDerivFromY(double y) const {
        switch (hiddenAct) {
        case HACT_TANH:  return tanhDerivFromY(y);
        case HACT_RELU:  return reluDerivFromY(y);
        default:         return sigmoidDerivFromY(y);
        }
    }

    void softmax(const double *z, double *y, int n) const;

    // Internal helpers
    int trueIndexFromOutputs(const LetterStructure &rec) const;

private:
    // Effective sizes (set by GUI)
    int H1;
    int H2;

    bool SHUFFLE;
    HiddenActivation hiddenAct;
    unsigned int RNG_SEED;

    // Buffers
    double inputs[INPUT_NEURONS];
    double target[OUTPUT_NEURONS];
    double actual[OUTPUT_NEURONS];

    double hidden1[MAX_H1_NEURONS];
    double hidden2[MAX_H2_NEURONS];

    double erro[OUTPUT_NEURONS];
    double errh1[MAX_H1_NEURONS];
    double errh2[MAX_H2_NEURONS];

    double logits[OUTPUT_NEURONS];

    // Weights (bias row at end)
    double wih[INPUT_NEURONS + 1][MAX_H1_NEURONS];  // in -> h1
    double whh[MAX_H1_NEURONS + 1][MAX_H2_NEURONS]; // h1 -> h2
    double who[MAX_H2_NEURONS + 1][OUTPUT_NEURONS]; // h2 -> out
};

#endif // BACKPROPAGATION_H
