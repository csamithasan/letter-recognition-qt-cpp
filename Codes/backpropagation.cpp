#include "backpropagation.h"
#include <limits>
#include <cmath>
#include <QFileInfo>

// Globals declared 
extern LetterStructure letters[20001];
extern LetterStructure testPattern;
extern bool patternsLoadedFromFile;
extern int MAX_EPOCHS;
extern double LEARNING_RATE;

static inline double RAND_WEIGHT_UNIF()
{
    return ( (double)rand() / (double)RAND_MAX ) * 0.2 - 0.1; // [-0.1, 0.1]
}

Backpropagation::Backpropagation()
{
    H1 = 64;
    H2 = 64;
    SHUFFLE   = true;
    hiddenAct = HACT_RELU;
    RNG_SEED  = 123;
    setSeed(RNG_SEED);
    assignRandomWeights();
}

void Backpropagation::setLayerSizes(int h1, int h2)
{
    if (h1 < 1) h1 = 1; if (h1 > MAX_H1_NEURONS) h1 = MAX_H1_NEURONS;
    if (h2 < 1) h2 = 1; if (h2 > MAX_H2_NEURONS) h2 = MAX_H2_NEURONS;
    H1 = h1; H2 = h2;
    qDebug() << "H1 =" << H1 << ", H2 =" << H2;
}

void Backpropagation::setHiddenActivation(HiddenActivation act)
{
    hiddenAct = act;
    qDebug() << "Hidden activation =" << (int)hiddenAct;
}

void Backpropagation::setLearningRate(double lr)
{
    if (lr <= 0) lr = 0.0001;
    LEARNING_RATE = lr; // keep your global
    qDebug() << "LEARNING_RATE =" << LEARNING_RATE;
}

void Backpropagation::setShuffleEnabled(bool on)
{
    SHUFFLE = on;
    qDebug() << "SHUFFLE =" << SHUFFLE;
}

void Backpropagation::setSeed(unsigned int s)
{
    RNG_SEED = s;
    srand(RNG_SEED);
    qDebug() << "RNG seed =" << RNG_SEED;
}

void Backpropagation::assignRandomWeights( void )
{
    for (int i=0;i<INPUT_NEURONS+1;++i)
        for (int j=0;j<MAX_H1_NEURONS;++j)
            wih[i][j] = RAND_WEIGHT_UNIF();

    for (int j=0;j<MAX_H1_NEURONS+1;++j)
        for (int k=0;k<MAX_H2_NEURONS;++k)
            whh[j][k] = RAND_WEIGHT_UNIF();

    for (int k=0;k<MAX_H2_NEURONS+1;++k)
        for (int o=0;o<OUTPUT_NEURONS;++o)
            who[k][o] = RAND_WEIGHT_UNIF();
}

void Backpropagation::softmax(const double *z, double *y, int n) const
{
    double m = z[0];
    for (int i=1;i<n;++i) if (z[i] > m) m = z[i];
    double s = 0.0;
    for (int i=0;i<n;++i) { y[i] = std::exp(z[i]-m); s += y[i]; }
    if (s <= 0.0) s = 1.0;
    for (int i=0;i<n;++i) y[i] /= s;
}

void Backpropagation::feedForward( )
{
    // hidden1
    for (int j=0;j<H1;++j) {
        double sum = wih[INPUT_NEURONS][j]; // bias
        for (int i=0;i<INPUT_NEURONS;++i)
            sum += inputs[i] * wih[i][j];
        hidden1[j] = actHidden(sum);
    }

    // hidden2
    for (int k=0;k<H2;++k) {
        double sum = whh[H1][k]; // bias
        for (int j=0;j<H1;++j)
            sum += hidden1[j] * whh[j][k];
        hidden2[k] = actHidden(sum);
    }

    // logits
    for (int o=0;o<OUTPUT_NEURONS;++o) {
        double sum = who[H2][o]; // bias
        for (int k=0;k<H2;++k)
            sum += hidden2[k] * who[k][o];
        logits[o] = sum;
    }

    // softmax
    softmax(logits, actual, OUTPUT_NEURONS);
}

void Backpropagation::backPropagate( void )
{
    // output gradient: softmax + CE => y_pred - y_true
    for (int o=0;o<OUTPUT_NEURONS;++o)
        erro[o] = (actual[o] - target[o]);

    // hidden2 error
    for (int k=0;k<H2;++k) {
        double g = 0.0;
        for (int o=0;o<OUTPUT_NEURONS;++o)
            g += erro[o] * who[k][o];
        errh2[k] = g * actHiddenDerivFromY(hidden2[k]);
    }

    // hidden1 error
    for (int j=0;j<H1;++j) {
        double g = 0.0;
        for (int k=0;k<H2;++k)
            g += errh2[k] * whh[j][k];
        errh1[j] = g * actHiddenDerivFromY(hidden1[j]);
    }

    // update h2->out
    for (int o=0;o<OUTPUT_NEURONS;++o) {
        for (int k=0;k<H2;++k)
            who[k][o] -= (LEARNING_RATE * erro[o] * hidden2[k]);
        who[H2][o] -= (LEARNING_RATE * erro[o]); // bias
    }

    // update h1->h2
    for (int k=0;k<H2;++k) {
        for (int j=0;j<H1;++j)
            whh[j][k] -= (LEARNING_RATE * errh2[k] * hidden1[j]);
        whh[H1][k] -= (LEARNING_RATE * errh2[k]); // bias
    }

    // update in->h1
    for (int j=0;j<H1;++j) {
        for (int i=0;i<INPUT_NEURONS;++i)
            wih[i][j] -= (LEARNING_RATE * errh1[j] * inputs[i]);
        wih[INPUT_NEURONS][j] -= (LEARNING_RATE * errh1[j]); // bias
    }
}

double Backpropagation::getSSEForPattern(double *vec, int targetIndex)
{
    for (int i=0;i<INPUT_NEURONS;++i) inputs[i] = vec[i];

    for (int k=0;k<OUTPUT_NEURONS;++k) target[k] = 0.0;
    if (targetIndex >=0 && targetIndex < OUTPUT_NEURONS) target[targetIndex] = 1.0;

    feedForward();

    double sse = 0.0;
    for (int k=0;k<OUTPUT_NEURONS;++k) {
        double d = (target[k] - actual[k]);
        sse += 0.5 * d * d;
    }
    return sse;
}

double* Backpropagation::testNetwork(LetterStructure testPattern)
{
    for (int j=0;j<INPUT_NEURONS;++j)
        inputs[j] = double(testPattern.f[j]);

    // zero target for inference
    for (int k=0;k<OUTPUT_NEURONS;++k) target[k] = 0.0;

    feedForward();
    return actual; // 26 scores
}

int Backpropagation::action( double *vector )
{
    int sel = 0;
    double maxv = vector[0];
    for (int i=1;i<OUTPUT_NEURONS;++i) {
        if (vector[i] > maxv) { maxv = vector[i]; sel = i; }
    }
    qDebug() << "Selected index =" << sel << ", with actual output=" << maxv;
    return sel;
}

// Helpers using one-hot y
int Backpropagation::trueIndexFromOutputs(const LetterStructure &rec) const
{
    const int *y = (const int*)(&rec.outputs);
    for (int j=0;j<OUTPUT_NEURONS;++j) {
        if (y[j] == 1) return j;
    }
    // Fallback: choose max
    int best=0; int bestv=y[0];
    for (int j=1;j<OUTPUT_NEURONS;++j){ if (y[j] > bestv){ bestv=y[j]; best=j; } }
    return best;
}

// Dataset evaluation

double Backpropagation::evaluateMSE_train()
{
    double sum = 0.0;
    for (int s=0; s<NUMBER_OF_TRAINING_PATTERNS; ++s) {
        for (int i=0;i<INPUT_NEURONS;++i) inputs[i] = double(letters[s].f[i]);
        // one-hot
        for (int k=0;k<OUTPUT_NEURONS;++k) target[k] = 0.0;
        int t = trueIndexFromOutputs(letters[s]);
        if (t>=0 && t<OUTPUT_NEURONS) target[t] = 1.0;
        feedForward();
        for (int k=0;k<OUTPUT_NEURONS;++k) {
            double d = (target[k] - actual[k]);
            sum += 0.5 * d * d;
        }
    }
    return sum / double(NUMBER_OF_TRAINING_PATTERNS);
}

double Backpropagation::evaluatePGC_train()
{
    int correct = 0;
    for (int s=0; s<NUMBER_OF_TRAINING_PATTERNS; ++s) {
        for (int i=0;i<INPUT_NEURONS;++i) inputs[i] = double(letters[s].f[i]);
        feedForward();
        int best = 0; double bestv = actual[0];
        for (int k=1;k<OUTPUT_NEURONS;++k) if (actual[k] > bestv){ bestv=actual[k]; best=k; }
        if (best == trueIndexFromOutputs(letters[s])) ++correct;
    }
    return 100.0 * double(correct) / double(NUMBER_OF_TRAINING_PATTERNS);
}

double Backpropagation::evaluateMSE_test()
{
    double sum = 0.0;
    const int start = NUMBER_OF_TRAINING_PATTERNS;
    const int end   = NUMBER_OF_TRAINING_PATTERNS + NUMBER_OF_TEST_PATTERNS;
    for (int s=start; s<end; ++s) {
        for (int i=0;i<INPUT_NEURONS;++i) inputs[i] = double(letters[s].f[i]);
        for (int k=0;k<OUTPUT_NEURONS;++k) target[k] = 0.0;
        int t = trueIndexFromOutputs(letters[s]);
        if (t>=0 && t<OUTPUT_NEURONS) target[t] = 1.0;
        feedForward();
        for (int k=0;k<OUTPUT_NEURONS;++k) {
            double d = (target[k] - actual[k]);
            sum += 0.5 * d * d;
        }
    }
    return sum / double(NUMBER_OF_TEST_PATTERNS);
}

double Backpropagation::evaluatePGC_test()
{
    int correct = 0;
    const int start = NUMBER_OF_TRAINING_PATTERNS;
    const int end   = NUMBER_OF_TRAINING_PATTERNS + NUMBER_OF_TEST_PATTERNS;
    for (int s=start; s<end; ++s) {
        for (int i=0;i<INPUT_NEURONS;++i) inputs[i] = double(letters[s].f[i]);
        feedForward();
        int best = 0; double bestv = actual[0];
        for (int k=1;k<OUTPUT_NEURONS;++k) if (actual[k] > bestv){ bestv=actual[k]; best=k; }
        if (best == trueIndexFromOutputs(letters[s])) ++correct;
    }
    return 100.0 * double(correct) / double(NUMBER_OF_TEST_PATTERNS);
}

// GUI-compatibility wrappers

// Trains ONE epoch over training set; returns avg SSE
double Backpropagation::trainNetwork()
{
    // Build shuffled indices if requested
    std::vector<int> idx(NUMBER_OF_TRAINING_PATTERNS);
    for (int i=0;i<NUMBER_OF_TRAINING_PATTERNS;++i) idx[i]=i;
    if (SHUFFLE) {
        for (int i=NUMBER_OF_TRAINING_PATTERNS-1;i>0;--i) {
            int j=getRand(i+1);
            std::swap(idx[i], idx[j]);
        }
    }

    double sse_sum = 0.0;

    for (int s=0;s<NUMBER_OF_TRAINING_PATTERNS;++s) {
        const LetterStructure &rec = letters[idx[s]];

        // set inputs
        for (int i=0;i<INPUT_NEURONS;++i) inputs[i] = double(rec.f[i]);

        // one-hot from outputs[]
        int t = trueIndexFromOutputs(rec);
        for (int k=0;k<OUTPUT_NEURONS;++k) target[k] = 0.0;
        if (t>=0 && t<OUTPUT_NEURONS) target[t] = 1.0;

        // forward/backward
        feedForward();

        // SSE before update
        double sse = 0.0;
        for (int k=0;k<OUTPUT_NEURONS;++k){
            double d = (target[k] - actual[k]);
            sse += 0.5 * d * d;
        }
        sse_sum += sse;

        backPropagate();
    }

    return sse_sum / double(NUMBER_OF_TRAINING_PATTERNS);
}

void Backpropagation::saveWeights(const QString &path)
{
    (void)saveWeight(path);
}

void Backpropagation::loadWeights(const QString &path)
{
    (void)loadWeight(path);
}

void Backpropagation::initialise()
{
    assignRandomWeights();
    qDebug() << "Network re-initialised.";
}

// Training with logging + early stop
// ==============================
bool Backpropagation::trainUntilAndLog(const QString &csvPath,
                                       int maxEpochs,
                                       double targetPGC,
                                       const QString &bestWeightsPath)
{
    // open CSV
    QFileInfo info(csvPath);
    const bool existsBefore = info.exists();

    QFile csv(csvPath);
    if (!csv.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Append)) {
        qWarning() << "Cannot open CSV for append:" << csvPath;
        return false;
    }
    QTextStream out(&csv);
    if (!existsBefore) {
        out << "epoch,train_MSE,train_PGC,test_MSE,test_PGC\n";
    }

    double bestTestPGC = -1.0;
    int bestEpoch = -1;

    for (int epoch = 1; epoch <= maxEpochs; ++epoch) {
        double avgSSE = trainNetwork(); // one epoch
        Q_UNUSED(avgSSE);

        const double trMSE = evaluateMSE_train();
        const double trPGC = evaluatePGC_train();
        const double teMSE = evaluateMSE_test();
        const double tePGC = evaluatePGC_test();

        out << epoch << "," << trMSE << "," << trPGC << "," << teMSE << "," << tePGC << "\n";
        out.flush();

        qDebug() << "Epoch" << epoch
                 << "train_PGC=" << trPGC
                 << "test_PGC="  << tePGC;

        // Track best
        if (tePGC > bestTestPGC) {
            bestTestPGC = tePGC;
            bestEpoch = epoch;
            if (!bestWeightsPath.isEmpty()) {
                saveWeight(bestWeightsPath);
            }
        }

        // Early stop
        if (tePGC >= targetPGC) {
            qDebug() << "Target PGC reached (" << tePGC << "%) at epoch" << epoch
                     << ". Best so far:" << bestTestPGC << "% @ epoch" << bestEpoch;
            csv.close();
            return true;
        }
    }

    qDebug() << "Finished" << maxEpochs << "epochs. Best test PGC:"
             << bestTestPGC << "% at epoch" << bestEpoch;
    csv.close();
    return true;
}


// Confusion Matrix (TEST set)
bool Backpropagation::writeConfusionMatrixCSV(const QString &csvPath)
{
    int cm[OUTPUT_NEURONS][OUTPUT_NEURONS];
    for (int r=0;r<OUTPUT_NEURONS;++r)
        for (int c=0;c<OUTPUT_NEURONS;++c)
            cm[r][c]=0;

    const int start = NUMBER_OF_TRAINING_PATTERNS;
    const int end   = NUMBER_OF_TRAINING_PATTERNS + NUMBER_OF_TEST_PATTERNS;

    for (int s=start; s<end; ++s) {
        for (int i=0;i<INPUT_NEURONS;++i) inputs[i] = double(letters[s].f[i]);
        feedForward();
        int pred = 0; double bestv = actual[0];
        for (int k=1;k<OUTPUT_NEURONS;++k) if (actual[k] > bestv){ bestv=actual[k]; pred=k; }
        int truth = trueIndexFromOutputs(letters[s]);
        cm[truth][pred] += 1;
    }

    QFile f(csvPath);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Cannot open for writing confusion matrix:" << csvPath;
        return false;
    }
    QTextStream out(&f);

    // header
    out << "true\\pred";
    for (int c=0;c<OUTPUT_NEURONS;++c) out << "," << (char)('A'+c);
    out << "\n";

    for (int r=0;r<OUTPUT_NEURONS;++r) {
        out << (char)('A'+r);
        for (int c=0;c<OUTPUT_NEURONS;++c) out << "," << cm[r][c];
        out << "\n";
    }

    f.close();
    qDebug() << "Confusion matrix written to" << csvPath;
    return true;
}

// Weight I/O
bool Backpropagation::saveWeight(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Cannot open for writing:" << path;
        return false;
    }
    QTextStream out(&file);

    out << "# H1 H2\n" << H1 << " " << H2 << "\n";
    out << "# HiddenAct (0=sigmoid,1=tanh,2=relu)\n" << (int)hiddenAct << "\n";
    out << "# LearningRate\n" << LEARNING_RATE << "\n";
    out << "# Seed\n" << RNG_SEED << "\n";

    out << "# W(in->h1) rows=" << (INPUT_NEURONS+1) << " cols=" << H1 << "\n";
    for (int i=0;i<INPUT_NEURONS+1;++i){
        for (int j=0;j<H1;++j) out << wih[i][j] << ",";
        out << "\n";
    }

    out << "# W(h1->h2) rows=" << (H1+1) << " cols=" << H2 << "\n";
    for (int j=0;j<H1+1;++j){
        for (int k=0;k<H2;++k) out << whh[j][k] << ",";
        out << "\n";
    }

    out << "# W(h2->out) rows=" << (H2+1) << " cols=" << OUTPUT_NEURONS << "\n";
    for (int k=0;k<H2+1;++k){
        for (int o=0;o<OUTPUT_NEURONS;++o) out << who[k][o] << ",";
        out << "\n";
    }

    file.close();
    qDebug() << "Weights saved to file.";
    return true;
}

bool Backpropagation::loadWeight(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Cannot open for reading:" << path;
        return false;
    }
    QTextStream in(&file);

    auto readLine = [&]() -> QString { return in.readLine(); };

    QString line;

    // ----- "# H1 H2" then "H1 H2" -----
    line = readLine();                 // "# H1 H2"
    if (line.isNull()) { file.close(); return false; }
    line = readLine();                 // e.g. "64 64"
    {
        int h1 = H1, h2 = H2;
        QTextStream ts(&line);
        ts >> h1 >> h2;
        setLayerSizes(h1, h2);
    }

    readLine();                        // "# HiddenAct (0=sigmoid,1=tanh,2=relu)"
    line = readLine();                 
    {
        int act = (int)hiddenAct;
        QTextStream ts(&line);
        ts >> act;
        setHiddenActivation(static_cast<HiddenActivation>(act));
    }

    readLine();                        // "# LearningRate"
    line = readLine();                 
    {
        double lr = LEARNING_RATE;
        QTextStream ts(&line);
        ts >> lr;
        setLearningRate(lr);
    }

    // "# Seed" then value
    readLine();                        // "# Seed"
    line = readLine();                 // e.g. "123"
    {
        unsigned int seed = RNG_SEED;
        QTextStream ts(&line);
        ts >> seed;
        setSeed(seed);
    }

    // "# W(in->h1) rows=.. cols=.."
    readLine();
    for (int i = 0; i < INPUT_NEURONS + 1; ++i) {
        line = readLine();
        if (line.isNull()) break;
        const QStringList parts = line.split(',', Qt::SkipEmptyParts);
        const int n = qMin(H1, parts.size());
        for (int j = 0; j < n; ++j)
            wih[i][j] = parts[j].toDouble();
    }

    // "# W(h1->h2) rows=.. cols=.." 
    readLine();
    for (int j = 0; j < H1 + 1; ++j) {
        line = readLine();
        if (line.isNull()) break;
        const QStringList parts = line.split(',', Qt::SkipEmptyParts);
        const int n = qMin(H2, parts.size());
        for (int k = 0; k < n; ++k)
            whh[j][k] = parts[k].toDouble();
    }

    // "# W(h2->out) rows=.. cols=.."
    readLine();
    for (int k = 0; k < H2 + 1; ++k) {
        line = readLine();
        if (line.isNull()) break;
        const QStringList parts = line.split(',', Qt::SkipEmptyParts);
        const int n = qMin(OUTPUT_NEURONS, parts.size());
        for (int o = 0; o < n; ++o)
            who[k][o] = parts[o].toDouble();
    }

    file.close();
    qDebug() << "Weights loaded.";
    return true;
}

