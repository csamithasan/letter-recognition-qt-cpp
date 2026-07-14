#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "globalVariables.h"
#include "backpropagation.h"

#include <QVector>
#include <QList>
#include <QString>
#include <QChar>
#include <QFileDialog>
#include <QScrollArea>
#include <QMessageBox>
#include <QDateTime>
#include <QScreen>
#include <QCursor>
#include <QInputDialog>
#include <QBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <cctype>

//--------------------------------------
// Global buffers
LetterStructure letters[20001];
LetterStructure testPattern;

bool patternsLoadedFromFile;
int MAX_EPOCHS;
double LEARNING_RATE;

// confidence threshold for one-sample GUI
static constexpr double TAU_UNKNOWN = 0.45;  // min confidence to accept any letter
static constexpr double MARGIN_MIN  = 0.05;  // min to keep that letter

// thresholded prediction ONLY for single-pattern GUI
// returns class index in [0..25], or -1 to mean "UNKNOWN".
static int predict_with_threshold(const double* outs, int len)
{
    int argmax = 0;
    double maxv = outs[0], secondv = -1.0;

    for (int k = 1; k < len; ++k) {
        double v = outs[k];
        if (v > maxv) { secondv = maxv; maxv = v; argmax = k; }
        else if (v > secondv) { secondv = v; }
    }

    if (maxv >= TAU_UNKNOWN && (maxv - secondv) >= MARGIN_MIN)
        return argmax;
    return -1; // "UNKNOWN" for GUI
}

// argmax for metrics
static int predict_argmax(const double* outs, int len)
{
    int idx = 0; double best = outs[0];
    for (int k = 1; k < len; ++k)
        if (outs[k] > best) { best = outs[k]; idx = k; }
    return idx;
}

///////////////////////////////////////////////////////
// Hidden activation: UI index <-> enum
static int actToIndex(HiddenActivation a) {
    switch (a) {
        case HACT_RELU:    return 0;
        case HACT_TANH:    return 1;
        case HACT_SIGMOID: return 2;
        default:           return 0;
    }
}
static HiddenActivation indexToAct(int idx) {
    switch (idx) {
        case 0:  return HACT_RELU;
        case 1:  return HACT_TANH;
        case 2:  return HACT_SIGMOID;
        default: return HACT_RELU;
    }
}


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    

    //---------------------------------------
    // initialisation of global variables
    LEARNING_RATE = 0.2;
    patternsLoadedFromFile = false;
    MAX_EPOCHS = 200;

    bp = new Backpropagation;

    bp->setHiddenActivation(HACT_TANH);
    bp->setLayerSizes(64, 64);
    bp->setLearningRate(LEARNING_RATE);

    //---------------------------------------
    // initialise widgets
    ui->spinBox_training_Epochs->setValue(MAX_EPOCHS);
    ui->horizScrollBar_LearningRate->setValue(int(LEARNING_RATE*100));

    QWidget *content = this->centralWidget();
    auto *scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setWidget(content);
    this->setCentralWidget(scroll);

    const QRect avail = screen()->availableGeometry();
    this->resize(qMin(1200, avail.width()-80), qMin(850, avail.height()-80));

    // shuffle checkbox
    addShuffleCheckboxNextToInit();
    addActivationChooserNextToReadFile();

    // remember last target
    lastTargetPGC = 0.0;
}

MainWindow::~MainWindow()
{
    delete bp;
    delete ui;
}

void MainWindow::addShuffleCheckboxNextToInit()
{
    if (chkShuffleUI) return;

    QPushButton* initBtn = ui->pushButton_Initialise_Network;
    if (!initBtn) return;

    QWidget* parentW = initBtn->parentWidget();
    QLayout* lay     = parentW ? parentW->layout() : nullptr;

    chkShuffleUI = new QCheckBox("Shuffle each epoch", parentW ? parentW : this);
    chkShuffleUI->setChecked(true);
    connect(chkShuffleUI, &QCheckBox::toggled, this, &MainWindow::handleShuffleToggled);

    // insert just after the Initialise button
    if (lay) {
        int idx = lay->indexOf(initBtn);
        if (auto *box = qobject_cast<QBoxLayout*>(lay)) {
            box->insertWidget(qMax(0, idx) + 1, chkShuffleUI);
        } else if (auto *grid = qobject_cast<QGridLayout*>(lay)) {
            int r,c,rs,cs;
            int bi = grid->indexOf(initBtn);
            if (bi >= 0) {
                grid->getItemPosition(bi, &r, &c, &rs, &cs);
                grid->addWidget(chkShuffleUI, r, c+1, 1, 1);
            } else {
                grid->addWidget(chkShuffleUI, 0, 0);
            }
        } else {
            // Unknown layout type: fallback to root
            if (auto sc = qobject_cast<QScrollArea*>(centralWidget())) {
                QWidget* content = sc->widget();
                if (content && content->layout())
                    content->layout()->addWidget(chkShuffleUI);
                else
                    chkShuffleUI->show();
            } else {
                chkShuffleUI->show();
            }
        }
    } else {
        // No layout: place it visually to the right as a last resort
        chkShuffleUI->move(initBtn->x() + initBtn->width() + 8, initBtn->y());
        chkShuffleUI->show();
    }

    // Apply initial choice to engine
    if (bp) bp->setShuffleEnabled(chkShuffleUI->isChecked());
}

void MainWindow::addActivationChooserNextToReadFile()
{
    if (comboHiddenActUI) return;

    QPushButton* readBtn = ui->pushButton_Read_File;
    if (!readBtn) return;

    QWidget* parentW = readBtn->parentWidget();
    QLayout* lay     = parentW ? parentW->layout() : nullptr;

    // Create tiny label + combo
    lblHiddenActUI = new QLabel("Activation Function:", parentW ? parentW : this);
    comboHiddenActUI = new QComboBox(parentW ? parentW : this);

    // Order: 0=tanh, 1=ReLU, 2=sigmoid
    comboHiddenActUI->addItem("tanh"); // 0
    comboHiddenActUI->addItem("ReLU");        // 1
    comboHiddenActUI->addItem("sigmoid");     // 2
    comboHiddenActUI->setCurrentIndex(0);     // default = tanh

    // Wire signal
    connect(comboHiddenActUI, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::handleActivationChanged);

    
    bool placed = false;
    if (lay) {
        if (auto *box = qobject_cast<QBoxLayout*>(lay)) {
            int idx = box->indexOf(readBtn);
            box->insertWidget(std::max(0, idx)+1, lblHiddenActUI);
            box->insertWidget(std::max(0, idx)+2, comboHiddenActUI);
            placed = true;
        } else if (auto *grid = qobject_cast<QGridLayout*>(lay)) {
            int r,c,rs,cs;
            int bi = grid->indexOf(readBtn);
            if (bi >= 0) {
                grid->getItemPosition(bi, &r, &c, &rs, &cs);
                grid->addWidget(lblHiddenActUI,   r, c+1, 1, 1);
                grid->addWidget(comboHiddenActUI, r, c+2, 1, 1);
                placed = true;
            }
        }
    }
    if (!placed) {
        
        QWidget* host = this->centralWidget();
        if (auto sa = qobject_cast<QScrollArea*>(host)) host = sa->widget();
        if (host) {
            if (!host->layout()) host->setLayout(new QVBoxLayout(host));
            host->layout()->addWidget(lblHiddenActUI);
            host->layout()->addWidget(comboHiddenActUI);
            placed = true;
        }
    }

    // default to engine right away
    if (bp) {
        bp->setHiddenActivation(HACT_TANH);
    }
}

void MainWindow::handleActivationChanged(int idx)
{
    if (!bp) return;
    HiddenActivation act = HACT_TANH;
    switch (idx) {
        case 0: act = HACT_TANH;    break; // tanh (best)
        case 1: act = HACT_RELU;    break; // ReLU
        case 2: act = HACT_SIGMOID; break; // sigmoid
        default: act = HACT_TANH;   break;
    }
    bp->setHiddenActivation(act);
}



void MainWindow::handleShuffleToggled(bool checked)
{
    if (bp) bp->setShuffleEnabled(checked);
}

void MainWindow::on_pushButton_Read_File_clicked()
{
    qDebug() << "\nReading file...";

    // file picker
    const QString fileName = QFileDialog::getOpenFileName(
        this,
        "Open dataset",
        QString(),
        "Text files (*.txt);;All files (*.*)"
    );

    if (fileName.isEmpty()) {
        patternsLoadedFromFile = false;
        qDebug() << "No file chosen.";
        return;
    }

    QFile file(fileName);

    if (!file.exists()) {
        patternsLoadedFromFile = false;
        qDebug() << "Data file does not exist!";
        return;
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        patternsLoadedFromFile = false;
        qDebug() << "Failed to open file:" << fileName;
        return;
    }

    QTextStream in(&file);

    // open split output files in the working directory
    // (no extra includes needed; writes where the app runs)
    QFile foutTrain("training_16000.txt");
    QFile foutTest("test_4000.txt");
    bool doExport = foutTrain.open(QIODevice::WriteOnly | QIODevice::Text)
                 && foutTest.open(QIODevice::WriteOnly | QIODevice::Text);
    QTextStream tsTrain(&foutTrain), tsTest(&foutTest);

    char t;
    char characterSymbol;
    QString line;

    QString lineOfData;
    QString msg;
    int i = 0;
    int counter[OUTPUT_NEURONS];
    for (int k = 0; k < OUTPUT_NEURONS; k++) {
        counter[k] = 0;
    }

    while (i < NUMBER_OF_PATTERNS && !in.atEnd()) {

        // read raw (unnormalised) integers first so we can export the original values
        int raw[INPUT_NEURONS];
        // Example line: T,2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8
        in >> characterSymbol >> t
           >> raw[0]  >> t >> raw[1]  >> t
           >> raw[2]  >> t >> raw[3]  >> t
           >> raw[4]  >> t >> raw[5]  >> t
           >> raw[6]  >> t >> raw[7]  >> t
           >> raw[8]  >> t >> raw[9]  >> t
           >> raw[10] >> t >> raw[11] >> t
           >> raw[12] >> t >> raw[13] >> t
           >> raw[14] >> t >> raw[15];

        line = in.readLine(); // consume rest of line

        // export the original (unnormalised) line to train/test file
        if (doExport) {
            QTextStream &ts = (i < NUMBER_OF_TRAINING_PATTERNS) ? tsTrain : tsTest;
            ts << characterSymbol;
            for (int jj = 0; jj < INPUT_NEURONS; ++jj) ts << "," << raw[jj];
            ts << "\n";
        }

        // normalise features to [0,1] assuming 0..15
        for (int j = 0; j < INPUT_NEURONS; ++j)
            letters[i].f[j] = raw[j] / 15.0f;

        // Assume uppercase letters only
        int symbolIndex;
        if ((characterSymbol >= 'A') && (characterSymbol <= 'Z'))
        {
            symbolIndex = characterSymbol - 'A';
            if (symbolIndex >= OUTPUT_NEURONS)
                symbolIndex = OUTPUT_NEURONS - 1;
            letters[i].symbol = static_cast<Symbol>(symbolIndex);
        }
        else
        {
            symbolIndex = OUTPUT_NEURONS - 1;
            letters[i].symbol = UNKNOWN;
        }

        // Set all outputs to zero
        for (int j = 0; j < OUTPUT_NEURONS; j++)
        {
            letters[i].outputs[j] = 0;
        }

        // Set the output we want to 1
        letters[i].outputs[symbolIndex] = 1;
        counter[symbolIndex]++;

        if (i == (NUMBER_OF_PATTERNS - 1)) {
            msg.clear();
            for (int j = 0; j < OUTPUT_NEURONS; j++)
            {
                lineOfData.clear();
                if (j == OUTPUT_NEURONS - 1)
                {
                    QTextStream(&lineOfData) << "number of patterns for UNKNOWN Letters = " << counter[j] << Qt::endl;
                }
                else
                {
                    QTextStream(&lineOfData) << "number of patterns for Letter " << (j + 'A') << " = "  << counter[j] << Qt::endl;
                }
                msg.append(lineOfData);
            }
            ui->plainTextEdit_results->setPlainText(msg);
            qApp->processEvents();
        }

        i++;
    }

    msg.append("done.");
    ui->plainTextEdit_results->setPlainText(msg);
    if (doExport) {
        ui->plainTextEdit_results->appendPlainText(
            "\nExported: training_16000.txt and test_4000.txt.");
    } else {
        ui->plainTextEdit_results->appendPlainText(
            "\nNote: could not export split files (write-permission?)");
    }
    qApp->processEvents();

    file.close();
    if (foutTrain.isOpen()) foutTrain.close();
    if (foutTest.isOpen())  foutTest.close();

    patternsLoadedFromFile = true;
}


void MainWindow::on_horizScrollBar_LearningRate_valueChanged(int value)
{
    ui->lcdNumber_LearningRate->setSegmentStyle(QLCDNumber::Filled);
    ui->lcdNumber_LearningRate->display(value/1000.0);
    LEARNING_RATE = value/1000.0;
}

QString generateLetter(int index) {
    if (index >= 0 && index < 26) {
        char letter = char('A' + index);
        return QString(QChar(letter));
    } else {
        return "UNKNOWN";
    }
}

void MainWindow::on_pushButton_Classify_Test_Pattern_clicked()
{
    char characterSymbol, t;
    QString *q;

    q = new QString(ui->plainTextEdit_Input_Pattern->toPlainText());
    if (q->trimmed().isEmpty()) {
        ui->label_Classification->setText(
            "Input is empty. Paste one line like:\nZ,2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8");
        delete q;
        return;
    }

    QTextStream line(q);

    line >> characterSymbol >> t
        >> testPattern.f[0]  >> t >> testPattern.f[1]  >> t
        >> testPattern.f[2]  >> t >> testPattern.f[3]  >> t
        >> testPattern.f[4]  >> t >> testPattern.f[5]  >> t
        >> testPattern.f[6]  >> t >> testPattern.f[7]  >> t
        >> testPattern.f[8]  >> t >> testPattern.f[9]  >> t
        >> testPattern.f[10] >> t >> testPattern.f[11] >> t
        >> testPattern.f[12] >> t >> testPattern.f[13] >> t
        >> testPattern.f[14] >> t >> testPattern.f[15];

    for (int j = 0; j < INPUT_NEURONS; ++j)
        testPattern.f[j] = testPattern.f[j] / 15.0f;

    // Assume uppercase letters only
    int symbolIndex;
    if ((characterSymbol >= 'A') && (characterSymbol <= 'Z')) {
        symbolIndex = characterSymbol - 'A';
        testPattern.symbol = static_cast<Symbol>(symbolIndex);
    } else {
        symbolIndex = OUTPUT_NEURONS - 1;
        testPattern.symbol = UNKNOWN;
    }

    // One-hot target
    for (int j=0; j<OUTPUT_NEURONS; ++j) testPattern.outputs[j] = 0;
    testPattern.outputs[symbolIndex] = 1;

    const double* classificationResults = bp->testNetwork(testPattern);

    ui->lcdNumber_A->display(classificationResults[0]);
    ui->lcdNumber_B->display(classificationResults[1]);
    ui->lcdNumber_C->display(classificationResults[2]);
    ui->lcdNumber_D->display(classificationResults[3]);
    ui->lcdNumber_E->display(classificationResults[4]);
    ui->lcdNumber_F->display(classificationResults[5]);
    ui->lcdNumber_G->display(classificationResults[6]);
    ui->lcdNumber_H->display(classificationResults[7]);
    ui->lcdNumber_I->display(classificationResults[8]);
    ui->lcdNumber_J->display(classificationResults[9]);
    ui->lcdNumber_K->display(classificationResults[10]);
    ui->lcdNumber_L->display(classificationResults[11]);
    ui->lcdNumber_M->display(classificationResults[12]);
    ui->lcdNumber_N->display(classificationResults[13]);
    ui->lcdNumber_O->display(classificationResults[14]);
    ui->lcdNumber_P->display(classificationResults[15]);
    ui->lcdNumber_Q->display(classificationResults[16]);
    ui->lcdNumber_R->display(classificationResults[17]);
    ui->lcdNumber_S->display(classificationResults[18]);
    ui->lcdNumber_T->display(classificationResults[19]);
    ui->lcdNumber_U->display(classificationResults[20]);
    ui->lcdNumber_V->display(classificationResults[21]);
    ui->lcdNumber_W->display(classificationResults[22]);
    ui->lcdNumber_X->display(classificationResults[23]);
    ui->lcdNumber_Y->display(classificationResults[24]);
    ui->lcdNumber_Z->display(classificationResults[25]);

    int predIdx = predict_with_threshold(classificationResults, OUTPUT_NEURONS);
    int desiredIdx = 0;
    for (int k=0; k<OUTPUT_NEURONS; ++k) if (testPattern.outputs[k] == 1) { desiredIdx = k; break; }

    QString desiredClass = generateLetter(desiredIdx);
    QString actualOutputClass = (predIdx < 0) ? "UNKNOWN" : generateLetter(predIdx);

    ui->label_Classification->setText("Desired class = " + desiredClass +
                                      ", actual output class = " + actualOutputClass);

    delete q;
}

void MainWindow::on_pushButton_Train_Network_Max_Epochs_clicked()
{
    if(!patternsLoadedFromFile) {
        ui->plainTextEdit_results->setPlainText("\nMissing training patterns.  Load data set first.\n");
        return;
    }

    
    if (LEARNING_RATE <= 0.0) LEARNING_RATE = 0.02;
    if (LEARNING_RATE > 1.0)  LEARNING_RATE = 1.0;

    // Disable heavy controls
    ui->pushButton_Train_Network_Max_Epochs->setEnabled(false);
    ui->pushButton_Test_All_Patterns->setEnabled(false);
    ui->pushButton_Classify_Test_Pattern->setEnabled(false);

    MAX_EPOCHS = ui->spinBox_training_Epochs->value();

    // read Shuffle + prompt for Target PGC
    if (chkShuffleUI) bp->setShuffleEnabled(chkShuffleUI->isChecked());
    bool ok = true;
    double targetPGC = QInputDialog::getDouble(
        this,
        "Target Test PGC",
        "Stop training when Test PGC ≥ this value\n(enter 0 to disable early stop):",
        lastTargetPGC,   // default
        0.0, 100.0, 2,
        &ok
    );
    if (!ok) targetPGC = 0.0;  // cancel = no early stop
    lastTargetPGC = targetPGC;

    // Per-epoch CSV (MSE + PGC for train/test)
    const QString csvPath = "training_log.csv";
    QFile csv(csvPath);
    if (!csv.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, "Error", "Cannot open training_log.csv for writing.");
        ui->pushButton_Train_Network_Max_Epochs->setEnabled(true);
        ui->pushButton_Test_All_Patterns->setEnabled(true);
        ui->pushButton_Classify_Test_Pattern->setEnabled(true);
        return;
    }
    QTextStream out(&csv);
    out << "epoch,train_MSE,train_PGC,test_MSE,test_PGC\n";

    // Best checkpoint tracking
    double bestTestPGC = -1.0;
    int     bestEpoch  = -1;
    const QString bestWeightsPath =
        ui->plainTextEdit_saveWeightsAs->toPlainText().trimmed().isEmpty()
        ? QString("best_weights.txt")
        : ui->plainTextEdit_saveWeightsAs->toPlainText().trimmed();

    QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));

    double lastSSE = 0.0;
    for(int e=1; e<=MAX_EPOCHS; ++e){
        QString msg;
        msg.append("Training in progress...\n");

        // Train exactly one epoch
        lastSSE = bp->trainNetwork();  // one-epoch function
        ui->lcdNumber_SSE->display(lastSSE);

        // Divergence guard
        if (!qIsFinite(lastSSE) || lastSSE > 1e12) {
            QApplication::restoreOverrideCursor();
            ui->plainTextEdit_results->appendPlainText("\nTraining stopped: SSE diverged. Reduce LR (0.02–0.2) and re-initialise.");
            break;
        }

        // Evaluate & log metrics
        const double trMSE = bp->evaluateMSE_train();
        const double trPGC = bp->evaluatePGC_train();
        const double teMSE = bp->evaluateMSE_test();
        const double tePGC = bp->evaluatePGC_test();

        out << e << "," << trMSE << "," << trPGC << "," << teMSE << "," << tePGC << "\n";
        out.flush();

        // Save best weights
        if (tePGC > bestTestPGC) {
            bestTestPGC = tePGC;
            bestEpoch   = e;
            bp->saveWeights(bestWeightsPath);
        }

        if (targetPGC > 0.0 && tePGC >= targetPGC) {
            // Show this epoch's line first
            msg.append("\nEpoch=");
            msg.append(QString::number(e));
            msg.append(QString("   Train PGC=%1   Test PGC=%2")
                    .arg(trPGC,0,'f',3).arg(tePGC,0,'f',3));
            ui->plainTextEdit_results->setPlainText(msg);
            qApp->processEvents();

            // Append the early-stop message
            ui->plainTextEdit_results->appendPlainText(
                QString("\nTarget reached: Test PGC %1% at epoch %2. Best so far %3% @ epoch %4.")
                    .arg(tePGC,0,'f',3).arg(e).arg(bestTestPGC,0,'f',3).arg(bestEpoch));
            qApp->processEvents();

            break;
        }

        
        qApp->processEvents();
        update();

        msg.append("\nEpoch=");
        msg.append(QString::number(e));
        msg.append(QString("   Train PGC=%1   Test PGC=%2").arg(trPGC,0,'f',3).arg(tePGC,0,'f',3));
        ui->plainTextEdit_results->setPlainText(msg);
    }

    csv.close();
    QApplication::restoreOverrideCursor();

    // Final summary in UI (last-epoch weights)
    const double trPGC_final = bp->evaluatePGC_train();
    const double tePGC_final = bp->evaluatePGC_test();
    ui->plainTextEdit_results->appendPlainText(
        QString("\nDone. Train PGC=%1  Test PGC=%2  (last SSE=%3)")
        .arg(trPGC_final,0,'f',3).arg(tePGC_final,0,'f',3).arg(lastSSE,0,'f',6));

    // Report the best epoch we saw during training
    if (bestEpoch > 0) {
        ui->plainTextEdit_results->appendPlainText(
            QString("Best Test PGC during run: %1% at epoch %2 (checkpoint: %3)")
            .arg(bestTestPGC,0,'f',3).arg(bestEpoch).arg(bestWeightsPath));
    }

    // Ensure confusion matrix uses the BEST
    bp->loadWeights(bestWeightsPath);

    // Confusion matrix (TEST set)
    const QString cmPath = "confusion_matrix.csv";
    auto cm = computeConfusionMatrixOnTestSet();
    saveConfusionMatrixCSV(cm, cmPath);
    ui->plainTextEdit_results->appendPlainText("Confusion matrix written to " + cmPath);

    // Re-enable controls
    ui->pushButton_Train_Network_Max_Epochs->setEnabled(true);
    ui->pushButton_Test_All_Patterns->setEnabled(true);
    ui->pushButton_Classify_Test_Pattern->setEnabled(true);
}

void MainWindow::on_pushButton_Initialise_Network_clicked()
{
    if (!patternsLoadedFromFile) {
        ui->plainTextEdit_results->setPlainText("Load data first (use 'Read File').");
        return;
    }

    HiddenActivation act = HACT_TANH; // default to best
    if (comboHiddenActUI) {
        int idx = comboHiddenActUI->currentIndex();
        switch (idx) {
            case 0: act = HACT_TANH;    break;
            case 1: act = HACT_RELU;    break;
            case 2: act = HACT_SIGMOID; break;
        }
    }
    bp->setHiddenActivation(act);

    // keep defaults
    bp->setLayerSizes(64, 64);
    bp->setLearningRate(LEARNING_RATE);

    bp->initialise();

    const char* name =
        (act == HACT_RELU)    ? "ReLU" :
        (act == HACT_TANH)    ? "tanh" :
                                "sigmoid";
    ui->plainTextEdit_results->setPlainText(
        QString("Network initialised (%1)")
            .arg(name));
}



void MainWindow::on_pushButton_Test_All_PatternS_clicked()
{
    on_pushButton_Test_All_PatternS_clicked();
}

void MainWindow::on_pushButton_Test_All_Patterns_clicked()
{
    int correctClassifications = 0;

    for (int i = NUMBER_OF_TRAINING_PATTERNS; i < NUMBER_OF_PATTERNS; ++i)
    {
        for (int j=0; j<INPUT_NEURONS; ++j)
            testPattern.f[j] = letters[i].f[j];

        int symbolIndex = static_cast<int>(letters[i].symbol);
        if (symbolIndex < 0 || symbolIndex >= OUTPUT_NEURONS) symbolIndex = OUTPUT_NEURONS - 1;
        for (int j=0; j<OUTPUT_NEURONS; ++j) testPattern.outputs[j] = 0;
        testPattern.outputs[symbolIndex] = 1;
        testPattern.symbol = static_cast<Symbol>(symbolIndex);

        const double* classificationResults = bp->testNetwork(testPattern);
        int predIdx = predict_argmax(classificationResults, OUTPUT_NEURONS);

        if (predIdx == symbolIndex) ++correctClassifications;
    }

    QString msg;
    QTextStream(&msg) << "TEST SET: correctClassifications = " << correctClassifications << "\n";
    ui->plainTextEdit_results->setPlainText(msg);

    double pgc = (double(correctClassifications)/double(NUMBER_OF_TEST_PATTERNS))*100.0;
    ui->lcdNumber_percentageOfGoodClassification->display(pgc);
    qDebug() << "TEST SET: correctClassifications = " << correctClassifications;
    qDebug() << "pgc = " << pgc << Qt::endl;

    qApp->processEvents();
    update();
}

void MainWindow::on_pushButton_Save_Weights_clicked()
{
    bp->saveWeights(
        ui->plainTextEdit_saveWeightsAs->toPlainText().trimmed().isEmpty()
            ? QString("best_weights.txt")
            : ui->plainTextEdit_saveWeightsAs->toPlainText()
        );

    QString msg;
    QString lineOfText;

    lineOfText = "weights saved to file: " + ui->plainTextEdit_saveWeightsAs->toPlainText();

    msg.append(lineOfText);

    ui->plainTextEdit_results->setPlainText(msg);
}

void MainWindow::on_pushButton_Load_Weights_clicked()
{
    bp->loadWeights(ui->plainTextEdit_fileNameLoadWeights->toPlainText());

    QString msg;
    msg.append("weights loaded.\n");
    ui->plainTextEdit_results->setPlainText(msg);
}

void MainWindow::on_pushButton_testNetOnTrainingSet_clicked()
{
    int correctClassifications = 0;

    for (int i = 0; i < NUMBER_OF_TRAINING_PATTERNS; ++i)
    {
        for (int j=0; j<INPUT_NEURONS; ++j)
            testPattern.f[j] = letters[i].f[j];

        int symbolIndex = static_cast<int>(letters[i].symbol);
        if (symbolIndex < 0 || symbolIndex >= OUTPUT_NEURONS) symbolIndex = OUTPUT_NEURONS - 1;
        for (int j=0; j<OUTPUT_NEURONS; ++j) testPattern.outputs[j] = 0;
        testPattern.outputs[symbolIndex] = 1;
        testPattern.symbol = static_cast<Symbol>(symbolIndex);

        const double* classificationResults = bp->testNetwork(testPattern);
        int predIdx = predict_argmax(classificationResults, OUTPUT_NEURONS);

        if (predIdx == symbolIndex) ++correctClassifications;
    }

    QString msg;
    QTextStream(&msg) << "TRAINING SET: correctClassifications = " << correctClassifications << "\n";
    ui->plainTextEdit_results->setPlainText(msg);

    double pgc = (double(correctClassifications)/double(NUMBER_OF_TRAINING_PATTERNS))*100.0;
    ui->lcdNumber_percentageOfGoodClassification->display(pgc);
    qDebug() << "TRAINING SET: correctClassifications = " << correctClassifications;
    qDebug() << "pgc = " << pgc << Qt::endl;

    qApp->processEvents();
    update();
}

void MainWindow::on_horizScrollBar_LearningRate_actionTriggered(int action)
{
    if(action == 0){
        
    }
}


void MainWindow::logSSEToCSV(int epoch, double trainSSE, double testSSE)
{
    QString path = "training_log_legacy_sse.csv";
    QFile f(path);
    bool writeHeader = !f.exists();
    if (f.open(QIODevice::Append | QIODevice::Text)) {
        QTextStream out(&f);
        if (writeHeader) out << "epoch,train_sse,test_sse\n";
        out << epoch << "," << trainSSE << "," << testSSE << "\n";
        f.close();
    }
}

QVector<QList<int>> MainWindow::computeConfusionMatrixOnTestSet()
{
    QVector<QList<int>> cm(OUTPUT_NEURONS, QList<int>(OUTPUT_NEURONS, 0));

    for (int i = NUMBER_OF_TRAINING_PATTERNS; i < NUMBER_OF_PATTERNS; ++i) {
        for (int j = 0; j < INPUT_NEURONS; ++j) {
            testPattern.f[j] = letters[i].f[j];
        }

        const double* outs = bp->testNetwork(testPattern);
        int predIdx = predict_argmax(outs, OUTPUT_NEURONS);

        int actualIdx = static_cast<int>(letters[i].symbol);
        if (actualIdx < 0 || actualIdx >= OUTPUT_NEURONS) actualIdx = OUTPUT_NEURONS - 1;

        cm[actualIdx][predIdx] += 1;
    }

    return cm;
}

void MainWindow::saveConfusionMatrixCSV(const QVector<QList<int>>& cm, const QString& path)
{
    QFile f(path);
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&f);
        out << "actual\\predicted";
        for (int j = 0; j < OUTPUT_NEURONS; ++j)
            out << ",C" << j;
        out << "\n";

        for (int i = 0; i < OUTPUT_NEURONS; ++i) {
            out << "C" << i;
            for (int j = 0; j < OUTPUT_NEURONS; ++j)
                out << "," << cm[i][j];
            out << "\n";
        }
        f.close();
    }
}
