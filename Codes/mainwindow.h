#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <QDataStream>
#include <QComboBox>
#include <QCheckBox>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QPushButton>
#include <QScrollArea>
#include <QLabel>
#include "backpropagation.h"

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_pushButton_Read_File_clicked();
    void on_horizScrollBar_LearningRate_valueChanged(int value);
    void on_pushButton_Classify_Test_Pattern_clicked();
    void on_pushButton_Train_Network_Max_Epochs_clicked();
    void on_pushButton_Initialise_Network_clicked();
    void on_pushButton_Test_All_PatternS_clicked();
    void on_pushButton_Test_All_Patterns_clicked();
    void on_pushButton_Save_Weights_clicked();
    void on_pushButton_Load_Weights_clicked();
    void on_pushButton_testNetOnTrainingSet_clicked();
    void on_horizScrollBar_LearningRate_actionTriggered(int action);
   
    void handleActivationChanged(int idx);
    void handleShuffleToggled(bool checked);

private:
    Ui::MainWindow *ui;
    Backpropagation *bp;

    QCheckBox* chkShuffleUI = nullptr;   // checkbox
    double     lastTargetPGC = 0.0;      // remember last target PGC from dialog
    void addShuffleCheckboxNextToInit(); // creates & inserts the checkbox

    QComboBox* comboHiddenActUI = nullptr;
    QLabel*    lblHiddenActUI   = nullptr;

    void addActivationChooserNextToReadFile(); // creates & inserts the activation function chooser


public:
    void logSSEToCSV(int epoch, double trainSSE, double testSSE);
    void saveConfusionMatrixCSV(const QVector<QList<int>>& cm, const QString& path);
    QVector<QList<int>> computeConfusionMatrixOnTestSet();
};

#endif // MAINWINDOW_H
