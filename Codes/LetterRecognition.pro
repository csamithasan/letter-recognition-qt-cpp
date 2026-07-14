#-------------------------------------------------
#
# Project created by QtCreator
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = LetterRecognition
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    backpropagation.cpp

HEADERS  += mainwindow.h \
    globalVariables.h \
    backpropagation.h

FORMS    += mainwindow.ui
