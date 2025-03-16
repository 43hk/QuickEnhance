#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QScreen>
#include <QMessageBox>

#include "opencv2/opencv.hpp"
using namespace cv;

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

enum class Mode
{
    Basic,
    Blur,
    GaussianBlur,
    MedianBlur,
    BilateralFilter,
    None
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void resizeEvent(QResizeEvent *event);

    void basicImageProcess(int brightness, float contrast);
    void blurImageProcess(int kernel);

    void imageDisplay();

private slots:
    void do_loadImage();

    void on_contrastSlider_sliderMoved(int position);
    void on_brightnessSlider_sliderMoved(int position);
    void on_blurSlider_sliderMoved(int position);

    void writeToImage();

    void do_resetContrast();
    void do_resetBrightness();
    void do_resetBlur();

    void do_resetAll();
    void do_resetBasicMode();
    void do_resetBlurMode();

    void do_saveImage();
private:
    Ui::MainWindow *ui;

    Mode nowMode;
    bool alreadySave;

    Mat imageSrc;
    Mat imageTemp;
    Mat imageDst;
    QImage myImage;

};
#endif // MAINWINDOW_H
