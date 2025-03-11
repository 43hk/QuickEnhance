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

enum class BlurMode
{
    Blur,
    GaussianBlur,
    MedianBlur,
    BilateralFilter
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void basicImageProcess(float contrast, int brightness);
    void blurImageProcess(int strength);

    void writeToImageTemp();

    BlurMode do_setBlurMode();
    void imageDisplay();

private slots:
    void on_contrastSlider_sliderMoved(int position);
    void on_contrastSlider_valueChanged(int value);
    void on_brightnessSlider_sliderMoved(int position);
    void on_brightnessSlider_valueChanged(int value);


    void on_blurSlider_sliderMoved(int position);
    void on_blurSlider_valueChanged(int value);

    void do_resetContrast();
    void do_resetBrightness();
    void do_resetBlur();

    void do_loadImage();
    void do_saveImage();
private:
    Ui::MainWindow *ui;

    Mat imageTemp;
    QImage myImage;

};
#endif // MAINWINDOW_H
