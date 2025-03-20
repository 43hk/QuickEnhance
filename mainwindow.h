#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QScreen>
#include <QMessageBox>

#include <memory>

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
    BilateralFilter,
    None
};
class MainWindow;
class GraphicData
{
public:
    GraphicData(MainWindow* windowPtr) :
        window(windowPtr),
        brightness(0),
        contrast(1.0f),
        kernel(1),
        sharpenAmount(0.0f),
        H(0),
        S(1.0f),
        V(1.0f),
        angle(0.0f),
        scale(1.0f){}
    ~GraphicData(){}

    MainWindow *window;

    void basicProcess();
    void colorProcess();
    void blurProcess();
    void sharpenProcess();
    void transfromerProcess();
    void process();


    int   brightness;
    float contrast;
    int   kernel;
    float sharpenAmount;
    int   H;
    float S;
    float V;
    float angle;
    float scale;

    Mat src;

    Mat tempBasic;
    Mat tempColor;
    Mat tempEqual;
    Mat tempHSV;
    Mat tempBlur;
    Mat tempSharpend;

    Mat dst;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void resizeEvent(QResizeEvent *event);
    BlurMode getBlurMode();
    bool isEqualizeHist();
    void imageDisplay();

private slots:
    void do_loadImage();

    void on_contrastSlider_sliderMoved(int position);
    void on_contrastSlider_valueChanged(int value);
    void on_brightnessSlider_sliderMoved(int position);
    void on_brightnessSlider_valueChanged(int value);

    void on_blurSlider_sliderMoved(int position);
    void on_blurSlider_valueChanged(int value);
    void on_sharpenSlider_sliderMoved(int position);
    void on_sharpenSlider_valueChanged(int value);

    void on_HSlider_sliderMoved(int position);
    void on_HSlider_valueChanged(int value);
    void on_SSlider_sliderMoved(int position);
    void on_SSlider_valueChanged(int value);
    void on_VSlider_sliderMoved(int position);
    void on_VSlider_valueChanged(int value);

    void on_minusButton_clicked();
    void on_plusButton_clicked();
    void on_leftRotButton_clicked();
    void on_rightRotButton_clicked();


    void do_resetAll();
    void do_resetBasicMode();
    void do_resetBlurMode();
    void do_resetColorMode();
    void do_resetTransformerMode();

    void do_saveImage();
private:
    Ui::MainWindow *ui;
    std::unique_ptr<GraphicData> image;

    QString originalImagePath;

    BlurMode nowMode = BlurMode::None;

    QImage myImage;

};
#endif // MAINWINDOW_H
