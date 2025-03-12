#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    contrast = 1.0;
    brightness = 0;
    kernel = 1;

    connect(ui->contrastResetButton, SIGNAL(clicked()), this, SLOT(do_resetContrast()));
    connect(ui->brightnessResetButton, SIGNAL(clicked()), this, SLOT(do_resetBrightness()));
    connect(ui->blurResetButton, SIGNAL(clicked()), this, SLOT(do_resetBlur()));

    connect(ui->blurRadioButton, SIGNAL(clicked()), this, SLOT(do_resetBlur()));
    connect(ui->gaussianRadioButton, SIGNAL(clicked()), this, SLOT(do_resetBlur()));
    connect(ui->medianRadioButton, SIGNAL(clicked()), this, SLOT(do_resetBlur()));
    connect(ui->bilateralRadioButton, SIGNAL(clicked()), this, SLOT(do_resetBlur()));

    connect(ui->actionload, SIGNAL(triggered()), this, SLOT(do_loadImage()));
    connect(ui->actionsave, SIGNAL(triggered()), this, SLOT(do_saveImage()));

    if (cv::cuda::getCudaEnabledDeviceCount() > 0) qDebug() << "CUDA is supported!";
    else qDebug() << "CUDA is not supported or no device found.";
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::do_loadImage()
{
    QString imagePath = QFileDialog::getOpenFileName(this, tr("打开图片"), "", tr("图片文件 (*.png *.jpg *.jpeg *.bmp);;All Files (*)"));
    if (!imagePath.isEmpty())
    {
        QPixmap pixmap(imagePath);
        if (!pixmap.isNull())
        {
            ui->image->clear();

            Mat imageData = imread(imagePath.toStdString());
            if(imageData.empty())
            {
                qDebug() << "Error: Failed to load image data.";
                return;
            }

            cvtColor(imageData, imageData, COLOR_BGR2RGB);
            imageSrc = imageData.clone();

            myImage = QImage((const unsigned char*)imageData.data, imageData.cols, imageData.rows, imageData.step, QImage::Format_RGB888).copy();
            if(myImage.isNull())
            {
                qDebug() << "Failed to convert cv::Mat to QImage.";
                return;
            }

            imageDisplay();
            qDebug() << "Selected file path:" << imagePath;
        }
        else {qDebug() << "Failed to load image.";}
    }
    //确保打开新文件的时候重置参数
    do_resetContrast();
    do_resetBrightness();
    do_resetBlur();
}


void MainWindow::imageDisplay()
{
    QSize labelSize = ui->image->size();

    ui->image->setPixmap(QPixmap::fromImage(myImage).scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation));
}


void MainWindow::imageProcess()
{   
    int kernelSize = (kernel % 2 == 1) ? kernel : kernel + 1;

    imageSrc.convertTo(imageTemp, -1, contrast, brightness);


    switch(nowMode)
    {
    case Mode::Blur:
        cv::blur(imageTemp, imageDst, Size(kernelSize, kernelSize), Point(-1, -1));
        break;

    case Mode::GaussianBlur:
        cv::GaussianBlur(imageTemp, imageDst, Size(kernelSize, kernelSize), 0, 0);
        break;

    case Mode::MedianBlur:
        cv::medianBlur(imageTemp, imageDst, kernelSize);
        break;

    case Mode::BilateralFilter:
        cv::bilateralFilter(imageTemp, imageDst, kernelSize, kernelSize * 2, kernelSize / 2);
        break;

    default:
        imageDst = imageTemp.clone();
        break;
    }

    myImage = QImage((const unsigned char*)(imageDst.data),
                     imageDst.cols, imageDst.rows,
                     imageDst.step,
                     QImage::Format_RGB888).copy();

    imageDisplay();
}


void MainWindow::on_brightnessSlider_valueChanged(int value)
{
    nowMode = Mode::Basic;
    brightness = value;
    imageProcess();
}

void MainWindow::on_contrastSlider_valueChanged(int value)
{
    nowMode = Mode::Basic;
    contrast = value / 33.0;
    imageProcess();
}


void MainWindow::on_blurSlider_valueChanged(int value)
{
    if      (ui->blurRadioButton->isChecked())      nowMode = Mode::Blur;
    else if (ui->gaussianRadioButton->isChecked())  nowMode = Mode::GaussianBlur;
    else if (ui->medianRadioButton->isChecked())    nowMode = Mode::MedianBlur;
    else if (ui->bilateralRadioButton->isChecked()) nowMode = Mode::BilateralFilter;

    kernel = value;
    imageProcess();
}

void MainWindow::do_resetBrightness()
{ui->brightnessSlider->setValue(0);}
void MainWindow::do_resetContrast()
{ui->contrastSlider->setValue(33);}
void MainWindow::do_resetBlur()
{ui->blurSlider->setValue(1);}

void MainWindow::do_saveImage()
{
    QString filename = QFileDialog::getSaveFileName(this, tr("保存图片"), "", tr("图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)"));

    // 检查用户是否取消了保存操作
    if (!filename.isEmpty())
    {
        bool saved = myImage.save(filename);

        if (saved) QMessageBox::information(this, tr("保存成功"), tr("图片已成功保存"));
        else QMessageBox::warning(this, tr("保存失败"), tr("无法保存图片，请检查文件路径和权限"));
    }
}
