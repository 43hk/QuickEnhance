#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->contrastResetButton, SIGNAL(clicked()), this, SLOT(do_resetContrast()));
    connect(ui->brightnessResetButton, SIGNAL(clicked()), this, SLOT(do_resetBrightness()));
    connect(ui->blurResetButton, SIGNAL(clicked()), this, SLOT(do_resetBlur()));

    connect(ui->actionload, SIGNAL(triggered()), this, SLOT(do_loadImage()));
    connect(ui->actionsave, SIGNAL(triggered()), this, SLOT(do_saveImage()));

    connect(ui->tabWidget, SIGNAL(currentChanged(int index)), this, SLOT(writeToImageTemp()));

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
            imageTemp = imageData; // 确保数据连续

            myImage = QImage((const unsigned char*)imageTemp.data, imageTemp.cols, imageTemp.rows, imageTemp.step, QImage::Format_RGB888).copy();
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


void MainWindow::basicImageProcess(float contrast, int brightness)
{
    // 重要！！！创建了一个独立于 cv::Mat 数据的 QImage 副本
    //避免了原始数据被释放或修改时影响到 QImage
    Mat imageSrc = imageTemp;
    //Mat imageSrc = imageTemp.clone();
    Mat imageDst;
    imageSrc.convertTo(imageDst, -1, contrast, brightness);

    myImage = QImage((const unsigned char*)(imageDst.data),
                     imageDst.cols, imageDst.rows,
                     imageDst.step,
                     QImage::Format_RGB888).copy();

    imageDisplay();
}


void MainWindow::blurImageProcess(int strength)
{
    if (imageTemp.empty())
    {
        qDebug() << "No Image." ;
        return;
    }
    Mat imageSrc = imageTemp;
    Mat imageDst;

    int kernelSize = (strength % 2 == 1) ? strength : strength + 1;


    switch(do_setBlurMode())
    {
        case BlurMode::Blur:
            cv::blur(imageSrc, imageDst, Size(kernelSize, kernelSize), Point(-1, -1));
            break;

        case BlurMode::GaussianBlur:
            cv::GaussianBlur(imageSrc, imageDst, Size(kernelSize, kernelSize), 0, 0);
            break;

        case BlurMode::MedianBlur:
            cv::medianBlur(imageSrc, imageDst, kernelSize);
            break;

        case BlurMode::BilateralFilter:
            cv::bilateralFilter(imageSrc, imageDst, kernelSize, kernelSize * 2, kernelSize / 2);
            break;

        default:
            break;
    }

    myImage = QImage((const unsigned char*)(imageDst.data),
                     imageDst.cols, imageDst.rows,
                     imageDst.step, QImage::Format_RGB888).copy();

    imageDisplay();
}


void MainWindow::writeToImageTemp()
{
}


void MainWindow::on_contrastSlider_sliderMoved(int position)
{
    basicImageProcess(position / 33.0, 0);
}

void MainWindow::on_contrastSlider_valueChanged(int value)
{
    basicImageProcess(value / 33.0, 0);
}
void MainWindow::on_brightnessSlider_sliderMoved(int position)
{
    basicImageProcess(1.0, position);

}
void MainWindow::on_brightnessSlider_valueChanged(int value)
{
    basicImageProcess(1.0, value);
}


BlurMode MainWindow::do_setBlurMode()
{
    if (ui->blurRadioButton->isChecked()) return BlurMode::Blur;
    else if (ui->gaussianRadioButton->isChecked()) return BlurMode::GaussianBlur;
    else if (ui->medianRadioButton->isChecked()) return BlurMode::MedianBlur;
    else if (ui->bilateralRadioButton->isChecked()) return BlurMode::BilateralFilter;
    else return BlurMode::Blur;
}


void MainWindow::on_blurSlider_sliderMoved(int position)
{
    blurImageProcess(position);
}
void MainWindow::on_blurSlider_valueChanged(int value)
{
    blurImageProcess(value);
}

void MainWindow::do_resetBrightness()
{
    ui->brightnessSlider->setValue(0);
}
void MainWindow::do_resetContrast()
{
    ui->contrastSlider->setValue(33);
}
void MainWindow::do_resetBlur()
{
    ui->blurSlider->setValue(1);
}

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
