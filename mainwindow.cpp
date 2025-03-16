#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    alreadySave = false;

    connect(ui->basicConfirmButton, SIGNAL(clicked()), this, SLOT(writeToImage()));
    connect(ui->blurConfirmButton, SIGNAL(clicked()), this, SLOT(writeToImage()));

    connect(ui->blurRadioButton, SIGNAL(clicked()), this, SLOT(do_resetBlurMode()));
    connect(ui->gaussianRadioButton, SIGNAL(clicked()), this, SLOT(do_resetBlurMode()));
    connect(ui->medianRadioButton, SIGNAL(clicked()), this, SLOT(do_resetBlurMode()));
    connect(ui->bilateralRadioButton, SIGNAL(clicked()), this, SLOT(do_resetBlurMode()));

    connect(ui->Tab, SIGNAL(currentChanged(int)), this, SLOT(on_Tab_Changed(int)));

    connect(ui->basicResetButton, SIGNAL(clicked()), this, SLOT(do_resetBasicMode()));
    connect(ui->blurResetButton, SIGNAL(clicked()), this, SLOT(do_resetBasicMode()));

    connect(ui->actionLoad, SIGNAL(triggered()), this, SLOT(do_loadImage()));
    connect(ui->actionSave, SIGNAL(triggered()), this, SLOT(do_saveImage()));
    connect(ui->actionReset, SIGNAL(triggered()), this, SLOT(do_resetAll()));

    if (cv::cuda::getCudaEnabledDeviceCount() > 0) qDebug() << "CUDA is supported!";
    else qDebug() << "CUDA is not supported or no device found.";
}

MainWindow::~MainWindow()
{
    delete ui;
}

// 重载窗口缩放的事件函数
void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
    imageDisplay(); // 窗口大小变化时更新图片显示
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
            imageTemp = imageSrc.clone();

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
    // 获取 QLabel 当前尺寸
    QSize labelSize = ui->image->size();

    // 原始图片的宽高比
    qreal imageAspectRatio = myImage.width() / (qreal)myImage.height();

    // 计算缩放后的尺寸，保持宽高比
    QSize scaledSize;
    if (imageAspectRatio > labelSize.width() / (qreal)labelSize.height())
    {
        // 如果图片更“扁”，以宽度为基准缩放
        scaledSize.setWidth(labelSize.width());
        scaledSize.setHeight(labelSize.width() / imageAspectRatio);
    }
    else
    {
        // 图片更“窄”，以高度为基准缩放
        scaledSize.setHeight(labelSize.height());
        scaledSize.setWidth(labelSize.height() * imageAspectRatio);
    }

    // 确保缩放后的尺寸不超过 QLabel 的大小
    scaledSize = scaledSize.boundedTo(labelSize);

    QPixmap scaledPixmap = QPixmap::fromImage(myImage).scaled(scaledSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    ui->image->setPixmap(scaledPixmap);
}


void MainWindow::basicImageProcess(int brightness, float contrast)
{
    imageTemp.convertTo(imageDst, -1, contrast, brightness);

    myImage = QImage((const unsigned char*)(imageDst.data),
                     imageDst.cols, imageDst.rows,
                     imageDst.step,
                     QImage::Format_RGB888).copy();

    imageDisplay();
}

void MainWindow::blurImageProcess(int kernel)
{
    int kernelSize = (kernel % 2 == 1) ? kernel : kernel + 1;

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
        break;
    }

    myImage = QImage((const unsigned char*)(imageDst.data),
                     imageDst.cols, imageDst.rows,
                     imageDst.step,
                     QImage::Format_RGB888).copy();

    imageDisplay();
}

void MainWindow::on_brightnessSlider_sliderMoved(int position)
{
    if (!alreadySave) do_resetBlurMode();
    nowMode = Mode::Basic;
    basicImageProcess(position, 1.0);
    alreadySave = false;
}
void MainWindow::on_contrastSlider_sliderMoved(int position)
{
    if (!alreadySave) do_resetBlurMode();
    nowMode = Mode::Basic;
    basicImageProcess(0, position / 33.3);
    alreadySave = false;
}
void MainWindow::on_blurSlider_sliderMoved(int position)
{
    if (!alreadySave) do_resetBasicMode();

    if      (ui->blurRadioButton->isChecked())      nowMode = Mode::Blur;
    else if (ui->gaussianRadioButton->isChecked())  nowMode = Mode::GaussianBlur;
    else if (ui->medianRadioButton->isChecked())    nowMode = Mode::MedianBlur;
    else if (ui->bilateralRadioButton->isChecked()) nowMode = Mode::BilateralFilter;

    blurImageProcess(position);
    alreadySave = false;
}


void MainWindow::writeToImage()
{
    alreadySave = true;

    switch(nowMode)
    {
    case Mode::Basic:
        do_resetBrightness();
        do_resetContrast();
        imageDst.copyTo(imageTemp);
        break;
    case Mode::Blur:
        ui->blurRadioButton->setChecked(false);
        do_resetBlur();
        imageDst.copyTo(imageTemp);
        break;
    case Mode::GaussianBlur:
        ui->gaussianRadioButton->setChecked(false);
        do_resetBlur();
        imageDst.copyTo(imageTemp);
        break;
    case Mode::MedianBlur:
        ui->medianRadioButton->setChecked(false);
        do_resetBlur();
        imageDst.copyTo(imageTemp);
        break;
    case Mode::BilateralFilter:
        ui->bilateralRadioButton->setChecked(false);
        do_resetBlur();
        imageDst.copyTo(imageTemp);
        break;
    default:
        break;
    }
}


void MainWindow::do_resetBrightness()
{ui->brightnessSlider->setValue(0);}
void MainWindow::do_resetContrast()
{ui->contrastSlider->setValue(33);}
void MainWindow::do_resetBlur()
{ui->blurSlider->setValue(1);}

void MainWindow::do_resetAll()
{
    do_resetBrightness();
    do_resetContrast();
    do_resetBlur();
    imageDst = imageSrc.clone();
    myImage = QImage((const unsigned char*)(imageDst.data),
                     imageDst.cols, imageDst.rows,
                     imageDst.step,
                     QImage::Format_RGB888).copy();
    imageDisplay();
}

void MainWindow::do_resetBasicMode()
{
    do_resetBrightness();
    do_resetContrast();
    imageDst = imageTemp.clone();
    myImage = QImage((const unsigned char*)(imageDst.data),
                     imageDst.cols, imageDst.rows,
                     imageDst.step,
                     QImage::Format_RGB888).copy();
    imageDisplay();
}

void MainWindow::do_resetBlurMode()
{
    do_resetBlur();
    imageDst = imageTemp.clone();
    myImage = QImage((const unsigned char*)(imageDst.data),
                     imageDst.cols, imageDst.rows,
                     imageDst.step,
                     QImage::Format_RGB888).copy();
    imageDisplay();
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
