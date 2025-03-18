#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , image(std::make_unique<GraphicData>())
{
    ui->setupUi(this);

    connect(ui->blurRadioButton,        SIGNAL(clicked()), this, SLOT(do_resetBlurMode()));
    connect(ui->gaussianRadioButton,    SIGNAL(clicked()), this, SLOT(do_resetBlurMode()));
    connect(ui->medianRadioButton,      SIGNAL(clicked()), this, SLOT(do_resetBlurMode()));
    connect(ui->bilateralRadioButton,   SIGNAL(clicked()), this, SLOT(do_resetBlurMode()));


    connect(ui->basicResetButton,       SIGNAL(clicked()), this, SLOT(do_resetBasicMode()));
    connect(ui->blurResetButton,        SIGNAL(clicked()), this, SLOT(do_resetBlurMode()));
    connect(ui->colorResetButton,       SIGNAL(clicked()), this, SLOT(do_resetColorMode()));
    connect(ui->transformerResetButton, SIGNAL(clicked()), this, SLOT(do_resetTransformerMode()));

    connect(ui->actionLoad,     SIGNAL(triggered()), this, SLOT(do_loadImage()));
    connect(ui->actionSave,     SIGNAL(triggered()), this, SLOT(do_saveImage()));
    connect(ui->actionReset,    SIGNAL(triggered()), this, SLOT(do_resetAll()));

    if (cv::cuda::getCudaEnabledDeviceCount() > 0) qDebug() << "CUDA is supported!";
    else qDebug() << "CUDA is not supported or no device found.";
}

MainWindow::~MainWindow()
{
    delete ui;
}

// ------------------------------
// 窗口的基础方法
// ------------------------------

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
            image->src = imageData.clone();
            image->temp = imageData.clone();
            image->dst = imageData.clone();
            do_resetAll();
            //imageDisplay();
            qDebug() << "Selected file path:" << imagePath;
        }
        else {qDebug() << "Failed to load image.";}
    }
}

void MainWindow::imageDisplay()
{
    myImage = QImage((const unsigned char*)(image->dst.data),
                     image->dst.cols, image->dst.rows,
                     image->dst.step,
                     QImage::Format_RGB888).copy();

    // 获取 QLabel 当前尺寸
    QSize labelSize = ui->image->size();

    // 原始图片的宽高比=
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

// ------------------------------
// 图像处理方法
// ------------------------------

void GraphicData::process(BlurMode nowMode)
{
    Mat tempBasic;
    Mat tempColor;
    Mat tempBlur;

    if (brightness != 0 || contrast != 1.0)
    {
        temp.convertTo(tempBasic, -1, contrast, brightness);
    }
    else
    {
        temp.copyTo(tempBasic);
    }

    if (H != 0 || S != 1.0 || V != 1.0)
    {
        adjustHSV(tempBasic).copyTo(tempColor);
    }
    else
    {
        tempBasic.copyTo(tempColor);
    }

    if (kernel != 1)
    {
        // 确保卷积核宽度为奇数
        int kernelSize = (kernel % 2 == 1) ? kernel : kernel + 1;
        switch(nowMode)
        {
        case BlurMode::Blur:
            cv::blur(tempColor, tempBlur, Size(kernelSize, kernelSize), Point(-1, -1));
            break;
        case BlurMode::GaussianBlur:
            cv::GaussianBlur(tempColor, tempBlur, Size(kernelSize, kernelSize), 0, 0);
            break;
        case BlurMode::MedianBlur:
            cv::medianBlur(tempColor, tempBlur, kernelSize);
            break;
        case BlurMode::BilateralFilter:
            cv::bilateralFilter(tempColor, tempBlur, kernelSize, kernelSize * 2, kernelSize / 2);
            break;
        case BlurMode::None:
        default:
            tempColor.copyTo(tempBlur);
            break;
        }

    }
    else
    {tempColor.copyTo(tempBlur);}

    if (angle != 0.0 || scale != 1.0)
    {
        Mat imageRot(2, 3, CV_32FC1);
        // 计算原图片的中心点
        Point centerPoint = Point(tempBlur.cols / 2, tempBlur.rows / 2);

        imageRot = getRotationMatrix2D(centerPoint, angle, scale);

        warpAffine(tempBlur, dst, imageRot, tempBlur.size());
    }
    else{tempBlur.copyTo(dst);}
}

Mat GraphicData::adjustHSV(Mat tempBasic)
{
    // 将图像从BGR颜色空间转换为HSV颜色空间
    cv::cvtColor(tempBasic, tempBasic, cv::COLOR_RGB2HSV);

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(tempBasic, channels);

    cv::Mat hue = channels[0];
    cv::Mat saturation = channels[1];
    cv::Mat value = channels[2];

    // 调整色调（hue），注意处理溢出
    if (H != 0)
    {
        hue += H;
        // 处理溢出: HSV中Hue范围是0-180在OpenCV中
        hue.setTo(0, hue < 0);
        hue.setTo(180, hue > 180);
    }

    // 调整饱和度（saturation）和亮度（value）
    if (S != 1.0)
    {
        saturation.convertTo(saturation, -1, S, 0);
        // 确保像素值在0到255之间
        cv::threshold(saturation, saturation, 255, 255, cv::THRESH_TRUNC);
        cv::threshold(saturation, saturation, 0, 0, cv::THRESH_TOZERO);
    }
    if (V != 1.0)
    {
        value.convertTo(value, -1, V, 0);
        // 确保像素值在0到255之间
        cv::threshold(value, value, 255, 255, cv::THRESH_TRUNC);
        cv::threshold(value, value, 0, 0, cv::THRESH_TOZERO);
    }

    // 合并通道
    std::vector<cv::Mat> adjustedChannels = {hue, saturation, value};
    cv::merge(adjustedChannels, tempBasic);

    Mat tempRGB;
    cv::cvtColor(tempBasic, tempRGB, cv::COLOR_HSV2RGB);

    return tempRGB;
}

// ------------------------------
// 控件响应事件
// ------------------------------

void MainWindow::on_brightnessSlider_sliderMoved(int position)
{
    image->brightness = position;
    image->process(nowMode);
    imageDisplay();
}
void MainWindow::on_brightnessSlider_valueChanged(int value)
{
    image->brightness = value;
    image->process(nowMode);
    imageDisplay();
}
void MainWindow::on_contrastSlider_sliderMoved(int position)
{
    image->contrast = position / 33.0;
    image->process(nowMode);
    imageDisplay();
}
void MainWindow::on_contrastSlider_valueChanged(int value)
{
    image->contrast = value / 33.0;
    image->process(nowMode);
    imageDisplay();
}
void MainWindow::on_blurSlider_sliderMoved(int position)
{
    if      (ui->blurRadioButton->isChecked())      nowMode = BlurMode::Blur;
    else if (ui->gaussianRadioButton->isChecked())  nowMode = BlurMode::GaussianBlur;
    else if (ui->medianRadioButton->isChecked())    nowMode = BlurMode::MedianBlur;
    else if (ui->bilateralRadioButton->isChecked()) nowMode = BlurMode::BilateralFilter;

    image->kernel = position;
    image->process(nowMode);
    imageDisplay();
}
void MainWindow::on_blurSlider_valueChanged(int value)
{
    if      (ui->blurRadioButton->isChecked())      nowMode = BlurMode::Blur;
    else if (ui->gaussianRadioButton->isChecked())  nowMode = BlurMode::GaussianBlur;
    else if (ui->medianRadioButton->isChecked())    nowMode = BlurMode::MedianBlur;
    else if (ui->bilateralRadioButton->isChecked()) nowMode = BlurMode::BilateralFilter;

    image->kernel = value;
    image->process(nowMode);
    imageDisplay();
}
void MainWindow::on_HSlider_sliderMoved(int position)
{
    image->H = position;
    image->process(nowMode);
    imageDisplay();
}
void MainWindow::on_HSlider_valueChanged(int value)
{
    image->H = value;
    image->process(nowMode);
    imageDisplay();
}
void MainWindow::on_SSlider_sliderMoved(int position)
{
    image->S = position / 33.0;
    image->process(nowMode);
    imageDisplay();
}
void MainWindow::on_SSlider_valueChanged(int value)
{
    image->S = value / 33.0;
    image->process(nowMode);
    imageDisplay();
}
void MainWindow::on_VSlider_sliderMoved(int position)
{
    image->V = position / 33.0;
    image->process(nowMode);
    imageDisplay();
}
void MainWindow::on_VSlider_valueChanged(int value)
{
    image->V = value / 33.0;
    image->process(nowMode);
    imageDisplay();
}

void MainWindow::on_minusButton_clicked()
{
    image->scale -= 0.1;
    image->process(nowMode);
    imageDisplay();
}
void MainWindow::on_plusButton_clicked()
{
    image->scale += 0.1;
    image->process(nowMode);
    imageDisplay();
}
void MainWindow::on_leftRotButton_clicked()
{
    image->angle += 5.0;
    image->process(nowMode);
    imageDisplay();
}
void MainWindow::on_rightRotButton_clicked()
{
    image->angle -= 5.0;
    image->process(nowMode);
    imageDisplay();
}

// ------------------------------
// 重置函数
// ------------------------------

void MainWindow::do_resetAll()
{
    ui->brightnessSlider->setValue(0);
    image->brightness = 0;
    ui->contrastSlider->setValue(33);
    image->contrast = 1.0;
    ui->blurSlider->setValue(1);
    image->kernel = 1;
    ui->HSlider->setValue(0);
    image->H = 0;
    ui->SSlider->setValue(33);
    image->S = 1.0;
    ui->VSlider->setValue(33);
    image->V = 1.0;

    image->angle = 0.0;
    image->scale = 1.0;

    image->process(BlurMode::None);
    imageDisplay();
}

void MainWindow::do_resetBasicMode()
{
    ui->brightnessSlider->setValue(0);
    image->brightness = 0;
    ui->contrastSlider->setValue(33);
    image->contrast = 1.0;

    image->process(nowMode);
    imageDisplay();
}
void MainWindow::do_resetBlurMode()
{
    ui->blurSlider->setValue(1);
    image->kernel = 1;

    image->process(BlurMode::None);
    imageDisplay();
}
void MainWindow::do_resetColorMode()
{
    ui->HSlider->setValue(0);
    image->H = 0;
    ui->SSlider->setValue(33);
    image->S = 1.0;
    ui->VSlider->setValue(33);
    image->V = 1.0;

    image->process(nowMode);
    imageDisplay();
}
void MainWindow::do_resetTransformerMode()
{
    image->angle = 0.0;
    image->scale = 1.0;

    image->process(nowMode);
    imageDisplay();
}

// ------------------------------
// 图片保存方法
// ------------------------------

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
