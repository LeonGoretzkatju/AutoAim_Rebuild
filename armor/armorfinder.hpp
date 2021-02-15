#pragma once

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include<opencv2/tracking.hpp>
#include<opencv2/highgui.hpp>
#include<Eigen/Dense>
#include "util.hpp"
#include "configurations.hpp"
#include "camview.hpp"
#include "dnnManager.hpp"
#include <vector>
#include "armor_finder.h"

using namespace std;
using namespace cv;
using namespace Eigen;

// class LightBlob {
// public:
//     cv::RotatedRect rect;   //灯条位置
//     double area_ratio;
//     double length;          //灯条长度
//     uint8_t blob_color;      //灯条颜色

//     LightBlob(cv::RotatedRect &r, double ratio, uint8_t color) : rect(r), area_ratio(ratio), blob_color(color) {
//         length = max(rect.size.height, rect.size.width);
//     };
//     LightBlob() = default;
// };

//typedef vector<LightBlob> LightBlobs;

// class ArmorBox{
// public:
//     typedef enum{
//         FRONT, SIDE, UNKNOWN
//     } BoxOrientation;

//     Rect2d rect;
//     LightBlobs light_blobs;
//     uint8_t box_color;
//     int id;
//     explicit ArmorBox(const cv::Rect &pos=cv::Rect2d(), const LightBlobs &blobs=LightBlobs(), uint8_t color=0, int i=0);

//    double getBlobsDistance() const {
//     if (light_blobs.size() == 2) {
//         auto &x = light_blobs[0].rect.center;
//         auto &y = light_blobs[1].rect.center;
//         return sqrt((x.x - y.x) * (x.x - y.x) + (x.y - y.y) * (x.y - y.y));
//     } else {
//         return 0;
//     }
// }
// };
// ArmorBox::ArmorBox(const cv::Rect &pos, const LightBlobs &blobs, uint8_t color, int i) :
//         rect(pos), light_blobs(blobs), box_color(color), id(i) {};

// typedef vector<ArmorBox> ArmorBoxs;

ArmorBox::ArmorBox(const cv::Rect &pos, const LightBlobs &blobs, uint8_t color, int i) :
         rect(pos), light_blobs(blobs), box_color(color), id(i) {};

cv::Point2f ArmorBox::getCenter() const {
    return cv::Point2f(
            rect.x + rect.width / 2,
            rect.y + rect.height / 2
    );
}

double ArmorBox::getBlobsDistance() const {
    if (light_blobs.size() == 2) {
        auto &x = light_blobs[0].rect.center;
        auto &y = light_blobs[1].rect.center;
        return sqrt((x.x - y.x) * (x.x - y.x) + (x.y - y.y) * (x.y - y.y));
    } else {
        return 0;
    }
}

class ArmorFinder{
public:
    ArmorFinder(CameraView* cameraview,DnnManager* dnnmanager) {
        this->cameraview=cameraview;
        this->dnnmanager=dnnmanager;
        if (ElectronicControlParams::teamInfo == 2)
        {
            this->enemy_color = (uint8_t)ARMOR_BLUE;
        }
        else
        {
            this->enemy_color = (uint8_t)ARMOR_RED;
        }
        
    }
    ~ArmorFinder()=default;

    Point2f update_frame(ImageData& frame,float dtTime){
        cout << "enter update_frame" << endl;
        double yaw,pitch;
        switch (state) {
            case SEARCHING_STATE:
                if (stateSearchingTarget(frame)) {
                    if ((target_box.rect & cv::Rect2d(0, 0, 640, 480)) == target_box.rect) { // 判断装甲板区域是否脱离图像区域
                        tracker = TrackerToUse::create();                       // 成功搜寻到装甲板，创建tracker对象
                        tracker->init(frame.image, target_box.rect);
                        state = TRACKING_STATE;
                        tracking_cnt = 0;
                    }
                }
                break;
            case TRACKING_STATE:
                if (!stateTrackingTarget(frame) || ++tracking_cnt > 100) {    // 最多追踪100帧图像
                    state = SEARCHING_STATE;
                }
                break;
             case STANDBY_STATE:
            default:
                stateStandBy(); // currently meaningless
        }

        Point2f shoot_of_angle;
        shoot_of_angle.x=yaw;
        shoot_of_angle.y=pitch;
        return shoot_of_angle;
    }

private:
    CameraView* cameraview;
    DnnManager* dnnmanager;
    typedef cv::TrackerKCF TrackerToUse;                // Tracker类型定义

    typedef enum{
        SEARCHING_STATE, TRACKING_STATE,STANDBY_STATE
    } State;                                            // 自瞄状态枚举定义

    systime frame_time;                                 // 当前帧对应时间
    uint8_t  enemy_color;                         // 敌方颜色，引用外部变量，自动变化
    //const uint8_t &is_anti_top;                         // 进入反陀螺，引用外部变量，自动变化
    State state;                                        // 自瞄状态对象实例
    ArmorBox target_box, last_box;                      // 目标装甲板
    int anti_switch_cnt;                                // 防止乱切目标计数器
    cv::Ptr<cv::Tracker> tracker;                       // tracker对象实例
    //Classifier classifier;                              // CNN分类器对象实例，用于数字识别
    int contour_area;                                   // 装甲区域亮点个数，用于数字识别未启用时判断是否跟丢（已弃用）
    int tracking_cnt;                                   // 记录追踪帧数，用于定时退出追踪
    //Serial &serial;                                     // 串口对象，引用外部变量，用于和能量机关共享同一个变量
    systime last_front_time;                            // 上次陀螺正对时间
    int anti_top_cnt;
    RoundQueue<double, 4> top_periodms;                 // 陀螺周期循环队列
    vector<systime> time_seq;                           // 一个周期内的时间采样点
    vector<float> angle_seq;                            // 一个周期内的角度采样点


    static bool angelJudge(const LightBlob &light_blob_i, const LightBlob &light_blob_j) {
    float angle_i = light_blob_i.rect.size.width > light_blob_i.rect.size.height ? light_blob_i.rect.angle :
                    light_blob_i.rect.angle - 90;
    float angle_j = light_blob_j.rect.size.width > light_blob_j.rect.size.height ? light_blob_j.rect.angle :
                    light_blob_j.rect.angle - 90;
    return abs(angle_i - angle_j) < 20;
    }
// 判断两个灯条的高度差
    static bool heightJudge(const LightBlob &light_blob_i, const LightBlob &light_blob_j) {
        cv::Point2f centers = light_blob_i.rect.center - light_blob_j.rect.center;
        return abs(centers.y) < 30;
    }

// 判断两个灯条的间距
    static bool lengthJudge(const LightBlob &light_blob_i, const LightBlob &light_blob_j) {
    double side_length;
    cv::Point2f centers = light_blob_i.rect.center - light_blob_j.rect.center;
    side_length = sqrt(centers.ddot(centers));
    return (side_length / light_blob_i.length < 10 && side_length / light_blob_i.length > 0.5);
    }
// 判断两个灯条的长度比
    static bool lengthRatioJudge(const LightBlob &light_blob_i, const LightBlob &light_blob_j) {
        return (light_blob_i.length / light_blob_j.length < 2.5
                && light_blob_i.length / light_blob_j.length > 0.4);
    }

    bool stateStandBy() {
    state = SEARCHING_STATE;
    return true;
}

/* 判断两个灯条的错位度，不知道英文是什么！！！ */
    static bool CuoWeiDuJudge(const LightBlob &light_blob_i, const LightBlob &light_blob_j) {
    float angle_i = light_blob_i.rect.size.width > light_blob_i.rect.size.height ? light_blob_i.rect.angle :
                    light_blob_i.rect.angle - 90;
    float angle_j = light_blob_j.rect.size.width > light_blob_j.rect.size.height ? light_blob_j.rect.angle :
                    light_blob_j.rect.angle - 90;
    float angle = (angle_i + angle_j) / 2.0 / 180.0 * 3.14159265459;
    if (abs(angle_i - angle_j) > 90) {
        angle += 3.14159265459 / 2;
    }
    Vector2f orientation(cos(angle), sin(angle));
    Vector2f p2p(light_blob_j.rect.center.x - light_blob_i.rect.center.x,
                 light_blob_j.rect.center.y - light_blob_i.rect.center.y);
    return abs(orientation.dot(p2p)) < 25;
    }

// 判断装甲板方向
    static bool boxAngleJudge(const LightBlob &light_blob_i, const LightBlob &light_blob_j) {
    float angle_i = light_blob_i.rect.size.width > light_blob_i.rect.size.height ? light_blob_i.rect.angle :
                    light_blob_i.rect.angle - 90;
    float angle_j = light_blob_j.rect.size.width > light_blob_j.rect.size.height ? light_blob_j.rect.angle :
                    light_blob_j.rect.angle - 90;
    float angle = (angle_i + angle_j) / 2.0;
    if (abs(angle_i - angle_j) > 90) {
        angle += 90.0;
    }
    return (-120.0 < angle && angle < -60.0) || (60.0 < angle && angle < 120.0);
    }

    bool findLightBlobs(const ImageData& frame,LightBlobs &Light_Blobs){
        cv::Mat color_channel;
        cv::Mat src_bin_light, src_bin_dim;
        std::vector<cv::Mat> channels;       // 通道拆分
        cv::split(frame.image,channels);
        if (enemy_color == ARMOR_BLUE)
        {        
            color_channel = channels[0];        /* 根据目标颜色进行通道提取 */
        } 
        else if (enemy_color == ARMOR_RED)
        {    
            color_channel = channels[2];        /************************/
        }
        int light_threshold;
        if(enemy_color == ARMOR_BLUE){
            light_threshold = 225;
        }
        else
        {
            light_threshold = 200;
        }
        cv::threshold(color_channel, src_bin_light, light_threshold, 255, CV_THRESH_BINARY); // 二值化对应通道
        if (src_bin_light.empty()) return false;
        imagePreProcess(src_bin_light);                                  // 开闭运算

        cv::threshold(color_channel, src_bin_dim, 140, 255, CV_THRESH_BINARY); // 二值化对应通道
        if (src_bin_dim.empty()) return false;
        imagePreProcess(src_bin_dim);                                  // 开闭运算
        std::vector<std::vector<cv::Point>> light_contours_light, light_contours_dim;
        LightBlobs  light_blobs_light , light_blobs_dim;
        std::vector<cv::Vec4i> hierarchy_light, hierarchy_dim;
        cv::findContours(src_bin_light, light_contours_light, hierarchy_light, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
        cv::findContours(src_bin_dim, light_contours_dim, hierarchy_dim, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
        for (int i = 0; i < light_contours_light.size(); i++) {
            if (hierarchy_light[i][2] == -1) {
                cv::RotatedRect rect = cv::minAreaRect(light_contours_light[i]);
                if (isValidLightBlob(light_contours_light[i], rect)) {
                    light_blobs_light.emplace_back(
                        rect, areaRatio(light_contours_light[i], rect), get_blob_color(frame.image, rect)
                );
                }
            }
        }
        for (int i = 0; i < light_contours_dim.size(); i++) {
            if (hierarchy_dim[i][2] == -1) {
                cv::RotatedRect rect = cv::minAreaRect(light_contours_dim[i]);
                if (isValidLightBlob(light_contours_dim[i], rect)) {
                    light_blobs_dim.emplace_back(
                            rect, areaRatio(light_contours_dim[i], rect), get_blob_color(frame.image, rect)
                    );
                }
            }
        }
        vector<int> light_to_remove, dim_to_remove;
        for (int l = 0; l != light_blobs_light.size(); l++) {
            for (int d = 0; d != light_blobs_dim.size(); d++) {
                if (isSameBlob(light_blobs_light[l], light_blobs_dim[d])) {
                    if (light_blobs_light[l].area_ratio > light_blobs_dim[d].area_ratio) {
                        dim_to_remove.emplace_back(d);
                    } else {
                        light_to_remove.emplace_back(l);
                    }
                }
            }
        }
        sort(light_to_remove.begin(), light_to_remove.end(), [](int a, int b) { return a > b; });
        sort(dim_to_remove.begin(), dim_to_remove.end(), [](int a, int b) { return a > b; });
        for (auto x : light_to_remove) {
            light_blobs_light.erase(light_blobs_light.begin() + x);
        }
        for (auto x : dim_to_remove) {
            light_blobs_dim.erase(light_blobs_dim.begin() + x);
        }
        for (const auto &light : light_blobs_light) {
            Light_Blobs.emplace_back(light);
        }
        for (const auto &dim : light_blobs_dim) {
            Light_Blobs.emplace_back(dim);
        }
        return Light_Blobs.size() >= 2;
    }

    bool findArmorBox(const ImageData& frame,ArmorBox& box){
        LightBlobs light_blobs;
        ArmorBoxs armor_boxs;
        box.rect = cv::Rect2d(0,0,0,0);
        box.id = -1;
        if(!findLightBlobs(frame,light_blobs)){
            return false;
        }
        // if(!matchArmorBoxes(frame,light_blobs,armor_boxs)){
        //     return false;
        // }
        if(state==SEARCHING_STATE){
            //imshow
        }
        ArmorDetailType armRes[20]; 
        float confidence[20];
        this->GetArmorTypes(frame.image,armRes,armor_boxs,confidence);
        box = this->Choose_Target(armor_boxs,armRes,confidence);
}

// 判断两个灯条是否可以匹配
static bool isCoupleLight(const LightBlob &light_blob_i, const LightBlob &light_blob_j, uint8_t enemy_color) {
    return light_blob_i.blob_color == enemy_color &&
           light_blob_j.blob_color == enemy_color &&
           lengthRatioJudge(light_blob_i, light_blob_j) &&
           lengthJudge(light_blob_i, light_blob_j) &&
           //           heightJudge(light_blob_i, light_blob_j) &&
           angelJudge(light_blob_i, light_blob_j) &&
           boxAngleJudge(light_blob_i, light_blob_j) &&
           CuoWeiDuJudge(light_blob_i, light_blob_j);

    }

    bool matchArmorBoxes(const ImageData& frame,const LightBlobs& light_blobs,ArmorBoxs& armor_boxs){
        armor_boxs.clear();
        Mat src = frame.image;
        for (int i = 0; i < light_blobs.size() - 1; ++i) {
            for (int j = i + 1; j < light_blobs.size(); ++j) {
                if (!isCoupleLight(light_blobs.at(i), light_blobs.at(j), enemy_color)) {
                    continue;
                }
                cv::Rect2d rect_left = light_blobs.at(static_cast<unsigned long>(i)).rect.boundingRect();
                cv::Rect2d rect_right = light_blobs.at(static_cast<unsigned long>(j)).rect.boundingRect();
                double min_x, min_y, max_x, max_y;
                min_x = fmin(rect_left.x, rect_right.x) - 4;
                max_x = fmax(rect_left.x + rect_left.width, rect_right.x + rect_right.width) + 4;
                min_y = fmin(rect_left.y, rect_right.y) - 0.5 * (rect_left.height + rect_right.height) / 2.0;
                max_y = fmax(rect_left.y + rect_left.height, rect_right.y + rect_right.height) +
                        0.5 * (rect_left.height + rect_right.height) / 2.0;
                if (min_x < 0 || max_x > src.cols || min_y < 0 || max_y > src.rows) {
                    continue;
                }
                if (state == SEARCHING_STATE && (max_y + min_y) / 2 < 120) continue;
                if ((max_x - min_x) / (max_y - min_y) < 0.8) continue;
                LightBlobs pair_blobs = {light_blobs.at(i), light_blobs.at(j)};
                armor_boxs.emplace_back(
                        cv::Rect2d(min_x, min_y, max_x - min_x, max_y - min_y),
                        pair_blobs,
                        enemy_color
                );
            }
        }
        return !armor_boxs.empty();
    }

    ArmorBox Choose_Target(ArmorBoxs armor_boxs,ArmorDetailType type[],float confidence[])
    {
        ArmorBox TargetBox;
        float typeScore = 0;
        int bestScore = 0;
        int id = -1;
        FOREACH(i,armor_boxs.size()){
            switch (armor_boxs[i].id)
            {
                case ARMOR_INFAN: typeScore = 250; break;
                case ARMOR_HERO: typeScore = 200; break;
                case ARMOR_ENGIN: typeScore = 50;break;
            }
            typeScore *= confidence[i];
            if(typeScore > bestScore || id == -1)
            {
                id = armor_boxs[i].id;
                bestScore = typeScore;
                TargetBox = armor_boxs[i];
            }
        }
        return TargetBox;
    }



    void GetArmorTypes(Mat src,ArmorDetailType res[],ArmorBoxs armor_boxs, float confidence[])
    {
        vector<Mat> imgs;
        FOREACH(i,armor_boxs.size())
        {
            // initialize res and confidence
            res[i] = ARMOR_TYPE_UNKNOWN; //armor_type 
            confidence[i] = 0;           //confidence
            Mat resized;                 //resize img to 28*28
            resized =src(armor_boxs[i].rect).clone();
            cv::resize(resized,resized,Size(28,28));
            imgs.push_back(resized);     //push to vector
        }
        dnnmanager->ClassifyArmors(imgs,res,confidence);
        // if(DEBUG_MODE)
        // {
        //     FOREACH(i,imgs.size())
        //     cout<<"armor_id:"<<res[i]<<"    armor_confidence:"<<confidence[i]<<endl;
        // }
        FOREACH(i,imgs.size())
        {
            switch(res[i])
            {
            case 3:case 4:case 5: res[i] = ARMOR_INFAN; break;
            case 1:res[i] = ARMOR_HERO; break;
            case 2:res[i] = ARMOR_ENGIN; break;
            default: res[i] = ARMOR_TYPE_UNKNOWN; break;
            armor_boxs[i].id = res[i];
            }
        }
    }

    bool stateSearchingTarget(ImageData& frame){
        cout << "enter stateSearchingTarget" << endl;
        if (findArmorBox(frame, target_box)) { // 在原图中寻找目标，并返回是否找到
        if (last_box.rect != cv::Rect2d() &&
            (getPointLength(last_box.getCenter() - target_box.getCenter()) > last_box.rect.height * 2.0) &&
            anti_switch_cnt++ < 3) { // 判断当前目标和上次有效目标是否为同一个目标
            target_box = ArmorBox(); // 并给３帧的时间，试图找到相同目标
            return false;            // 可以一定程度避免频繁多目标切换
        } else {
            anti_switch_cnt = 0;
            return true;
        }
        } else {
            target_box = ArmorBox();
            anti_switch_cnt++;
            return false;
        } 
    }
    bool stateTrackingTarget(ImageData& frame) {
        cout << "enter stateTrackingTarget" << endl;
        auto pos = target_box.rect;
        if(!tracker->update(frame.image, pos)){ // 使用KCFTracker进行追踪
            target_box = ArmorBox();
            return false;
        }
        if((pos & cv::Rect2d(0, 0, 640, 480)) != pos){//out of image
            target_box = ArmorBox();
            return false;
        }

        // 获取相较于追踪区域两倍长款的区域，用于重新搜索，获取灯条信息
        cv::Rect2d bigger_rect;
        bigger_rect.x = pos.x - pos.width / 2.0;
        bigger_rect.y = pos.y - pos.height / 2.0;
        bigger_rect.height = pos.height * 2;
        bigger_rect.width  = pos.width * 2;
        bigger_rect &= cv::Rect2d(0, 0, 640, 480);
        cv::Mat interesting = frame.image(bigger_rect).clone();

        ImageData roi;
        frame.copyTo(roi);
        roi.image=interesting;

        ArmorBox box;
        // 在区域内重新搜索。
        if(findArmorBox(roi, box)) { // 如果成功获取目标，则利用搜索区域重新更新追踪器
            target_box = box;
            target_box.rect.x += bigger_rect.x; //　添加roi偏移量
            target_box.rect.y += bigger_rect.y;
            for(auto &blob : target_box.light_blobs){
                blob.rect.center.x += bigger_rect.x;
                blob.rect.center.y += bigger_rect.y;
            }
            tracker = TrackerToUse::create();
            tracker->init(frame.image, target_box.rect);
        }else{    // 如果没有成功搜索目标，则使用判断是否跟丢。
            roi.image = frame.image(pos).clone();
            if(/*classifier*/1){ // 分类器可用，使用分类器判断。
                cv::resize(roi.image, roi.image, cv::Size(48, 36));
                // if(classifier(roi) == 0){
                //     target_box = ArmorBox();
                //     return false;
                // }
            }else{ //　分类器不可用，使用常规方法判断
            // ??????maybe useless  nonsense!!!
                cv::Mat roi_gray;
                cv::cvtColor(roi.image, roi_gray, CV_RGB2GRAY);
                cv::threshold(roi_gray, roi_gray, 180, 255, cv::THRESH_BINARY);
                contour_area = cv::countNonZero(roi_gray);
                if(abs(cv::countNonZero(roi_gray) - contour_area) > contour_area * 0.3){
                    target_box = ArmorBox();
                    return false;
                }
            }
            target_box.rect = pos;
            target_box.light_blobs.clear();
        }
        return true;
    }
    static void imagePreProcess(cv::Mat &src) {
    static cv::Mat kernel_erode = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5));
    erode(src, src, kernel_erode);

    static cv::Mat kernel_dilate = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5));
    dilate(src, src, kernel_dilate);

    static cv::Mat kernel_dilate2 = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5));
    dilate(src, src, kernel_dilate2);

    static cv::Mat kernel_erode2 = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5));
    erode(src, src, kernel_erode2);
    }
    static bool isValidLightBlob(const std::vector<cv::Point> &contour, const cv::RotatedRect &rect) {
    return (1.2 < lw_rate(rect) && lw_rate(rect) < 10) &&
           //           (rect.size.area() < 3000) &&
           ((rect.size.area() < 50 && areaRatio(contour, rect) > 0.4) ||
            (rect.size.area() >= 50 && areaRatio(contour, rect) > 0.6));
    }
    static double areaRatio(const std::vector<cv::Point> &contour, const cv::RotatedRect &rect) {
    return cv::contourArea(contour) / rect.size.area();
    }

    static double lw_rate(const cv::RotatedRect &rect) {
    return rect.size.height > rect.size.width ?
           rect.size.height / rect.size.width :
           rect.size.width / rect.size.height;
    }

    static bool isSameBlob(LightBlob blob1, LightBlob blob2) {
    auto dist = blob1.rect.center - blob2.rect.center;
    return (dist.x * dist.x + dist.y * dist.y) < 9;
    }

    static uint8_t get_blob_color(const cv::Mat &src, const cv::RotatedRect &blobPos) {
    auto region = blobPos.boundingRect();
    region.x -= fmax(3, region.width * 0.1);
    region.y -= fmax(3, region.height * 0.05);
    region.width += 2 * fmax(3, region.width * 0.1);
    region.height += 2 * fmax(3, region.height * 0.05);
    region &= cv::Rect(0, 0, src.cols, src.rows);
    cv::Mat roi = src(region);
    int red_cnt = 0, blue_cnt = 0;
    for (int row = 0; row < roi.rows; row++) {
        for (int col = 0; col < roi.cols; col++) {
            red_cnt += roi.at<cv::Vec3b>(row, col)[2];
            blue_cnt += roi.at<cv::Vec3b>(row, col)[0];
        }
    }
    if (red_cnt > blue_cnt) {
        return BLOB_RED;
    } else {
        return BLOB_BLUE;
    }
    }
};