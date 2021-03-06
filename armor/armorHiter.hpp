/*
 * 这是 TJU Robomasters 上位机源码，未经管理层允许严禁传播给其他人（包括队内以及队外）
 *
 * armorHiter类控制机器人进行打击操作
 */


#pragma once

#include "armorfinder.hpp"
#include "serial.hpp"
#include "util.hpp"
#include "configurations.hpp"
#include "MillHiter.hpp"
#include "shoothome.hpp"

using namespace cv;
using namespace std;
using namespace millhiter;

extern bool flag ;//anti_top
// 模块ID为2
//yong de dou shi pai sheng lei
class ArmorHiter : public ModuleBase{

public:
    ArmorHiter(SerialManager* serial,ArmorFinder* armor_finder):ModuleBase(2)
    {
        this->serial = serial;
        this->armor_finder = armor_finder;
    }
protected:
    ArmorFinder* armor_finder;
    SerialManager *serial;
    MillHiter* mill_hiter;
    ShootHome* shoot_home;
    
};

// // 适用于哨兵的打击程序
// class SentinelArmorHiter : public ArmorHiter
// {
// public:

//     double PatrolAimLowAngle = -30;
//     double PatrolAimHighAngle = -10;
//     double PatrolPitchSpeed = 50;
//     double PatrolMaxPosition = 10;
//     double PatrolMinPosition = -10;
//     double PatrolAimUpdateTime = 0.5;
//     double PatrolAimDeltaAngle = 40;
//     double TrackMaxMotionSpeed = 0.5;
//     double TrackMinMotionSpeed = 0.2;
//     double TargetLostTimeLimit = 0.5;
//     double ShootAngleThresh = 1;

//     SentinelArmorHiter(SerialManager *_serial,ArmorTrackerBase *_armor_tracker,ArmorTrackerBase*_anti_top_ptr) : ArmorHiter(_serial,_armor_tracker,_anti_top_ptr)
//     {
//         SET_CONFIG_DOUBLE_VARIABLE(PatrolAimLowAngle,-30);
//         SET_CONFIG_DOUBLE_VARIABLE(PatrolAimHighAngle,-10);
//         SET_CONFIG_DOUBLE_VARIABLE(PatrolPitchSpeed,50);
//         SET_CONFIG_DOUBLE_VARIABLE(PatrolMaxPosition,10);
//         SET_CONFIG_DOUBLE_VARIABLE(PatrolMinPosition,-10);
//         SET_CONFIG_DOUBLE_VARIABLE(PatrolAimUpdateTime,0.5);
//         SET_CONFIG_DOUBLE_VARIABLE(PatrolAimDeltaAngle,40);
//         SET_CONFIG_DOUBLE_VARIABLE(TrackMaxMotionSpeed,0.5);
//         SET_CONFIG_DOUBLE_VARIABLE(TrackMinMotionSpeed,0.2);
//         SET_CONFIG_DOUBLE_VARIABLE(TargetLostTimeLimit,0.5);
//         SET_CONFIG_DOUBLE_VARIABLE(ShootAngleThresh,1);

//         patrolPitchAngle = PatrolAimLowAngle;
//     }

//     void Update(ImageData &frame,float dtTime)
//     {
//         patrolMotionPosition = frame.worldPosition.x;
//         Point2f trackResult = armor_tracker->UpdateFrame(frame,dtTime);
//         bool foundTarget = armor_tracker->trackState;
//         if (controlState == 0)
//         {
//             if (foundTarget)
//                 controlState = 1;
//             else
//             {
//                 // 积分得到一个大致的巡逻位置
//                 //patrolMotionPosition += patrolMotionDirection * dtTime;
//                 // 到达巡逻的边界位置则改变巡逻方向
//                 if ((patrolMotionDirection > 0 && patrolMotionPosition > PatrolMaxPosition) ||
//                     (patrolMotionDirection < 0 && patrolMotionPosition < PatrolMinPosition))
//                     patrolMotionDirection = -patrolMotionDirection;
//                 // 更新云台扫描方向
//                 aimCurtime += dtTime;
//                 if (aimCurtime >= PatrolAimUpdateTime)
//                 {
//                     aimCurtime = 0;
//                     if ((patrolAimDirection > 0 && patrolAimPosition + PatrolAimDeltaAngle > 180) ||
//                         (patrolAimDirection < 0 && patrolAimPosition - PatrolAimDeltaAngle < -180))
//                         patrolAimDirection *= -1;
//                     patrolAimPosition += PatrolAimDeltaAngle;
//                 }
//                 // 更新云台俯仰
//                 patrolPitchAngle += dtTime * PatrolPitchSpeed;
//                 //float pitch = PingPong(patrolPitchAngle,PatrolAimLowAngle,PatrolAimHighAngle);
//                 // 相关控制信息发送给电控
//                 serial->SendFiringOrder(false,false);
//                 serial->SendMotionControl(patrolMotionDirection,0,0);
//                 serial->SendPTZAbsoluteAngle(patrolAimPosition,patrolPitchAngle);
//             }
//         }
//         else if (controlState == 2)
//         {
//             if (foundTarget)
//                 controlState = 1;
//             else
//             {
//                 serial->SendFiringOrder(false,true);
//                 serial->SendMotionControl(0,0,0);
//                 // count time 
//                 targetLostTime += dtTime;
//                 if (targetLostTime > TargetLostTimeLimit)
//                 {
//                     controlState = 0;
//                     // 设置巡逻参数
//                     float t = 0, bst = 0 ,opr = frame.ptzAngle.x > 0 ? -1 : 1;
//                     patrolAimPosition = int(frame.ptzAngle.x / PatrolAimDeltaAngle) * PatrolAimDeltaAngle;
//                     aimCurtime = 0;
//                     patrolPitchAngle = frame.ptzAngle.y;
//                 }
//             }
//         }
//         // 跟踪打击
//         if (controlState == 1)
//         {
//             if (!foundTarget)
//             {
//                 controlState = 2;
//                 targetLostTime = 0;
//                 return ;
//             }
//             // 寻找最佳的射击位置
//             Point2f tgtPTZAngle = armor_tracker->ptzOffAngle + frame.ptzAngle;
//             float yawoff = abs(LoopTo(tgtPTZAngle.x,-180,180)),movement = 0;
//             if (yawoff < 85) // move right
//                 movement = CalcMotionSpeed(90 - yawoff);
//             else if (yawoff > 95)
//                 movement = -CalcMotionSpeed(yawoff - 90);
//             else movement = patrolMotionDirection * TrackMinMotionSpeed;

//             patrolMotionDirection = movement > 0 ? 1 : -1;
//             // 发送消息
//             serial->SendFiringOrder(Length(armor_tracker->shootOffAngle) < ShootAngleThresh ,true);
//             // outof bounds
//             if ((movement > 0 && patrolMotionPosition > PatrolMaxPosition) ||
//                     (movement < 0 && patrolMotionPosition < PatrolMinPosition))
//                 serial ->SendMotionControl(0,0,0);
//             else
//                 serial->SendMotionControl(movement,0,0);

//             float curdistance;
//             curdistance=armor_tracker->lastArmorDistance ;
//             Point2f Corction=corection(armor_tracker,&curdistance);

//             serial->SendPTZAbsoluteAngle(trackResult.x + frame.ptzAngle.x+Corction.x,trackResult.y + frame.ptzAngle.y+Corction.y);
//             serial->SendRecommandedFireSpeed(20);
//             // 计算位置
//             //patrolMotionPosition += movement * dtTime;
//         }
//     }
// protected:
//     // 控制状态：0表示巡逻  1表示目标跟踪 2表示丢失等待中
//     int controlState = 0;
//     // patrolMotionDirection 巡逻的移动方向 -1向左 1向右
//     // patrolAimDirection 云台扫描方向 1顺时针 -1逆时针
//     int patrolMotionDirection = -1, patrolAimDirection = 1;
//     float patrolMotionPosition = 0,patrolAimPosition = 0,aimCurtime = 0,patrolPitchAngle = 0;
//     float targetLostTime = 0;
   
//     Point2f corection(ArmorTrackerBase* armor,float* curdistance){
//         double yaw=armor->ptzOffAngle.x;
//         float distance = *(curdistance);
//         double detla_x=distance*sin(yaw);
//         double detla_y=distance*cos(yaw);                                                                                                 
//         Point2f result;
//         result.y=0;
//         result.x=(2*TrackMinMotionSpeed*detla_x-1+sqrt(((2*TrackMinMotionSpeed*detla_x-1)*(2*TrackMinMotionSpeed*detla_x-1)-4*TrackMinMotionSpeed*TrackMinMotionSpeed*(detla_x*detla_x+detla_y*detla_y-400))))/(2*TrackMinMotionSpeed);
//         return result;
//     }

//     /*float PingPong(float &val,float min,float max)
//     {
//         float l = max - min,l2 = l*2;
//         float t = val - l;
//         while(1){
//             if (t < 0) t+=l2;
//             else if(t > l2) t-=l2;
//             else break;
//         }
//         val = l + t;
//         return t < l ? t + min : max + t - l2;
//     }*/

//     float LoopTo(float val,float min,float max)
//     {
//         float l = max - min;
//         float t = val - min;
//         while(1)
//         {
//             if (t < 0) t+= l;
//             else if (t > l) t-=l;
//             else break;
//         }
//         return t + min;
//     }

//     inline float CalcMotionSpeed(float off)
//     {
//         float rt = off < 45 ? off / 45.0f : 1;
//         return TrackMinMotionSpeed + (TrackMaxMotionSpeed - TrackMinMotionSpeed) * rt;
//     }


// };


// //适用于步兵的打击程序
// class InfancyArmorHiter : public ArmorHiter
// {
// public:

//     double ShootAngleThresh = 1;
    
//     InfancyArmorHiter(SerialManager *_serial,ArmorTrackerBase *_rmor_tracker,ArmorTrackerBase*_anti_top_ptr) : ArmorHiter(serial,armor_finder)
//     {
//         SET_CONFIG_DOUBLE_VARIABLE(ShootAngleThresh,1)
//     }


//     void Update(ImageData &frame,float dtTime)
//     {
//         Point2f result = armor_tracker->UpdateFrame(frame,dtTime);
//         if (armor_tracker->trackState) // found target
//         {
//             // 发送消息
//             serial->SendFiringOrder(Length(armor_tracker->shootOffAngle) < ShootAngleThresh ,true);
//             serial->SendPTZAbsoluteAngle(result.x + frame.ptzAngle.x,result.y + frame.ptzAngle.y);
//         }
//     }

// protected:
    
// };


//适用于英雄的打击程序
class HeroArmorHiter : public ArmorHiter
{
public:

    double ShootAngleThresh = 1;
    
    HeroArmorHiter(SerialManager* serial,ArmorFinder* armor_finder) : ArmorHiter(serial,armor_finder)
    {
        SET_CONFIG_DOUBLE_VARIABLE(ShootAngleThresh,1)
    }

    void Update(ImageData &frame,float dtTime)
    {
        // if(frame.shoot_mill){
        //     mill_hiter->update(frame,dtTime)
        // }
        // if(shoot_home){
        //     //shoot_home->update(frame,dtTime);
        // }
        // else{
            cout << "enter hiter update function" << endl;
            Point2f result=armor_finder->update_frame(frame,dtTime);
            serial->SendPTZAbsoluteAngle(result.x + frame.ptzAngle.x,result.y + frame.ptzAngle.y);
        // }
    }

protected:
    
};

