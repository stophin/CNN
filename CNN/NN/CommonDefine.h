//CommonDefine.h
//

#ifndef _COMMON_DEFINE_H_
#define _COMMON_DEFINE_H_

#include "../common/MultiLinkList.h"

//Define Maxium Hidden Layer
#define MAX_HIDDEN_LAYER		5

//Define Maxium Simulation Range
#define MAX_SIMULATION_RANGE_INPUT	1.0
#define MAX_SIMULATION_RANGE_OUTPUT	1.0

#define WEIGHT	(((EFTYPE)(rand() % 100))/100.0)//初始化权值为0，也可以初始化随机值 
#define BIAS	0//初始化阀值为0，也可以初始化随机值

#define ETA_W    0.0035   //权值调整率 adjustWeight
#define ETA_B    0.001    //阀值调整率 adjustBias

#define T_ERROR	0.00002		//单个样本允许的误差
#define T_TIMES	10000000	//训练次数
#endif