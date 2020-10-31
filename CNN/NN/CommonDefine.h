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

#define WEIGHT	((2.0*(double)rand()/RAND_MAX) - 1)//(((EFTYPE)(rand() % 10))/100.0)//初始化权值为0，也可以初始化随机值 
#define BIAS	((2.0*(double)rand()/RAND_MAX) - 1)//0.1//初始化阀值为0，也可以初始化随机值

#define ETA_W    0.90   //权值调整率 adjustWeight
#define ETA_B    0.01    //阀值调整率 adjustBias

#define T_ERROR	0.075		//单个样本允许的误差
#define T_TIMES	10000000	//训练次数

#if 0
#ifdef _NANOC_WINDOWS_
#include<opencv2\opencv.hpp>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>

#ifndef max
#define max(x, y) (x > y ? x : y)
#endif
#ifndef min
#define min(x, y) (x < y? x : y);
#endif


//Get sample_number samples in XML file,from the start column. 
void get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start=0);
#endif
#endif
#endif