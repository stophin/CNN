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

#define WEIGHT	(((EFTYPE)(rand() % 100))/100.0)//��ʼ��ȨֵΪ0��Ҳ���Գ�ʼ�����ֵ 
#define BIAS	0//��ʼ����ֵΪ0��Ҳ���Գ�ʼ�����ֵ

#define ETA_W    0.0035   //Ȩֵ������ adjustWeight
#define ETA_B    0.001    //��ֵ������ adjustBias

#define T_ERROR	0.00002		//����������������
#define T_TIMES	10000000	//ѵ������
#endif