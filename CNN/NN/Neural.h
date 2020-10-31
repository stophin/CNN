//NEURAL.h
//

#ifndef __NEURAL_H_
#define __NEURAL_H_

#include "CommonDefine.h"

typedef class Neural Neural;
typedef class Connector Connector;
class Connector {
public:
	Connector(EFTYPE v) :
		back(NULL),
		forw(NULL),
		weight(v){
	}
	~Connector() {
		back = NULL;
		forw = NULL;
	}

	Neural * back;
	Neural * forw;

	EFTYPE weight;
	EFTYPE delta;

	EFTYPE deltaSum;

	EFTYPE *_deltaSum;
	EFTYPE *_delta;

	// for multilinklist
	// The number of Connectors will be the max hidden layer numbers + 1
#define Connector_Size MAX_HIDDEN_LAYER + 1
	INT uniqueID;
	Connector * prev[Connector_Size];
	Connector * next[Connector_Size];
	void operator delete(void * _ptr){
		if (_ptr == NULL)
		{
			return;
		}
		for (INT i = 0; i < Connector_Size; i++)
		{
			if (((Connector*)_ptr)->prev[i] != NULL || ((Connector*)_ptr)->next[i] != NULL)
			{
				return;
			}
		}
		delete(_ptr);
	}
};

class Neural {
public:
	Neural(EFTYPE value) :
		uniqueID(value),
		value(value),
		output(0),
		delta(0),
		bias(0),
		conn(0),
		sum(0){
	}
	~Neural() {
		conn.~MultiLinkList();
	}

	union {
		EFTYPE value;
		EFTYPE weight;
	};
	EFTYPE output;
	EFTYPE delta;
	EFTYPE bias;
	EFTYPE sum;

	EFTYPE biasSum;

	EFTYPE *_biasSum;
	EFTYPE *_delta;
	EFTYPE *_value;
	EFTYPE *_output;

	MultiLinkList<Connector> conn;

	// for multilinklist
#define Neural_Size 2
	INT uniqueID;
	Neural * prev[Neural_Size];
	Neural * next[Neural_Size];
	void operator delete(void * _ptr){
		if (_ptr == NULL)
		{
			return;
		}
		for (INT i = 0; i < Neural_Size; i++)
		{
			if (((Neural*)_ptr)->prev[i] != NULL || ((Neural*)_ptr)->next[i] != NULL)
			{
				return;
			}
		}
		delete(_ptr);
	}
};


#endif