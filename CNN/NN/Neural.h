//NEURAL.h
//

#ifndef __NEURAL_H_
#define __NEURAL_H_

#include "CommonDefine.h"

typedef struct _Sample
{
	double *data;
	double *label;

	int sample_w;
	int sample_h;
	int sample_count;
} Sample;
typedef struct _Kernel
{
	double *W;
	double *dW;
} Kernel;

typedef struct _Map
{
	double *data;
	double *error;
	double *label;
	double  b;
	double  db;
} Map;

//mode:
//0-normal
//1-input
//2-convolutional
//3-pooling(max)
//4-pooling(avg)
//5-output
enum LayerMode {
	Normal = 0,
	Input,
	Conv,
	MaxPool,
	AvgPool,
	Output
};

typedef class Neural Neural;
typedef class Connector Connector;
class Connector {
public:
	Connector(EFTYPE v) :
		back(NULL),
		forw(NULL),
		weight(v){
		kernel.W = NULL;
		kernel.dW = NULL;
	}
	~Connector() {
		back = NULL;
		forw = NULL;
		uninit_cnn_neural();
	}

	Neural * back;
	Neural * forw;

	EFTYPE weight;
	EFTYPE deltaSum;
	EFTYPE *_deltaSum;

	int kernel_w;
	int kernel_h;
	Kernel kernel;
	Kernel *_kernel;

	void uninit_cnn_neural() {
		if (this->kernel.W) {
			free(this->kernel.W);
			this->kernel.W = NULL;
		}
		if (this->kernel.dW) {
			free(this->kernel.dW);
			this->kernel.dW = NULL;
		}
	}

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
		map_common = NULL;
		map.data = NULL;
		map.error = NULL;
		map.label = NULL;
	}
	~Neural() {
		conn.~MultiLinkList();
		uninit_cnn_neural();
	}
	void* layer;

	EFTYPE value;
	EFTYPE output;
	EFTYPE delta;
	EFTYPE bias;
	EFTYPE sum;

	EFTYPE biasSum;

	//the following is for multi thread
	EFTYPE *_biasSum;
	EFTYPE *_delta;
	EFTYPE *_value;
	EFTYPE *_output;

	//when it is convolutional, value and output will become matrix
	int map_w;
	int map_h;
	int map_count;
	Map map;
	Map *_map;

	double *map_common;
	double **_map_common;

	void init_cnn_neural(Neural* neural, int map_count, int kernel_w, int kernel_h, int mode) {
		int prevlayer_map_count = 0;
		int map_w = 0, map_h = 0;
		if (neural == NULL) {
			//input layer
			prevlayer_map_count = 0;
			map_w = kernel_w;
			map_h = kernel_h;
			kernel_w = 0;
			kernel_h = 0;
		}
		else {
			prevlayer_map_count = neural->map_count;
			if (mode == LayerMode::Conv) {
				map_w = neural->map_w - kernel_w + 1;
				map_h = neural->map_h - kernel_h + 1;
			}
			else if (mode == LayerMode::Output) {
				map_w = 1;
				map_h = 1;
			}
			else {//pooling
				map_w = neural->map_w / 2;
				map_h = neural->map_h / 2;
			}
		}
		int mem_size = 0;

		const double scale = 6.0;
		int fan_in = 0;
		int fan_out = 0;
		//is pooling
		if (mode >= LayerMode::MaxPool && mode <= LayerMode::AvgPool)
		{
			fan_in = 4;
			fan_out = 1;
		}
		else
		{
			fan_in = prevlayer_map_count * kernel_w * kernel_h;
			fan_out = map_count * kernel_w * kernel_h;
		}
		int denominator = fan_in + fan_out;
		double weight_base = (denominator != 0) ? sqrt(scale / (double)denominator) : 0.5;


		this->map_count = map_count;
		this->map_w = map_w;
		this->map_h = map_h;
		mem_size = this->map_w * this->map_h * sizeof(double);
		if (mem_size) {
			this->map.b = 0.0;
			this->map.db = 0.0;
			this->map.data = (double *)malloc(mem_size);
			this->map.label = (double *)malloc(mem_size);
			this->map.error = (double *)malloc(mem_size);
			memset(this->map.data, 0, mem_size);
			memset(this->map.label, 0, mem_size);
			memset(this->map.error, 0, mem_size);
			this->map_common = (double *)malloc(mem_size);
			memset(this->map_common, 0, mem_size);
		}
		this->_map = NULL;
		this->_map_common = NULL;
		this->_biasSum = NULL;
		this->_delta = NULL;
		this->_output = NULL;
		this->_value = NULL;
	}

	void uninit_cnn_neural() {

		if (this->map.data) {
			free(this->map.data);
			this->map.data = NULL;
		}
		if (this->map.label) {
			free(this->map.label);
			this->map.label = NULL;
		}
		if (this->map.error) {
			free(this->map.error);
			this->map.error = NULL;
		}
		if (this->map_common) {
			free(this->map_common);
			this->map_common = NULL;
		}
	}

	void init_cnn_connector(Connector *connector, Neural* neural, int map_count, int kernel_w, int kernel_h, int mode) {
		int prevlayer_map_count = 0;
		int map_w = 0, map_h = 0;
		if (neural == NULL) {
			//input layer
			prevlayer_map_count = 0;
			map_w = kernel_w;
			map_h = kernel_h;
			kernel_w = 0;
			kernel_h = 0;
		}
		else {
			prevlayer_map_count = neural->map_count;
			if (mode == LayerMode::Conv) {
				map_w = neural->map_w - kernel_w + 1;
				map_h = neural->map_h - kernel_h + 1;
			}
			else if (mode == LayerMode::Output) {
				map_w = 1;
				map_h = 1;
			}
			else {//pooling
				map_w = neural->map_w / 2;
				map_h = neural->map_h / 2;
			}
		}
		int mem_size = 0;

		const double scale = 6.0;
		int fan_in = 0;
		int fan_out = 0;
		//is pooling
		if (mode >= LayerMode::MaxPool && mode <= LayerMode::AvgPool)
		{
			fan_in = 4;
			fan_out = 1;
		}
		else
		{
			fan_in = prevlayer_map_count * kernel_w * kernel_h;
			fan_out = map_count * kernel_w * kernel_h;
		}
		int denominator = fan_in + fan_out;
		double weight_base = (denominator != 0) ? sqrt(scale / (double)denominator) : 0.5;

		connector->kernel_w = kernel_w;
		connector->kernel_h = kernel_h;
		mem_size = connector->kernel_w * connector->kernel_h * sizeof(double);
		if (mem_size > 0) {
			connector->kernel.W = (double*)malloc(mem_size);
			int size = connector->kernel_w*connector->kernel_h;
			for (int i = 0; i < size; i++)
			{
				connector->kernel.W[i] = (genrand_real1() - 0.5) * 2 * weight_base; //0.5;// ((2.0*(double)rand() / RAND_MAX) - 1)* weight_base;
			}
			connector->kernel.dW = (double *)malloc(mem_size);
			memset(connector->kernel.dW, 0, mem_size);
		}
		connector->_kernel = NULL;
		connector->_deltaSum = NULL;
	}


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