// CNN.cpp : 定义控制台应用程序的入口点。
//

#include "common/MultiLinkList.h"

#define EFTYPE float

#define MAX_LINK	10

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

	// for multilinklist
#define Connector_Size 5
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
		value(value),
		output(0),
		delta(0),
		bias(0),
		conn(0){
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


typedef class Layer Layer;
class Layer {
public:
	Layer() :
		neurals(0){
	}
	~Layer() {
		neurals.~MultiLinkList();
	};

	MultiLinkList<Neural> neurals;

	void addNeural(EFTYPE value, EFTYPE bias = 0) {
		Neural * neural = new Neural(value);
		neural->bias = bias;
		this->neurals.insertLink(neural);
	}

	void setNeural(EFTYPE values[], int size) {
		if (size < this->neurals.linkcount){
			return;
		}
		for (int i = 0; i < this->neurals.linkcount; i++) {
			this->neurals.getPos(i)->value = values[i];
		}
		return;
	}

	void makeConnection(Layer& layer, INT index) {
		Neural * _neural = layer.neurals.link;
		if (_neural) {
			do {
				_neural->conn.linkindex = index;

				_neural = layer.neurals.next(_neural);
			} while (_neural && _neural != layer.neurals.link);
		}
		Neural * neural = this->neurals.link;
		if (neural) {
			do {

				_neural = layer.neurals.link;
				if (_neural) {
					do {
						//initialized weight is 0
						Connector * conn = new Connector(0);
						conn->back = neural;
						conn->forw = _neural;
						neural->conn.insertLink(conn);
						_neural->conn.insertLink(conn);

						_neural = layer.neurals.next(_neural);
					} while (_neural && _neural != layer.neurals.link);
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	void getOutput() {
		Neural * neural = this->neurals.link;
		if (neural) {
			do {
				INT c = 0;
				EFTYPE t = 0;
				Connector * conn = neural->conn.link;
				if (conn) {
					do {
						//for all neural that links before this neural
						if (conn->forw == neural) {
							Neural * _neural = conn->back;
							if (_neural) {
								t += conn->weight * _neural->output;
							}
							c++;
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (c > 0) {
					//formula:
					//S(i) = SUM[j=0~m-1](w(ij)x(j)) - BIAS[i]
					//OUTPUT(i) = F(NET(i))
					//bias
					t += neural->bias;
					t = eva_fun(t);
				}
				else {
					//input layer
					t = neural->value;
				}
				neural->output = t;

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	void getDelta() {
		Neural * neural = this->neurals.link;
		if (neural) {
			do {
				INT c = 0;
				EFTYPE t = 0;
				Connector * conn = neural->conn.link;
				if (conn) {
					do {
						//for all neurals that links after this neural
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								t += conn->weight * _neural->delta;
							}
							c++;
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (c > 0) {
					//formula:
					//delta[ki] = SUM[j=0~n-1](delta[ij] * w[ij] * F_1(S[i]))
					// F_1(S[i]) will be multipied in here
					t = t * eva_fun_1(neural->output);
				}
				else {
					//output layer
					//formula:
					//delta[ij] = (d[j] - y[j]) * F_1(S[j]
					t = (neural->output - neural->value) * eva_fun_1(neural->output);
				}
				neural->delta = t;

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

#define ETA_W    0.0035   //权值调整率
	void adjustWeight() {
		Neural * neural = this->neurals.link;
		if (neural) {
			do {
				INT c = 0;
				Connector * conn = neural->conn.link;
				if (conn) {
					do {
						//for all the neurals that links after this neural
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								//formula:
								//w[ij] = w[ij] - lamda1 * delta[ij] * x[i]
								conn->weight -= ETA_W * _neural->delta * neural->output;
							}
							c++;
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (c > 0) {
				}
				else {
					//output layer
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
#define ETA_B    0.001    //阀值调整率
	void adjustBias() {
		Neural * neural = this->neurals.link;
		if (neural) {
			INT c = 0;
			do {
				//formula:
				//b[j] = b[j] - lamda2 * delta[ij]
				neural->bias -= ETA_B * neural->delta;

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	EFTYPE getError() {
		Neural * neural = this->neurals.link;
		EFTYPE ans = 0;
		if (neural) {
			INT c = 0;
			do {
				ans += 0.5 * (neural->output - neural->value) * (neural->output - neural->value);

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}

		return ans;
	}

	EFTYPE eva_fun(EFTYPE x) {
		return sigmod(x);
	}
	EFTYPE eva_fun_1(EFTYPE S) {
		return sigmod_1(S);
	}
#define A        30.0
#define B        10.0 //A和B是激活函数的参数
	EFTYPE sigmod(EFTYPE x) {
		return A / (1 + exp(-x / B));
	}

	EFTYPE sigmod_1(EFTYPE S) {
		return S * (A - S) / (A * B);
	}

	EFTYPE atanh(EFTYPE x) {
		return  A * atan(x);
	}
	EFTYPE atanh_1(EFTYPE S) {
		return 1 / (S * S + 1) / B;
	}

	// for multilinklist
#define Layer_Size 2
	INT uniqueID;
	Layer * prev[Layer_Size];
	Layer * next[Layer_Size];
	void operator delete(void * _ptr){
		if (_ptr == NULL)
		{
			return;
		}
		for (INT i = 0; i < Layer_Size; i++)
		{
			if (((Layer*)_ptr)->prev[i] != NULL || ((Layer*)_ptr)->next[i] != NULL)
			{
				return;
			}
		}
		delete(_ptr);
	}
};

class Network {
public:
	Network() :
		hiddens(0),
		layers(1),
		input(*(new Layer())),
		output(*(new Layer())){
		layers.insertLink(&input);
		layers.insertLink(&output);
	}
	~Network() {
		hiddens.~MultiLinkList();
		layers.~MultiLinkList();
	}

	Layer &input;
	Layer &output;
	MultiLinkList<Layer> hiddens;
	MultiLinkList<Layer> layers;

#define T_ERROR	0.002//单个样本允许的误差
	void Train() {
		printf("Target:");
		Neural * neural = input.neurals.link;
		if (neural) {
			do {

				printf("%.2f ", neural->value);

				neural = input.neurals.next(neural);
			} while (neural && neural != input.neurals.link);
		}
		printf("->");
		neural = output.neurals.link;
		if (neural) {
			do {

				printf("%.2f ", neural->value);

				neural = output.neurals.next(neural);
			} while (neural && neural != output.neurals.link);
		}
		printf("\n");


		INT count = 0;
		EFTYPE error;
		while (count < 10000) {
			ForwardTransfer();

			error = output.getError();
			if (error < T_ERROR) {
				break;
			}
			neural = output.neurals.link;
			if (neural) {
				do {

					printf("%.2f->", neural->output);

					neural = output.neurals.next(neural);
				} while (neural && neural != output.neurals.link);
			}
			printf("[%5d]Error is: %f\n", count, error);
			count++;

			ReverseTrasfer();
		}

		printf("Target:");
		neural = input.neurals.link;
		if (neural) {
			do {

				printf("%.2f ", neural->value);

				neural = input.neurals.next(neural);
			} while (neural && neural != input.neurals.link);
		}
		printf("->");
		neural = output.neurals.link;
		if (neural) {
			do {

				printf("%.2f ", neural->value);

				neural = output.neurals.next(neural);
			} while (neural && neural != output.neurals.link);
		}
		printf("\n");
		printf("Result:");
		neural = output.neurals.link;
		if (neural) {
			do {

				printf("%.2f ", neural->output);

				neural = output.neurals.next(neural);
			} while (neural && neural != output.neurals.link);
		}
		printf("\n");
		printf("[%5d]Error is: %f\n", count, error);
	}

	void ForwardTransfer() {
		input.getOutput();
		Layer * hidden = this->hiddens.link;
		if (hidden) {
			do {
				hidden->getOutput();

				hidden = this->hiddens.next(hidden);
			} while (hidden && hidden != this->hiddens.link);
		}
		output.getOutput();
	}

	void ReverseTrasfer(){
		GetDelta();
		UpdateNetwork();
	}

	void GetDelta() {
		output.getDelta();
		Layer * _hidden = this->hiddens.prev(this->hiddens.link);
		Layer * hidden = _hidden;
		if (hidden) {
			do {
				hidden->getDelta();

				hidden = this->hiddens.prev(hidden);
			} while (hidden && hidden != _hidden);
		}
	}

	void UpdateNetwork() {
		output.adjustBias();
		Layer * _hidden = this->hiddens.prev(this->hiddens.link);
		Layer * hidden = _hidden;
		if (hidden) {
			do {
				hidden->adjustWeight();
				hidden->adjustBias();

				hidden = this->hiddens.prev(hidden);
			} while (hidden && hidden != _hidden);
		}
		input.adjustWeight();
	}

	void Forecast(Layer& in) {
		if (input.neurals.linkcount != in.neurals.linkcount) {
			return;
		}
		//exchange values
		Layer temp;
		Neural * neural = input.neurals.link;
		Neural * _neural = in.neurals.link;
		if (neural) {
			do {

				EFTYPE temp = neural->value;
				neural->value = _neural->value;
				_neural->value = temp;

				_neural = in.neurals.next(_neural);
				neural = input.neurals.next(neural);
			} while (neural && neural != input.neurals.link &&
					_neural && _neural != in.neurals.link);
		}

		ForwardTransfer();

		//output
		neural = output.neurals.link;
		if (neural) {
			do {
				printf("%.2f->", neural->output);

				neural = output.neurals.next(neural);
			} while (neural && neural != output.neurals.link);
		}
		printf("\n");

		//restore values
		neural = input.neurals.link;
		_neural = in.neurals.link;
		if (neural) {
			do {

				EFTYPE temp = neural->value;
				neural->value = _neural->value;
				_neural->value = temp;

				_neural = in.neurals.next(_neural);
				neural = input.neurals.next(neural);
			} while (neural && neural != input.neurals.link &&
				_neural && _neural != in.neurals.link);
		}

	}

	void Traverse() {
		//Test
		Neural * neural = this->input.neurals.link;
		if (neural) {
			do {

				printf("%.2f--->", neural->value);

				Connector * conn = neural->conn.link;
				if (conn) {
					do {

						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								printf("%.2f(%.2f, %.2f)-->", _neural->delta, _neural->delta, conn->weight);

								Connector * _conn = _neural->conn.link;
								if (_conn) {
									do {
										if (_conn->back == _neural) {
											Neural * __neural = _conn->forw;
											if (__neural) {
												printf("%.2f(%.2f)->", __neural->value, _conn->weight);
											}
										}

										_conn = _neural->conn.next(_conn);
									} while (_conn && _conn != _neural->conn.link);
								}
								printf("\n");
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				printf("\n");

				neural = this->input.neurals.next(neural);
			} while (neural && neural != this->input.neurals.link);
		}
	}
};

EFTYPE sample[41][4] =
{
	{ 0, 0, 0, 0 },
	{ 5, 1, 4, 19.020 },
	{ 5, 3, 3, 14.150 },
	{ 5, 5, 2, 14.360 },
	{ 5, 3, 3, 14.150 },
	{ 5, 3, 2, 15.390 },
	{ 5, 3, 2, 15.390 },
	{ 5, 5, 1, 19.680 },
	{ 5, 1, 2, 21.060 },
	{ 5, 3, 3, 14.150 },
	{ 5, 5, 4, 12.680 },
	{ 5, 5, 2, 14.360 },
	{ 5, 1, 3, 19.610 },
	{ 5, 3, 4, 13.650 },
	{ 5, 5, 5, 12.430 },
	{ 5, 1, 4, 19.020 },
	{ 5, 1, 4, 19.020 },
	{ 5, 3, 5, 13.390 },
	{ 5, 5, 4, 12.680 },
	{ 5, 1, 3, 19.610 },
	{ 5, 3, 2, 15.390 },
	{ 1, 3, 1, 11.110 },
	{ 1, 5, 2, 6.521 },
	{ 1, 1, 3, 10.190 },
	{ 1, 3, 4, 6.043 },
	{ 1, 5, 5, 5.242 },
	{ 1, 5, 3, 5.724 },
	{ 1, 1, 4, 9.766 },
	{ 1, 3, 5, 5.870 },
	{ 1, 5, 4, 5.406 },
	{ 1, 1, 3, 10.190 },
	{ 1, 1, 5, 9.545 },
	{ 1, 3, 4, 6.043 },
	{ 1, 5, 3, 5.724 },
	{ 1, 1, 2, 11.250 },
	{ 1, 3, 1, 11.110 },
	{ 1, 3, 3, 6.380 },
	{ 1, 5, 2, 6.521 },
	{ 1, 1, 1, 16.000 },
	{ 1, 3, 2, 7.219 },
	{ 1, 5, 3, 5.724 }
};

EFTYPE train_sample(EFTYPE x, EFTYPE y, EFTYPE z) {
	return x + y + z;
}

int _tmain(int argc, _TCHAR* argv[])
{
	INT i, j;

	Network nets;

#define WEIGHT	0
#define BIAS	0//初始化权值和阀值为0，也可以初始化随机值
	nets.input.addNeural(1);
	nets.input.addNeural(2);
	nets.input.addNeural(3);

	nets.output.addNeural(1, BIAS);

	for (i = 0; i < 1; i++) {
		Layer * hidden = new Layer();
		
		//formula of perfect hidden num:
		//sqrt(in_num + out_num) + a
		hidden->addNeural(1 + (i + 1) * 1000, BIAS);
		hidden->addNeural(2 + (i + 1) * 1000, BIAS);
		hidden->addNeural(3 + (i + 1) * 1000, BIAS);
		hidden->addNeural(4 + (i + 1) * 1000, BIAS);
		hidden->addNeural(5 + (i + 1) * 1000, BIAS);
		hidden->addNeural(6 + (i + 1) * 1000, BIAS);
		hidden->addNeural(7 + (i + 1) * 1000, BIAS);

		nets.hiddens.insertLink(hidden);
		nets.layers.insertLink(hidden, &nets.output, NULL);

		hidden = nets.layers.link;
		if (hidden) {
			Layer * _hidden = nets.layers.next(hidden);
			if (_hidden && _hidden != nets.layers.link) {
				do {

					printf("%p, %p->", hidden, _hidden);

					hidden = _hidden;
					_hidden = nets.layers.next(_hidden);
				} while (_hidden && _hidden != nets.layers.link);
				printf("\n");
			}
		}

		hidden = nets.hiddens.link;
		if (hidden) {
			do {

				printf("%p=>", hidden);

				hidden = nets.hiddens.next(hidden);
			} while (hidden && hidden != nets.hiddens.link);
			printf("\n");
		}
	}

	Layer * hidden = nets.layers.link;
	if (hidden) {
		i = 0;
		Layer * _hidden = nets.layers.next(hidden);
		if (_hidden && _hidden != nets.layers.link) {
			do {

				hidden->makeConnection(*_hidden, ++i);

				hidden = _hidden;
				_hidden = nets.layers.next(_hidden);
			} while (_hidden && _hidden != nets.layers.link);
		}
	}

	nets.Traverse();

	getch();

	Layer input;
	input.addNeural(1);
	input.addNeural(2);
	input.addNeural(3);

	EFTYPE temp[] = { 1, 5, 2, 10 };
	for (i = 0; i < 10000; i++) {
		printf("Training: %d\n", i + 1);
		//nets.input.setNeural(sample[i], 3);
		//nets.output.setNeural(sample[i] + 3, 1);
		temp[0] = (EFTYPE)(rand() % 10) + 1;
		temp[1] = (EFTYPE)(rand() % 10) + 1;
		temp[2] = (EFTYPE)(rand() % 10) + 1;
		temp[3] = train_sample(temp[0], temp[1], temp[2]);
		nets.input.setNeural(temp, 3);
		nets.output.setNeural(temp + 3, 1);

		nets.Train();

		if (kbhit()) {
			while (1) {
				for (j = 0; j < 3; j++){
					scanf("%f", &temp[j]);
					if (ISZERO(temp[j])) {
						break;
					}
				}
				if (j < 3) {
					break;
				}

				input.setNeural(temp, 3);
				nets.Forecast(input);
			}
		}
	}


	while (1) {
		for (j = 0; j < 3; j++){
			scanf("%f", &temp[j]);
			if (ISZERO(temp[j])) {
				break;
			}
		}
		if (j < 3) {
			break;
		}

		input.setNeural(temp, 3);
		nets.Forecast(input);
	}

	return 0;
}

