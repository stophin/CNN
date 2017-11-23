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
#define Connector_Size 3
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
			INT c = 0;
			do {
				EFTYPE t = 0;
				Connector * conn = neural->conn.link;
				if (conn) {
					do {
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
					//bias
					t += neural->bias;

					t = sigmod(t);
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
			INT c = 0;
			do {
				EFTYPE t = 0;
				Connector * conn = neural->conn.link;
				if (conn) {
					do {
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
					t = t * sigmod_1(neural->output);
				}
				else {
					//output layer
					t = (neural->output - neural->value) * sigmod_1(neural->output);
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
			INT c = 0;
			do {
				Connector * conn = neural->conn.link;
				if (conn) {
					do {
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
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
				neural->bias -= ETA_B * neural->delta;

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	EFTYPE getError() {
		Neural * neural = this->neurals.link;
		EFTYPE ans = 0;
		if (neural) {
			do {
				ans += 0.5 * (neural->output - neural->value) * (neural->output - neural->value);

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}

		return ans;
	}

#define A        30.0
#define B        10.0 //A和B是S型函数的参数
	EFTYPE sigmod(EFTYPE x) {
		return A / (1 + exp(-x / B));
	}

	EFTYPE sigmod_1(EFTYPE S) {
		return S * (A - S) / (A * B);
	}
};

class Network {
public:
	Network(){
	}
	~Network() {
	}

	Layer input;
	Layer hidden;
	Layer output;

#define T_ERROR	0.002//单个样本允许的误差
	void Train() {
		INT count = 0;
		while (count < 1000) {

			input.getOutput();
			hidden.getOutput();
			output.getOutput();

			EFTYPE error = output.getError();
			if (error < T_ERROR) {
				break;
			}
			printf("[%5d]Error is: %f\n", count++, error);

			output.getDelta();
			hidden.getDelta();

			output.adjustBias();
			hidden.adjustWeight();
			hidden.adjustBias();
			input.adjustWeight();
		}
	}

	void Forecast(Layer& in) {
		if (input.neurals.linkcount != in.neurals.linkcount) {
			return;
		}
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

		input.getOutput();
		hidden.getOutput();
		output.getOutput();

		neural = output.neurals.link;
		if (neural) {
			do {
				printf("%.2f->", neural->output);

				neural = output.neurals.next(neural);
			} while (neural && neural != output.neurals.link);
		}

		//restore
		neural = input.neurals.link;
		_neural = in.neurals.link;
		if (neural) {
			do {

				neural->value = _neural->value;

				_neural = in.neurals.next(_neural);
				neural = input.neurals.next(neural);
			} while (neural && neural != input.neurals.link &&
				_neural && _neural != in.neurals.link);
		}

	}

	void Traverse() {
		//Test
		Neural * neural = this->output.neurals.link;
		if (neural) {
			do {

				printf("%.2f--->", neural->value);

				Connector * conn = neural->conn.link;
				if (conn) {
					do {

						if (conn->forw == neural) {
							Neural * _neural = conn->back;
							if (_neural) {
								printf("%.2f(%.2f, %.2f)-->", _neural->delta, _neural->delta, conn->weight);

								Connector * _conn = _neural->conn.link;
								if (_conn) {
									do {
										if (_conn->forw == _neural) {
											Neural * __neural = _conn->back;
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

				neural = this->output.neurals.next(neural);
			} while (neural && neural != this->output.neurals.link);
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

int _tmain(int argc, _TCHAR* argv[])
{

	Network nets;

#define WEIGHT	0
#define BIAS	-1
	nets.input.addNeural(1);
	nets.input.addNeural(2);
	nets.input.addNeural(3);

	nets.hidden.addNeural(4, BIAS);
	nets.hidden.addNeural(5, BIAS);
	nets.hidden.addNeural(6, BIAS);

	nets.output.addNeural(1, BIAS);

	nets.input.makeConnection(nets.hidden, 1);
	nets.hidden.makeConnection(nets.output, 2);


	nets.Traverse();

	for (int i = 0; i < 41; i++) {
		nets.input.setNeural(sample[i], 3);
		nets.output.setNeural(sample[i] + 3, 1);

		nets.Train();
	}

	EFTYPE temp[] = { 5, 1, 4 };
	Layer input;
	input.addNeural(1);
	input.addNeural(2);
	input.addNeural(3);

	input.setNeural(temp, 3);
	nets.Forecast(input);

	getch();
	return 0;
}

