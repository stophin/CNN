//LAYER.h
//

#ifndef __LAYER_H_
#define __LAYER_H_

#include "Neural.h"

typedef class Layer Layer;
class Layer {
public:
	Layer() :
		mode(0),
		neurals(0) {
	}
	Layer(int mode) :
		mode(mode),
		neurals(0) {
	}
	~Layer() {
		neurals.~MultiLinkList();
	};

	MultiLinkList<Neural> neurals;

	//mode:
	//0-normal
	//1-convolutional
	int mode;

	void addNeural(EFTYPE value) {
		Neural * neural = new Neural(value);
		neural->bias = BIAS;
		neural->mode = this->mode;
		this->neurals.insertLink(neural);
	}

	void setNeural(EFTYPE values[], int size) {
		if (size < this->neurals.linkcount){
			return;
		}
		//for (int i = 0; i < this->neurals.linkcount; i++) {
		//	this->neurals.getPos(i)->value = values[i];
		//}
		int i = 0;
		Neural * _neural = this->neurals.link;
		if (_neural) {
			do {
				_neural->value = values[i];
				i++;
				if (i >= size) {
					break;
				}

				_neural = this->neurals.next(_neural);
			} while (_neural && _neural != this->neurals.link);
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
						conn->weight = WEIGHT;
						neural->conn.insertLink(conn);
						_neural->conn.insertLink(conn);

						_neural = layer.neurals.next(_neural);
					} while (_neural && _neural != layer.neurals.link);
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	void setScale(EFTYPE scale) {
		Neural * neural = this->neurals.link;
		if (neural) {
			do {
				neural->value *= scale;

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
					neural->sum = t;
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
								t += conn->weight * neural->delta;
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
					t = t *eva_fun_1(neural->output);
					//t = t * neural->output * (1 - neural->output);
				}
				else {
					//output layer
					//formula:
					//delta[ij] = (d[j] - y[j]) * F_1(S[j]
					t = (neural->output - neural->value) * eva_fun_1(neural->output);// / this->neurals.linkcount;
					//t = (neural->output - neural->value) * neural->output * (1 - neural->output);
				}
				neural->delta = t;

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	void adjustWeight() {
		Neural * neural = this->neurals.link;
		if (neural) {
			do {
				INT c = 0;
				Connector * conn = neural->conn.link;
				if (conn) {
					do {
						//for all the neurals that links after this neural
						if (conn->forw == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								//formula:
								//w[ij] = w[ij] - lamda1 * delta[ij] * x[i]
								conn->weight += ETA_W * _neural->delta * neural->output;// *eva_fun_1(_neural->output);//_neural->delta * neural->output;
							}
							c++;
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (c > 0) {
				}
				else {
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	void adjustBias() {
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
								neural->bias += ETA_B * _neural->delta;// _neural->delta;
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
					//formula:
					//b[j] = b[j] - lamda2 * delta[ij]
					neural->bias += ETA_B * neural->delta;
				}

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
				ans += 0.5 * (neural->value - neural->output) * (neural->value - neural->output) / this->neurals.linkcount;

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}

		return ans;
	}

	void resetDeltaSum() {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				conn = neural->conn.link;
				if (conn) {
					do {
						//for all the neurals that links after this neural
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								_neural->deltaSum = 0;
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				neural->biasSum = 0;

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	void resetBiasSum() {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				neural->biasSum = 0;

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	void updateDeltaSum() {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				conn = neural->conn.link;
				if (conn) {
					do {
						//for all the neurals that links after this neural
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								_neural->deltaSum += neural->output * _neural->delta;
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				neural->biasSum += neural->delta;

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	void updateBiasSum() {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				neural->biasSum += neural->delta;

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	void updateWeightWithDeltaSum(int size) {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				conn = neural->conn.link;
				if (conn) {
					do {
						//for all the neurals that links after this neural
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								conn->weight -= ETA_W * _neural->deltaSum / size;
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				neural->bias -= ETA_W * neural->biasSum / size;

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	void updateBiasWithBiasSum(int size) {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				neural->bias -= ETA_W * neural->biasSum / size;

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}


	//mode:
	//1-alloc memory and init
	//2-delete alloced memory
	//0-init alloced memory
	void resetNeural(int tc, int mode) {
		Neural * _neural = this->neurals.link;
		if (_neural) {
			do {
				if (mode > 0) {
					if (_neural->_value) {
						delete[] _neural->_value;
						_neural->_value = NULL;
					}
				}
				if (mode == 1) {
					_neural->_value = new EFTYPE[tc];
				}
				if (mode != 2) {
					for (int i = 0; i < tc; i++) {
						_neural->_value[i] = 0;
					}
				}

				if (mode > 0) {
					if (_neural->_output) {
						delete[] _neural->_output;
						_neural->_output = NULL;
					}
				}
				if (mode == 1) {
					_neural->_output = new EFTYPE[tc];
				}
				if (mode != 2) {
					for (int i = 0; i < tc; i++) {
						_neural->_output[i] = 0;
					}
				}

				_neural = this->neurals.next(_neural);
			} while (_neural && _neural != this->neurals.link);
		}
		return;
	}
	//mode:
	//1-alloc memory and init
	//2-delete alloced memory
	//0-init alloced memory
	void resetDeltaSum(int tc, int mode) {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				conn = neural->conn.link;
				if (conn) {
					do {
						//for all the neurals that links after this neural
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								if (mode > 0) {
									if (_neural->_deltaSum) {
										delete[] _neural->_deltaSum;
										_neural->_deltaSum = NULL;
									}
								}
								if (mode == 1) {
									_neural->_deltaSum = new EFTYPE[tc];
								}
								if (mode != 2) {
									for (int i = 0; i < tc; i++) {
										_neural->_deltaSum[i] = 0;
									}
								}
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (mode > 0) {
					if (neural->_biasSum) {
						delete[] neural->_biasSum;
						neural->_biasSum = NULL;
					}
				}
				if (mode == 1) {
					neural->_biasSum = new EFTYPE[tc];
				}
				if (mode != 2) {
					for (int i = 0; i < tc; i++) {
						neural->_biasSum[i] = 0;
					}
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	//mode:
	//1-alloc memory and init
	//2-delete alloced memory
	//0-init alloced memory
	void resetBiasSum(int tc, int mode) {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				if (mode > 0) {
					if (neural->_biasSum) {
						delete[] neural->_biasSum;
						neural->_biasSum = NULL;
					}
				}
				if (mode == 1) {
					neural->_biasSum = new EFTYPE[tc];
				}
				if (mode != 2) {
					for (int i = 0; i < tc; i++) {
						neural->_biasSum[i] = 0;
					}
				}


				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	//mode:
	//1-alloc memory and init
	//2-delete alloced memory
	//0-init alloced memory
	void resetDelta(int tc, int mode) {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				conn = neural->conn.link;
				if (conn) {
					do {
						//for all the neurals that links after this neural
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								if (mode > 0) {
									if (_neural->_delta) {
										delete[] _neural->_delta;
										_neural->_delta = NULL;
									}
								}
								if (mode == 1) {
									_neural->_delta = new EFTYPE[tc];
								}
								if (mode != 2) {
									for (int i = 0; i < tc; i++) {
										_neural->_delta[i] = 0;
									}
								}
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (mode > 0) {
					if (neural->_delta) {
						delete[] neural->_delta;
						neural->_delta = NULL;
					}
				}
				if (mode == 1) {
					neural->_delta = new EFTYPE[tc];
				}
				if (mode != 2) {
					for (int i = 0; i < tc; i++) {
						neural->_delta[i] = 0;
					}
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	void setNeural(EFTYPE values[], int size, int tid) {
		if (size < this->neurals.linkcount) {
			return;
		}
		//for (int i = 0; i < this->neurals.linkcount; i++) {
		//	this->neurals.getPos(i)->value = values[i];
		//}
		int i = 0;
		Neural * _neural = this->neurals.link;
		if (_neural) {
			do {
				_neural->_value[tid] = values[i];
				i++;
				if (i >= size) {
					break;
				}

				_neural = this->neurals.next(_neural);
			} while (_neural && _neural != this->neurals.link);
		}
		return;
	}
	void setScale(EFTYPE scale, int tid) {
		Neural * neural = this->neurals.link;
		if (neural) {
			do {
				neural->_value[tid] *= scale;

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	void updateDeltaSum(int tid) {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				conn = neural->conn.link;
				if (conn) {
					do {
						//for all the neurals that links after this neural
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								_neural->_deltaSum[tid] += neural->_output[tid] * _neural->_delta[tid];
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				neural->_biasSum[tid] += neural->_delta[tid];

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	void updateBiasSum(int tid) {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				neural->_biasSum[tid] += neural->_delta[tid];

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	void accumulateDeltaSum(int tc) {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				conn = neural->conn.link;
				if (conn) {
					do {
						//for all the neurals that links after this neural
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								for (int i = 0; i < tc; i++) {
									_neural->deltaSum += _neural->_deltaSum[i];
								}
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				for (int i = 0; i < tc; i++) {
					neural->biasSum += neural->_biasSum[i];
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	void accumulateBiasSum(int tc) {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {
				for (int i = 0; i < tc; i++) {
					neural->biasSum += neural->_biasSum[i];
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	void getOutput(int tid) {
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
								t += conn->weight * _neural->_output[tid];
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
					//neural->sum = t;
					t = eva_fun(t);
				}
				else {
					//input layer
					t = neural->_value[tid];
				}
				neural->_output[tid] = t;

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	void getDelta(int tid) {
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
								t += conn->weight * _neural->_delta[tid];
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
					t = t * eva_fun_1(neural->_output[tid]);
					//t = t * neural->output * (1 - neural->output);
				}
				else {
					//output layer
					//formula:
					//delta[ij] = (d[j] - y[j]) * F_1(S[j]
					t = (neural->_output[tid] - neural->_value[tid]) * eva_fun_1(neural->_output[tid]);// / this->neurals.linkcount;
					//t = (neural->output - neural->value) * neural->output * (1 - neural->output);
				}
				neural->_delta[tid] = t;

				conn = neural->conn.link;
				if (conn) {
					do {
						//for all neurals that links to this neural
						if (conn->forw == neural) {
							Neural * _neural = conn->back;
							if (_neural) {
								//conn->_delta[tid] = neural->_delta[tid];// *eva_fun_1(_neural->output);//*_nerual->output;
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	EFTYPE getError(int tid) {
		Neural * neural = this->neurals.link;
		EFTYPE ans = 0;
		if (neural) {
			INT c = 0;
			do {
				ans += 0.5 * (neural->_value[tid] - neural->_output[tid]) * (neural->_value[tid] - neural->_output[tid]) / this->neurals.linkcount;

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}

		return ans;
	}

	Neural * getMax() {
		double max = 0;
		Neural * neural = this->neurals.link;
		Neural * predict = neural;
		if (neural) {
			do {

				if (max < neural->output) {
					max = neural->output;
					predict = neural;
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
		return predict;
	}

	//activation function
	EFTYPE eva_fun(EFTYPE x) {
		return sigmod(x);
	}
	EFTYPE eva_fun_1(EFTYPE S) {
		return sigmod_1(S);
	}
	//A和B是激活函数的参数
#define A        MAX_SIMULATION_RANGE_OUTPUT
#define B        MAX_SIMULATION_RANGE_INPUT
	//sigmoid function
	EFTYPE sigmod(EFTYPE x) {
		return A / (1 + exp(-x / B));
	}

	EFTYPE sigmod_1(EFTYPE S) {
		return S * (A - S) / (A * B);
	}

	//arctan function
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

#endif