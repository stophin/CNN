//LAYER.h
//

#ifndef __LAYER_H_
#define __LAYER_H_

#include "Neural.h"

typedef class Layer Layer;
class Layer {
public:
	Layer() :
		mode(LayerMode::Normal),
		neurals(0),
		gates(0){
	}
	Layer(LayerMode mode) :
		mode(mode),
		neurals(0),
		gates(0) {
	}
	~Layer() {
		neurals.~MultiLinkList();
	};

	//for RNN
	MultiLinkList<NeuralGate> gates;
	NeuralGate * gate;

	MultiLinkList<Neural> neurals;

	LayerMode mode;

	EFTYPE learning_rate;

	void *nets;

	void addNeural(EFTYPE value) {
		Neural * neural = new Neural(value);
		neural->bias = BIAS;
		neural->layer = this;
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
				if (this->mode == LayerMode::Normal) {
					_neural->value = values[i];
					i++;
					if (i >= size) {
						break;
					}
				}
				else {
					for (int j = 0; j < _neural->map_w * _neural->map_h; j++) {
						_neural->map.label[j] = values[i];
						i++;
						if (i >= size) {
							break;
						}
					}
					if (i >= size) {
						break;
					}
				}

				_neural = this->neurals.next(_neural);
			} while (_neural && _neural != this->neurals.link);
		}
		return;
	}
	void setNeuralMatrix(EFTYPE data[], int mem_size) {
		int i = 0;
		Neural * _neural = this->neurals.link;
		if (_neural) {
			do {

				if (this->mode == LayerMode::Normal) {
					//forbidden
				}
				else {
					//memcpy(_neural->map.data, data, mem_size * sizeof(double));
					memcpy(_neural->map.label, data, mem_size * sizeof(double));
				}

				_neural = this->neurals.next(_neural);
			} while (_neural && _neural != this->neurals.link);
		}
		return;
	}
	void makeRecursiveConnection() {
		int i = 0;
		Neural * _neural = this->neurals.link;
		if (_neural) {
			do {
				_neural->rconn.linkindex += i;
				i++;

				_neural = this->neurals.next(_neural);
			} while (_neural && _neural != this->neurals.link);
		}
		Neural * neural = this->neurals.link;
		if (neural) {
			do {
				//connect to itself
				Layer& layer = *this;
				_neural = layer.neurals.link;
				if (_neural) {
					do {
						//initialized weight is 0
						Connector * conn = new Connector(0);
						conn->back = neural;
						conn->forw = _neural;
						conn->weight = WEIGHT;
						conn->gate.reset();
						neural->rconn.insertLink(conn);
						if (neural != _neural) {
							_neural->rconn.insertLink(conn);
						}

						_neural = layer.neurals.next(_neural);
					} while (_neural && _neural != layer.neurals.link);
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	void makeConnection(Layer& layer, INT index, bool *pconnection = NULL) {
		Neural * _neural = layer.neurals.link;
		if (_neural) {
			do {
				_neural->conn.linkindex = index;

				_neural = layer.neurals.next(_neural);
			} while (_neural && _neural != layer.neurals.link);
		}
		int i = 0, j = 0, w = this->neurals.linkcount;
		Neural * neural = this->neurals.link;
		if (neural) {
			do {
				i++;
				j = 0;
				_neural = layer.neurals.link;
				if (_neural) {
					do {
						j++;
						//when pooling/output, make sure one neural connect to only one neural
						int do_connection = 1;
						Layer * _layer = (Layer*)_neural->layer;
						if ((_layer->mode >= LayerMode::MaxPool && _layer->mode <= LayerMode::AvgPool)) {
							do_connection = 2;
							if (_neural->conn.linkcount > 0) {
								do_connection = 0;
							}
						}
						else if (_layer->mode == LayerMode::Conv) {
							do_connection = 3;
							if (pconnection && pconnection[i * w + j]) {
								do_connection = 0;
							}
						}
						if (do_connection) {
							//initialized weight is 0
							Connector * conn = new Connector(0);
							conn->back = neural;
							conn->forw = _neural;
							conn->weight = WEIGHT;
							neural->conn.insertLink(conn);
							_neural->conn.insertLink(conn);
							if (do_connection == 2) {
								break;
							}
						}

						_neural = layer.neurals.next(_neural);
					} while (_neural && _neural != layer.neurals.link);
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	void makeMatrix(int kernel_w, int kernel_h) {
		if (this->mode == LayerMode::Normal) {
			return;
		}
		Neural * neural = this->neurals.link;
		if (neural) {
			do {
				Neural * _neural = NULL;
				Connector * conn = neural->conn.link;
				if (conn) {
					do {
						//for all neural that links before this neural
						if (conn->forw == neural) {
							_neural = conn->back;
							_neural->init_cnn_connector(conn, _neural, this->neurals.linkcount, kernel_w, kernel_h, this->mode);
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				//malloc matrix for convolutional/pooling neural
				//when the layer is input, _neural is NULL , otherwise there must be one neural linked before
				//note that all neurals in prev layer is at the same matrix, so we just choose one
				neural->init_cnn_neural(_neural, this->neurals.linkcount, kernel_w, kernel_h, this->mode);

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	void setScale(EFTYPE scale) {
		Neural * neural = this->neurals.link;
		if (neural) {
			do {
				if (mode == LayerMode::Normal) {
					neural->value *= scale;
				}
				else {
					for (int i = neural->map_w * neural->map_h - 1; i >= 0; i--) {
						neural->map.label[i] *= scale;
					}
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	// 克罗内克积 start--------------------------------------------------------------------------------------------/
	void kronecker(double *in_data, int in_map_w, int in_map_h, double *out_data, int out_map_w)
	{
		for (int i = 0; i < in_map_h; i++)
		{
			for (int j = 0; j < in_map_w; j++)
			{
				for (int n = 2 * i; n < 2 * (i + 1); n++)
				{
					for (int m = 2 * j; m < 2 * (j + 1); m++)
					{
						out_data[n*out_map_w + m] = in_data[i*in_map_w + j];
					}
				}
			}
		}
	}
	// 克罗内克积 end--------------------------------------------------------------------------------------------/
	// 卷积 start--------------------------------------------------------------------------------------------------/
	void convn_valid(double *in_data, int in_w, int in_h, double *kernel, int kernel_w, int kernel_h, double *out_data, int out_w, int out_h)
	{
		double sum = 0.0;
		for (int i = 0; i < out_h; i++)
		{
			for (int j = 0; j < out_w; j++)
			{
				sum = 0.0;
				for (int n = 0; n < kernel_h; n++)
				{
					for (int m = 0; m < kernel_w; m++)
					{
						sum += in_data[(i + n)*in_w + j + m] * kernel[n*kernel_w + m];
					}
				}
				out_data[i*out_w + j] += sum;
			}
		}
	}
	// 卷积 end--------------------------------------------------------------------------------------------------/
	double gradient_descent(double W, double dW, double alpha, double lambda)
	{
		return W - alpha * (dW + lambda * W);
	}
	void getOutput() {
		Neural * neural = this->neurals.link;
		if (neural) {
			do {
				INT c = 0;
				EFTYPE t = 0;
				INT size = 0;
				double scale_factor = 0.25;
				if (mode == LayerMode::Conv) {
					size = neural->map_w * neural->map_h;
					memset(neural->map_common, 0, size * sizeof(double));
				}
				else if (mode == LayerMode::MaxPool) {
				} else if (mode == LayerMode::AvgPool) {
				} else if (mode == LayerMode::Output) {
				}
				Connector * conn = neural->conn.link;
				if (conn) {
					do {
						//for all neural that links before this neural
						if (conn->forw == neural) {
							Neural * _neural = conn->back;
							if (_neural) {
								if (this->mode == LayerMode::Normal) {
									t += conn->weight * _neural->output;
								}
								else {
									if (mode == LayerMode::Conv) {
										convn_valid(_neural->map.data, _neural->map_w, _neural->map_h,
											conn->kernel.W, conn->kernel_w, conn->kernel_h,
											neural->map_common, neural->map_w, neural->map_h);
									} else if (mode == LayerMode::MaxPool) {
										for (int i = 0; i < neural->map_h; i++) {
											for (int j = 0; j < neural->map_w; j++) {
												double max_value = _neural->map.data[2 * i * _neural->map_w + 2 * j];
												for (int n = 2 * i; n < 2 * (i + 1); n++){
													for (int m = 2 * j; m < 2 * (j + 1); m++){
														max_value = MAX(max_value, _neural->map.data[n * _neural->map_w + m]);
													}
												}
												neural->map.data[i * neural->map_w + j] = tan_h(max_value);
											}
										}
									}
									else if (mode == LayerMode::AvgPool) {
										for (int i = 0; i < neural->map_h; i++) {
											for (int j = 0; j < neural->map_w; j++) {
												double sum = 0.0;
												for (int n = 2 * i; n < 2 * (i + 1); n++) {
													for (int m = 2 * j; m < 2 * (j + 1); m++) {
														sum += _neural->map.data[n * _neural->map_w + m] * conn->kernel.W[0];
													}
												}
												sum *= scale_factor;
												sum += neural->map.b;
												neural->map.data[i * neural->map_w + j] = tan_h(sum);
											}
										}
									}
									else if (mode == LayerMode::Output) {
										t += _neural->map.data[0] * conn->kernel.W[0];
									}
								}
							}
							c++;
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (this->mode == LayerMode::Normal) {
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
				}
				else {
					if (c > 0) {
						if (mode == LayerMode::Conv) {
							for (int i = 0; i < size; i++) {
								neural->map.data[i] = tan_h(neural->map_common[i] + neural->map.b);
							}
						} else if (mode == LayerMode::MaxPool) {
						}else if (mode == LayerMode::AvgPool) {
						} else if (mode == LayerMode::Output) {
							t += neural->map.b;
							neural->map.data[0] = tan_h(t);
							//make it to output
							neural->output = neural->map.data[0];
						}
					}
					else {
						//input_layer
						memcpy(neural->map.data, neural->map.label, neural->map_w * neural->map_h * sizeof(double));
					}
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	void getDelta(LayerMode mode=LayerMode::Normal) {
		Neural * neural = this->neurals.link;
		if (neural) {
			do {
				INT c = 0;
				EFTYPE t = 0;
				INT size = 0;
				double scale_factor = 0.25;
				if (mode == LayerMode::Conv) {
					size = neural->map_w * neural->map_h;
					memset(neural->map_common, 0, size * sizeof(double));
				}
				else if (mode == LayerMode::MaxPool) {
				}
				else if (mode == LayerMode::AvgPool) {
					size = neural->map_w * neural->map_h;
				}
				else if (mode == LayerMode::Output) {
					neural->map.error[0] = 0;
				}
				Connector * conn = neural->conn.link;
				if (conn) {
					do {
						//for all neurals that links after this neural
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								if (this->mode == LayerMode::Normal) {
									t += conn->weight * _neural->delta;
								}
								else {
									if (mode == LayerMode::Conv) {
										for (int n = 0; n < _neural->map_h; n++) {
											for (int m = 0; m < _neural->map_w; m++) {
												double error = _neural->map.error[n * _neural->map_w + m];
												for (int ky = 0; ky < conn->kernel_h; ky++) {
													for (int kx = 0; kx < conn->kernel_w; kx++) {
														neural->map_common[(n + ky) * neural->map_w + m + kx] += error * conn->kernel.W[ky * conn->kernel_w + kx];
													}
												}
											}
										}
									}
									else if (mode == LayerMode::MaxPool) {
										for (int i = 0; i < _neural->map_h; i++) {
											for (int j = 0; j < _neural->map_w; j++) {
												int row = 2 * i, col = 2 * j;
												double max_value = neural->map.data[row * neural->map_w + col];
												for (int n = 2 * i; n < 2 * (i + 1); n++) {
													for (int m = 2 * j; m < 2 * (j + 1); m++) {
														if (neural->map.data[n * neural->map_w + m] > max_value) {
															row = n;
															col = m;
															max_value = neural->map.data[n * neural->map_w + m];
														}
														else {
															neural->map.error[n * neural->map_w + m] = 0;
														}
													}
												}
												neural->map.error[row * neural->map_w + col] = _neural->map.error[i * _neural->map_w + j] * tan_h_1(max_value);
											}
										}
									}
									else if (mode == LayerMode::AvgPool) {
										kronecker(_neural->map.error, neural->map_w, neural->map_h, neural->map_common, neural->map_w);
										t += conn->kernel.W[0];
									}
									else if (mode == LayerMode::Output) {
										t += _neural->map.error[0] * conn->kernel.W[0];
									}
									else if (mode == LayerMode::Normal) {
										t += _neural->delta;
									}
								}
							}
							c++;
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (this->mode == LayerMode::Normal) {
					if (c > 0) {
						//formula:
						//delta[ki] = SUM[j=0~n-1](delta[ij] * w[ij] * F_1(S[i]))
						// F_1(S[i]) will be multipied in here
						t = t * eva_fun_1(neural->output);
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
				}
				else {
					if (c > 0) {
						if (mode == LayerMode::Conv) {
							for (int i = 0; i < size; i++) {
								neural->map.error[i] = neural->map_common[i] * tan_h_1(neural->map.data[i]);
							}
						}
						else if (mode == LayerMode::MaxPool) {
						}
						else if (mode == LayerMode::AvgPool) {
							for (int k = 0; k < size; k++) {
								double delta = t * neural->map_common[k];
								neural->map.error[k] = delta * scale_factor * tan_h_1(neural->map.data[k]);
							}
						}
						else if (mode == LayerMode::Output) {
							neural->map.error[0] = t * tan_h_1(neural->map.data[0]);
						}
						else if (mode == LayerMode::Normal) {
							neural->map.error[0] = t;
						}
					}
					else {
						//output_layer
						neural->map.error[0] = (neural->map.data[0] - neural->map.label[0]) * tan_h_1(neural->map.data[0]);
					}
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}


	EFTYPE getError() {
		Neural * neural = this->neurals.link;
		EFTYPE ans = 0;
		EFTYPE cur = 0;
		if (neural) {
			INT c = 0;
			do {
				if (mode == LayerMode::Normal) {
					cur = 0.5 * (neural->value - neural->output) * (neural->value - neural->output) / this->neurals.linkcount;
				}
				else {
					cur = 0.5 * (neural->map.data[0] - neural->map.label[0]) * (neural->map.data[0] - neural->map.label[0]) / this->neurals.linkcount;
				}
				//ans += cur;
				if (ans < cur) {
					ans = cur;
				}

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

				INT k_size = 0;
				double lambda = 0.0;
				if (mode == LayerMode::Normal) {
				}
				else {
				}
				conn = neural->conn.link;
				if (conn) {
					do {
						//for all the neurals that links after this neural
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								if (mode == LayerMode::Normal) {
									conn->deltaSum = 0;
								}
								else {
									k_size = conn->kernel_w * conn->kernel_h;
									memset(conn->kernel.dW, 0, k_size * sizeof(double));
								}
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (mode == LayerMode::Normal) {
					neural->biasSum = 0;
				}
				else {
					neural->map.db = 0;
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	void resetBiasSum() {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				if (mode == LayerMode::Normal) {
					neural->biasSum = 0;
				}
				else {
					neural->map.db = 0;
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	void updateDeltaSum(LayerMode mode=LayerMode::Normal) {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				INT c = 0;
				INT size = 0;
				double scale_factor = 0.25;
				if (this->mode == LayerMode::Conv) {
					size = neural->map_w * neural->map_h;
				}
				else if (this->mode == LayerMode::MaxPool) {
					size = neural->map_w * neural->map_h;
				}
				else if (this->mode == LayerMode::AvgPool) {
					size = neural->map_w * neural->map_h;
				}
				else if (this->mode == LayerMode::Output) {
					size = neural->map_w * neural->map_h;
				}
				conn = neural->conn.link;
				if (conn) {
					do {
						//for all the neurals that links after this neural
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								if (this->mode == LayerMode::Normal) {
									conn->deltaSum += neural->output * _neural->delta;
								}
								else {
									if (mode == LayerMode::Conv) {
										convn_valid(neural->map.data, neural->map_w, neural->map_h,
											_neural->map.error, _neural->map_w, _neural->map_h,
											conn->kernel.dW, conn->kernel_w, conn->kernel_h);
									}
									else if (mode == LayerMode::MaxPool) {
									}
									else if (mode == LayerMode::AvgPool) {
										double sum = 0;
										for (int i = 0; i < size; i++) {
											sum += neural->map.data[i] * neural->map_common[i];
										}
										conn->kernel.dW[0] += sum * scale_factor;
									}
									else if (mode == LayerMode::Output) {
										conn->kernel.dW[0] += _neural->map.error[0] * neural->map.data[0];
									}
								}
								c++;
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (this->mode == LayerMode::Normal) {
					neural->biasSum += neural->delta;
				}
				else {
					if (c > 0) {
						if (this->mode == LayerMode::Conv) {
							double sum = 0;
							for (int i = 0; i < size; i++) {
								sum += neural->map.error[i];
							}
							neural->map.db += sum;
						}
						else if (this->mode == LayerMode::MaxPool) {
						}
						else if (this->mode == LayerMode::AvgPool) {
							double sum = 0;
							for (int i = 0; i < size; i++) {
								sum += neural->map.error[i];
							}
							neural->map.db += sum;
						}
						else if (this->mode == LayerMode::Output) {
						}
					}
					else {
						//output_layer
						neural->map.db += neural->map.error[0];
					}
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	void updateBiasSum() {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				INT size = 0;
				if (mode == LayerMode::Conv) {
					size = neural->map_w * neural->map_h;
				}
				else if (mode == LayerMode::MaxPool) {
					size = neural->map_w * neural->map_h;
				}
				else if (mode == LayerMode::AvgPool) {
					size = neural->map_w * neural->map_h;
				}
				else if (mode == LayerMode::Output) {
				}


				if (mode == LayerMode::Normal) {
					neural->biasSum += neural->delta;
				}
				else {
					if (mode == LayerMode::Conv) {
						double sum = 0;
						for (int i = 0; i < size; i++) {
							sum += neural->map.error[i];
						}
						neural->map.db += sum;
					}
					else if (mode == LayerMode::MaxPool) {
					}
					else if (mode == LayerMode::AvgPool) {
						double sum = 0;
						for (int i = 0; i < size; i++) {
							sum += neural->map.error[i];
						}
						neural->map.db += sum;
					}
					else if (mode == LayerMode::Output) {
						neural->map.db += neural->map.error[0];
					}
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	void updateWeightWithDeltaSum(int size) {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				INT k_size = 0;
				double lambda = 0.0;
				if (mode == LayerMode::Normal) {
				}
				else {
				}
				conn = neural->conn.link;
				if (conn) {
					do {
						//for all the neurals that links after this neural
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								if (mode == LayerMode::Normal) {
									conn->weight -= learning_rate * conn->deltaSum / size;//TODO
								}
								else {
									k_size = conn->kernel_w * conn->kernel_h;
									for (int i = 0; i < k_size; i++) {
										conn->kernel.W[i] = gradient_descent(conn->kernel.W[i], conn->kernel.dW[i] / size, learning_rate, lambda);
									}
								}
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (mode == LayerMode::Normal) {
					neural->bias -= learning_rate * neural->biasSum / size;
				}
				else {
					neural->map.b = gradient_descent(neural->map.b, neural->map.db / size, learning_rate, lambda);
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	void updateBiasWithBiasSum(int size) {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {
				double lambda = 0.0;

				if (mode == LayerMode::Normal) {
					neural->bias -= learning_rate * neural->biasSum / size;
				}
				else {
					neural->map.b = gradient_descent(neural->map.b, neural->map.db / size, learning_rate, lambda);
				}

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
	void resetMapKernel(int tc, int mode) {
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
								if (this->mode == LayerMode::Normal) {
								}
								else {
									if (mode > 0) {
										if (conn->_kernel) {
											delete[] conn->_kernel;
											conn->_kernel = NULL;
										}
									}
									if (mode == 1) {
										conn->_kernel = new Kernel[tc];
									}
									if (mode != 2) {
										for (int i = 0; i < tc; i++) {
											conn->_kernel[i].dW = NULL;
											conn->_kernel[i].W = NULL;
										}
									}
								}
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (this->mode == LayerMode::Normal) {
				}
				else {
					if (mode > 0) {
						if (neural->_map) {
							delete[] neural->_map;
							neural->_map = NULL;
						}
					}
					if (mode == 1) {
						neural->_map = new Map[tc];
					}
					if (mode != 2) {
						for (int i = 0; i < tc; i++) {
							neural->_map[i].data = NULL;
							neural->_map[i].error = NULL;
							neural->_map[i].label = NULL;
						}
					}
					if (mode > 0) {
						if (neural->_map) {
							for (int i = 0; i < tc; i++) {
								if (neural->_map[i].data) {
									delete[] neural->_map[i].data;
									neural->_map[i].data = NULL;
								}
							}
						}
					}
					if (mode == 1) {
						for (int i = 0; i < tc; i++) {
							neural->_map[i].data = new double[neural->map_w * neural->map_h];
						}
					}
					if (mode != 2) {
						for (int i = 0; i < tc; i++) {
							memset(neural->_map[i].data, 0, sizeof(double) * neural->map_w * neural->map_h);
						}
					}
					if (mode > 0) {
						if (neural->_map) {
							for (int i = 0; i < tc; i++) {
								if (neural->_map[i].label) {
									delete[] neural->_map[i].label;
									neural->_map[i].label = NULL;
								}
							}
						}
					}
					if (mode == 1) {
						for (int i = 0; i < tc; i++) {
							neural->_map[i].label = new double[neural->map_w * neural->map_h];
						}
					}
					if (mode != 2) {
						for (int i = 0; i < tc; i++) {
							memset(neural->_map[i].label, 0, sizeof(double) * neural->map_w * neural->map_h);
						}
					}
					if (mode > 0) {
						if (neural->_map_common) {
							for (int i = 0; i < tc; i++) {
								if (neural->_map_common[i]) {
									delete[] neural->_map_common[i];
									neural->_map_common[i] = NULL;
								}
							}
							delete[] neural->_map_common;
							neural->_map_common = NULL;
						}
					}
					if (mode == 1) {
						neural->_map_common = new double*[tc];
						for (int i = 0; i < tc; i++) {
							neural->_map_common[i] = new double[neural->map_w * neural->map_h];
						}
					}
					if (mode != 2) {
						for (int i = 0; i < tc; i++) {
							memset(neural->_map_common[i], 0, sizeof(double) * neural->map_w * neural->map_h);
						}
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
								if (this->mode == LayerMode::Normal) {
									if (mode > 0) {
										if (conn->_deltaSum) {
											delete[] conn->_deltaSum;
											conn->_deltaSum = NULL;
										}
									}
									if (mode == 1) {
										conn->_deltaSum = new EFTYPE[tc];
									}
									if (mode != 2) {
										for (int i = 0; i < tc; i++) {
											conn->_deltaSum[i] = 0;
										}
									}
								}
								else {
									if (mode > 0) {
										if (conn->_kernel) {
											for (int i = 0; i < tc; i++) {
												if (conn->_kernel[i].dW) {
													delete[] conn->_kernel[i].dW;
													conn->_kernel[i].dW = NULL;
												}
											}
										}
									}
									if (mode == 1) {
										for (int i = 0; i < tc; i++) {
											conn->_kernel[i].dW = new double[conn->kernel_w * conn->kernel_h];
										}
									}
									if (mode != 2) {
										for (int i = 0; i < tc; i++) {
											memset(conn->_kernel[i].dW, 0, sizeof(double)*conn->kernel_w * conn->kernel_h);
										}
									}
								}
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (this->mode == LayerMode::Normal) {
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
				}
				else {
					if (mode > 0) {
					}
					if (mode == 1) {
					}
					if (mode != 2) {
						for (int i = 0; i < tc; i++) {
							neural->_map[i].db = 0;
						}
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

				if (this->mode == LayerMode::Normal) {
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

				}
				else {
					if (mode > 0) {
					}
					if (mode == 1) {
					}
					if (mode != 2) {
						for (int i = 0; i < tc; i++) {
							neural->_map[i].db = 0;
						}
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
		if (neural) {
			do {
				if (this->mode == LayerMode::Normal) {
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
				}
				else {
					if (mode > 0) {
						if (neural->_map) {
							for (int i = 0; i < tc; i++) {
								if (neural->_map[i].error) {
									delete[] neural->_map[i].error;
									neural->_map[i].error = NULL;
								}
							}
						}
					}
					if (mode == 1) {
						for (int i = 0; i < tc; i++) {
							neural->_map[i].error = new double[neural->map_w * neural->map_h];
						}
					}
					if (mode != 2) {
						for (int i = 0; i < tc; i++) {
							memset(neural->_map[i].error, 0, sizeof(double) * neural->map_w * neural->map_h);
						}
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
				if (this->mode == LayerMode::Normal) {
					_neural->_value[tid] = values[i];
					i++;
					if (i >= size) {
						break;
					}
				} else {
					for (int j = 0; j < _neural->map_w * _neural->map_h; j++) {
						_neural->_map[tid].label[j] = values[i];
						i++;
						if (i >= size) {
							break;
						}
					}
					if (i >= size) {
						break;
					}
				}

				_neural = this->neurals.next(_neural);
			} while (_neural && _neural != this->neurals.link);
		}
		return;
	}
	void setNeuralMatrix(EFTYPE data[], int mem_size, int tid) {
		int i = 0;
		Neural * _neural = this->neurals.link;
		if (_neural) {
			do {
				if (this->mode == LayerMode::Normal) {
					//forbidden
				}
				else {
					//memcpy(_neural->_map[tid].data, data, mem_size * sizeof(double));
					memcpy(_neural->_map[tid].label, data, mem_size * sizeof(double));
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
				if (this->mode == LayerMode::Normal) {
					neural->_value[tid] *= scale;
				}
				else {
					for (int i = neural->map_w * neural->map_h - 1; i >= 0; i--) {
						neural->_map[tid].label[i] *= scale;
					}
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	void updateDeltaSum(int tid, LayerMode mode = LayerMode::Normal) {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				INT c = 0;
				INT size = 0;
				double scale_factor = 0.25;
				if (this->mode == LayerMode::Conv) {
					size = neural->map_w * neural->map_h;
				}
				else if (this->mode == LayerMode::MaxPool) {
					size = neural->map_w * neural->map_h;
				}
				else if (this->mode == LayerMode::AvgPool) {
					size = neural->map_w * neural->map_h;
				}
				else if (this->mode == LayerMode::Output) {
					size = neural->map_w * neural->map_h;
				}
				conn = neural->conn.link;
				if (conn) {
					do {
						//for all the neurals that links after this neural
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								if (this->mode == LayerMode::Normal) {
									conn->_deltaSum[tid] += neural->_output[tid] * _neural->_delta[tid];
								}
								else {
									if (mode == LayerMode::Conv) {
										convn_valid(neural->_map[tid].data, neural->map_w, neural->map_h,
											_neural->_map[tid].error, _neural->map_w, _neural->map_h,
											conn->_kernel[tid].dW, conn->kernel_w, conn->kernel_h);
									}
									else if (mode == LayerMode::MaxPool) {
									}
									else if (mode == LayerMode::AvgPool) {
										double sum = 0;
										for (int i = 0; i < size; i++) {
											sum += neural->_map[tid].data[i] * neural->_map_common[tid][i];
										}
										conn->_kernel[tid].dW[0] += sum * scale_factor;
									}
									else if (mode == LayerMode::Output) {
										conn->_kernel[tid].dW[0] += _neural->_map[tid].error[0] * neural->_map[tid].data[0];
									}
								}
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (this->mode == LayerMode::Normal) {
					neural->_biasSum[tid] += neural->_delta[tid];
				}
				else {
					if (c > 0) {
						if (this->mode == LayerMode::Conv) {
							double sum = 0;
							for (int i = 0; i < size; i++) {
								sum += neural->_map[tid].error[i];
							}
							neural->_map[tid].db += sum;
						}
						else if (this->mode == LayerMode::MaxPool) {
						}
						else if (this->mode == LayerMode::AvgPool) {
							double sum = 0;
							for (int i = 0; i < size; i++) {
								sum += neural->_map[tid].error[i];
							}
							neural->_map[tid].db += sum;
						}
						else if (this->mode == LayerMode::Output) {
						}
					}
					else {
						//output_layer
						neural->_map[tid].db += neural->_map[tid].error[0];
					}
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}
	void updateBiasSum(int tid) {
		Neural *neural = this->neurals.link;
		Connector * conn;
		if (neural) {
			do {

				INT size = 0;
				if (mode == LayerMode::Conv) {
					size = neural->map_w * neural->map_h;
				}
				else if (mode == LayerMode::MaxPool) {
					size = neural->map_w * neural->map_h;
				}
				else if (mode == LayerMode::AvgPool) {
					size = neural->map_w * neural->map_h;
				}
				else if (mode == LayerMode::Output) {
				}


				if (mode == LayerMode::Normal) {
					neural->_biasSum[tid] += neural->_delta[tid];
				}
				else {
					if (mode == LayerMode::Conv) {
						double sum = 0;
						for (int i = 0; i < size; i++) {
							sum += neural->_map[tid].error[i];
						}
						neural->_map[tid].db += sum;
					}
					else if (mode == LayerMode::MaxPool) {
					}
					else if (mode == LayerMode::AvgPool) {
						double sum = 0;
						for (int i = 0; i < size; i++) {
							sum += neural->_map[tid].error[i];
						}
						neural->_map[tid].db += sum;
					}
					else if (mode == LayerMode::Output) {
						neural->_map[tid].db += neural->_map[tid].error[0];
					}
				}

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
								if(this->mode == LayerMode::Normal) {
									for (int i = 0; i < tc; i++) {
										conn->deltaSum += conn->_deltaSum[i];
									}
								} else {
									for (int i = 0; i < tc; i++) {
										for (int j = conn->kernel_w * conn->kernel_h - 1; j >= 0; j--) {
											conn->kernel.dW[j] += conn->_kernel[i].dW[j];
										}
									}
								}
							}
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (this->mode == LayerMode::Normal) {
					for (int i = 0; i < tc; i++) {
						neural->biasSum += neural->_biasSum[i];
					}
				}
				else {
					for (int i = 0; i < tc; i++) {
						neural->map.db += neural->_map[i].db;
					}
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
				if (this->mode == LayerMode::Normal) {
					for (int i = 0; i < tc; i++) {
						neural->biasSum += neural->_biasSum[i];
					}
				}else {
					for (int i = 0; i < tc; i++) {
						neural->map.db += neural->_map[i].db;
					}
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
				INT size = 0;
				double scale_factor = 0.25;
				if (mode == LayerMode::Conv) {
					size = neural->map_w * neural->map_h;
					memset(neural->_map_common[tid], 0, size * sizeof(double));
				}
				else if (mode == LayerMode::MaxPool) {
				}
				else if (mode == LayerMode::AvgPool) {
				}
				else if (mode == LayerMode::Output) {
				}
				Connector * conn = neural->conn.link;
				if (conn) {
					do {
						//for all neural that links before this neural
						if (conn->forw == neural) {
							Neural * _neural = conn->back;
							if (_neural) {
								if (this->mode == LayerMode::Normal) {
									t += conn->weight * _neural->_output[tid];
								}
								else {
									if (mode == LayerMode::Conv) {
										convn_valid(_neural->_map[tid].data, _neural->map_w, _neural->map_h,
											conn->kernel.W, conn->kernel_w, conn->kernel_h,
											neural->_map_common[tid], neural->map_w, neural->map_h);
									}
									else if (mode == LayerMode::MaxPool) {
										for (int i = 0; i < neural->map_h; i++) {
											for (int j = 0; j < neural->map_w; j++) {
												double max_value = _neural->_map[tid].data[2 * i * _neural->map_w + 2 * j];
												for (int n = 2 * i; n < 2 * (i + 1); n++) {
													for (int m = 2 * j; m < 2 * (j + 1); m++) {
														max_value = MAX(max_value, _neural->_map[tid].data[n * _neural->map_w + m]);
													}
												}
												neural->_map[tid].data[i * neural->map_w + j] = tan_h(max_value);
											}
										}
									}
									else if (mode == LayerMode::AvgPool) {
										for (int i = 0; i < neural->map_h; i++) {
											for (int j = 0; j < neural->map_w; j++) {
												double sum = 0.0;
												for (int n = 2 * i; n < 2 * (i + 1); n++) {
													for (int m = 2 * j; m < 2 * (j + 1); m++) {
														sum += _neural->_map[tid].data[n * _neural->map_w + m] * conn->kernel.W[0];
													}
												}
												sum *= scale_factor;
												sum += neural->map.b;
												neural->_map[tid].data[i * neural->map_w + j] = tan_h(sum);
											}
										}
									}
									else if (mode == LayerMode::Output) {
										t += _neural->_map[tid].data[0] * conn->kernel.W[0];
									}
								}
							}
							c++;
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (this->mode == LayerMode::Normal) {
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
				}
				else {
					if (c > 0) {
						if (mode == LayerMode::Conv) {
							for (int i = 0; i < size; i++) {
								neural->_map[tid].data[i] = tan_h(neural->_map_common[tid][i] + neural->map.b);
							}
						}
						else if (mode == LayerMode::MaxPool) {
						}
						else if (mode == LayerMode::AvgPool) {
						}
						else if (mode == LayerMode::Output) {
							t += neural->map.b;
							neural->_map[tid].data[0] = tan_h(t);
							//make it to output
							neural->output = neural->_map[tid].data[0];
						}
					}
					else {
						//input_layer
						memcpy(neural->_map[tid].data, neural->_map[tid].label, neural->map_w * neural->map_h * sizeof(double));
					}
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}
	}

	void getDelta(int tid, LayerMode mode = LayerMode::Normal) {
		Neural * neural = this->neurals.link;
		if (neural) {
			do {
				INT c = 0;
				EFTYPE t = 0;
				INT size = 0;
				double scale_factor = 0.25;
				if (mode == LayerMode::Conv) {
					size = neural->map_w * neural->map_h;
					memset(neural->_map_common[tid], 0, size * sizeof(double));
				}
				else if (mode == LayerMode::MaxPool) {
				}
				else if (mode == LayerMode::AvgPool) {
					size = neural->map_w * neural->map_h;
				}
				else if (mode == LayerMode::Output) {
					neural->_map[tid].error[0] = 0;
				}
				Connector * conn = neural->conn.link;
				if (conn) {
					do {
						//for all neurals that links after this neural
						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								if (this->mode == LayerMode::Normal) {
									t += conn->weight * _neural->_delta[tid];
								}
								else {
									if (mode == LayerMode::Conv) {
										for (int n = 0; n < _neural->map_h; n++) {
											for (int m = 0; m < _neural->map_w; m++) {
												double error = _neural->_map[tid].error[n * _neural->map_w + m];
												for (int ky = 0; ky < conn->kernel_h; ky++) {
													for (int kx = 0; kx < conn->kernel_w; kx++) {
														neural->_map_common[tid][(n + ky) * neural->map_w + m + kx] += error * conn->kernel.W[ky * conn->kernel_w + kx];
													}
												}
											}
										}
									}
									else if (mode == LayerMode::MaxPool) {
										for (int i = 0; i < _neural->map_h; i++) {
											for (int j = 0; j < _neural->map_w; j++) {
												int row = 2 * i, col = 2 * j;
												double max_value = neural->_map[tid].data[row * neural->map_w + col];
												for (int n = 2 * i; n < 2 * (i + 1); n++) {
													for (int m = 2 * j; m < 2 * (j + 1); m++) {
														if (neural->_map[tid].data[n * neural->map_w + m] > max_value) {
															row = n;
															col = m;
															max_value = neural->_map[tid].data[n * neural->map_w + m];
														}
														else {
															neural->_map[tid].error[n * neural->map_w + m] = 0;
														}
													}
												}
												neural->_map[tid].error[row * neural->map_w + col] = _neural->_map[tid].error[i * _neural->map_w + j] * tan_h_1(max_value);
											}
										}
									}
									else if (mode == LayerMode::AvgPool) {
										kronecker(_neural->_map[tid].error, neural->map_w, neural->map_h, neural->_map_common[tid], neural->map_w);
										t += conn->kernel.W[0];
									}
									else if (mode == LayerMode::Output) {
										t += _neural->_map[tid].error[0] * conn->kernel.W[0];
									}
									else if (mode == LayerMode::Normal) {
										t += _neural->_delta[tid];
									}
								}
							}
							c++;
						}

						conn = neural->conn.next(conn);
					} while (conn && conn != neural->conn.link);
				}
				if (this->mode == LayerMode::Normal) {
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
				}
				else {
					if (c > 0) {
						if (mode == LayerMode::Conv) {
							for (int i = 0; i < size; i++) {
								neural->_map[tid].error[i] = neural->_map_common[tid][i] * tan_h_1(neural->_map[tid].data[i]);
							}
						}
						else if (mode == LayerMode::MaxPool) {
						}
						else if (mode == LayerMode::AvgPool) {
							for (int k = 0; k < size; k++) {
								double delta = t * neural->_map_common[tid][k];
								neural->_map[tid].error[k] = delta * scale_factor * tan_h_1(neural->_map[tid].data[k]);
							}
						}
						else if (mode == LayerMode::Output) {
							neural->_map[tid].error[0] = t * tan_h_1(neural->_map[tid].data[0]);
						}
						else if (mode == LayerMode::Normal) {
							neural->_map[tid].error[0] = t;
						}
					}
					else {
						//output_layer
						neural->_map[tid].error[0] = (neural->_map[tid].data[0] - neural->_map[tid].label[0]) * tan_h_1(neural->_map[tid].data[0]);
					}
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
				if (this->mode == LayerMode::Normal) {
					ans += 0.5 * (neural->_value[tid] - neural->_output[tid]) * (neural->_value[tid] - neural->_output[tid]) / this->neurals.linkcount;
				}
				else {
					ans += 0.5 * (neural->_map[tid].label[0] - neural->_map[tid].data[0])* (neural->_map[tid].label[0] - neural->_map[tid].data[0]) / this->neurals.linkcount;
				}

				neural = this->neurals.next(neural);
			} while (neural && neural != this->neurals.link);
		}

		return ans;
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
								conn->weight += learning_rate * _neural->delta * neural->output;// *eva_fun_1(_neural->output);//_neural->delta * neural->output;
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

	/* scale: -0.8 ~ 0.8 和label初始值对应 */
	EFTYPE tan_h(double val)
	{
		double ep = exp(val);
		double em = exp(-val);

		return (ep - em) / (ep + em);
	}

	EFTYPE tan_h_1(double val)
	{
		return 1.0 - val * val;
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
	void clear() {
		void * _ptr = this;
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
		delete(this);
	}
};

#endif