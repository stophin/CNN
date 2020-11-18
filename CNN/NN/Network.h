//Network.h
//

#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "Layer.h"

#ifdef _CNN_SHOW_LOSS_
extern const int show_width;
extern const int show_height;
#endif

struct ThreadParam{
	int tid;
	int size;
	int in_size;
	int out_size;
	int* indice;
	int start;
	int end;
	double** X;
	double** Y;
	Sample * sample;
	int tc;
	void *nets;
	HANDLE_MUTEX mutex;
	HANDLE_MUTEX main_mutex;
	HANDLE thread;
	double error;
};

class Network {
public:
	Network() :
		divrange(1),
		divoutrange(1),
		hiddens(0),
		layers(1),
		input(*(new Layer())),
		output(*(new Layer())){
		layers.insertLink(&input);
		layers.insertLink(&output);
	}
	Network(LayerMode mode, LayerMode mode1) :
		divrange(1),
		divoutrange(1),
		hiddens(0),
		layers(1),
		input(*(new Layer(mode))),
		output(*(new Layer(mode1))) {
		layers.insertLink(&input);
		layers.insertLink(&output);
	}
	~Network() {
		hiddens.~MultiLinkList();
		layers.~MultiLinkList();
	}

	void setLearningRate(EFTYPE r) {
		Layer * layer = this->layers.link;
		if (layer) {
			do {

				layer->learning_rate = r;

				layer = this->layers.next(layer);
			} while (layer && layer != this->layers.link);
		}
	}

	EFTYPE divrange;
	EFTYPE divoutrange;

	Layer &input;
	Layer &output;
	MultiLinkList<Layer> hiddens;
	//manage all the layers
	MultiLinkList<Layer> layers;

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
		output.getDelta(output.mode);
		Layer * _hidden = this->hiddens.prev(this->hiddens.link);
		Layer * hidden = _hidden;
		if (hidden) {
			do {
				Layer *__hidden = this->hiddens.next(hidden);
				if (__hidden == this->hiddens.link) {
					__hidden = &output;
				}
				hidden->getDelta(__hidden->mode);

				hidden = this->hiddens.prev(hidden);
			} while (hidden && hidden != _hidden);
		}
	}

	void UpdateNetwork() {
		//output.adjustBias();
		output.adjustWeight();
		Layer * _hidden = this->hiddens.prev(this->hiddens.link);
		Layer * hidden = _hidden;
		if (hidden) {
			do {
				hidden->adjustWeight();
				//hidden->adjustBias();

				hidden = this->hiddens.prev(hidden);
			} while (hidden && hidden != _hidden);
		}
		input.adjustWeight();
	}

	void Scale(EFTYPE dir, EFTYPE dor) {
		this->divrange = dir;
		this->divoutrange = dor ;
	}

	void Train() {

		//set scale
		input.setScale(1.0 / divrange);
		output.setScale(1.0 / divoutrange);

		printf("Target:");
		Neural * neural = input.neurals.link;
		if (neural) {
			do {

				printf("%e ", neural->value * divrange);

				neural = input.neurals.next(neural);
			} while (neural && neural != input.neurals.link);
		}
		printf("->");
		neural = output.neurals.link;
		if (neural) {
			do {

				printf("%e ", neural->value * divoutrange);

				neural = output.neurals.next(neural);
			} while (neural && neural != output.neurals.link);
		}
		printf("\n");


		INT count = 0;
		EFTYPE error;
		while (count < T_TIMES) {
			ForwardTransfer();

			error = output.getError() * divoutrange * divoutrange;
			if (error < T_ERROR) {
				break;
			}
			neural = input.neurals.link;
			if (neural) {
				do {

					printf("%e ", neural->value * divrange);

					neural = input.neurals.next(neural);
				} while (neural && neural != input.neurals.link);
			}
			neural = output.neurals.link;
			if (neural) {
				do {

					printf(" %e->%e", neural->value * divoutrange, neural->output * divoutrange);

					neural = output.neurals.next(neural);
				} while (neural && neural != output.neurals.link);
			}
			printf("[%5d]Error is: %e\r", count, error);
			count++;

			ReverseTrasfer();
		}

		printf("\nTarget:");
		neural = input.neurals.link;
		if (neural) {
			do {

				printf("%e ", neural->value * divrange);

				neural = input.neurals.next(neural);
			} while (neural && neural != input.neurals.link);
		}
		printf("->");
		neural = output.neurals.link;
		if (neural) {
			do {

				printf("%e ", neural->value * divoutrange);

				neural = output.neurals.next(neural);
			} while (neural && neural != output.neurals.link);
		}
		printf("\n");
		printf("Result:");
		neural = output.neurals.link;
		if (neural) {
			do {

				printf("%e ", neural->output * divoutrange);

				neural = output.neurals.next(neural);
			} while (neural && neural != output.neurals.link);
		}
		printf("\n");
		printf("[%5d]Error is: %e\n", count, error);
	}

	void Train(double **X, double **Y, int size, int in_size, int out_size, double threshold) {
		EFTYPE error;
		Layer * layer;
		while (true) {
			//initialize delta sum
			input.resetDeltaSum();
			layer = hiddens.link;
			if (layer) {
				do {
					layer->resetDeltaSum();

					layer = hiddens.next(layer);
				} while (layer && layer != hiddens.link);
			}
			output.resetBiasSum();

			error = 0;
			for (int iter = 0; iter < size; iter++) {
				input.setNeural((double*)((double*)X + iter * in_size), in_size);
				output.setNeural((double*)((double*)Y + iter * out_size), out_size);

				input.setScale(1.0 / divrange);
				output.setScale(1.0 / divoutrange);

				ForwardTransfer();
				GetDelta();
				error += output.getError() * divrange * divoutrange;

				input.updateDeltaSum();
				layer = hiddens.link;
				if (layer) {
					do {
						layer->updateDeltaSum();

						layer = hiddens.next(layer);
					} while (layer && layer != hiddens.link);
				}
				output.updateBiasSum();
			}

			input.updateWeightWithDeltaSum(size);
			layer = hiddens.link;
			if (layer) {
				do {
					layer->updateWeightWithDeltaSum(size);

					layer = hiddens.next(layer);
				} while (layer && layer != hiddens.link);
			}
			output.updateBiasWithBiasSum(size);

			printf("[ 0]Error is: %e\r", error);
			if (error < threshold) {
				printf("\n");
				break;
			}
		}
	}
	void TrainRNN(double **X, double **Y, int size, int in_size, int out_size, double threshold, int serial_size) {
		EFTYPE error;
		Layer * layer;
		Layer *hidden, *_hidden;
		Neural * neural, *_neural;
		Connector *conn, *_conn;
		NeuralGate *pgate, *lpgate;
		//empty gate
		hidden = hiddens.link;
		if (hidden) {
			do {

				neural = hidden->neurals.link;
				if (neural) {
					do {
						pgate = new NeuralGate();
						pgate->t = -1;
						neural->gates.insertLink(pgate);
						neural->gate = pgate;

						neural = hidden->neurals.next(neural);
					} while (neural && neural != hidden->neurals.link);
				}

				hidden = hiddens.next(hidden);
			} while (hidden && hidden != hiddens.link);
		}
		//serial_size gates
		for (int i = 0; i < serial_size; i++) {
			//input
			neural = input.neurals.link;
			if (neural) {
				do {
					pgate = new NeuralGate();
					pgate->t = i;
					neural->gates.insertLink(pgate);
					neural->gate = pgate;

					neural = input.neurals.next(neural);
				} while (neural && neural != input.neurals.link);
			}
			//output
			neural = output.neurals.link;
			if (neural) {
				do {
					pgate = new NeuralGate();
					pgate->t = i;
					neural->gates.insertLink(pgate);
					neural->gate = pgate;

					neural = output.neurals.next(neural);
				} while (neural && neural != output.neurals.link);
			}
			//output layer
			pgate = new NeuralGate();
			pgate->t = i;
			output.gates.insertLink(pgate);
			output.gate = pgate;
			//hidden
			hidden = hiddens.link;
			if (hidden) {
				do {

					neural = hidden->neurals.link;
					if (neural) {
						do {

							pgate = new NeuralGate();
							pgate->t = i;
							neural->gates.insertLink(pgate);
							neural->gate = pgate;

							neural = hidden->neurals.next(neural);
						} while (neural && neural != hidden->neurals.link);
					}

					hidden = hiddens.next(hidden);
				} while (hidden && hidden != hiddens.link);
			}
		}
		//empty gate
		hidden = hiddens.link;
		if (hidden) {
			do {

				neural = hidden->neurals.link;
				if (neural) {
					do {
						pgate = new NeuralGate();
						pgate->t = -1;
						neural->gates.insertLink(pgate);
						neural->gate = pgate;

						neural = hidden->neurals.next(neural);
					} while (neural && neural != hidden->neurals.link);
				}

				hidden = hiddens.next(hidden);
			} while (hidden && hidden != hiddens.link);
		}
		//hidden gates and output layer gates ara as follows:
		//hidden: | empty gate | t=0   gate | t=1   gate | t=2   gate | ......     | t-1   gate | t     gate | empty gate |
		//output:                           | t=0   gate | t=1   gate | t=2   gate | ......     | t-1   gate | t     gate |

		int count = 0;
		while (true) {
			error = 0;
			for (int iter = 0; iter < size; iter++) {
				//reset gate to t=0
				//input
				neural = input.neurals.link;
				if (neural) {
					do {
						neural->gate = neural->gates.link;

						neural = input.neurals.next(neural);
					} while (neural && neural != input.neurals.link);
				}
				//output
				neural = output.neurals.link;
				if (neural) {
					do {
						neural->gate = neural->gates.link;

						neural = output.neurals.next(neural);
					} while (neural && neural != output.neurals.link);
				}
				//output layer
				output.gate = output.gates.link;
				//hidden
				hidden = hiddens.link;
				if (hidden) {
					do {

						neural = hidden->neurals.link;
						if (neural) {
							do {
								//skip empty gate
								neural->gate = neural->gates.next(neural->gates.link);

								neural = hidden->neurals.next(neural);
							} while (neural && neural != hidden->neurals.link);
						}

						hidden = hiddens.next(hidden);
					} while (hidden && hidden != hiddens.link);
				}
				int i, j, k;
				//for each serial_size
				for (int p = 0; p < serial_size; p++) {
					//put input
					i = 0;
					neural = input.neurals.link;
					if (neural) {
						do {
							neural->value = *(double*)((double*)X + iter * in_size * serial_size + i * serial_size + p);
							if (neural->gate) {
								neural->gate->in = neural->value;
							}
							i++;

							neural = input.neurals.next(neural);
						} while (neural && neural != input.neurals.link);
					}
					//put output
					i = 0;
					neural = output.neurals.link;
					if (neural) {
						do {
							neural->value = *(double*)((double*)Y + iter * out_size * serial_size + i * serial_size + p);
							if (neural->gate) {
								neural->gate->in = neural->value;
							}
							i++;

							neural = output.neurals.next(neural);
						} while (neural && neural != output.neurals.link);
					}

					input.setScale(1.0 / divrange);
					output.setScale(1.0 / divoutrange);

					//forawrd transfer
					//input
					neural = input.neurals.link;
					if (neural) {
						do {
							int c = 0;
							EFTYPE t;
							if (c > 0) {
							}
							else {
								//input layer
								t = neural->value;
							}
							neural->output = t;
							if (neural->gate) {
								neural->gate->out = neural->output;
							}

							neural = input.neurals.next(neural);
						} while (neural && neural != input.neurals.link);
					}
					//hidden
					hidden = hiddens.link;
					if (hidden) {
						do {

							neural = hidden->neurals.link;
							if (neural) {
								do {
									double inGate = 0.0;
									double outGate = 0.0;
									double forgetGate = 0.0;
									double gGate = 0.0;

									//from input
									conn = neural->conn.link;
									if (conn) {
										do {
											//for all neural that links before this neural
											if (conn->forw == neural) {
												_neural = conn->back;
												if (_neural) {
													inGate += _neural->output * conn->gate.W_I;
													outGate += _neural->output * conn->gate.W_O;
													forgetGate += _neural->output * conn->gate.W_F;
													gGate += _neural->output * conn->gate.W_G;
												}
											}

											conn = neural->conn.next(conn);
										} while (conn && conn != neural->conn.link);
									}
									
									//for hidden itself
									conn = neural->rconn.link;
									if (conn) {
										do {
											//for all neural that links after this neural
											if (conn->back == neural) {
												_neural = conn->forw;
												if (_neural) {
													if (_neural->gate) {
														//get previous gate
														NeuralGate& prev_gate = *_neural->gates.prev(_neural->gate);
														inGate += prev_gate.h * conn->gate.U_I;
														outGate += prev_gate.h * conn->gate.U_O;
														forgetGate += prev_gate.h * conn->gate.U_F;
														gGate += prev_gate.h * conn->gate.U_G;
													}
												}
											}

											conn = neural->rconn.next(conn);
										} while (conn && conn != neural->rconn.link);
									}

									//accumulate
									if (neural->gate) {
										NeuralGate& gate = *neural->gate;
										gate.in_gate = hidden->sigmod(inGate);
										gate.out_gate = hidden->sigmod(outGate);
										gate.forget_gate = hidden->sigmod(forgetGate);
										gate.g_gate = hidden->sigmod(gGate);

										NeuralGate& prev_gate = *neural->gates.prev(neural->gate);
										gate.state = gate.forget_gate * prev_gate.state + gate.g_gate * gate.in_gate;
										gate.h = gate.in_gate * hidden->tan_h(gate.state);
									}

									neural = hidden->neurals.next(neural);
								} while (neural && neural != hidden->neurals.link);
							}

							hidden = hiddens.next(hidden);
						} while (hidden && hidden != hiddens.link);
					}

					//output
					neural = output.neurals.link;
					if (neural) {
						do {
							INT c = 0;
							EFTYPE t = 0;
							INT size = 0;
							conn = neural->conn.link;
							if (conn) {
								do {
									//for all neural that links before this neural
									if (conn->forw == neural) {
										Neural * _neural = conn->back;
										if (_neural) {
											if (_neural->gate) {
												NeuralGate& gate = *_neural->gate;
												t += conn->weight * gate.h;
											}
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
								//t += neural->bias;
								neural->sum = t;
								//t = output.eva_fun(t);
								t = output.sigmod(t);
							}
							else {
								//input layer
								t = neural->value;
							}
							neural->output = t;
							if (neural->gate) {
								neural->gate->out = neural->output;
							}

							neural = output.neurals.next(neural);
						} while (neural && neural != output.neurals.link);
					}

					//delta
					if (output.gate) {
						NeuralGate& gate = *output.gate;
						EFTYPE e = 0;
						gate.y_delta = 0;
						neural = output.neurals.link;
						if (neural) {
							do {
								gate.y_delta += (neural->value - neural->output) * output.sigmod_1(neural->output);
								e += fabs(neural->value - neural->output);

								neural = output.neurals.next(neural);
							} while (neural && neural != output.neurals.link);
						}
						error += e;
					}

					//move forawrd t+1
					//input
					neural = input.neurals.link;
					if (neural) {
						do {
							if (neural->gate) {
								neural->gate = neural->gates.next(neural->gate);
							}

							neural = input.neurals.next(neural);
						} while (neural && neural != input.neurals.link);
					}
					//output
					neural = output.neurals.link;
					if (neural) {
						do {
							if (neural->gate) {
								neural->gate = neural->gates.next(neural->gate);
							}

							neural = output.neurals.next(neural);
						} while (neural && neural != output.neurals.link);
					}
					//output layer
					output.gate = output.gates.next(output.gate);
					//hidden
					hidden = hiddens.link;
					if (hidden) {
						do {

							neural = hidden->neurals.link;
							if (neural) {
								do {
									if (neural->gate) {
										neural->gate = neural->gates.next(neural->gate);
									}

									neural = hidden->neurals.next(neural);
								} while (neural && neural != hidden->neurals.link);
							}

							hidden = hiddens.next(hidden);
						} while (hidden && hidden != hiddens.link);
					}
				}

				//backward propagation
				//move gate to t
				//input
				neural = input.neurals.link;
				if (neural) {
					do {
						neural->gate = neural->gates.prev(neural->gates.link);

						neural = input.neurals.next(neural);
					} while (neural && neural != input.neurals.link);
				}
				//output
				neural = output.neurals.link;
				if (neural) {
					do {
						neural->gate = neural->gates.prev(neural->gates.link);

						neural = output.neurals.next(neural);
					} while (neural && neural != output.neurals.link);
				}
				//output layer
				output.gate = output.gates.prev(output.gates.link);
				//hidden
				hidden = hiddens.link;
				if (hidden) {
					do {

						neural = hidden->neurals.link;
						if (neural) {
							do {
								//skip empty gate
								neural->gate = neural->gates.prev(neural->gates.prev(neural->gates.link));

								neural = hidden->neurals.next(neural);
							} while (neural && neural != hidden->neurals.link);
						}

						hidden = hiddens.next(hidden);
					} while (hidden && hidden != hiddens.link);
				}
				//for each t gate
				pgate = output.gate;
				if (pgate) {
					do {
						//output
						neural = output.neurals.link;
						if (neural) {
							do {
								conn = neural->conn.link;
								if (conn) {
									do {
										//for all neural that links before this neural
										if (conn->forw == neural) {
											Neural * _neural = conn->back;
											if (_neural) {
												if (_neural->gate) {
													NeuralGate& gate = *_neural->gate;
													conn->weight += output.learning_rate * pgate->y_delta * gate.h;
												}
											}
										}

										conn = neural->conn.next(conn);
									} while (conn && conn != neural->conn.link);
								}

								neural = output.neurals.next(neural);
							} while (neural && neural != output.neurals.link);
						}


						//hidden
						hidden = hiddens.link;
						if (hidden) {
							do {

								neural = hidden->neurals.link;
								if (neural) {
									do {
										pgate->h_delta = 0;
										//from output
										conn = neural->conn.link;
										if (conn) {
											do {
												//for all neural that links after this neural
												if (conn->back == neural) {
													_neural = conn->forw;
													if (_neural) {
														pgate->h_delta += pgate->y_delta * conn->weight;
													}
												}

												conn = neural->conn.next(conn);
											} while (conn && conn != neural->conn.link);
										}

										//for hidden itself
										conn = neural->rconn.link;
										if (conn) {
											do {
												//for all neural that links after this neural
												if (conn->back == neural) {
													_neural = conn->forw;
													if (_neural) {
														if (_neural->gate) {
															//get next gate
															NeuralGate& future_gate = *_neural->gates.next(_neural->gate);
															pgate->h_delta += future_gate.I_delta * conn->gate.U_I;
															pgate->h_delta += future_gate.F_delta * conn->gate.U_F;
															pgate->h_delta += future_gate.O_delta * conn->gate.U_O;
															pgate->h_delta += future_gate.G_delta * conn->gate.U_G;
														}
													}
												}

												conn = neural->rconn.next(conn);
											} while (conn && conn != neural->rconn.link);
										}

										//accumulate delta
										if (neural->gate) {
											NeuralGate& gate = *neural->gate;
											NeuralGate& prev_gate = *neural->gates.prev(neural->gate);
											NeuralGate& future_gate = *neural->gates.next(neural->gate);
											gate.O_delta = pgate->h_delta * hidden->tan_h(gate.state) * hidden->sigmod_1(gate.out_gate);
											gate.state_delta = pgate->h_delta * gate.out_gate * hidden->tan_h_1(hidden->tan_h(gate.state)) +
												future_gate.state_delta * future_gate.forget_gate;
											gate.F_delta = gate.state_delta * prev_gate.state * hidden->sigmod_1(gate.forget_gate);
											gate.I_delta = gate.state_delta * gate.g_gate * hidden->sigmod_1(gate.in_gate);
											gate.G_delta = gate.state_delta * gate.in_gate * hidden->sigmod_1(gate.g_gate);

											//adjust weight
											//for hidden itself
											conn = neural->rconn.link;
											if (conn) {
												do {
													//for all neural that links after this neural
													if (conn->back == neural) {
														_neural = conn->forw;
														if (_neural) {
															if (_neural->gate) {
																conn->gate.U_I += hidden->learning_rate * gate.I_delta * prev_gate.h;
																conn->gate.U_F += hidden->learning_rate * gate.F_delta * prev_gate.h;
																conn->gate.U_O += hidden->learning_rate * gate.O_delta * prev_gate.h;
																conn->gate.U_G += hidden->learning_rate * gate.G_delta * prev_gate.h;
															}
														}
													}

													conn = neural->rconn.next(conn);
												} while (conn && conn != neural->rconn.link);
											}
											//for input
											conn = neural->conn.link;
											if (conn) {
												do {
													//for all neural that links before this neural
													if (conn->forw == neural) {
														_neural = conn->back;
														if (_neural) {
															conn->gate.W_I += hidden->learning_rate * gate.I_delta * _neural->output;
															conn->gate.W_F += hidden->learning_rate * gate.F_delta * _neural->output;
															conn->gate.W_O += hidden->learning_rate * gate.O_delta * _neural->output;
															conn->gate.W_G += hidden->learning_rate * gate.G_delta * _neural->output;
														}
													}

													conn = neural->conn.next(conn);
												} while (conn && conn != neural->conn.link);
											}
										}

										neural = hidden->neurals.next(neural);
									} while (neural && neural != hidden->neurals.link);
								}

								hidden = hiddens.next(hidden);
							} while (hidden && hidden != hiddens.link);
						}

						//move to t - 1
						//input
						neural = input.neurals.link;
						if (neural) {
							do {
								if (neural->gate) {
									neural->gate = neural->gates.prev(neural->gate);
								}

								neural = input.neurals.next(neural);
							} while (neural && neural != input.neurals.link);
						}
						//output
						neural = output.neurals.link;
						if (neural) {
							do {
								if (neural->gate) {
									neural->gate = neural->gates.prev(neural->gate);
								}

								neural = output.neurals.next(neural);
							} while (neural && neural != output.neurals.link);
						}
						//hidden
						hidden = hiddens.link;
						if (hidden) {
							do {

								neural = hidden->neurals.link;
								if (neural) {
									do {
										if (neural->gate) {
											neural->gate = neural->gates.prev(neural->gate);
										}

										neural = hidden->neurals.next(neural);
									} while (neural && neural != hidden->neurals.link);
								}

								hidden = hiddens.next(hidden);
							} while (hidden && hidden != hiddens.link);
						}

						pgate = output.gates.prev(pgate);
					} while (pgate && pgate != output.gate);
				}

				if (iter % 1000 == 0) {
					printf("\n");
					int out = 0;
					for (int i = 0; i < in_size; i++) {
						int in = 0;
						for (int p = serial_size - 1; p >= 0; p--) {
							double value = *(double*)((double*)X + iter * in_size * serial_size + i * serial_size + p);
							printf("%.2f ", value);
							in += value * pow(2, p);
						}
						printf("\t%d\n", in);
						out += in;
					}
					printf("\t\t\t\t%d\n", out);
					for (int i = 0; i < out_size; i++) {
						int in = 0;
						for (int p = serial_size - 1; p >= 0; p--) {
							double value = *(double*)((double*)Y + iter * out_size * serial_size + i * serial_size + p);
							printf("%.2f ", value);
							in += value * pow(2, p);
						}
						printf("\t%d\n", in);
					}
					//put input
					out = 0;
					neural = input.neurals.link;
					if (neural) {
						do {
							//for each serial
							int in = 0;
							int p = serial_size - 1;
							lpgate = neural->gates.prev(neural->gates.link);
							pgate = lpgate;
							if (pgate) {
								do {
									double value = pgate->out;
									printf("%.2f ", value);
									in += value * pow(2, p);
									p--;

									pgate = neural->gates.prev(pgate);
								} while (pgate && pgate != lpgate);
							}
							printf("\t%d\n", in);
							out += in;

							neural = input.neurals.next(neural);
						} while (neural && neural != input.neurals.link);
					}
					printf("\t\t\t\t%d\n", out);
					//put output
					neural = output.neurals.link;
					if (neural) {
						do {
							//for each serial
							int in = 0;
							int p = serial_size - 1;
							lpgate = neural->gates.prev(neural->gates.link);
							pgate = lpgate;
							if (pgate) {
								do {
									double value = pgate->out;
									printf("%.2f ", value);
									in += value * pow(2, p);
									p--;

									pgate = neural->gates.prev(pgate);
								} while (pgate && pgate != lpgate);
							}
							printf("\t%d\n", in);

							neural = output.neurals.next(neural);
						} while (neural && neural != output.neurals.link);
					}

					//put output
					neural = output.neurals.link;
					if (neural) {
						do {
							//for each serial
							int in = 0;
							int p = serial_size - 1;
							lpgate = neural->gates.prev(neural->gates.link);
							pgate = lpgate;
							if (pgate) {
								do {
									double value = (int)(pgate->out + 0.5);
									printf("%.2f ", value);
									in += value * pow(2, p);
									p--;

									pgate = neural->gates.prev(pgate);
								} while (pgate && pgate != lpgate);
							}
							printf("\t%d\n", in);

							neural = output.neurals.next(neural);
						} while (neural && neural != output.neurals.link);
					}
					getch();
				}

				printf("[ %d/%d]Error is: %e\r", iter, count, error);
				if (error < threshold) {
					printf("\n");
					break;
				}
			}
			count++;
		}

		//release
		//input
		neural = input.neurals.link;
		if (neural) {
			do {
				neural->gates.~MultiLinkList();

				neural = input.neurals.next(neural);
			} while (neural && neural != input.neurals.link);
		}
		//output
		neural = output.neurals.link;
		if (neural) {
			do {
				neural->gates.~MultiLinkList();

				neural = output.neurals.next(neural);
			} while (neural && neural != output.neurals.link);
		}
		//output layer
		output.gates.~MultiLinkList();
		//hidden
		hidden = hiddens.link;
		if (hidden) {
			do {

				neural = hidden->neurals.link;
				if (neural) {
					do {

						neural->gates.~MultiLinkList();

						neural = hidden->neurals.next(neural);
					} while (neural && neural != hidden->neurals.link);
				}

				hidden = hiddens.next(hidden);
			} while (hidden && hidden != hiddens.link);
		}
	}

	void TrainCNN(Sample* sample, int size, int train_size, int in_size, int out_size, double threshold) {
		EFTYPE error;
		Layer * layer, *_hidden, *__hidden, *hidden;
		while (true) {
			//initialize delta sum
			input.resetDeltaSum();
			layer = hiddens.link;
			if (layer) {
				do {
					layer->resetDeltaSum();

					layer = hiddens.next(layer);
				} while (layer && layer != hiddens.link);
			}
			output.resetBiasSum();



			error = 0;
			for (int iter = 0; iter < size; iter++) {
				input.setNeuralMatrix(sample[iter].data, in_size);
				output.setNeural(sample[iter].label, out_size);

				input.setScale(1.0 / divrange);
				output.setScale(1.0 / divoutrange);

				ForwardTransfer();
				output.getDelta(output.mode);
				_hidden = this->hiddens.prev(this->hiddens.link);
				hidden = _hidden;
				if (hidden) {
					do {
						__hidden = this->hiddens.next(hidden);
						if (__hidden == this->hiddens.link) {
							__hidden = &output;
						}
						hidden->getDelta(__hidden->mode);

						hidden = this->hiddens.prev(hidden);
					} while (hidden && hidden != _hidden);
				}
				input.getDelta(this->hiddens.link->mode);

				error += output.getError() * divrange * divoutrange;

				output.updateDeltaSum(output.mode);
				_hidden = this->hiddens.prev(this->hiddens.link);
				hidden = _hidden;
				if (hidden) {
					do {
						Layer *__hidden = this->hiddens.next(hidden);
						if (__hidden == this->hiddens.link) {
							__hidden = &output;
						}
						hidden->updateDeltaSum(__hidden->mode);

						hidden = this->hiddens.prev(hidden);
					} while (hidden && hidden != _hidden);
				}
				input.updateDeltaSum(this->hiddens.link->mode);
			}

			input.updateWeightWithDeltaSum(size);
			layer = hiddens.link;
			if (layer) {
				do {
					layer->updateWeightWithDeltaSum(size);

					layer = hiddens.next(layer);
				} while (layer && layer != hiddens.link);
			}
			output.updateBiasWithBiasSum(size);

			printf("[ 0]Error is: %e\n", error);
			if (error < threshold) {
				printf("\n");
				break;
			}
		}
	}
	static __NANOC_THREAD_FUNC_BEGIN__(TrainThread) {
		ThreadParam *param = (ThreadParam*)pv;
		int tid = param->tid;
		int in_size = param->in_size;
		int out_size = param->out_size;
		int start = param->start;
		int end = param->end;
		double **X = param->X;
		double **Y = param->Y;
		Network * nets = (Network*)param->nets;
		HANDLE_MUTEX &mutex = param->mutex;
		HANDLE_MUTEX &main_mutex = param->main_mutex;
		Layer &input = nets->input;
		Layer &output = nets->output;
		Layer * hidden, *_hidden, *layer;
		EFTYPE divrange = nets->divrange;
		EFTYPE divoutrange = nets->divoutrange;
		EFTYPE &error = param->error;

		while (true) {
			__NANOC_THREAD_MUTEX_LOCK__(mutex);
			tid = param->tid;
			if (tid < 0) {
				__NANOC_THREAD_MUTEX_UNLOCK__(mutex);
				__NANOC_THREAD_MUTEX_UNLOCK__(main_mutex);
				break;
			}
			//printf("tid:%d\n", tid);
			for (int iter = start; iter < end; iter++) {
				int ind = param->indice[iter];
				input.setNeural((double*)((double*)X + ind * in_size), in_size, tid);
				output.setNeural((double*)((double*)Y + ind * out_size), out_size, tid);

				input.setScale(1.0 / divrange, tid);
				output.setScale(1.0 / divoutrange, tid);

				//ForwardTransfer();
				input.getOutput(tid);
				hidden = nets->hiddens.link;
				if (hidden) {
					do {
						hidden->getOutput(tid);

						hidden = nets->hiddens.next(hidden);
					} while (hidden && hidden != nets->hiddens.link);
				}
				output.getOutput(tid);
				//GetDelta();
				output.getDelta(tid);
				_hidden = nets->hiddens.prev(nets->hiddens.link);
				hidden = _hidden;
				if (hidden) {
					do {
						hidden->getDelta(tid);

						hidden = nets->hiddens.prev(hidden);
					} while (hidden && hidden != _hidden);
				}

				error += output.getError(tid) * divrange * divoutrange;

				input.updateDeltaSum(tid);
				layer = nets->hiddens.link;
				if (layer) {
					do {
						layer->updateDeltaSum(tid);

						layer = nets->hiddens.next(layer);
					} while (layer && layer != nets->hiddens.link);
				}
				output.updateBiasSum(tid);
			}
			__NANOC_THREAD_MUTEX_UNLOCK__(main_mutex);
		}

		__NANOC_THREAD_FUNC_END__(0);
	}

	void Train(double **X, double **Y, int size, int in_size, int out_size, double threshold, int thx, int thy, int sample_size = 0, int train_count = 0) {
		int tc = thx * thy;
		EFTYPE error;
		Layer * layer;
		Layer * hidden, *_hidden;
		//alloc and initialize delta sum
		input.resetDeltaSum(tc, 1);
		layer = hiddens.link;
		if (layer) {
			do {
				layer->resetDeltaSum(tc, 1);

				layer = hiddens.next(layer);
			} while (layer && layer != hiddens.link);
		}
		output.resetBiasSum(tc, 1);
		//alloc and initialize neural
		input.resetNeural(tc, 1);
		layer = hiddens.link;
		if (layer) {
			do {
				layer->resetNeural(tc, 1);

				layer = hiddens.next(layer);
			} while (layer && layer != hiddens.link);
		}
		output.resetNeural(tc, 1);
		//alloc and initialze delta
		input.resetDelta(tc, 1);
		layer = hiddens.link;
		if (layer) {
			do {
				layer->resetDelta(tc, 1);

				layer = hiddens.next(layer);
			} while (layer && layer != hiddens.link);
		}
		output.resetDelta(tc, 1);

		//adjust parameter
		if (sample_size <= 0) {
			sample_size = size;
		}

		//init thread
		ThreadParam * params = new ThreadParam[tc];
		int *indice = new int[size];
		int *indice_root = new int[sample_size];
		int div = size / tc;
		int divl = size - div * tc;
		int divd = 0;
		for (int i = 0; i < tc; i++) {
			params[i].tid = i;
			params[i].size = size;
			params[i].in_size = in_size;
			params[i].out_size = out_size;
			params[i].indice = indice;
			params[i].X = X;
			params[i].Y = Y;
			params[i].tc = tc;
			params[i].nets = this;
			params[i].start = (div > 0 ? div * i + divd : i);
			params[i].end = params[i].start + div;
			if (divl > 0) {
				params[i].end++;
				divl--;
				divd++;
			}
			__NANOC_THREAD_MUTEX_INIT__(mutex, (&params[i]));
			__NANOC_THREAD_MUTEX_INIT__(main_mutex, (&params[i]));
			__NANOC_THREAD_MUTEX_LOCK__(params[i].mutex);
			__NANOC_THREAD_MUTEX_LOCK__(params[i].main_mutex);
			__NANOC_THREAD_BEGIN__(params[i].thread, TrainThread, &params[i]);
			printf("%d %d %d\n", i, params[i].start, params[i].end);
		}
		//getch();


		//randomize
		srand(time(NULL));
		for (int i = 0; i < sample_size; i++) {
			indice_root[i] = i;
		}
		for (int i = sample_size - 1; i > 0; i--) {
			int ind = rand() % i;
			int t = indice_root[i];
			indice_root[i] = indice_root[ind];
			indice_root[ind] = t;
		}

		int count = 0;
		int count_div = sample_size / size;
		if (train_count > 0) {
			count_div = train_count;
		}
#ifdef _CNN_SHOW_LOSS_
		EFTYPE *show_error = new EFTYPE[count_div];
		EFTYPE max_error = 0;
		int lcount = 0;
#endif
		printf("%d %d %d\n", sample_size, size, count_div);
		while (true) {
			//endless loop when count_div is 1, say sample_size equals size
			if (count_div > 1 && count >= count_div) {
				break;
			}
			//initialze alloced delta sum
			input.resetDeltaSum(tc, 0);
			layer = hiddens.link;
			if (layer) {
				do {
					layer->resetDeltaSum(tc, 0);

					layer = hiddens.next(layer);
				} while (layer && layer != hiddens.link);
			}
			output.resetBiasSum(tc, 0);
			//initialize delta sum
			input.resetDeltaSum();
			layer = hiddens.link;
			if (layer) {
				do {
					layer->resetDeltaSum();

					layer = hiddens.next(layer);
				} while (layer && layer != hiddens.link);
			}
			output.resetBiasSum();

			/*
			int tid = 0;
			error = 0;
			for (int iter = 0; iter < size; iter++) {
				input.setNeural((double*)((double*)X + iter * in_size), in_size, tid);
				output.setNeural((double*)((double*)Y + iter * out_size), out_size, tid);

				input.setScale(1.0 / divrange, tid);
				output.setScale(1.0 / divoutrange, tid);

				//ForwardTransfer();
				input.getOutput(tid);
				hidden = this->hiddens.link;
				if (hidden) {
					do {
						hidden->getOutput(tid);

						hidden = this->hiddens.next(hidden);
					} while (hidden && hidden != this->hiddens.link);
				}
				output.getOutput(tid);
				//GetDelta();
				output.getDelta(tid);
				_hidden = this->hiddens.prev(this->hiddens.link);
				hidden = _hidden;
				if (hidden) {
					do {
						hidden->getDelta(tid);

						hidden = this->hiddens.prev(hidden);
					} while (hidden && hidden != _hidden);
				}

				error += output.getError(tid) * divrange * divoutrange;

				input.updateDeltaSum(tid);
				layer = hiddens.link;
				if (layer) {
					do {
						layer->updateDeltaSum(tid);

						layer = hiddens.next(layer);
					} while (layer && layer != hiddens.link);
				}
				output.updateBiasSum(tid);
			}*/
			//reset error
			error = 0;
			for (int i = 0; i < tc; i++) {
				params[i].error = 0;
			}

			//randomise indice
			int ind = rand() % sample_size;
			for (int i = 0; i < size; i++) {
				indice[i] = indice_root[(ind + i) % sample_size];
			}

			//release sem
			for (int i = 0; i < tc; i++) {
				__NANOC_THREAD_MUTEX_UNLOCK__(params[i].mutex);
			}

			//wait for sem
			for (int i = 0; i < tc; i++) {
				__NANOC_THREAD_MUTEX_LOCK__(params[i].main_mutex);
			}

			//accumulate error
			for (int i = 0; i < tc; i++) {
				error += params[i].error;
			}

			//accumulate delta sum
			input.accumulateDeltaSum(tc);
			layer = hiddens.link;
			if (layer) {
				do {
					layer->accumulateDeltaSum(tc);

					layer = hiddens.next(layer);
				} while (layer && layer != hiddens.link);
			}
			output.accumulateDeltaSum(tc);


			input.updateWeightWithDeltaSum(size);
			layer = hiddens.link;
			if (layer) {
				do {
					layer->updateWeightWithDeltaSum(size);

					layer = hiddens.next(layer);
				} while (layer && layer != hiddens.link);
			}
			output.updateBiasWithBiasSum(size);

#ifdef _CNN_SHOW_LOSS_
			if (count < count_div) {
				show_error[count] = error;
				if (max_error < error) {
					max_error = error;
				}
				if (count > 1) {
					EFTYPE width_r = (EFTYPE)count / show_width;
					if (count - lcount >= width_r) {
						lcount = count;
						EFTYPE height_r = (EFTYPE)max_error / show_height;
						EP_ClearDevice();
						EP_SetColor(WHITE);
						ege::setlinewidth(1);
						for (int i = 1; i < count; width_r > 1 ? i += (int)width_r : i++) {
							EP_Line(i / width_r, show_error[i] / height_r, (i - 1) / width_r, show_error[i - 1] / height_r);
						}
#endif
#ifdef _CNN_SHOW_GUI_
						width_r = (EFTYPE)sample_size / show_width;
						height_r = (EFTYPE)this->divoutrange / show_height;
						//EP_ClearDevice();
						int x, y, ex = 0, ey = 0, ex1 = 0, ey1 = 0;
						for (int i = 1; i < sample_size; i++) {
							input.setNeural((double*)((double*)X + i * in_size), in_size);
							Forecast(input, &output);
							x = i / width_r;
							y = *((double*)((double*)Y + i * out_size)) / height_r;
							if (x <= show_width && y <= show_height) {
								EP_SetColor(GREEN);
								EP_Line(x, y, ex, ey);
								ex = x;
								ey = y;
							}
							y = output.neurals.link->output / height_r;
							if (x <= show_width && y <= show_height) {
								EP_SetColor(RED);
								EP_Line(x, y, ex1, ey1);
								ex1 = x;
								ey1 = y;
							}
						}
#endif
#ifdef _CNN_SHOW_LOSS_
						EP_RenderFlush();
					}
				}
			}
#endif

			printf("[ %d]Error is: %e\r", count, error);
			count++;
			if (error < threshold) {
				printf("\n");
				break;
			}
		}
		delete[] indice_root;
		delete[] indice;
#ifdef _CNN_SHOW_LOSS_
		delete[] show_error;
#endif
		//release sem
		for (int i = 0; i < tc; i++) {
			//send end singal
			params[i].tid = -1;
			__NANOC_THREAD_MUTEX_UNLOCK__(params[i].mutex);
		}

		//wait for sem
		for (int i = 0; i < tc; i++) {
			__NANOC_THREAD_MUTEX_LOCK__(params[i].main_mutex);
		}
		//wait thread
		for (int i = 0; i < tc; i++) {
			__NANOC_THREAD_WAIT__(params[i].thread);
		}
		//end thread
		for (int i = 0; i < tc; i++) {
			//__NANOC_THREAD_END__(params[i].thread);
			__NANOC_THREAD_CLOSE__(params[i].thread);
			__NANOC_THREAD_CLOSE__(params[i].mutex);
			__NANOC_THREAD_CLOSE__(params[i].main_mutex);
		}

		//unalloc delta sum
		input.resetDeltaSum(tc, 2);
		layer = hiddens.link;
		if (layer) {
			do {
				layer->resetDeltaSum(tc, 2);

				layer = hiddens.next(layer);
			} while (layer && layer != hiddens.link);
		}
		output.resetBiasSum(tc, 2);
		//unalloc neural
		input.resetNeural(tc, 2);
		layer = hiddens.link;
		if (layer) {
			do {
				layer->resetNeural(tc, 2);

				layer = hiddens.next(layer);
			} while (layer && layer != hiddens.link);
		}
		output.resetNeural(tc, 2);
		//unalloc delta
		input.resetDelta(tc, 2);
		layer = hiddens.link;
		if (layer) {
			do {
				layer->resetDelta(tc, 2);

				layer = hiddens.next(layer);
			} while (layer && layer != hiddens.link);
		}
		output.resetDelta(tc, 2);
		//delete thread
		delete[]params;
	}



	static __NANOC_THREAD_FUNC_BEGIN__(TrainCNNThread) {
		ThreadParam *param = (ThreadParam*)pv;
		int tid = param->tid;
		int in_size = param->in_size;
		int out_size = param->out_size;
		int start = param->start;
		int end = param->end;
		Sample * sample = param->sample;
		Network * nets = (Network*)param->nets;
		HANDLE_MUTEX &mutex = param->mutex;
		HANDLE_MUTEX &main_mutex = param->main_mutex;
		Layer &input = nets->input;
		Layer &output = nets->output;
		Layer * hidden, *_hidden, *layer, *__hidden;
		EFTYPE divrange = nets->divrange;
		EFTYPE divoutrange = nets->divoutrange;
		EFTYPE &error = param->error;

		while (true) {
			__NANOC_THREAD_MUTEX_LOCK__(mutex);
			tid = param->tid;
			if (tid < 0) {
				__NANOC_THREAD_MUTEX_UNLOCK__(mutex);
				__NANOC_THREAD_MUTEX_UNLOCK__(main_mutex);
				break;
			}
			//printf("tid:%d\n", tid);
			for (int iter = start; iter < end; iter++) {
				int ind = param->indice[iter];
				input.setNeuralMatrix(sample[ind].data, in_size, tid);
				output.setNeural(sample[ind].label, out_size, tid);

				input.setScale(1.0 / divrange, tid);
				output.setScale(1.0 / divoutrange, tid);

				//ForwardTransfer();
				input.getOutput(tid);
				hidden = nets->hiddens.link;
				if (hidden) {
					do {
						hidden->getOutput(tid);

						hidden = nets->hiddens.next(hidden);
					} while (hidden && hidden != nets->hiddens.link);
				}
				output.getOutput(tid);
				//GetDelta();
				output.getDelta(tid, output.mode);
				_hidden = nets->hiddens.prev(nets->hiddens.link);
				hidden = _hidden;
				if (hidden) {
					do {
						__hidden = nets->hiddens.next(hidden);
						if (__hidden == nets->hiddens.link) {
							__hidden = &output;
						}
						hidden->getDelta(tid, __hidden->mode);

						hidden = nets->hiddens.prev(hidden);
					} while (hidden && hidden != _hidden);
				}
				input.getDelta(tid, nets->hiddens.link->mode);

				error += output.getError(tid) * divrange * divoutrange;

				output.updateDeltaSum(tid, output.mode);
				_hidden = nets->hiddens.prev(nets->hiddens.link);
				hidden = _hidden;
				if (hidden) {
					do {
						Layer *__hidden = nets->hiddens.next(hidden);
						if (__hidden == nets->hiddens.link) {
							__hidden = &output;
						}
						hidden->updateDeltaSum(tid, __hidden->mode);

						hidden = nets->hiddens.prev(hidden);
					} while (hidden && hidden != _hidden);
				}
				input.updateDeltaSum(tid, nets->hiddens.link->mode);
			}
			__NANOC_THREAD_MUTEX_UNLOCK__(main_mutex);
		}

		__NANOC_THREAD_FUNC_END__(0);
	}

	void TrainCNN(Sample *sample, int size, int train_size, int in_size, int out_size, double threshold, int thx, int thy, int train_count = 0) {
		int tc = thx * thy;
		EFTYPE error;
		Layer * layer;
		Layer * hidden, *_hidden;
		//alloc and initialize map and kernel, must be called first
		input.resetMapKernel(tc, 1);
		layer = hiddens.link;
		if (layer) {
			do {
				layer->resetMapKernel(tc, 1);

				layer = hiddens.next(layer);
			} while (layer && layer != hiddens.link);
		}
		output.resetMapKernel(tc, 1);
		//alloc and initialize delta sum
		input.resetDeltaSum(tc, 1);
		layer = hiddens.link;
		if (layer) {
			do {
				layer->resetDeltaSum(tc, 1);

				layer = hiddens.next(layer);
			} while (layer && layer != hiddens.link);
		}
		output.resetBiasSum(tc, 1);
		//alloc and initialize neural
		input.resetNeural(tc, 1);
		layer = hiddens.link;
		if (layer) {
			do {
				layer->resetNeural(tc, 1);

				layer = hiddens.next(layer);
			} while (layer && layer != hiddens.link);
		}
		output.resetNeural(tc, 1);
		//alloc and initialze delta
		input.resetDelta(tc, 1);
		layer = hiddens.link;
		if (layer) {
			do {
				layer->resetDelta(tc, 1);

				layer = hiddens.next(layer);
			} while (layer && layer != hiddens.link);
		}
		output.resetDelta(tc, 1);

		//init thread
		ThreadParam * params = new ThreadParam[tc];
		int *indice = new int[size];
		int *indice_root = new int[train_size];
		int div = size / tc;
		int divl = size - div * tc;
		int divd = 0;
		for (int i = 0; i < tc; i++) {
			params[i].tid = i;
			params[i].size = size;
			params[i].in_size = in_size;
			params[i].out_size = out_size;
			params[i].sample = sample;
			params[i].tc = tc;
			params[i].nets = this;
			params[i].start = (div > 0 ? div * i + divd : i);
			params[i].end = params[i].start + div;
			params[i].indice = indice;
			if (divl > 0) {
				params[i].end++;
				divl--;
				divd++;
			}
			__NANOC_THREAD_MUTEX_INIT__(mutex, (&params[i]));
			__NANOC_THREAD_MUTEX_INIT__(main_mutex, (&params[i]));
			__NANOC_THREAD_MUTEX_LOCK__(params[i].mutex);
			__NANOC_THREAD_MUTEX_LOCK__(params[i].main_mutex);
			__NANOC_THREAD_BEGIN__(params[i].thread, TrainCNNThread, &params[i]);
			printf("%d %d %d\n", i, params[i].start, params[i].end);
		}
		//getch();

		//randomize
		srand(time(NULL));
		for (int i = 0; i < train_size; i++) {
			indice_root[i] = i;
		}
		for (int i = train_size - 1; i > 0; i--) {
			int ind = rand() % i;
			int t = indice_root[i];
			indice_root[i] = indice_root[ind];
			indice_root[ind] = t;
		}

		int count = 0;
		int count_div = train_size / size;
		if (train_count > 0) {
			count_div = train_count;
		}
#ifdef _CNN_SHOW_LOSS_
		EFTYPE *show_error = new EFTYPE[count_div];
		EFTYPE max_error = 0;
#endif
		printf("%d %d %d\n", train_size, size, count_div);
#ifdef _CNN_SHOW_CONV_
		char c = 0;
#endif
		while (count < count_div) {
			//initialze alloced delta sum
			input.resetDeltaSum(tc, 0);
			layer = hiddens.link;
			if (layer) {
				do {
					layer->resetDeltaSum(tc, 0);

					layer = hiddens.next(layer);
				} while (layer && layer != hiddens.link);
			}
			output.resetBiasSum(tc, 0);
			//initialize delta sum
			input.resetDeltaSum();
			layer = hiddens.link;
			if (layer) {
				do {
					layer->resetDeltaSum();

					layer = hiddens.next(layer);
				} while (layer && layer != hiddens.link);
			}
			output.resetBiasSum();

			/*
			int tid = 0;
			error = 0;
			for (int iter = 0; iter < size; iter++) {
				input.setNeural((double*)((double*)X + iter * in_size), in_size, tid);
				output.setNeural((double*)((double*)Y + iter * out_size), out_size, tid);

				input.setScale(1.0 / divrange, tid);
				output.setScale(1.0 / divoutrange, tid);

				//ForwardTransfer();
				input.getOutput(tid);
				hidden = this->hiddens.link;
				if (hidden) {
					do {
						hidden->getOutput(tid);

						hidden = this->hiddens.next(hidden);
					} while (hidden && hidden != this->hiddens.link);
				}
				output.getOutput(tid);
				//GetDelta();
				output.getDelta(tid);
				_hidden = this->hiddens.prev(this->hiddens.link);
				hidden = _hidden;
				if (hidden) {
					do {
						hidden->getDelta(tid);

						hidden = this->hiddens.prev(hidden);
					} while (hidden && hidden != _hidden);
				}

				error += output.getError(tid) * divrange * divoutrange;

				input.updateDeltaSum(tid);
				layer = hiddens.link;
				if (layer) {
					do {
						layer->updateDeltaSum(tid);

						layer = hiddens.next(layer);
					} while (layer && layer != hiddens.link);
				}
				output.updateBiasSum(tid);
			}*/
			//reset error
			error = 0;
			for (int i = 0; i < tc; i++) {
				params[i].error = 0;
			}

			//set random
			int ind = rand() % train_size;
			for (int i = 0; i < size; i++) {
				indice[i] = indice_root[(ind + i) % train_size];
			}

			//release sem
			for (int i = 0; i < tc; i++) {
				__NANOC_THREAD_MUTEX_UNLOCK__(params[i].mutex);
			}

			//wait for sem
			for (int i = 0; i < tc; i++) {
				__NANOC_THREAD_MUTEX_LOCK__(params[i].main_mutex);
			}

			//accumulate error
			for (int i = 0; i < tc; i++) {
				error += params[i].error;
			}

			//accumulate delta sum
			input.accumulateDeltaSum(tc);
			layer = hiddens.link;
			if (layer) {
				do {
					layer->accumulateDeltaSum(tc);

					layer = hiddens.next(layer);
				} while (layer && layer != hiddens.link);
			}
			output.accumulateDeltaSum(tc);


			input.updateWeightWithDeltaSum(size);
			layer = hiddens.link;
			if (layer) {
				do {
					layer->updateWeightWithDeltaSum(size);

					layer = hiddens.next(layer);
				} while (layer && layer != hiddens.link);
			}
			output.updateBiasWithBiasSum(size);

#ifdef _CNN_SHOW_CONV_
			double scale_max = 1.0;
			double scale_min = -1.0;
			int l = 0;
			int n = 0;
			for (int t = 0; t < tc; t++) {
				if (c == 'c' || c == 'q') {
					break;
				}
				layer = this->layers.link;
				if (layer) {
					do {
						l++;
						n = 0;
						Neural *neural = layer->neurals.link;
						if (neural) {
							do {
								n++;
								EP_ClearDevice();
								EFTYPE width_r = (EFTYPE)neural->map_w / show_width;
								EFTYPE height_r = (EFTYPE)neural->map_h / show_height;
								for (int i = 0; i < show_width; i++) {
									for (int j = 0; j < show_height; j++) {
										int ind = (int)((INT)(j * height_r)* neural->map_h + (INT)(i * width_r));
										double v = 0;
										if (ind < neural->map_w * neural->map_h) {
											v = neural->_map[t].data[ind];
										}
										ECOLOR color = (ECOLOR)(((v - scale_min) / (scale_max - scale_min)) * 255.0);
										EP_SetPixel(i, j, color);
									}
								}
								EP_RenderFlush();

								printf("tid: %d/%d, layer: %d/%d, neural: %d/%d\n", t + 1, tc, l, this->layers.linkcount, n, layer->neurals.linkcount);
								if (!(c == 'q' || c == 'Q')) {
									c = getchar();
								}
								if (c == 'c' || c == 'q') {
									break;
								}
								if (c == 't' || c == 'T') {
									break;
								}
								if (c == 'q' || c == 'Q') {
									break;
								}

								neural = layer->neurals.next(neural);
							} while (neural && neural != layer->neurals.link);
						}

						if (c == 't' || c == 'T') {
							break;
						}
						if (c == 'q' || c == 'Q') {
							break;
						}

						layer = this->layers.next(layer);
					} while (layer && layer != this->layers.link);
				}
				if (c == 'q' || c == 'Q') {
					break;
				}
			}
#endif

#ifdef _CNN_SHOW_LOSS_
			if (count < count_div) {
				show_error[count] = error;
				if (max_error < error) {
					max_error = error;
				}
				if (count > 1) {
					EP_ClearDevice();
					EFTYPE width_r = (EFTYPE)count / show_width;
					EFTYPE height_r = (EFTYPE)max_error / show_height;
					EP_SetColor(WHITE);
					ege::setlinewidth(1);
					for (int i = 1; i < count; width_r > 1 ? i += (int)width_r : i++) {
						EP_Line(i / width_r, show_error[i] / height_r, (i - 1) / width_r, show_error[i - 1] / height_r);
					}
					EP_RenderFlush();
				}
			}
#endif

			printf("[ %d]Error is: %e\n", count, error);
			count++;
			if (error < threshold) {
				printf("\n");
				break;
			}
		}
#ifdef _CNN_SHOW_LOSS_
		delete[] show_error;
#endif
		//release sem
		for (int i = 0; i < tc; i++) {
			//send end singal
			params[i].tid = -1;
			__NANOC_THREAD_MUTEX_UNLOCK__(params[i].mutex);
		}

		//wait for sem
		for (int i = 0; i < tc; i++) {
			__NANOC_THREAD_MUTEX_LOCK__(params[i].main_mutex);
		}
		//wait thread
		for (int i = 0; i < tc; i++) {
			__NANOC_THREAD_WAIT__(params[i].thread);
		}
		//end thread
		for (int i = 0; i < tc; i++) {
			//__NANOC_THREAD_END__(params[i].thread);
			__NANOC_THREAD_CLOSE__(params[i].thread);
			__NANOC_THREAD_CLOSE__(params[i].mutex);
			__NANOC_THREAD_CLOSE__(params[i].main_mutex);
		}

		//unalloc delta sum
		input.resetDeltaSum(tc, 2);
		layer = hiddens.link;
		if (layer) {
			do {
				layer->resetDeltaSum(tc, 2);

				layer = hiddens.next(layer);
			} while (layer && layer != hiddens.link);
		}
		output.resetBiasSum(tc, 2);
		//unalloc neural
		input.resetNeural(tc, 2);
		layer = hiddens.link;
		if (layer) {
			do {
				layer->resetNeural(tc, 2);

				layer = hiddens.next(layer);
			} while (layer && layer != hiddens.link);
		}
		output.resetNeural(tc, 2);
		//unalloc delta
		input.resetDelta(tc, 2);
		layer = hiddens.link;
		if (layer) {
			do {
				layer->resetDelta(tc, 2);

				layer = hiddens.next(layer);
			} while (layer && layer != hiddens.link);
		}
		output.resetDelta(tc, 2);
		//unalloc map kernel, must be called last
		input.resetMapKernel(tc, 2);
		layer = hiddens.link;
		if (layer) {
			do {
				layer->resetMapKernel(tc, 2);

				layer = hiddens.next(layer);
			} while (layer && layer != hiddens.link);
		}
		output.resetMapKernel(tc, 2);
		//delete thread
		delete[] indice;
		delete[] indice_root;
		delete[]params;
	}

	void Forecast(Layer& in, Layer * out = NULL) {
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

		input.setScale(1.0 / divrange);

		ForwardTransfer();

		//output
		/*
		neural = output.neurals.link;
		if (neural) {
			do {
				printf("%e->", neural->output * divoutrange);

				neural = output.neurals.next(neural);
			} while (neural && neural != output.neurals.link);
		}
		printf("\n");*/

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

		//save out
		if (out) {
			neural = output.neurals.link;
			_neural = out->neurals.link;
			if (neural) {
				do {

					_neural->output = neural->output * this->divoutrange;

					_neural = out->neurals.next(_neural);
					neural = output.neurals.next(neural);
				} while (neural && neural != output.neurals.link &&
					_neural && _neural != out->neurals.link);
			}
		}
	}

	void Traverse() {
		//Test
		Neural * neural = this->input.neurals.link;
		if (neural) {
			do {

				printf("%e--->", neural->value);

				Connector * conn = neural->conn.link;
				if (conn) {
					do {

						if (conn->back == neural) {
							Neural * _neural = conn->forw;
							if (_neural) {
								printf("%e(%e, %e)-->", _neural->value, _neural->delta, conn->weight);

								Connector * _conn = _neural->conn.link;
								if (_conn) {
									do {
										if (_conn->back == _neural) {
											Neural * __neural = _conn->forw;
											if (__neural) {
												printf("%e(%e)->", __neural->value, _conn->weight);
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


#endif