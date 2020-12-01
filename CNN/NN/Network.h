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
	void prepareRNN(int serial_size) {
		Neural * neural;
		Layer * hidden;
		NeuralGate * pgate;
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

	}
	void releaseRNN() {
		Neural * neural;
		Layer * hidden;
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
	void TrainRNN(double **X, double **Y, int size, int in_size, int out_size, double threshold, int serial_size, int sample_size, int train_count) {
		EFTYPE error;
		Layer * layer;
		Layer *hidden, *_hidden;
		Neural * neural, *_neural;
		Connector *conn, *_conn;
		NeuralGate *pgate, *lpgate;

		int *indice = new int[size];
		int *indice_root = new int[sample_size];

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
		while (true) {
			//endless loop when count_div is 1, say sample_size equals size
			if (count_div > 1 && count >= count_div) {
				break;
			}
			//randomise indice
			int ind = rand() % sample_size;
			for (int i = 0; i < size; i++) {
				indice[i] = indice_root[(ind + i) % sample_size];
			}

			error = 0;
			for (int iter = 0; iter < size; iter++) {
				int ind = indice[iter];
				EFTYPE e = 0;
				//reset gate to t=0
				//input
				input.resetGate();
				//output
				output.resetGate();
				//hidden
				hidden = hiddens.link;
				if (hidden) {
					do {
						hidden->resetGate();

						hidden = hiddens.next(hidden);
					} while (hidden && hidden != hiddens.link);
				}
				int i, j, k;
				//for each serial_size
				for (int p = 0; p < serial_size; p++) {
					//put input
					input.setNeuralSerial((double*)((double*)X + ind * in_size * serial_size), serial_size, p);
					//put output
					output.setNeuralSerial((double*)((double*)Y + ind * out_size * serial_size), serial_size, p);

					input.setScale(1.0 / divrange);
					output.setScale(1.0 / divoutrange);

					//forawrd transfer
					//input
					input.getOutput();
					//hidden
					hidden = hiddens.link;
					if (hidden) {
						do {
							hidden->getOutput();

							hidden = hiddens.next(hidden);
						} while (hidden && hidden != hiddens.link);
					}
					//output
					output.getOutput();

					//delta
					e += output.getError();

					//move forawrd t+1
					//input
					input.nextGate();
					//output
					output.nextGate();
					//hidden
					hidden = hiddens.link;
					if (hidden) {
						do {
							hidden->nextGate();

							hidden = hiddens.next(hidden);
						} while (hidden && hidden != hiddens.link);
					}
				}

				//backward propagation
				//move gate to t
				//input
				input.backGate();
				//output
				output.backGate();
				//hidden
				hidden = hiddens.link;
				if (hidden) {
					do {
						hidden->backGate();

						hidden = hiddens.next(hidden);
					} while (hidden && hidden != hiddens.link);
				}
				//for each t gate
				for (int p = serial_size - 1; p >= 0 ; p--) {
						//output
						neural = output.neurals.link;
						if (neural) {
							do {
								pgate = neural->gate;
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
								if (hidden->mode == LayerMode::LSTM) {
									neural = hidden->neurals.link;
									if (neural) {
										do {
											//accumulate delta
											if (neural->gate) {
												NeuralGate& gate = *neural->gate;
												NeuralGate& prev_gate = *neural->gates.prev(neural->gate);
												NeuralGate& future_gate = *neural->gates.next(neural->gate);
												gate.h_delta = 0;
												//from output
												conn = neural->conn.link;
												if (conn) {
													do {
														//for all neural that links after this neural
														if (conn->back == neural) {
															_neural = conn->forw;
															if (_neural) {
																if (_neural->gate) {
																	gate.h_delta += _neural->gate->y_delta * conn->weight;
																}
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
																	NeuralGate& _future_gate = *_neural->gates.next(_neural->gate);
																	gate.h_delta += _future_gate.I_delta * conn->gate.U_I;
																	gate.h_delta += _future_gate.F_delta * conn->gate.U_F;
																	gate.h_delta += _future_gate.O_delta * conn->gate.U_O;
																	gate.h_delta += _future_gate.G_delta * conn->gate.U_G;
																}
															}
														}

														conn = neural->rconn.next(conn);
													} while (conn && conn != neural->rconn.link);
												}

												gate.O_delta = gate.h_delta * hidden->tan_h(gate.state) * hidden->sigmod_1(gate.out_gate);
												gate.state_delta = gate.h_delta * gate.out_gate * hidden->tan_h_1(hidden->tan_h(gate.state)) +
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
																conn->gate.W_I += hidden->learning_rate * gate.I_delta * _neural->gate->out;
																conn->gate.W_F += hidden->learning_rate * gate.F_delta * _neural->gate->out;
																conn->gate.W_O += hidden->learning_rate * gate.O_delta * _neural->gate->out;
																conn->gate.W_G += hidden->learning_rate * gate.G_delta * _neural->gate->out;
															}
														}

														conn = neural->conn.next(conn);
													} while (conn && conn != neural->conn.link);
												}
											}

											neural = hidden->neurals.next(neural);
										} while (neural && neural != hidden->neurals.link);
									}
								} else if (hidden->mode == LayerMode::GRU) {
									neural = hidden->neurals.link;
									if (neural) {
										do {
											//accumulate delta
											if (neural->gate) {


												NeuralGate& gate = *neural->gate;
												NeuralGate& prev_gate = *neural->gates.prev(neural->gate);
												NeuralGate& future_gate = *neural->gates.next(neural->gate);
												gate.d_delta = future_gate.h_delta;
												//from output
												conn = neural->conn.link;
												if (conn) {
													do {
														//for all neural that links after this neural
														if (conn->back == neural) {
															_neural = conn->forw;
															if (_neural) {
																if (_neural->gate) {
																	gate.d_delta += _neural->gate->y_delta * conn->weight;
																}
															}
														}

														conn = neural->conn.next(conn);
													} while (conn && conn != neural->conn.link);
												}

												gate.H_delta = gate.d_delta * gate.z_gate * hidden->tan_h_1(gate.h_gate);
												gate.Z_delta = gate.d_delta * ( - prev_gate.h + gate.h_gate) * hidden->sigmod_1(gate.z_gate);

												gate.r_delta = 0;
												//for hidden itself
												conn = neural->rconn.link;
												if (conn) {
													do {
														//for all neural that links after this neural
														if (conn->back == neural) {
															_neural = conn->forw;
															if (_neural) {
																if (_neural->gate) {
																	gate.r_delta += conn->gate.U_H * gate.H_delta;
																}
															}
														}

														conn = neural->rconn.next(conn);
													} while (conn && conn != neural->rconn.link);
												}
												gate.R_delta = gate.r_delta * prev_gate.h * hidden->sigmod_1(gate.r_gate);

												gate.h_delta = 0;
												//for hidden itself
												conn = neural->rconn.link;
												if (conn) {
													do {
														//for all neural that links after this neural
														if (conn->back == neural) {
															_neural = conn->forw;
															if (_neural) {
																if (_neural->gate) {
																	gate.h_delta += conn->gate.U_Z * gate.Z_delta;
																	gate.h_delta += conn->gate.U_R * gate.R_delta;
																}
															}
														}

														conn = neural->rconn.next(conn);
													} while (conn && conn != neural->rconn.link);
												}
												gate.h_delta += gate.d_delta * (1 - gate.z_gate);
												gate.h_delta += gate.r_delta * prev_gate.h  * gate.r_gate;

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
																	conn->gate.U_Z += hidden->learning_rate * gate.Z_delta * prev_gate.h;
																	conn->gate.U_R += hidden->learning_rate * gate.R_delta * prev_gate.h;
																	conn->gate.U_H += hidden->learning_rate * gate.H_delta * (gate.r_gate * prev_gate.h);
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
																conn->gate.W_Z += hidden->learning_rate * gate.Z_delta * _neural->gate->out;
																conn->gate.W_R += hidden->learning_rate * gate.R_delta * _neural->gate->out;
																conn->gate.W_H += hidden->learning_rate * gate.H_delta * _neural->gate->out;
															}
														}

														conn = neural->conn.next(conn);
													} while (conn && conn != neural->conn.link);
												}
											}

											neural = hidden->neurals.next(neural);
										} while (neural && neural != hidden->neurals.link);
									}
								}

								hidden = hiddens.next(hidden);
							} while (hidden && hidden != hiddens.link);
						}

						//move to t - 1
						//input
						input.prevGate();
						//output
						output.prevGate();
						//hidden
						hidden = hiddens.link;
						if (hidden) {
							do {
								hidden->prevGate();

								hidden = hiddens.next(hidden);
							} while (hidden && hidden != hiddens.link);
						}
				}

				if (0 && iter % 1000 == 0) {
					printf("\n");
					int out = 0;
					for (int i = 0; i < in_size; i++) {
						int in = 0;
						for (int p = serial_size - 1; p >= 0; p--) {
							double value = *(double*)((double*)X + ind * in_size * serial_size + i * serial_size + p);
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
							double value = *(double*)((double*)Y + ind * out_size * serial_size + i * serial_size + p);
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
							printf("\t%d\t%d\t%d\n", in, out, out - in);

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
							printf("\t%d\t%d\t%d\n", in, out, out - in);

							neural = output.neurals.next(neural);
						} while (neural && neural != output.neurals.link);
					}
					//getchar();
				}
				error += e;
				//printf("[ %d/%d]Error is: %e\r", iter, count, error);
			}
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
						for (int i = 1; i < count; i += (width_r > 1.0 ? width_r : 1)) {
							EP_Line(i / width_r, show_error[i] / height_r, (i - 1) / width_r, show_error[i - 1] / height_r);
						}
#endif
#ifdef _CNN_SHOW_CURV_
						int show_start = 1024;
						int show_size = 100;
						width_r = (EFTYPE)show_size / show_width;
						height_r = (EFTYPE)pow(2, serial_size) / show_height;
						//EP_ClearDevice();
						int x, y, ex = 0, ey = 0, ex1 = 0, ey1 = 0;
						for (int i = show_start; i < show_start + show_size; i += (width_r > 1.0 ? width_r : 1)) {
							int ind = i;
							int result = 0;
							int predict = 0;
							//reset gate to t=0
							//input
							input.resetGate();
							//output
							output.resetGate();
							//hidden
							hidden = hiddens.link;
							if (hidden) {
								do {
									hidden->resetGate();

									hidden = hiddens.next(hidden);
								} while (hidden && hidden != hiddens.link);
							}
							//for each serial_size
							for (int p = 0; p < serial_size; p++) {
								//put input
								input.setNeuralSerial((double*)((double*)X + ind * in_size * serial_size), serial_size, p);
								//put output
								output.setNeuralSerial((double*)((double*)Y + ind * out_size * serial_size), serial_size, p);

								input.setScale(1.0 / divrange);
								output.setScale(1.0 / divoutrange);

								//forawrd transfer
								//input
								input.getOutput();
								//hidden
								hidden = hiddens.link;
								if (hidden) {
									do {
										hidden->getOutput();

										hidden = hiddens.next(hidden);
									} while (hidden && hidden != hiddens.link);
								}
								//output
								output.getOutput();

								result += output.neurals.link->gate->in * pow(2, p);
								predict += ((int)(output.neurals.link->gate->out + 0.5)) * pow(2, p);

								//move forawrd t+1
								//input
								input.nextGate();
								//output
								output.nextGate();
								//hidden
								hidden = hiddens.link;
								if (hidden) {
									do {
										hidden->nextGate();

										hidden = hiddens.next(hidden);
									} while (hidden && hidden != hiddens.link);
								}
							}
							x = (i - show_start) / width_r;
							y = result / height_r;
							if (x <= show_width && y <= show_height) {
								EP_SetColor(GREEN);
								EP_Line(x, y, x, 0);
								ex = x;
								ey = y;
							}
							y = predict / height_r;
							if (x <= show_width && y <= show_height) {
								EP_SetColor(RED);
								EP_Line(x, y, x, 0);
								ex1 = x;
								ey1 = y;
							}
							y = show_height - abs(result - predict) / height_r;
							if (x <= show_width && y <= show_height) {
								EP_SetColor(BLUE);
								EP_Line(x, y, x, show_height);
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
			if (kbhit_console()) {
				getch_console();
				printf("\n");
				char c = getch_console();
				if (c == 'q' || c == 'Q') {
					break;
				}
				if (c == 's' || c == 'S') {
#ifdef _NANO_LSTM_
					this->Save("LSTM.txt");
#else
					this->Save("GRU.txt");
#endif
				}
			}

			count++;
			printf("\n[ %d]Error is: %e\r", count, error);
			if (error < threshold) {
				printf("\n");
				break;
			}
		}
#ifdef _CNN_SHOW_LOSS_
		delete[] show_error;
#endif
		delete[] indice;
		delete[] indice_root;
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
						for (int i = 1; i < count; i += (width_r > 1.0 ? width_r : 1)) {
							EP_Line(i / width_r, show_error[i] / height_r, (i - 1) / width_r, show_error[i - 1] / height_r);
						}
#endif
#ifdef _CNN_SHOW_GUI_
						width_r = (EFTYPE)sample_size / show_width;
						height_r = (EFTYPE)this->divoutrange / show_height;
						//EP_ClearDevice();
						int x, y, ex = 0, ey = 0, ex1 = 0, ey1 = 0;
						for (int i = 1; i < sample_size; i+=(width_r > 1.0 ? width_r : 1)) {
							input.setNeural((double*)((double*)X + i * in_size), in_size);
							output.setNeural((double*)((double*)Y + i * out_size), out_size);
							Forecast(input, &output);
							x = i / width_r;
							y = output.neurals.link->value / height_r;
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
			if (kbhit_console()) {
				getch_console();
				printf("\n");
				char c = getch_console();
				if (c == 'q' || c == 'Q') {
					break;
				}
				if (c == 's' || c == 'S') {
#ifdef _NANO_LINEAR1_
					this->Save("LINE1.txt");
#endif
#ifdef _NANO_LINEAR2_
					this->Save("LINE2.txt");
#endif
#ifdef _NANO_LINEAR_
					this->Save("LINE.txt");
#endif
				}
			}

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
								for (int i = 0; i < show_width; i+= (width_r > 1.0 ? width_r : 1)) {
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
								if (c == 's' || c == 'S') {
									this->Save("CNN.txt");
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
					for (int i = 1; i < count; i += (width_r > 1.0 ? width_r : 1)) {
						EP_Line(i / width_r, show_error[i] / height_r, (i - 1) / width_r, show_error[i - 1] / height_r);
					}
					EP_RenderFlush();
				}
			}
#endif
			if (kbhit_console()) {
				getch_console();
				printf("\n");
				c = getch_console();
				if (c == 'q' || c == 'Q') {
					break;
				}
			}

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

	void Save(const char * filename) {
		FILE *fp = NULL;
		fopen_s(&fp, filename, "w");
		if (!fp) {
			printf("File not exist: %s\n", filename);
			return;
		}
		printf("Write out %s\n", filename);
		Neural * neural, *_neural;
		Connector * conn, *rconn;
		Layer* layer = this->layers.link;
		if (layer) {
			do {
				fprintf(fp, "layer\t%d\t%d\n", layer->mode, layer->neurals.linkcount);

				neural = layer->neurals.link;
				if (neural) {
					do {

						fprintf(fp, "\tneural\t%d\t%e\t%d\t%d\t%d\t%e\n", neural->uniqueID, neural->bias, neural->map_w, neural->map_h, neural->map_count, neural->map.b);

						fprintf(fp, "\t\tconn\t%d\n", neural->conn.linkcount);
						conn = neural->conn.link;
						if (conn) {
							do {

								if (conn->back == neural) {
									_neural = conn->forw;
									if (_neural) {
										fprintf(fp, "\t\t\tneural\t%d\n", _neural->uniqueID);
										fprintf(fp, "\t\t\t\tweight\t%e\n", conn->weight);
										fprintf(fp, "\t\t\t\tkernel\t%d\t%d", conn->kernel_w, conn->kernel_h);
										for (int i = 0; i < conn->kernel_w; i++) {
											fprintf(fp, "\n\t\t\t\t\tkernel\t");
											for (int j = 0; j < conn->kernel_h; j++) {
												fprintf(fp, "%e\t", conn->kernel.W[i * conn->kernel_h + j]);
											}
										}
										fprintf(fp, "\n");
									}
								}

								conn = neural->conn.next(conn);
							} while (conn && conn != neural->conn.link);
						}

						fprintf(fp, "\t\trconn\t%d\n", neural->rconn.linkcount);
						rconn = neural->rconn.link;
						if (rconn) {
							do {

								if (rconn->back == neural) {
									_neural = rconn->forw;
									if (_neural) {
										fprintf(fp, "\t\t\t\tgate\t%d\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",
											_neural->uniqueID,
											rconn->gate.W_I, rconn->gate.U_I, rconn->gate.W_F, rconn->gate.U_F,
											rconn->gate.W_O, rconn->gate.U_O, rconn->gate.W_G, rconn->gate.U_G,
											rconn->gate.W_Z, rconn->gate.U_Z, rconn->gate.W_R, rconn->gate.U_R,
											rconn->gate.W_H, rconn->gate.U_H);
									}
								}
								rconn = neural->rconn.next(rconn);
							} while (rconn && rconn != neural->rconn.link);
						}

						neural = layer->neurals.next(neural);
					} while (neural && neural != layer->neurals.link);
				}

				layer = this->layers.next(layer);
			} while (layer && layer != this->layers.link);
		}
		fclose(fp);
	}

#define MAX_STR 1024
#define MAX_PAR	20
	INT parseParameter(CHAR buffer[], CHAR command[], CHAR parameters[][MAX_STR], INT maxP = MAX_PAR)
	{
		int i, j;
		int cp = 0;
		int pp = 0;
		int pc = 0;
		int flag = 0;
		int lspc = 0;
		for (i = 0; buffer[i] != '\0' && (buffer[i] == '\t' || buffer[i] == ' '); i++);
		for (j = i; buffer[j] != '\0'; j++)
		{
			if (buffer[j] == '\n')
			{
				buffer[j] = '\0';
				break;
			}
		}
		for (; buffer[i] != '\0'; i++)
		{
			if (buffer[i] == '\t' || buffer[i] == ' ')
			{
				if (lspc == 0)
				{
					if (flag)
					{
						parameters[pc][pp] = '\0';
						pc++;
						if (maxP && pc >= maxP - 1) {
							break;
						}
						pp = 0;
					}
					else
					{
						command[cp] = '\0';
						flag = 1;
					}
				}
				lspc++;
				continue;
			}
			else
			{
				lspc = 0;
			}
			if (flag)
			{
				parameters[pc][pp++] = buffer[i];
				if (pp >= MAX_STR)
				{
					break;
				}
			}
			else
			{
				command[cp++] = buffer[i];
				if (cp >= MAX_STR)
				{
					break;
				}
			}
		}
		if (flag)
		{
			parameters[pc][pp] = '\0';
		}
		else
		{
			command[cp] = '\0';
			pc = pc - 1;
		}
		return pc + 1;
	}
	void Load(const char * filename) {
		FILE *rfp = NULL;
		fopen_s(&rfp, filename, "r");
		if (!rfp) {
			printf("File not exist: %s\n", filename);
			return;
		}
		printf("Read from %s\n", filename);
		CHAR buffer[MAX_STR];
		CHAR command[MAX_STR];
		CHAR attrs[MAX_PAR][MAX_STR];
		int paramCount;
		sprintf_s(buffer, "test_%s", filename);
		FILE *fp = NULL;
		fopen_s(&fp, buffer, "w");
		if (!fp) {
			printf("File not exist: %s\n", filename);
			return;
		}

#define LOAD_DEBUG

		Neural * neural, *_neural;
		Connector * conn, *rconn;
		Layer* layer = this->layers.link;
		int uniqueID, _uniqueID;
		int linkcount;
		if (layer) {
			do {
				fgets(buffer, MAX_STR, rfp);
				paramCount = parseParameter(buffer, command, attrs);

#ifdef LOAD_DEBUG
				fprintf(fp, "layer\t%d\t%d\n", layer->mode, layer->neurals.linkcount);
#endif

				for (int i = 0; i < layer->neurals.linkcount; i++) {
					uniqueID = -1;
					do {
						fgets(buffer, MAX_STR, rfp);
						paramCount = parseParameter(buffer, command, attrs);
						if (paramCount >= 1) {
							uniqueID = atoi(attrs[0]);
						}
					} while (uniqueID == -1);

					neural = layer->neurals.getLink(uniqueID);
					if (neural) {
						if (paramCount >= 6) {
							neural->bias = atof(attrs[1]);
							neural->map_w = atoi(attrs[2]);
							neural->map_h = atoi(attrs[3]);
							neural->map_count = atoi(attrs[4]);
							neural->map.b = atof(attrs[5]);
						}
#ifdef LOAD_DEBUG
						fprintf(fp, "\tneural\t%d\t%e\t%d\t%d\t%d\t%e\n", neural->uniqueID, neural->bias, neural->map_w, neural->map_h, neural->map_count, neural->map.b);
#endif

						fgets(buffer, MAX_STR, rfp);
						paramCount = parseParameter(buffer, command, attrs);
#ifdef LOAD_DEBUG
						fprintf(fp, "\t\tconn\t%d\n", neural->conn.linkcount);
#endif

						linkcount = 0;
						conn = neural->conn.link;
						if (conn) {
							do {

								if (conn->back == neural) {
									_neural = conn->forw;
									if (_neural) {
										linkcount++;
									}
								}

								conn = neural->conn.next(conn);
							} while (conn && conn != neural->conn.link);
						}

						for (int j = 0; j < linkcount; j++) {
							_uniqueID = -1;
							do {
								fgets(buffer, MAX_STR, rfp);
								paramCount = parseParameter(buffer, command, attrs);
								if (paramCount >= 1) {
									_uniqueID = atoi(attrs[0]);
								}
							} while (_uniqueID == -1);

							_neural = NULL;
							conn = neural->conn.link;
							if (conn) {
								do {

									if (conn->back == neural) {
										_neural = conn->forw;
										if (_neural) {
											if (_neural->uniqueID == _uniqueID) {
												break;
											}
										}
									}

									conn = neural->conn.next(conn);
								} while (conn && conn != neural->conn.link);
							}
							if (_neural && conn) {

								if (conn->back == neural) {
									_neural = conn->forw;
									if (_neural) {
#ifdef LOAD_DEBUG
										fprintf(fp, "\t\t\tneural\t%d\n", _neural->uniqueID);
#endif
										fgets(buffer, MAX_STR, rfp);
										paramCount = parseParameter(buffer, command, attrs);
										if (paramCount >= 1) {
											conn->weight = atof(attrs[0]);
										}
#ifdef LOAD_DEBUG
										fprintf(fp, "\t\t\t\tweight\t%e\n", conn->weight);
#endif
										fgets(buffer, MAX_STR, rfp);
										paramCount = parseParameter(buffer, command, attrs);
										if (paramCount >= 2) {
											conn->kernel_w = atoi(attrs[0]);
											conn->kernel_h = atoi(attrs[1]);
										}
#ifdef LOAD_DEBUG
										fprintf(fp, "\t\t\t\tkernel\t%d\t%d", conn->kernel_w, conn->kernel_h);
#endif
										for (int i = 0; i < conn->kernel_w; i++) {
#ifdef LOAD_DEBUG
											fprintf(fp, "\n\t\t\t\t\tkernel\t");
#endif
											fgets(buffer, MAX_STR, rfp);
											paramCount = parseParameter(buffer, command, attrs);
											if (paramCount >= conn->kernel_h) {
												for (int j = 0; j < conn->kernel_h; j++) {
													conn->kernel.W[i * conn->kernel_h + j] = atof(attrs[j]);
#ifdef LOAD_DEBUG
													fprintf(fp, "%e\t", conn->kernel.W[i * conn->kernel_h + j]);
#endif
												}
											}
										}
#ifdef LOAD_DEBUG
										fprintf(fp, "\n");
#endif
									}
								}
							}
						}

						fgets(buffer, MAX_STR, rfp);
						paramCount = parseParameter(buffer, command, attrs);
#ifdef LOAD_DEBUG
						fprintf(fp, "\t\trconn\t%d\n", neural->rconn.linkcount);
#endif

						linkcount = 0;
						rconn = neural->rconn.link;
						if (rconn) {
							do {

								if (rconn->back == neural) {
									_neural = rconn->forw;
									if (_neural) {
										linkcount++;
									}
								}

								rconn = neural->rconn.next(rconn);
							} while (rconn && rconn != neural->rconn.link);
						}

						for (int j = 0; j < linkcount; j++) {
							_uniqueID = -1;
							do {
								fgets(buffer, MAX_STR, rfp);
								paramCount = parseParameter(buffer, command, attrs);
								if (paramCount >= 1) {
									_uniqueID = atoi(attrs[0]);
								}
							} while (_uniqueID == -1);


							_neural = NULL;
							rconn = neural->rconn.link;
							if (rconn) {
								do {

									if (rconn->back == neural) {
										_neural = rconn->forw;
										if (_neural) {
											if (_neural->uniqueID == _uniqueID) {
												break;
											}
										}
									}

									rconn = neural->rconn.next(rconn);
								} while (rconn && rconn != neural->rconn.link);
							}
							if (_neural && rconn) {

								if (rconn->back == neural) {
									_neural = rconn->forw;
									if (_neural) {
										if (paramCount >= 15) {
											rconn->gate.W_I = atof(attrs[1]);
											rconn->gate.U_I = atof(attrs[2]);
											rconn->gate.W_F = atof(attrs[3]);
											rconn->gate.U_F = atof(attrs[4]);
											rconn->gate.W_O = atof(attrs[5]);
											rconn->gate.U_O = atof(attrs[6]);
											rconn->gate.W_G = atof(attrs[7]);
											rconn->gate.U_G = atof(attrs[8]);
											rconn->gate.W_Z = atof(attrs[9]);
											rconn->gate.U_Z = atof(attrs[10]);
											rconn->gate.W_R = atof(attrs[11]);
											rconn->gate.U_R = atof(attrs[12]);
											rconn->gate.W_H = atof(attrs[13]);
											rconn->gate.U_H = atof(attrs[14]);
										}
#ifdef LOAD_DEBUG
										fprintf(fp, "\t\t\t\tgate\t%d\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",
											_neural->uniqueID,
											rconn->gate.W_I, rconn->gate.U_I, rconn->gate.W_F, rconn->gate.U_F,
											rconn->gate.W_O, rconn->gate.U_O, rconn->gate.W_G, rconn->gate.U_G,
											rconn->gate.W_Z, rconn->gate.U_Z, rconn->gate.W_R, rconn->gate.U_R,
											rconn->gate.W_H, rconn->gate.U_H);
#endif
									}
								}
							}
						}
					}
				}

				layer = this->layers.next(layer);
			} while (layer && layer != this->layers.link);
		}
		fclose(rfp);
		fclose(fp);
	}
};


#endif