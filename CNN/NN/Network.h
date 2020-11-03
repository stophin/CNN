//Network.h
//

#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "Layer.h"

struct ThreadParam{
	int tid;
	int size;
	int in_size;
	int out_size;
	int start;
	int end;
	double** X;
	double** Y;
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

	void Train(Sample* sample, int size, int in_size, int out_size, double threshold) {
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
				input.getDelta(hiddens.link->mode);

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
				input.updateDeltaSum(hiddens.link->mode);
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
				input.setNeural((double*)((double*)X + iter * in_size), in_size, tid);
				output.setNeural((double*)((double*)Y + iter * out_size), out_size, tid);

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

	void Train(double **X, double **Y, int size, int in_size, int out_size, double threshold, int thx, int thy) {
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

		//init thread
		ThreadParam * params = new ThreadParam[tc];
		int div = size / tc;
		int divl = size - div * tc;
		int divd = 0;
		for (int i = 0; i < tc; i++) {
			params[i].tid = i;
			params[i].size = size;
			params[i].in_size = in_size;
			params[i].out_size = out_size;
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
		getch();

		int count = 0;
		while (true) {
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

			printf("[ %d]Error is: %e\r", count, error);
			count++;
			if (error < threshold) {
				printf("\n");
				break;
			}
		}
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
			__NANOC_THREAD_END__(params[i].thread);
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
		neural = output.neurals.link;
		if (neural) {
			do {
				printf("%e->", neural->output * divoutrange);

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

		//save out
		if (out) {
			neural = output.neurals.link;
			_neural = out->neurals.link;
			if (neural) {
				do {

					_neural->output = neural->output;

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