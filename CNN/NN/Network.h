//Network.h
//

#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "Layer.h"

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

#define T_ERROR	0.002//单个样本允许的误差
	void Train() {

		//set scale
		input.setScale(1.0 / divrange);
		output.setScale(1.0 / divoutrange);

		printf("Target:");
		Neural * neural = input.neurals.link;
		if (neural) {
			do {

				printf("%.2f ", neural->value * divrange);

				neural = input.neurals.next(neural);
			} while (neural && neural != input.neurals.link);
		}
		printf("->");
		neural = output.neurals.link;
		if (neural) {
			do {

				printf("%.2f ", neural->value * divoutrange);

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

					printf("%.2f->", neural->output * divoutrange);

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

				printf("%.2f ", neural->value * divrange);

				neural = input.neurals.next(neural);
			} while (neural && neural != input.neurals.link);
		}
		printf("->");
		neural = output.neurals.link;
		if (neural) {
			do {

				printf("%.2f ", neural->value * divoutrange);

				neural = output.neurals.next(neural);
			} while (neural && neural != output.neurals.link);
		}
		printf("\n");
		printf("Result:");
		neural = output.neurals.link;
		if (neural) {
			do {

				printf("%.2f ", neural->output * divoutrange);

				neural = output.neurals.next(neural);
			} while (neural && neural != output.neurals.link);
		}
		printf("\n");
		printf("[%5d]Error is: %f\n", count, error);
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

		input.setScale(1.0 / divrange);

		ForwardTransfer();

		//output
		neural = output.neurals.link;
		if (neural) {
			do {
				printf("%.2f->", neural->output * divoutrange);

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
								printf("%.2f(%.2f, %.2f)-->", _neural->value, _neural->delta, conn->weight);

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


#endif