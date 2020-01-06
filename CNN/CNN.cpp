// CNN.cpp : 定义控制台应用程序的入口点。
//


#include "NN/Network.h"

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


int g_indexM[3] = { 10, 10, 10 };
int g_index = 0;
EFTYPE train_sample_min() {
	//return 1;
	return 1;
}
EFTYPE train_sample_max() {
	//return 100;
	return 5;
}
EFTYPE train_sample_output_min() {
	//return 1;
	return 1;
}
EFTYPE train_sample_output_max() {
	//return 300;
	return 22;
}
EFTYPE train_sample_input(INT index, EFTYPE range_min, EFTYPE range_max) {
	//return g_indexM[index];
	return sample[g_index][index];
}
EFTYPE train_sample_output(EFTYPE x, EFTYPE y, EFTYPE z) {
	//return x + y + z;
	return sample[g_index % 41][3];
}
void train_sample_index() {
	//g_indexM[g_index]++;
	if (g_indexM[g_index] >= 100) {
		g_indexM[g_index] = 0;
		g_index++;
		if (g_index >= 3) {
			g_index = 0;
		}
	}
	++g_index;
}

EFTYPE train_sample(EFTYPE x, EFTYPE y, EFTYPE z) {
	//return x + y + z;
	int _x = (int)x;
	int _y = (int)y;
	int _z = (int)z;
	for (int i = 0; i < 41; i++) {
		if (sample[i][0] == _x && sample[i][1] == _y && sample[i][2] == _z) {
			return sample[i][3];
		}
	}
	return 0;
}

int test() {
	INT i, j, k;

	Network nets;

	//inputs
	nets.input.addNeural(1);
	nets.input.addNeural(2);
	nets.input.addNeural(3);

	//outputs
	nets.output.addNeural(1);

	//layers
	for (i = 0; i < 3; i++) {
		Layer * hidden = new Layer();
		
		//formula of perfect hidden num:
		//sqrt(in_num + out_num) + a
		//a is 5
		hidden->addNeural(1 + (i + 1) * 1000);
		hidden->addNeural(2 + (i + 1) * 1000);
		hidden->addNeural(3 + (i + 1) * 1000);
		hidden->addNeural(4 + (i + 1) * 1000);
		hidden->addNeural(5 + (i + 1) * 1000);
		hidden->addNeural(6 + (i + 1) * 1000);
		hidden->addNeural(7 + (i + 1) * 1000);

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

	//make connections
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

	//test input
	Layer input;
	input.addNeural(1);
	input.addNeural(2);
	input.addNeural(3);

	EFTYPE temp[] = { 1, 5, 2, 10 };

	//get scale range to limit input/ouptput to 0~1
	EFTYPE range_min = train_sample_min();
	EFTYPE outrange_min = train_sample_output_min();
	EFTYPE range_max = train_sample_max();
	EFTYPE outrange_max = train_sample_output_max();
	EFTYPE divrange = ((EFTYPE)(range_max - range_min)) / MAX_SIMULATION_RANGE_INPUT;
	EFTYPE divoutrange = ((EFTYPE)(outrange_max - outrange_min)) / MAX_SIMULATION_RANGE_OUTPUT;
	range_max = max(range_min, range_max);
	range_min = min(range_min, range_max);
	//nets.Scale(1.0, 1.0);
	nets.Scale(divrange, divoutrange);
	k = 0;
	i = 0;
	
	while(1) {
		i++;
		//k = i % 41;
		//nets.input.setNeural(sample[k], 3);
		//nets.output.setNeural(sample[k] + 3, 1);
		//nets.input.setNeural(sample[k], 3);
		//nets.output.setNeural(sample[k] + 3, 1);

		temp[0] = train_sample_input(0, range_min, range_max);// (EFTYPE)(rand() % range_max) + range_min;
		temp[1] = train_sample_input(1, range_min, range_max);// (EFTYPE)(rand() % range_max) + range_min;
		temp[2] = train_sample_input(2, range_min, range_max);// (EFTYPE)(rand() % range_max) + range_min;
		temp[3] = train_sample_output(temp[0], temp[1], temp[2]);
		train_sample_index();
		nets.input.setNeural(temp, 3);
		nets.output.setNeural(temp + 3, 1);

		nets.Train();

		//nets.Traverse();

		if (kbhit()) 
		{
			printf("Training: %d\n", i);
			while (1) {
				for (j = 0; j < 3; j++){
					scanf("%lf", &temp[j]);
				}
				if (ISZERO(temp[0]) || ISZERO(temp[1]) || ISZERO(temp[2])) {
					break;
				}

				input.setNeural(temp, 3);
				nets.Forecast(input);
				temp[3] = train_sample(temp[0], temp[1], temp[2]);
				printf("Actual: %e\n", temp[3]);
			}
			if (ISZERO(temp[0]) && ISZERO(temp[1]) && ISZERO(temp[2])) {
				break;
			}
		}
	}
	return 0;
}

int main(int argc, _TCHAR* argv[])
{
	while (1) {
		test();
	}
}
