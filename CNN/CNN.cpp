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

EFTYPE train_sample(EFTYPE x, EFTYPE y, EFTYPE z) {
	//return  1 / x + 1 / y + 1 / z;
	//return x + y + z;
	return x * y * z;
}

int _tmain(int argc, _TCHAR* argv[])
{
	INT i, j, k;

	Network nets;

#define WEIGHT	0
#define BIAS	0//初始化权值和阀值为0，也可以初始化随机值
	nets.input.addNeural(1);
	nets.input.addNeural(2);
	nets.input.addNeural(3);

	nets.output.addNeural(1, BIAS);

	for (i = 0; i < 2; i++) {
		Layer * hidden = new Layer();
		
		//formula of perfect hidden num:
		//sqrt(in_num + out_num) + a
		//a is 5
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

	INT range_min = 1;
	INT outrange_min = train_sample(range_min, range_min, range_min);
	INT range_max = 10;
	INT outrange_max = train_sample(range_max, range_max, range_max);
	nets.divrange = ((EFTYPE)(range_max - range_min)) / MAX_SIMULATION_RANGE_INPUT;
	nets.divoutrange = ((EFTYPE)(outrange_max - outrange_min)) / MAX_SIMULATION_RANGE_OUTPUT;
	range_max = max(range_min, range_max);
	range_min = min(range_min, range_max);
	//nets.divrange = 1.0;
	//nets.divoutrange = 1.0;
	k = 0;
	i = 0;
	while(1) {
		i++;
		k = i % 41;
		//nets.input.setNeural(sample[k], 3);
		//nets.output.setNeural(sample[k] + 3, 1);
		temp[0] = (EFTYPE)(rand() % range_max) + range_min;
		temp[1] = (EFTYPE)(rand() % range_max) + range_min;
		temp[2] = (EFTYPE)(rand() % range_max) + range_min;
		temp[3] = train_sample(temp[0], temp[1], temp[2]);

		//nets.input.setNeural(sample[k], 3);
		//nets.output.setNeural(sample[k] + 3, 1);
		nets.input.setNeural(temp, 3);
		nets.output.setNeural(temp + 3, 1);

		nets.Train();

		if (kbhit()) {
			printf("Training: %d\n", i);
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

	return 0;
}

