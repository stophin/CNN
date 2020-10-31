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
	return 1;
	//return 1;
}
EFTYPE train_sample_max() {
	return 100;
	//return 5;
}
EFTYPE train_sample_output_min() {
	return 1;
	//return 1;
}
EFTYPE train_sample_output_max() {
	return 300;
	//return 22;
}
EFTYPE train_sample_input(INT index, EFTYPE range_min, EFTYPE range_max) {
	return g_indexM[index];
	//return sample[g_index % 41][index];
}
EFTYPE train_sample_output(EFTYPE x, EFTYPE y, EFTYPE z) {
	return x + y + z;
	//return sample[g_index % 41][3];
}
void train_sample_index() {
	g_indexM[g_index]++;
	if (g_indexM[g_index] >= 100) {
		g_indexM[g_index] = 10;
		g_index++;
		if (g_index >= 3) {
			g_index = 0;
		}
	}
	//++g_index;
	//if (g_index >= 41) {
	//	g_index = 1;
	//}
}
EFTYPE train_sample(EFTYPE x, EFTYPE y, EFTYPE z) {
	return x + y + z;
	//int _x = (int)x;
	//int _y = (int)y;
	//int _z = (int)z;
	//for (int i = 0; i < 41; i++) {
	//	if (sample[i][0] == _x && sample[i][1] == _y && sample[i][2] == _z) {
	//		return sample[i][3];
	//	}
	//}
	//return 0;
}
#ifdef _NANOC_WINDOWS_
#include <float.h>
unsigned int fp_control_state = _controlfp_s(&fp_control_state, _EM_INEXACT, _MCW_EM);
#endif


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

int test0() {
	INT i, j, k;

	Network nets;

	//inputs
	for (int i = 0; i < 100; i++) {
		EFTYPE input = train_sample_input(0, 0, 0);
		EFTYPE output = train_sample_output(input, 0, 0);
		nets.input.addNeural(input);
		nets.output.addNeural(output);
	}
	//nets.input.addNeural(1);
	//nets.input.addNeural(2);
	//nets.input.addNeural(3);

	//outputs
	//nets.output.addNeural(1);

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
	//input.addNeural(2);
	//input.addNeural(3);

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
		//temp[1] = train_sample_input(1, range_min, range_max);// (EFTYPE)(rand() % range_max) + range_min;
		//temp[2] = train_sample_input(2, range_min, range_max);// (EFTYPE)(rand() % range_max) + range_min;
		temp[3] = train_sample_output(temp[0], temp[1], temp[2]);
		train_sample_index();
		//nets.input.setNeural(temp, 1);
		//nets.output.setNeural(temp + 3, 1);

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

				input.setNeural(temp, 1);
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
#if 0
#ifdef _NANOC_WINDOWS_
#pragma comment(lib, "opencv_core401d.lib")
#pragma comment(lib, "opencv_highgui401d.lib")
#pragma comment(lib, "opencv_imgproc401d.lib")
using namespace cv;
void get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start)
{
	cv::FileStorage fs;
	fs.open(filename, cv::FileStorage::READ);

	printf("FS:%d\n", fs.isOpened());

	cv::Mat input_, target_;
	fs["input"] >> input_;
	fs["target"] >> target_;
	fs.release();
	input = input_(cv::Rect(start, 0, sample_num, input_.rows));
	label = target_(cv::Rect(start, 0, sample_num, target_.rows));
}
int test1() {
	Network nets;


	//Get test samples and test samples 
	Mat sample, label, test_sample, test_label;
	int sample_number = 800;
	get_input_label("H:\\Neural\\Neural\\data\\input_label_1000.xml", sample, label, sample_number);
	get_input_label("H:\\Neural\\Neural\\data\\input_label_1000.xml", test_sample, test_label, 200, 800);

	//inputs
	for (int i = 0; i < sample.rows; i++) {
		nets.input.addNeural(i+1);
	}
	for (int i = 0; i < label.rows; i++) {
		nets.output.addNeural((i+1) + 9000);
	}

	//layers
	for (int i = 0; i < 1; i++) {
		Layer * hidden = new Layer();

		//formula of perfect hidden num:
		//sqrt(in_num + out_num) + a
		//a is 5
		for (int j = 0; j < 100; j++) {
			hidden->addNeural(j+1 + (i + 1) * 1000);
		}

		nets.hiddens.insertLink(hidden);
		nets.layers.insertLink(hidden, &nets.output, NULL);
	}

	//make connections
	Layer * hidden = nets.layers.link;
	if (hidden) {
		int i = 0;
		Layer * _hidden = nets.layers.next(hidden);
		if (_hidden && _hidden != nets.layers.link) {
			do {

				hidden->makeConnection(*_hidden, ++i);

				hidden = _hidden;
				_hidden = nets.layers.next(_hidden);
			} while (_hidden && _hidden != nets.layers.link);
		}
	}

	//nets.Traverse();

	//getch();

	//test input
	Layer input;
	for (int i = 0; i < sample.rows; i++) {
		input.addNeural(i + 1);
	}

	nets.Scale(1.0, 1.0);
	//nets.Scale(100.0, 100.0);

	double **_sample = new double*[sample.cols];
	for (int i = 0; i < sample.cols; i++) {
		_sample[i] = new double[sample.rows];
	}
	for (int i = 0; i < sample.cols; i++) {
		for (int j = 0; j < sample.rows; j++) {
			_sample[i][j] = sample.at<FLOAT>(j, i);
		}
	}
	double **_label = new double*[label.cols];
	for (int i = 0; i < label.cols; i++) {
		_label[i] = new double[label.rows];
	}
	for (int i = 0; i < label.cols; i++) {
		for (int j = 0; j < label.rows; j++) {
			_label[i][j] = label.at<FLOAT>(j, i);
		}
	}
	double **_test_sample = new double*[test_sample.cols];
	for (int i = 0; i < test_sample.cols; i++) {
		_test_sample[i] = new double[test_sample.rows];
	}
	for (int i = 0; i < test_sample.cols; i++) {
		for (int j = 0; j < test_sample.rows; j++) {
			_test_sample[i][j] = test_sample.at<FLOAT>(j, i);
		}
	}
	int col = 0;
	EFTYPE error = 0;
	while (1) {
		nets.input.setNeural(_sample[col], sample.rows);
		nets.output.setNeural(_label[col], label.rows);
		col++;
		if (col >= sample.cols) {
			col = 0;
			printf("\nBatch error:%e\n", error);
			if (error < T_ERROR) {
				break;
			}
			error = 0;
		}

		//error += nets.Train();

		nets.input.setScale(1.0 / nets.divrange);
		nets.output.setScale(1.0 / nets.divoutrange);

		nets.ForwardTransfer();
		error += nets.output.getError() * nets.divoutrange * nets.divoutrange;
		printf("[%5d]Error is: %e\r", col, error);
		nets.ReverseTrasfer();

		//nets.Traverse();

		if (kbhit())
		{
			printf("Training: %d\n", col);
			int _col = 0;
			while (1) {
				int actual = 0;
				double max = 0;
				for (int i = 0; i < test_label.rows; i++) {
					if (max < test_label.at<FLOAT>(i, _col)) {
						max = test_label.at<FLOAT>(i, _col);
						actual = i;
					}
				}

				input.setNeural(_test_sample[_col], test_sample.rows);
				nets.Forecast(input);

				Neural *predict = nets.output.getMax();

				printf("Predict: %d, Actual: %d\n", predict->uniqueID, actual );

				_col++;
				if (_col >= test_sample.cols) {
					break;
				}
			}
			getch();
			getch();
		}
	}
	for (int i = 0; i < sample.cols; i++) {
		delete[] _sample[i];
	}
	delete[] _sample;
	for (int i = 0; i < label.cols; i++) {
		delete[] _label[i];
	}
	delete[] _label;
	for (int i = 0; i < test_sample.cols; i++) {
		delete[] _test_sample[i];
	}
	delete[] _test_sample;
	getch();
	return 0;
}
#endif
#endif

int test1() {
	INT i, j, k;

	Network nets;

	//inputs
	nets.input.addNeural(1);
	nets.input.addNeural(2);

	//outputs
	nets.output.addNeural(1);

	//layers
	for (i = 0; i < 1; i++) {
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
	Layer output;
	output.addNeural(1);

	//data
	/*
	EFTYPE X[][2] = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
		{0.1, 0.2},
		{0.15, 0.9},
		{1.0, 0.01},
		{0.88, 1.03}
	};
	EFTYPE Y[][1] = {
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
		{0}
	};
	INT sample_size = 1;
	INT sample_size_real = 8;
	INT in_size = 2;
	INT out_size = 1;
	EFTYPE divx = 1.0;
	EFTYPE divy = 1.0;
	nets.Scale(divx, divy); */

	EFTYPE X[][2] = {
		{0.356649128, 0.030306376},
		{0.105260929, 0.876207066},
		{0.481199704, 0.253876948},
		{0.471508224, 0.54439978},
		{0.077497426, 0.885208},
		{0.255285458, 0.785259903},
		{0.391494441, 0.069935833},
		{0.933032415, 0.301373387},
		{0.523691793, 0.796247175},
		{0.779717102, 0.713375742},
		{0.076675163, 0.410487185},
		{0.962224658, 0.016073412},
		{0.515055289, 0.72873525},
		{0.322424475, 0.730628336},
		{0.688762677, 0.755388034},
		{0.260675482, 0.13143853},
		{0.778240922, 0.093408653},
		{0.619528796, 0.673211656},
		{0.615551417, 0.035100893},
		{0.854474454, 0.815391365},
		{0.217608904, 0.701204949},
		{0.235063898, 0.863031668},
		{0.119538688, 0.597635479},
		{0.938469809, 0.046644211},
		{0.424230246, 0.954724505},
		{0.2804779, 0.59883102},
		{0.019132831, 0.634812852},
		{0.761891727, 0.364283401},
		{0.220551241, 0.398446811},
		{0.576858876, 0.735632906},
		{0.125193284, 0.740642657},
		{0.240168653, 0.6018266},
		{0.513267996, 0.462949566},
		{0.1428799, 0.87131965},
		{0.858053973, 0.602338272},
		{0.587654877, 0.385143152},
		{0.516003448, 0.988193208},
		{0.660521998, 0.270327062},
		{0.214826163, 0.70650369},
		{0.726414412, 0.514509073},
		{0.898426694, 0.627019375},
		{0.773045407, 0.161891909}
	};
	EFTYPE Y[][1] = {
		{0},{0},{0},{1},{0},{1},{0},{1},{1},{1},{0},{0},{1},{1},{1},{0},{0},{1},{0},{1},{0},{1},{0},{0},{1},{0},{0},{1},{0},{1},{0},{0},{0},{1},{1},{0},{1},{0},{0},{1},{1},{0}
	};
	INT sample_size = 30;
	INT sample_size_real = 42;
	INT in_size = 2;
	INT out_size = 1;
	EFTYPE divx = 1.0;
	EFTYPE divy = 1.0;
	nets.Scale(divx, divy);

	int count = 0;
	while (1) {
		count++;

		//nets.Train((double**)X, (double**)Y, sample_size, in_size, out_size, 0.0001);
		nets.Train((double**)X, (double**)Y, sample_size, in_size, out_size, 0.001, 3, 10);
		sample_size++;
		if (sample_size > 4) {
			sample_size = 4;
		}

		//if (kbhit())
		{
			printf("Training: %d\n", count);
			nets.Traverse();
			while (1) {
				INT ind;
				while (scanf("%d", &ind) != 1) {
					getchar();
					fflush(stdin);
				}
				if (ind < 0 || ind >= sample_size_real) {
					break;
				}

				input.setNeural(X[ind], in_size);
				nets.Forecast(input, &output);
				printf("\n");
				for (i = 0; i < in_size; i++) {
					printf("%e ", X[ind][i]);
				}
				printf("\n");
				for (i = 0; i < out_size; i++) {
					printf("%e ", Y[ind][i]);
				}
				printf("\n");
				printf("Actual:\n");

				Neural * neural = output.neurals.link;
				EFTYPE e = 0;
				i = 0;
				if (neural) {
					do {
						printf("%e %e", Y[ind][i], neural->output);
						EFTYPE f = Y[ind][i] - neural->output;
						f = f * f / (2 * divy);
						e += f;
						printf(" error: %lf\n", f);
						i++;

						neural = output.neurals.next(neural);
					} while (neural && neural != output.neurals.link);
				}
				printf("total error: %lf\n", e);
			}
		}
	}
	return 0;
}
int main(int argc, _TCHAR* argv[])
{
	while (1) {
		test1();
	}
}
