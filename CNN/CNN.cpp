// CNN.cpp : 定义控制台应用程序的入口点。
//

#include "NN/Network.h"

#ifdef _CNN_SHOW_GUI_
const int show_width = 300;
const int show_height = 200;
#endif

const int classes_count = 10;
const int width = 32;
const int height = 32;
const int train_sample_count = 60000;
const int test_sample_count = 10000;
// 读取数据----------------------------------------------------------------------------------------------/
// 高低位转换
int SwapEndien_32(int i)
{
	return ((i & 0x000000FF) << 24) | ((i & 0x0000FF00) << 8) | ((i & 0x00FF0000) >> 8) | ((i & 0xFF000000) >> 24);
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;

	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

// 读训练/测试集数据
void read_mnist_data(Sample *sample, const char *file_name)
{
	FILE *fp = NULL;
	fopen_s(&fp, file_name, "rb");

	int magic_number = 0;
	int sample_count = 0;
	int n_rows = 0, n_cols = 0, padding = 2;

	fread((char*)&magic_number, sizeof(magic_number), 1, fp);
	magic_number = SwapEndien_32(magic_number);
	fread((char*)&sample_count, sizeof(sample_count), 1, fp);
	sample_count = SwapEndien_32(sample_count);
	fread((char*)&n_rows, sizeof(n_rows), 1, fp);
	n_rows = SwapEndien_32(n_rows);
	fread((char*)&n_cols, sizeof(n_cols), 1, fp);
	n_cols = SwapEndien_32(n_cols);

	double scale_max = 1.0;
	double scale_min = -1.0;
	unsigned char temp = 0;
	int size = width * height;
	int mem_size = size * sizeof(double);
	for (int k = 0; k < sample_count; k++)
	{
		sample[k].data = (double *)malloc(mem_size);

		for (int i = 0; i < size; i++)
		{
			sample[k].data[i] = scale_min;
		}

		for (int i = 0; i < n_rows; i++)
		{
			for (int j = 0; j < n_cols; j++)
			{
				fread((char*)&temp, sizeof(temp), 1, fp);
				sample[k].data[(i + padding)*width + j + padding] = ((double)temp / 255.0) * (scale_max - scale_min) + scale_min;
			}
		}
	}

	fclose(fp);
	fp = NULL;
}

// 读训练/测试集标签
void read_mnist_label(Sample *sample, const char *file_name)
{
	FILE *fp = NULL;
	fopen_s(&fp, file_name, "rb");

	int magic_number = 0;
	int sample_count = 0;

	fread((char*)&magic_number, sizeof(magic_number), 1, fp);
	magic_number = SwapEndien_32(magic_number);

	fread((char*)&sample_count, sizeof(sample_count), 1, fp);
	sample_count = SwapEndien_32(sample_count);

	unsigned char label = 0;
	int mem_size = classes_count * sizeof(double);
	for (int k = 0; k < sample_count; k++)
	{
		sample[k].label = (double *)malloc(mem_size);
		for (int i = 0; i < classes_count; i++)
		{
			sample[k].label[i] = -0.8;
		}

		fread((char*)&label, sizeof(label), 1, fp);
		sample[k].label[label] = 0.8;
	}

	fclose(fp);
	fp = NULL;
}
//*-------------------------------------------------------------------------------------------------------/


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

	//getch();

	//test input
	Layer input;
	input.addNeural(1);
	input.addNeural(2);
	Layer output;
	output.addNeural(1);

	//data
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
	nets.Scale(divx, divy); 

		/*
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
	nets.Scale(divx, divy);*/

	clock_t start, end;
	int count = 0;
	while (1) {
		count++;

		nets.setLearningRate(0.95);

		start = clock();
		//nets.Train((double**)X, (double**)Y, sample_size, in_size, out_size, 0.01);
		nets.Train((double**)X, (double**)Y, sample_size, in_size, out_size, 1e-5, 3, 10, sample_size_real, 5000);
		end = clock();
		printf("\ntime=%f\n", (double)(end - start) / CLK_TCK);

		//if (kbhit())
		{
			printf("\nTraining: %d\n", count);
			nets.Traverse();

#ifdef _CNN_SHOW_GUI_
			EFTYPE width_r = (EFTYPE)sample_size_real / show_width;
			EFTYPE height_r = (EFTYPE)divy / show_height;
			//EP_ClearDevice();
			int x, y, ex = 0, ey = 0, ex1 = 0, ey1 = 0;
			for (int i = 1; i < sample_size_real; i++) {
				input.setNeural(X[i], in_size);
				nets.Forecast(input, &output);
				x = i / width_r;
				y = Y[i][0] / height_r;
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
			EP_RenderFlush();
#endif
			int lind = -1;
			while (1) {
				INT ind;
				while (scanf("%d", &ind) != 1) {
					getchar();
					fflush(stdin);
				}
				if (ind < 0 || ind >= sample_size_real) {
					break;
				}

#ifdef _CNN_SHOW_GUI_
				if (lind >= 0) {
					x = lind / width_r;
					for (int i = 0; i < show_height; i++) {
						if (EP_GetPixel(x, i) == YELLOW) {
							EP_SetPixel(x, i, BLACK);
						}
					}
				}
				x = ind / width_r;
				for (int i = 0; i < show_height; i++) {
					if (EP_GetPixel(x, i) == BLACK) {
						EP_SetPixel(x, i, YELLOW);
					}
				}
				EP_RenderFlush();
#endif
				lind = ind;

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
						f = f * f / (divy * divy);
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

#ifdef _CNN_SHOW_GUI_
VOID onKeyDown(WPARAM wParam)
{
}
EPointF drag;
VOID onDrag(FLOAT x, FLOAT y, INT mode)
{
	if (mode == 1) // mouse down
	{
		drag.X = x;
		drag.Y = y;
	}
	else if (mode == 2) // mouse move
	{
		if (EP_NTZERO(drag.X) && EP_NTZERO(drag.Y))
		{
			EP_SetColor(BLUE);
			ege::setlinewidth(20);
			EP_Line(x, y, drag.X, drag.Y);
			drag.X = x;
			drag.Y = y;
		}
	}
	else	// mouse up
	{
		drag.X = 0;
		drag.Y = 0;
	}
}
__NANOC_THREAD_FUNC_BEGIN__(GUIThread) {
	ThreadParam *param = (ThreadParam*)pv;
	HANDLE_MUTEX &mutex = param->mutex;
	HANDLE_MUTEX &main_mutex = param->main_mutex;
	int tid = param->tid;
	int pull = 0;
	while (true) {
		tid = param->tid;
		if (tid < 0) {
			__NANOC_THREAD_MUTEX_UNLOCK__(main_mutex);
			break;
		}
		if (tid > 0) {
			while (EP_MouseHit())
			{
				EP_MouseMsg msg = EP_GetMouseMsg();
				if (EP_IsLeft(msg)) {
					if (pull > 0) {
						pull = 0;
						EP_ClearDevice();
						param->tid = 2;
					}
					if (EP_IsDown(msg))
					{
						onDrag(msg.x, msg.y, 1);
					}
					else if (EP_IsUp(msg))
					{
						onDrag(msg.x, msg.y, 0);
					}
				}
				if (EP_IsMove(msg))
				{
					onDrag(msg.x, msg.y, 2);
				}
			}
			if (EP_KBMsg()) {
				EP_MSG msg = EP_GetKBMsg();
				if (EP_KBIsDown(msg)) {
					onKeyDown(EP_GetKey(msg));
				}
			}
			EP_RenderFlush();
		}
		else {
			pull = 1;
			::Sleep(100);
		}
	}

	__NANOC_THREAD_FUNC_END__(0);
}
#endif
int test_sample() {

	// 训练数据
	Sample *train_sample = (Sample *)malloc(train_sample_count * sizeof(Sample));
	memset(train_sample, 0, train_sample_count * sizeof(Sample));
	train_sample->sample_w = width;
	train_sample->sample_h = height;
	train_sample->sample_count = train_sample_count;
	read_mnist_data(train_sample, "./mnist/train-images.idx3-ubyte");
	read_mnist_label(train_sample, "./mnist/train-labels.idx1-ubyte");

	// 测试数据
	Sample *test_sample = (Sample *)malloc(test_sample_count * sizeof(Sample));
	memset(test_sample, 0, test_sample_count * sizeof(Sample));
	test_sample->sample_w = width;
	test_sample->sample_h = height;
	test_sample->sample_count = test_sample_count;
	read_mnist_data(test_sample, "./mnist/t10k-images.idx3-ubyte");
	read_mnist_label(test_sample, "./mnist/t10k-labels.idx1-ubyte");

	INT i, j, k;

	Network nets(LayerMode::Input, LayerMode::Output);

	//inputs
	nets.input.addNeural(1);

	//outputs
	nets.output.addNeural(1);
	nets.output.addNeural(2);
	nets.output.addNeural(3);
	nets.output.addNeural(4);
	nets.output.addNeural(5);
	nets.output.addNeural(6);
	nets.output.addNeural(7);
	nets.output.addNeural(8);
	nets.output.addNeural(9);
	nets.output.addNeural(10);

	//layers
	INT layers[][4] = {
		{LayerMode::Input, 1, width, height},//input
		{LayerMode::Conv, 6, 5, 5},//convolutional 1
		{LayerMode::MaxPool, 6, 1, 1},//pooling 2
		{LayerMode::Conv, 16, 5, 5},//convolutional 3
		{LayerMode::MaxPool, 16, 1, 1},//pooling 4
		{LayerMode::Conv, 120, 5, 5},//convolutional 5
		{LayerMode::Output, 10, 1, 1},//output
		{LayerMode::Normal, 10, 1, 1},//normal
		{LayerMode::Normal, 10, 1, 1},//normal
	};
	for (i = 0; i < 6; i++) {
		if (layers[i][0] == LayerMode::Input) {
			continue;
		}
		Layer * hidden = new Layer((LayerMode)layers[i][0]);

		//formula of perfect hidden num:
		//sqrt(in_num + out_num) + a
		//a is 5
		for (j = 0; j < layers[i][1]; j++) {
			hidden->addNeural(1 + j + (i + 1) * 1000);
		}

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
#define O true
#define X false
	bool connection_table[6 * 16] =
	{
		O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
		O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
		O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
		X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
		X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
		X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
	};
#undef O
#undef X
	Layer * hidden = nets.layers.link;
	if (hidden) {
		i = 0;
		Layer * _hidden = nets.layers.next(hidden);
		if (_hidden && _hidden != nets.layers.link) {
			do {

				if (hidden->neurals.linkcount == 16 && _hidden->neurals.linkcount == 6) {
					hidden->makeConnection(*_hidden, ++i, connection_table);
				}
				else {
					hidden->makeConnection(*_hidden, ++i);
				}

				hidden = _hidden;
				_hidden = nets.layers.next(_hidden);
			} while (_hidden && _hidden != nets.layers.link);
		}
	}

	//make matrix
	INT layer_index = 0;
	hidden = nets.layers.link;
	if (hidden) {
		do {
			hidden->makeMatrix(layers[layer_index][2], layers[layer_index][3]);
			layer_index++;

			hidden = nets.layers.next(hidden);
		} while (hidden && hidden != nets.layers.link);
	}

	nets.Traverse();

	//getch();

	//test input
	Layer input(LayerMode::Input);
	input.addNeural(1);
	Layer output(LayerMode::Output);
	output.addNeural(1);
	output.addNeural(2);
	output.addNeural(3);
	output.addNeural(4);
	output.addNeural(5);
	output.addNeural(6);
	output.addNeural(7);
	output.addNeural(8);
	output.addNeural(9);
	output.addNeural(10);

	input.makeMatrix(width, height);
	output.makeMatrix(1, 1);

	INT sample_size = train_sample_count;
	INT sample_size_real = test_sample_count;
	INT in_size = train_sample->sample_w * train_sample->sample_h;
	INT out_size = 10;
	EFTYPE divx = 1.0;
	EFTYPE divy = 1.0;
	nets.Scale(divx, divy);

#ifdef _CNN_SHOW_GUI_
	ThreadParam param;
	param.tid = 0;
	__NANOC_THREAD_MUTEX_INIT__(main_mutex, (&param));
	__NANOC_THREAD_MUTEX_LOCK__(param.main_mutex);
	__NANOC_THREAD_BEGIN__(param.thread, GUIThread, &param);
	EFTYPE *draw = new EFTYPE[width * height];
#endif
	clock_t start, end;
	int count = 0;
	while (1) {
		count++;

		nets.setLearningRate(0.03);

		start = clock();
		//nets.TrainCNN(train_sample, 300, sample_size, in_size, out_size, 0.1);
		nets.TrainCNN(train_sample, 300, sample_size, in_size, out_size, 0.0001, 3, 10, 0);
		//nets.TrainCNN(train_sample, 10, 100, in_size, out_size, 0.0001, 3, 10);
		end = clock();
		printf("\ntime=%f\n", (double)(end - start) / CLK_TCK);

		//if (kbhit())
		{
			printf("\nTraining: %d\n", count);
			nets.Traverse();
			INT ind = 0;
			while (1) {
#ifdef _CNN_SHOW_GUI_
				param.tid = 1;
#endif
				while (scanf("%d", &ind) != 1) {
					getchar();
					fflush(stdin);
				}
#ifdef _CNN_SHOW_GUI_
				double scale_max = 1.0;
				double scale_min = -1.0;
				int is_input = 0;
				if (param.tid == 2) {
					is_input = 1;
				}
				param.tid = 0;

				if (is_input) {
					printf("input: %d\n", ind);

					EFTYPE width_r = (EFTYPE)width / show_width;
					EFTYPE height_r = (EFTYPE)height / show_height;
					for (int i = 0; i < width; i++) {
						for (int j = 0; j < height; j++) {
							int ind = i * width + j;
							int x = (int)(j / width_r);
							int y = (int)(i / height_r);
							ECOLOR color = EP_GetPixel(x, y);
							draw[ind] = ((double)color / 255.0) * (scale_max - scale_min) + scale_min;
						}
					}
					nets.input.setNeuralMatrix(draw, in_size);
					for (int i = 0; i < out_size; i++) {
						ECOLOR temp = 0;
						if (i == ind) {
							draw[i] = 0.8;
						}
						else {
							draw[i] = -0.8;
						}
					}
					nets.output.setNeural(draw, out_size);
				}
				else {
					if (ind < 0 || ind >= sample_size_real) {
						break;
					}

					nets.input.setNeuralMatrix(test_sample[ind].data, in_size);
					nets.output.setNeural(test_sample[ind].label, out_size);
				}
#else
				if (ind < 0 || ind >= sample_size_real) {
					break;
				}

				nets.input.setNeuralMatrix(test_sample[ind].data, in_size);
				nets.output.setNeural(test_sample[ind].label, out_size);
#endif
				//nets.input.setNeuralMatrix(train_sample[ind].data, in_size);
				//nets.output.setNeural(train_sample[ind].label, out_size);
				nets.Forecast(input, &output);

#ifdef _CNN_SHOW_CONV_
				char c = 0;
				Layer * layer;
				int tc = 1;
				int l = 0;
				int n = 0;
				for (int t = 0; t < tc; t++) {
					layer = nets.layers.link;
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
												v = neural->map.data[ind];
											}
											ECOLOR color = (ECOLOR)(((v - scale_min) / (scale_max - scale_min)) * 255.0);
											EP_SetPixel(i, j, color);
										}
									}
									EP_RenderFlush();

									printf("tid: %d/%d, layer: %d/%d, neural: %d/%d\n", t + 1, tc, l, nets.layers.linkcount, n, layer->neurals.linkcount);
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

							layer = nets.layers.next(layer);
						} while (layer && layer != nets.layers.link);
					}
					if (c == 'q' || c == 'Q') {
						break;
					}
				}
#endif

				printf("\n");
				/*
				for (i = 0; i < in_size; i++) {
					printf("%e ", train_sample[ind].data[i]);
				}*/
				printf("\n");
				printf("Actual:\n");

				Neural * neural = nets.output.neurals.link;
				EFTYPE result_softmax = 0;
				EFTYPE predict_softmax = 0;
				if (neural) {
					do {
						result_softmax += exp(neural->map.label[0]);
						predict_softmax += exp(neural->map.data[0]);

						neural = nets.output.neurals.next(neural);
					} while (neural && neural != nets.output.neurals.link);
				}
				EFTYPE e = 0;
				int result = -1;
				int predict = -1;
				EFTYPE softmax;
				EFTYPE result_softmax_max = 0;
				EFTYPE predict_softmax_max = 0;
				i = 0;
				if (neural) {
					do {
						printf("%e %e", neural->map.label[0], neural->map.data[0]);
						EFTYPE f = neural->map.label[0] - neural->map.data[0];
						f = f * f / (divy * divy);
						e += f;
						printf(" error: %lf\n", f);
						softmax = exp(neural->map.label[0]) / result_softmax;
						if (softmax > result_softmax_max) {
							result_softmax_max = softmax;
							result = i;
						}
						softmax = exp(neural->map.data[0]) / result_softmax;
						if (softmax > predict_softmax_max) {
							predict_softmax_max = softmax;
							predict = i;
						}
						i++;

						neural = nets.output.neurals.next(neural);
					} while (neural && neural != nets.output.neurals.link);
				}
				printf("result: %d  predict: %d\n", result, predict);
				printf("total error: %lf\n", e);
			}
			if (ind < 0) {
				break;
			}
		}
	}
#ifdef _CNN_SHOW_GUI_
	//release sem
	//send end singal
	param.tid = -1;
	__NANOC_THREAD_MUTEX_LOCK__(param.main_mutex);
	__NANOC_THREAD_CLOSE__(param.thread, GUIThread, &param);
	delete[] draw;
#endif
	// 释放资源
	for (int i = 0; i < train_sample_count; i++)
	{
		free(train_sample[i].data);
		free(train_sample[i].label);
		train_sample[i].data = NULL;
		train_sample[i].label = NULL;
	}
	free(train_sample);

	for (int i = 0; i < test_sample_count; i++)
	{
		free(test_sample[i].data);
		free(test_sample[i].label);
		test_sample[i].data = NULL;
		test_sample[i].label = NULL;
	}
	free(test_sample);
	return 0;
}
//将一个10进制整数转换为2进制数
void int2binary(double _n, double *arr, int binary_dim)
{
	int i = 0;
	int n = (int)_n;
	while (n)
	{
		arr[i++] = n % 2;
		n /= 2;
	}
	while (i < binary_dim)
		arr[i++] = 0;
}

int test_rnn() {
	INT i, j, k;
	
	Network nets(LayerMode::RecursiveIn, LayerMode::RecursiveOut);

	//inputs
	nets.input.addNeural(1);
	nets.input.addNeural(2);

	//outputs
	nets.output.addNeural(1);
	
	int hidden_size = 26;
	Layer * hidden = new Layer(LayerMode::Recursive);
	for (i = 0; i < hidden_size; i++) {
		hidden->addNeural(1 + i);
	}
	nets.hiddens.insertLink(hidden);
	nets.layers.insertLink(hidden, &nets.output, NULL);

	//make connections
	hidden = nets.layers.link;
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

	hidden = nets.hiddens.link;
	if (hidden) {
		do {

			hidden->makeRecursiveConnection();

			hidden = nets.hiddens.next(hidden);
		} while (hidden && hidden != nets.hiddens.link);
	}

	nets.Traverse();

	INT sample_size = 1000;
	INT sample_size_real = 11000;
	INT in_size = 2;
	INT out_size = 1;
	EFTYPE divx = 1.0;
	EFTYPE divy = 1.0;
	nets.Scale(divx, divy);

	INT serial_size = 8;
	double * X = new double[sample_size_real * in_size * serial_size];
	double * Y = new double[sample_size_real * out_size * serial_size];

	int largest_number = pow(2, serial_size);


	nets.prepareRNN(serial_size);

	clock_t start, end;
	int count = 0;
	while (1) {
		count++;

		nets.setLearningRate(0.1);

		for (i = 0; i < sample_size_real; i++) {
			X[i * in_size * serial_size + 0 * serial_size + 0] = (int)((double)rand() / RAND_MAX * largest_number / 2.0);
			X[i * in_size * serial_size + 1 * serial_size + 0] = (int)((double)rand() / RAND_MAX * largest_number / 2.0);
			Y[i * out_size * serial_size + 0 * serial_size + 0] = X[i * in_size * serial_size + 0 * serial_size + 0] + X[i * in_size * serial_size + 1 * serial_size + 0];
			if (i < 10) {
				printf("%.2f %.2f %.2f\n", X[i * in_size * serial_size + 0 * serial_size + 0], X[i * in_size * serial_size + 1 * serial_size + 0], Y[i * out_size * serial_size + 0 * serial_size + 0]);
			}
			int2binary(X[i * in_size * serial_size + 0 * serial_size + 0], &X[i * in_size * serial_size + 0 * serial_size + 0], serial_size);
			int2binary(X[i * in_size * serial_size + 1 * serial_size + 0], &X[i * in_size * serial_size + 1 * serial_size + 0], serial_size);
			int2binary(Y[i * out_size * serial_size + 0 * serial_size + 0], &Y[i * out_size * serial_size + 0 * serial_size + 0], serial_size);
		}

		start = clock();
		nets.TrainRNN((double**)X, (double**)Y, sample_size, in_size, out_size, 1e1, serial_size, sample_size_real, 10000);
		end = clock();
		printf("\ntime=%f\n", (double)(end - start) / CLK_TCK);


		//if (kbhit())
		{
			//EP_ClearDevice();
			int lind = -1;
			while (1) {
				INT ind;
				while (scanf("%d", &ind) != 1) {
					getchar();
					fflush(stdin);
				}
				if (ind < 0 || ind >= sample_size_real) {
					break;
				}

#ifdef _CNN_SHOW_GUI_
				int x;
				EFTYPE width_r = (EFTYPE)sample_size_real / show_width;
				EFTYPE height_r = (EFTYPE)divy / show_height;
				if (lind >= 0) {
					x = lind / width_r;
					for (int i = 0; i < show_height; i++) {
						if (EP_GetPixel(x, i) == YELLOW) {
							EP_SetPixel(x, i, BLACK);
						}
					}
				}
				x = ind / width_r;
				for (int i = 0; i < show_height; i++) {
					if (EP_GetPixel(x, i) == BLACK) {
						EP_SetPixel(x, i, YELLOW);
					}
				}
				EP_RenderFlush();
#endif
				lind = ind;

				Layer& input = nets.input;
				Layer& output = nets.output;
				MultiLinkList<Layer>& hiddens = nets.hiddens;
				Layer* hidden;
				Neural * neural;
				NeuralGate *pgate, *lpgate;
				EFTYPE divoutrange = nets.divoutrange;
				EFTYPE divrange = nets.divrange;
				
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
			}
		}
	}
	delete[] X;
	delete[] Y;

	nets.releaseRNN();

	return 0;
}
int main(int argc, _TCHAR* argv[])
{
#ifdef _CNN_SHOW_GUI_
	EP_Init(show_width, show_height);
#endif
	while (1) {
		//test1();
		//test_sample();
		test_rnn();
	}
}
