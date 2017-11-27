// Convolute.h
//

#ifndef _CONVOLUTE_H_
#define _CONVOLUTE_H_

#include "CommonDefine.h"

class Filter {
public:
	Filter():
		filter(NULL){
		filter = new EFTYPE[width * height];
	}
	~Filter() {
		if (filter) {
			delete[] filter;
		}
	}
	
	INT width;
	INT height;
	EFTYPE *filter;
};

class Convolute {
public:
	Convolute() :
		image(NULL),
		tar_image(NULL){
	}
	~Convolute() {
	}

	INT width;
	INT height;
	EFTYPE *image;
	EFTYPE *tar_image;
};

#endif