//EPaintEx.h
//Paint external functions' declaration
//Using lpng and ljpeg to load image
//Author: Stophin
//2021.12.23
//Ver: 0.01
//
#ifndef _EPAINTEX_H_
#define _EPAINTEX_H_

#include "ELibrary.h"
typedef DWORD * PDWORD;

#define HAVE_UNSIGNED_SHORT
#define HAVE_BOOLEAN
#include "../ljpeg/jconfigint.h"
#include "../ljpeg/jmorecfg.h"
#include "../ljpeg/jpeglib.h"
#include "../lpng/png.h"
#include "../lpng/pnginfo.h"

class IMAGE_EX {
public:
	IMAGE_EX() : m_pBuffer(NULL) {}
	~IMAGE_EX() {
		if (m_pBuffer) {
			delete[] m_pBuffer;
			m_pBuffer = NULL;
		}
	}
	int         m_width;
	int         m_height;
	PDWORD      m_pBuffer;

	int  resize(int width, int height);
	int  getimage(LPCSTR pImgFile, int zoomWidth = 0, int zoomHeight = 0);
	int getimage_pngfile(LPCSTR  filename, int zoomWidth = 0, int zoomHeight = 0);
	int  getpngimg(FILE* fp, int zoomWidth = 0, int zoomHeight = 0);
	void getimage_from_png_struct(void*, void*, int zoomWidth = 0, int zoomHeight = 0);
	int getimage_jpgfile(LPCSTR  filename, int zoomWidth = 0, int zoomHeight = 0);
	int  getjpgimg(FILE* fp, int zoomWidth = 0, int zoomHeight = 0);
	void getimage_from_jpg_struct(void*, void*, int zoomWidth = 0, int zoomHeight = 0);
};
typedef IMAGE_EX* PIMAGE_EX;
typedef const IMAGE_EX* PCIMAGE_EX;

typedef unsigned int uint32;

#define USING_EPAINT_TEST

#endif //end of _EPAINTEX_H_
//end of file
