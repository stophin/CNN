//EPaintEx.cpp
//Paint external functions
//Author: Stophin
//2021.12.23
//Ver: 0.01
//


#include "EPaintEx.h"

int
IMAGE_EX::resize(int width, int height) {

	// 截止到 0
	if (width < 0) width = 0;
	if (height < 0) height = 0;

	if (m_pBuffer) {
		delete[] m_pBuffer;
	}

	m_width = width;
	m_height = height;
	m_pBuffer = new DWORD[width * height];

	return 0;
}

void IMAGE_EX::getimage_from_png_struct(void* vpng_ptr, void* vinfo_ptr, int zoomWidth, int zoomHeight) {
	png_structp png_ptr = (png_structp)vpng_ptr;
	png_infop info_ptr = (png_infop)vinfo_ptr;
	png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_BGR | PNG_TRANSFORM_EXPAND, NULL);
	png_set_expand(png_ptr);
	this->resize(zoomWidth, zoomHeight); //png_get_IHDR

	PDWORD m_pBuffer = this->m_pBuffer;
	const png_uint_32 width = info_ptr->width;
	const png_uint_32 height = info_ptr->height;
	const png_uint_32 depth = info_ptr->pixel_depth;
	png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);

	/*
	for (uint32 i = 0; i < height; ++i) {
		if (depth == 24) {
			for (uint32 j = 0; j < width; ++j) {
				m_pBuffer[i * width + j] = 0xFFFFFF & (DWORD&)row_pointers[i][j * 3];
			}
		}
		else if (depth == 32) {
			for (uint32 j = 0; j < width; ++j) {
				m_pBuffer[i * width + j] = ((DWORD*)(row_pointers[i]))[j];
				if ((m_pBuffer[i * width + j] & 0xFF000000) == 0) {
					m_pBuffer[i * width + j] = 0;
				}
			}
		}
	}
	*/
	float ratioW = width  / (float)zoomWidth;
	float ratioH = height / (float)zoomHeight;
	for (uint32 i = 0; i < zoomHeight; ++i) {
		if (depth == 24) {
			for (uint32 j = 0; j < zoomWidth; ++j) {
				int ii = i * ratioH;
				int jj = j * ratioW;
				m_pBuffer[i * zoomWidth + j] = 0xFFFFFF & (DWORD&)row_pointers[ii][jj * 3];
			}
		}
		else if (depth == 32) {
			for (uint32 j = 0; j < zoomWidth; ++j) {
				int ii = i * ratioH;
				int jj = j * ratioW;
				m_pBuffer[i * zoomWidth + j] = ((DWORD*)(row_pointers[ii]))[jj];
				if ((m_pBuffer[i * zoomWidth + j] & 0xFF000000) == 0) {
					m_pBuffer[i * zoomWidth + j] = 0;
				}
			}
		}
	}
}
int
IMAGE_EX::getpngimg(FILE* fp, int zoomWidth, int zoomHeight) {
	png_structp png_ptr;
	png_infop info_ptr;

	{
		char header[16];
		uint32 number = 8;
		fread(header, 1, number, fp);
		int isn_png = png_sig_cmp((png_const_bytep)header, 0, number);

		if (isn_png) {
			return grIOerror;
		}
		fseek(fp, 0, SEEK_SET);
	}

	png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (png_ptr == NULL) {
		return grAllocError;
	}
	info_ptr = png_create_info_struct(png_ptr);
	if (info_ptr == NULL) {
		png_destroy_read_struct(&png_ptr, NULL, NULL);
		return grAllocError;
	}

	png_init_io(png_ptr, fp);
	this->getimage_from_png_struct(png_ptr, info_ptr, zoomWidth, zoomHeight);

	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
	return grOk;
}
int
IMAGE_EX::getimage_pngfile(LPCSTR  filename, int zoomWidth, int zoomHeight) {
	FILE* fp = NULL;
	int ret;
	fp = fopen(filename, "rb");
	if (fp == NULL) return grFileNotFound;
	ret = this->getpngimg(fp, zoomWidth, zoomHeight);
	fclose(fp);
	return ret;
}



void IMAGE_EX::getimage_from_jpg_struct(void* vpng_ptr, void* vinfo_ptr, int zoomWidth, int zoomHeight) {
	png_structp png_ptr = (png_structp)vpng_ptr;
	png_infop info_ptr = (png_infop)vinfo_ptr;
	png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_BGR | PNG_TRANSFORM_EXPAND, NULL);
	png_set_expand(png_ptr);
	this->resize(zoomWidth, zoomHeight); //png_get_IHDR

	PDWORD m_pBuffer = this->m_pBuffer;
	const png_uint_32 width = info_ptr->width;
	const png_uint_32 height = info_ptr->height;
	const png_uint_32 depth = info_ptr->pixel_depth;
	png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);

	/*
	for (uint32 i = 0; i < height; ++i) {
		if (depth == 24) {
			for (uint32 j = 0; j < width; ++j) {
				m_pBuffer[i * width + j] = 0xFFFFFF & (DWORD&)row_pointers[i][j * 3];
			}
		}
		else if (depth == 32) {
			for (uint32 j = 0; j < width; ++j) {
				m_pBuffer[i * width + j] = ((DWORD*)(row_pointers[i]))[j];
				if ((m_pBuffer[i * width + j] & 0xFF000000) == 0) {
					m_pBuffer[i * width + j] = 0;
				}
			}
		}
	}
	*/
	float ratioW = width / (float)zoomWidth;
	float ratioH = height / (float)zoomHeight;
	for (uint32 i = 0; i < zoomHeight; ++i) {
		if (depth == 24) {
			for (uint32 j = 0; j < zoomWidth; ++j) {
				int ii = i * ratioH;
				int jj = j * ratioW;
				m_pBuffer[i * zoomWidth + j] = 0xFFFFFF & (DWORD&)row_pointers[ii][jj * 3];
			}
		}
		else if (depth == 32) {
			for (uint32 j = 0; j < width; ++j) {
				int ii = i * ratioH;
				int jj = j * ratioW;
				m_pBuffer[i * width + j] = ((DWORD*)(row_pointers[ii]))[jj];
				if ((m_pBuffer[i * width + j] & 0xFF000000) == 0) {
					m_pBuffer[i * width + j] = 0;
				}
			}
		}
	}
}
int
IMAGE_EX::getjpgimg(FILE* fp, int zoomWidth, int zoomHeight) {

	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);

	int row_stride;
	JSAMPARRAY buffer;

	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, fp);

	(void)jpeg_read_header(&cinfo, TRUE);

	(void)jpeg_start_decompress(&cinfo);

	/*
	row_stride = cinfo.output_width * cinfo.output_components;

	int cols = cinfo.output_width;
	int rows = cinfo.output_height;

	buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);


	unsigned int width = cinfo.output_width;
	unsigned int height = cinfo.output_height;


	unsigned char* dstImg = NULL;
	if (cinfo.output_components == 3)
	{
		dstImg = new unsigned char[width * height * 3 + 1];

	}//gray
	else {
		dstImg = new unsigned char[width * height + 1];
	}

	while (cinfo.output_scanline < cinfo.output_height) {
		(void)jpeg_read_scanlines(&cinfo, buffer, 1);
		for (size_t i = 0; i < cinfo.output_width; i++)
		{
			//rgb
			if (cinfo.output_components == 3)
			{
				dstImg[(cinfo.output_scanline - 1) * width * 3 + i * 3] = buffer[0][i * 3];
				dstImg[(cinfo.output_scanline - 1) * width * 3 + i * 3 + 1] = buffer[0][i * 3 + 1];
				dstImg[(cinfo.output_scanline - 1) * width * 3 + i * 3 + 2] = buffer[0][i * 3 + 2];
			}
			else
			{
				dstImg[(cinfo.output_scanline - 1) * width + i] = buffer[0][i];
			}
		}
	}
	*/


	row_stride = cinfo.output_width * cinfo.output_components;

	/* 3.跳过读取的头文件字节Make a one-row-high sample array that will go away when done with image */
	buffer = (*cinfo.mem->alloc_sarray)
		((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

	unsigned char* output_buffer = (unsigned char*)malloc(row_stride * cinfo.output_height);
	memset(output_buffer, 0, row_stride * cinfo.output_height);
	unsigned char* tmp = output_buffer;

	/* 4.Process data由左上角从上到下行行扫描 */
	while (cinfo.output_scanline < cinfo.output_height) {
		(void)jpeg_read_scanlines(&cinfo, buffer, 1);

		memcpy(tmp, *buffer, row_stride);
		tmp += row_stride;
	}


	this->resize(zoomWidth, zoomHeight); //
	PDWORD m_pBuffer = this->m_pBuffer;

	float ratioW = cinfo.output_width / (float)zoomWidth;
	float ratioH = cinfo.output_height / (float)zoomHeight;
	for (uint32 i = 0; i < zoomHeight; ++i) {
		if (cinfo.output_components == 3) {
			for (uint32 j = 0; j < zoomWidth; ++j) {
				int ii = i * ratioH;
				int jj = j * ratioW;
				m_pBuffer[i * zoomWidth + j] = 0xFFFFFF & (DWORD&)output_buffer[ii * row_stride + jj * 3];
			}
		}
		else  {
			for (uint32 j = 0; j < zoomWidth; ++j) {
				int ii = i * ratioH;
				int jj = j * ratioW;
				m_pBuffer[i * zoomWidth + j] = (DWORD)(EGERGB(output_buffer[ii * row_stride + jj], output_buffer[ii * row_stride + jj], output_buffer[ii * row_stride + jj]));
			}
		}
	}
	if (output_buffer) {
		free(output_buffer);
		output_buffer = NULL;
	}

	(void)jpeg_finish_decompress(&cinfo);

	jpeg_destroy_decompress(&cinfo);

	return grOk;
}
int
IMAGE_EX::getimage_jpgfile(LPCSTR  filename, int zoomWidth, int zoomHeight) {
	FILE* fp = NULL;
	int ret;
	fp = fopen(filename, "rb");
	if (fp == NULL) return grFileNotFound;
	ret = this->getjpgimg(fp, zoomWidth, zoomHeight);
	fclose(fp);
	return ret;
}

int
IMAGE_EX::getimage(LPCSTR filename, int zoomWidth, int zoomHeight) {
	{
		int ret = this->getimage_pngfile(filename, zoomWidth, zoomHeight);
		if (ret == 0) return 0;
	}

	this->getimage_jpgfile(filename, zoomWidth, zoomHeight);
	return 0;
}

//end of file
