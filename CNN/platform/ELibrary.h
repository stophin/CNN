//ELibrary.h
//now we're using ege library tool
//you may change this
//once changed, associated files should be changed
//Author: Stophin
//2014.01.08
//Ver: 0.01
//
#ifndef _ELIBRARY_H_
#define _ELIBRARY_H_

#include <Windows.h>
#include <stdio.h>
#include <tchar.h>
#include <conio.h>

#include <math.h>

#define getch _getch
#define scanf scanf_s
#define kbhit _kbhit

#define ZERO	1e-6
#define ISZERO(x) (x > -ZERO && x < ZERO)

#define EFTYPE float

#endif	//end of _ELIBRARY_H_
//end of file
