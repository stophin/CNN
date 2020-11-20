
#include "conio.h"

#ifndef _NANO_LINUX_
//resolve ege conio.h conflit, just compile this file in windows
int getch_console() {
	return _getch();
}
int kbhit_console() {
	return _kbhit();
}
#endif