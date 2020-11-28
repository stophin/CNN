
#include "conio.h"

#ifndef _NANO_LINUX_
//resolve ege conio.h conflit, just compile this file in windows
int getch_console() {
#ifdef _NANO_MINGW_
	return getch();
#else
	return _getch();
#endif
}
int kbhit_console() {
#ifdef _NANO_MINGW_
	return kbhit();
#else
	return _kbhit();
#endif
}
#endif