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


#ifndef _NANO_LINUX_
#define _NANOC_WINDOWS_
#endif

#define _THREAD_SEM_

#ifndef MIN
#define MIN(A,B)	(((A) <= (B)) ? (A) : (B))
#endif

#ifndef MAX
#define MAX(A,B)	(((A) >= (B)) ? (A) : (B))
#endif

#ifdef _NANOC_WINDOWS_
#include <Windows.h>
#include <stdio.h>
#include <tchar.h>
#include <conio.h>
#include <time.h>

#include <math.h>

#define getch _getch
#define scanf scanf_s
#define kbhit _kbhit
#include <process.h>

typedef void *HANDLE;
typedef HANDLE HANDLE_MUTEX;
#define __NANOC_THREAD_FUNC_DECLARE(hHandle, pFuncName)\
	public:\
	HANDLE hHandle;\
	static unsigned int __stdcall pFuncName(void *pv);
#define __NANOC_THREAD_FUNC_BEGIN__(pFuncName) unsigned int __stdcall pFuncName(void *pv)
#define __NANOC_THREAD_FUNC_END__(nReturn) return nReturn

#define __NANOC_THREAD_BEGIN__(hHandle, pFuncName, pParam) hHandle = (HANDLE)_beginthreadex(NULL, 0, &pFuncName, pParam, 0, 0)
#define __NANOC_THREAD_WAIT__(hHandle) WaitForSingleObject(hHandle, 1000)
#define __NANOC_THREAD_END__(hHandle) TerminateThread(hHandle, 0)

#ifndef _THREAD_SEM_
#define __NANOC_THREAD_MUTEX_INIT__(hMutex, obj) obj->hMutex = CreateMutex(NULL, FALSE, NULL)
#define __NANOC_THREAD_MUTEX_LOCK__(hMutex) WaitForSingleObject(hMutex, INFINITE)
#define __NANOC_THREAD_MUTEX_UNLOCK__(hMutex) ReleaseMutex(hMutex);
#else
#define __NANOC_THREAD_MUTEX_INIT__(hMutex, obj) obj->hMutex = CreateSemaphore(NULL, 1, 1, NULL)
#define __NANOC_THREAD_MUTEX_LOCK__(hMutex) WaitForSingleObject(hMutex, INFINITE)
#define __NANOC_THREAD_MUTEX_UNLOCK__(hMutex) ReleaseSemaphore(hMutex, 1, NULL);
#endif
#else 

#include <stdlib.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>

#include <math.h>

void changemode(int dir)
{
	static struct termios oldt, newt;

	if (dir == 1)
	{
		tcgetattr(STDIN_FILENO, &oldt);
		newt = oldt;
		newt.c_lflag &= ~(ICANON | ECHO);
		tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	}
	else
		tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
}

int kbhit(void)
{
	struct timeval tv;
	fd_set rdfs;

	tv.tv_sec = 0;
	tv.tv_usec = 0;

	FD_ZERO(&rdfs);
	FD_SET(STDIN_FILENO, &rdfs);

	select(STDIN_FILENO + 1, &rdfs, NULL, NULL, &tv);
	return FD_ISSET(STDIN_FILENO, &rdfs);
}

int getch(void)
{
	struct termios tm, tm_old;
	int fd = 0, ch;

	if (tcgetattr(fd, &tm) < 0) {//保存现在的终端设置
		return -1;
	}

	tm_old = tm;
	cfmakeraw(&tm);//更改终端设置为原始模式，该模式下所有的输入数据以字节为单位被处理
	if (tcsetattr(fd, TCSANOW, &tm) < 0) {//设置上更改之后的设置
		return -1;
	}

	ch = getchar();
	if (tcsetattr(fd, TCSANOW, &tm_old) < 0) {//更改设置为最初的样子
		return -1;
	}

	return ch;
}

#define INT		int
#define VOID	void
#define _TCHAR	char

#define min(x, y) (x > y ? y : x)
#define max(x, y) (x > y ? x : y)

#define fopen_s(fp, filename, mode) *fp = fopen(filename, mode)
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>

typedef void * HINSTANCE;
typedef pthread_t HANDLE;
#define __NANOC_THREAD_FUNC_DECLARE(hHandle, pFuncName)\
	private:\
	HANDLE hHandle; \
	static void * pFuncName(void *pv);
#define __NANOC_THREAD_FUNC_BEGIN__(pFuncName) void * pFuncName(void *pv)
#define __NANOC_THREAD_FUNC_END__(nReturn) pthread_exit((void *)nReturn)

#define __NANOC_THREAD_BEGIN__(hHandle, pFuncName, pParam) pthread_create(&hHandle, NULL, pFuncName, pParam)
#define __NANOC_THREAD_WAIT__(hHandle) pthread_cancel(hHandle)
#define __NANOC_THREAD_END__(hHandle) pthread_kill(hHandle, 0)

#ifndef _THREAD_SEM_
typedef pthread_mutex_t HANDLE_MUTEX;
#define __NANOC_THREAD_MUTEX_INIT__(hMutex, obj) pthread_mutex_init(&obj->hMutex, NULL)
#define __NANOC_THREAD_MUTEX_LOCK__(hMutex)  pthread_mutex_lock(&hMutex)
#define __NANOC_THREAD_MUTEX_UNLOCK__(hMutex) pthread_mutex_unlock(&hMutex)
#else
typedef sem_t HANDLE_MUTEX;
#define __NANOC_THREAD_MUTEX_INIT__(hMutex, obj) sem_init(&obj->hMutex, 0, 0);sem_post(&obj->hMutex)
#define __NANOC_THREAD_MUTEX_LOCK__(hMutex)  sem_wait(&hMutex)
#define __NANOC_THREAD_MUTEX_UNLOCK__(hMutex) sem_post(&hMutex)
#endif
#endif

#define ZERO	1e-10
#define ISZERO(x) (x > -ZERO && x < ZERO)

#define EFTYPE double

#endif	//end of _ELIBRARY_H_
//end of file
