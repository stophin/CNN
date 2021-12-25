
LJPEG_FILE_EX=$(wildcard ljpeg/*.c)
LPNG_FILE_EX=$(wildcard lpng/*.c)
LJPEG_FILE=$(patsubst ljpeg/jstdhuff.c,,$(patsubst ljpeg/jdmrg565.c,,$(patsubst ljpeg/jccolext.c,,$(patsubst ljpeg/jdcol565.c,,$(patsubst ljpeg/jdmrgext.c,,$(patsubst ljpeg/jdcolext.c,,$(LJPEG_FILE_EX)))))))
LPNG_FILE=$(LPNG_FILE_EX)
SRC_FILE=CNN.cpp NN/mt19937ar.cpp common/MultiLinkList.cpp platform/EPaint.cpp platform/EPaintEx.cpp platform/EPoint.cpp $(LJPEG_FILE:.c=.o) 
LIB_FILE=-lgraphics -lgdiplus -limm32 -lmsimg32 -lole32 -loleaut32 -lwinmm -luuid -mwindows -mconsole
SRC_FILE_LINUX=CNN.cpp NN/mt19937ar.cpp platform/EPaintEx.cpp $(LJPEG_FILE:.c=.o) $(LPNG_FILE:.c=.o) 
LIB_FILE_LINUX=-lpthread -DXMD_H
INC_FOLDER=-I ljpeg -I lpng -DBMP_SUPPORTED -DPPM_SUPPORTED
RM=rm

%.o:%.c
	gcc -c -o $@ $^ ${INC_FOLDER}

linux_gru:${SRC_FILE_LINUX}
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_GRU_ -o CNN_gru ${SRC_FILE_LINUX} ${LIB_FILE_LINUX} 
linux_lstm:${SRC_FILE_LINUX}
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_LSTM_ -o CNN_lstm ${SRC_FILE_LINUX} ${LIB_FILE_LINUX}
linux_cnn:${SRC_FILE_LINUX}
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_CNN_ -o CNN_cnn ${SRC_FILE_LINUX} ${LIB_FILE_LINUX}
linux_cifar:${SRC_FILE_LINUX}
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_CNN_ -D_NANO_CIFAR_ -D_CIFAR_FULL_DATA_ -o CNN_cifar ${SRC_FILE_LINUX} ${LIB_FILE_LINUX}
linux_line1:${SRC_FILE_LINUX}
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_LINEAR1_ -o CNN_line1 ${SRC_FILE_LINUX} ${LIB_FILE_LINUX}
linux_line2:${SRC_FILE_LINUX}
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_LINEAR2_ -o CNN_line2 ${SRC_FILE_LINUX} ${LIB_FILE_LINUX}
linux_line:${SRC_FILE_LINUX}
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_LINEAR_ -o CNN_line ${SRC_FILE_LINUX} ${LIB_FILE_LINUX}
	
gru:${SRC_FILE}
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_GRU_ -o CNN_gru.exe ${SRC_FILE} ${LIB_FILE}
lstm:${SRC_FILE}
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_LSTM_ -o CNN_lstm.exe ${SRC_FILE} ${LIB_FILE}
cnn:${SRC_FILE}
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_CNN_ -o CNN_cnn.exe ${SRC_FILE} ${LIB_FILE}
cifar:${SRC_FILE}
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_CNN_ -D_NANO_CIFAR_ -D_CIFAR_FULL_DATA_ -o CNN_cifar.exe ${SRC_FILE} ${LIB_FILE}
line1:${SRC_FILE}
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_LINEAR1_ -o CNN_line1.exe ${SRC_FILE} ${LIB_FILE}
line2:${SRC_FILE}
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_LINEAR2_ -o CNN_line2.exe ${SRC_FILE} ${LIB_FILE}
line:${SRC_FILE}
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_LINEAR_ -o CNN_line.exe ${SRC_FILE} ${LIB_FILE}
	
clean_obj:
	${RM} ljpeg/*.o
	${RM} lpng/*.o
clean: clean_obj
	${RM} ./CNN_gru.exe
	${RM} ./CNN_lstm.exe
	${RM} ./CNN_cnn.exe
	${RM} ./CNN_cifar.exe
	${RM} ./CNN_line1.exe
	${RM} ./CNN_line2.exe
	${RM} ./CNN_line.exe
	${RM} ./CNN_gru
	${RM} ./CNN_lstm
	${RM} ./CNN_line1
	${RM} ./CNN_lin2
	${RM} ./CNN_line
	
gru_run:
	./CNN_gru
lstm_run:
	./CNN_lstm
cnn_run:
	./CNN_cnn
cifar_run:
	./CNN_cifar
line1_run:
	./CNN_line1
line2_run:
	./CNN_line2
line_run:
	./CNN_line
	
	
	
