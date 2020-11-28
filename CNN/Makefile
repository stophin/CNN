
SRC_FILE=CNN.cpp NN/mt19937ar.cpp common/MultiLinkList.cpp platform/EPaint.cpp platform/EPoint.cpp
LIB_FILE=-lgraphics -lgdiplus -limm32 -lmsimg32 -lole32 -loleaut32 -lwinmm -luuid -mwindows -mconsole

linux_gru:
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_GRU_ -o CNN CNN.cpp NN/mt19937ar.cpp -lpthread
linux_lstm:
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_LSTM_ -o CNN CNN.cpp NN/mt19937ar.cpp -lpthread
linux_cnn:
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_CNN_ -o CNN CNN.cpp NN/mt19937ar.cpp -lpthread
linux_line1:
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_LINEAR1_ -o CNN CNN.cpp NN/mt19937ar.cpp -lpthread
linux_line2:
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_LINEAR2_ -o CNN CNN.cpp NN/mt19937ar.cpp -lpthread
linux_line:
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_LINEAR_ -o CNN CNN.cpp NN/mt19937ar.cpp -lpthread
	
gru:
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_GRU_ -o CNN.exe ${SRC_FILE} ${LIB_FILE}
lstm:
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_LSTM_ -o CNN.exe ${SRC_FILE} ${LIB_FILE}
cnn:
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_CNN_ -o CNN.exe ${SRC_FILE} ${LIB_FILE}
line1:
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_LINEAR1_ -o CNN.exe ${SRC_FILE} ${LIB_FILE}
line2:
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_LINEAR2_ -o CNN.exe ${SRC_FILE} ${LIB_FILE}
line:
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_LINEAR_ -o CNN.exe ${SRC_FILE} ${LIB_FILE}
	
clean:
	rm ./CNN.exe
	rm ./CNN
	
run:
	./CNN
