
SRC_FILE=CNN.cpp NN/mt19937ar.cpp common/MultiLinkList.cpp platform/EPaint.cpp platform/EPoint.cpp
LIB_FILE=-lgraphics -lgdiplus -limm32 -lmsimg32 -lole32 -loleaut32 -lwinmm -luuid -mwindows -mconsole

linux_gru:
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_GRU_ -o CNN_gru CNN.cpp NN/mt19937ar.cpp -lpthread
linux_lstm:
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_LSTM_ -o CNN_lstm CNN.cpp NN/mt19937ar.cpp -lpthread
linux_cnn:
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_CNN_ -o CNN_cnn CNN.cpp NN/mt19937ar.cpp -lpthread
linux_line1:
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_LINEAR1_ -o CNN_line1 CNN.cpp NN/mt19937ar.cpp -lpthread
linux_line2:
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_LINEAR2_ -o CNN_line2 CNN.cpp NN/mt19937ar.cpp -lpthread
linux_line:
	g++ -g -O3 -D_NANO_LINUX_ -D_NANO_LINEAR_ -o CNN_line CNN.cpp NN/mt19937ar.cpp -lpthread
	
gru:
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_GRU_ -o CNN_gru.exe ${SRC_FILE} ${LIB_FILE}
lstm:
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_LSTM_ -o CNN_lstm.exe ${SRC_FILE} ${LIB_FILE}
cnn:
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_CNN_ -o CNN_cnn.exe ${SRC_FILE} ${LIB_FILE}
line1:
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_LINEAR1_ -o CNN_line1.exe ${SRC_FILE} ${LIB_FILE}
line2:
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_LINEAR2_ -o CNN_line2.exe ${SRC_FILE} ${LIB_FILE}
line:
	g++ -g -O3 -D_NANO_MINGW_ -D_NANO_LINEAR_ -o CNN_line.exe ${SRC_FILE} ${LIB_FILE}
	
clean:
	rm ./CNN_gru.exe
	rm ./CNN_lstm.exe
	rm ./CNN_cnn.exe
	rm ./CNN_line1.exe
	rm ./CNN_line2.exe
	rm ./CNN_line.exe
	rm ./CNN_gru
	rm ./CNN_lstm
	rm ./CNN_line1
	rm ./CNN_lin2
	rm ./CNN_line
	
gru_run:
	./CNN_gru
lstm_run:
	./CNN_lstm
cnn_run:
	./CNN_cnn
line1_run:
	./CNN_line1
line2_run:
	./CNN_line2
line_run:
	./CNN_line
	
	
	
