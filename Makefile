
all: sock_cli_serv.o add.o
	nvcc add.o sock_cli_serv.o -o app 

add.o: cuda_add.cu 
	nvcc -c cuda_add.cu -o add.o

sock_cli_serv.o: sock_cli_serv.cpp
	nvcc -c sock_cli_serv.cpp

