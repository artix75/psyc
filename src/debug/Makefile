include ../Makefile.common.mk
#OBJS=../psyc.o ../io.o ../utils.o ../maths.o ../log.o ../convolutional.o ../recurrent.o ../lstm.o ../mnist.o ../debug.o


default: all

profile: $(OBJS) profile.o
	$(CC) -o profile $(OBJS) profile.o $(LDFLAGS)
	valgrind --leak-check=yes ./profile
all: profile
        
