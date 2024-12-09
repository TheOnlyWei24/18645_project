CC		= gcc
CC_FLAGS 	= -O3 -std=c99 -march=native -mfma -mavx -mavx2
RM 		= rm -f

default: all assemble

all: kernel

kernel: kernel_driver.x
	./kernel_driver.x

kernel_driver.x:
	$(CC) $(CC_FLAGS) -o kernel_driver.x kernel_driver.c 

assemble:
	objdump -s -d -f --source ./kernel_driver.x > kernel_driver.S

python:
	$(CC) $(CC_FLAGS) -fPIC -shared kernel_driver.c -o kernel.so

clean:
	rm -rf *.x *.S