CC		= gcc
CC_FLAGS 	= -O3 -std=c99 -march=native -mfma -mavx -mavx2 -fopenmp
RM 		= rm -f

default: all assemble

all: kernel

kernel: kernel_driver.x
	./kernel_driver.x

kernel_driver.x:
	$(CC) $(CC_FLAGS) -o kernel_driver.x kernel_driver.c

assemble:
	objdump -s -d -f --source ./kernel_driver.x > kernel_driver.S

clean:
	rm -rf *.x *.S