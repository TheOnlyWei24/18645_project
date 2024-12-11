CC		= gcc
CC_FLAGS 	= -O3 -std=c99 -march=native -mfma -mavx -mavx2 -fopenmp
RM 		= rm -f

all: serial 

serial: kernel_driver_serial.x
	./kernel_driver_serial.x

omp: kernel_driver_omp.x
	./kernel_driver_omp.x

kernel_driver_serial.x:
	$(CC) $(CC_FLAGS) -o kernel_driver_serial.x kernel_driver_serial.c 

kernel_driver_omp.x:
	$(CC) $(CC_FLAGS) -o kernel_driver_omp.x kernel_driver_omp.c 

assemble:
	objdump -s -d -f --source ./kernel_driver_serial.x > kernel_driver_serial.S

clean:
	rm -rf *.x *.S