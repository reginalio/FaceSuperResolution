CC = gcc
TARGET = bin/lcr
SOURCES = src/main.c src/load_pgm.c src/util.c src/cg.c

GSL_HEADERS = -I/mnt/applications/gsl/1.16/include
GSL_LDFLAGS = -L/mnt/applications/gsl/1.16/lib
GSL_LDLIBS = -lgsl -lgslcblas -lm


bin/lcr : 
	mkdir -p bin
	$(CC) $(SOURCES) -o $(TARGET) $(GSL_HEADERS) $(GSL_LDFLAGS) $(GSL_LDLIBS)

clean:
	rm -rf $(TARGET)
