# Define the compiler to use
CC = g++

# Define the compiler flags
CFLAGS = -std=c++17

# Define the target executable name
TARGET = progam

# Define the source files
SRC = main.cpp \
      include/read_obj.cpp \
      include/gaussian_curvature.cpp \
      include/extract_features.cpp \
      include/general_operations.cpp \
      include/mean_curvature.cpp \
      include/save_result.cpp 

# Default target when 'make' is executed
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(SRC)
	$(CC) -o ./bin/$(TARGET) $(SRC) $(CFLAGS)

# Rule to run the executable
run:
	./bin/$(TARGET)

# Rule to clean up the build artifacts
clean:
	rm -f $(TARGET)
