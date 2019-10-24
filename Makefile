CC=g++
CFLAGS=-c -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment
LDFLAGS=

#Executable name
EXECUTABLE=CBWH

#Source files to compile
SRC_DIR=./src/
SRC_PATTERN ='*.cpp'
SOURCES = $(shell find $(SRC_DIR) -name $(SRC_PATTERN))
OBJECTS=$(SOURCES:.cpp=.o)

#Includes
OPENCV_INCLUDES = /opt/installation/OpenCV-3.4.4/include
LOCAL_INCLUDES = ./src/
INCLUDE=$(OPENCV_INCLUDES) $(LOCAL_INCLUDES)
INC_PARAMS=$(foreach d, $(INCLUDE), -I$d) #Add '-I' to each include

#libs
PATH_LIB = /opt/installation/OpenCV-3.4.4/lib
LIBS = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_ml -lopencv_videoio -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_flann

#Rules for compilation and linking
all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS)
	@echo $(SOURCES)
	@echo $(OBJECTS)
	$(CC) $(LDFLAGS) $(INC_PARAMS) $(OBJECTS) -o $@ -L$(PATH_LIB) $(LIBS) -lm

.cpp.o:
	$(CC) $(CFLAGS) $(INC_PARAMS) $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
