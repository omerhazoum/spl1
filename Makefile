.PHONY: all clean

all: bin/server

bin/server: bin/GPU.o bin/Main.o bin/MemoryBlock.o bin/Model.o bin/Request.o bin/SchedulingPolicy.o bin/Server.o
	g++ -o bin/server bin/GPU.o bin/Main.o bin/MemoryBlock.o bin/Model.o bin/Request.o bin/SchedulingPolicy.o bin/Server.o

bin/GPU.o: src/GPU.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/GPU.cpp -o bin/GPU.o

bin/Main.o: src/Main.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/Main.cpp -o bin/Main.o

bin/MemoryBlock.o: src/MemoryBlock.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/MemoryBlock.cpp -o bin/MemoryBlock.o

bin/Model.o: src/Model.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/Model.cpp -o bin/Model.o

bin/Request.o: src/Request.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/Request.cpp -o bin/Request.o

bin/SchedulingPolicy.o: src/SchedulingPolicy.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/SchedulingPolicy.cpp -o bin/SchedulingPolicy.o

bin/Server.o: src/Server.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/Server.cpp -o bin/Server.o

clean:
	rm -f bin/*