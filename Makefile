all: bin/server

bin/server: bin/Main.o bin/Model.o bin/Server.o
	g++ -o bin/server bin/Main.o bin/Model.o bin/Server.o

bin/Main.o: src/Main.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/Main.cpp -o bin/Main.o

bin/Model.o: src/Model.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/Model.cpp -o bin/Model.o

bin/Server.o: src/Server.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/Server.cpp -o bin/Server.o

clean:
	rm -f bin/*