https://people.sc.fsu.edu/~jburkardt/cpp_src/sobol/sobol.html

```c++
// compile sobol library, compile demo app, link the two
g++ -c -Wall -I. sobol.cpp  
g++ -c -Wall -I. demo.cpp  
g++ demo.o sobol.o -lm -o demo    
./demo
```