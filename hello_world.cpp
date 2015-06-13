#include <python_interpreter.hpp>
#include <vector>


int main()
{
    const PythonInterpreter& python = PythonInterpreter::instance();
    python.callFunction("functions", "hello_world");
}