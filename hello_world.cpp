#include <python_interpreter.hpp>


int main()
{
    const PythonInterpreter& python = PythonInterpreter::instance();
    python.callFunction("functions", "hello_world");
}