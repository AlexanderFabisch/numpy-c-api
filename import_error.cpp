#include <python_interpreter.hpp>


int main()
{
    const PythonInterpreter& python = PythonInterpreter::instance();
    python.callFunction("bla", "hello_world");
}