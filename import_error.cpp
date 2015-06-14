#include <python_interpreter.hpp>


int main()
{
    const PythonInterpreter& python = PythonInterpreter::instance();
    python.import("bla")->function("hello_world").call();
}