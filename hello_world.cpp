#include <python_interpreter.hpp>


int main()
{
    const PythonInterpreter& python = PythonInterpreter::instance();
    python.addToPythonpath(".");
    python.import("functions")->function("hello_world").call();
}