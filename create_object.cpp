#include <python_interpreter.hpp>
#include <iostream>


int main()
{
    const PythonInterpreter& python = PythonInterpreter::instance();
    python.import("classes")->function("factory").call().returnObject()->method("pr").call();
}
