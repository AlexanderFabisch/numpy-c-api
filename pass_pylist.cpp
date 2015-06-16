#include <python_interpreter.hpp>
#include <iostream>


int main()
{
    const PythonInterpreter& python = PythonInterpreter::instance();
    std::string str = "Test string";
    Object list = *python.listBuilder()->pass(DOUBLE).pass(INT).pass(STRING).build(2.0, 1, &str);
    python.import("functions")->function("print_list").pass(OBJECT).call(&list);
}
