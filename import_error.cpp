#include <python_interpreter.hpp>
#include <stdexcept>
#include <iostream>


int main()
{
    const PythonInterpreter& python = PythonInterpreter::instance();
    try
    {
        python.import("bla");
    }
    catch (std::runtime_error e)
    {
        std::cout << e.what() << std::endl;
    }

    try
    {
        python.import("functions")->function("raise_import_error").call();
    }
    catch (std::runtime_error e)
    {
        std::cout << e.what() << std::endl;
    }
}
