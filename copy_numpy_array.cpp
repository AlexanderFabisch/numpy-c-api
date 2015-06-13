#include <python_interpreter.hpp>
#include <vector>
#include <iostream>


int main()
{
    const PythonInterpreter& python = PythonInterpreter::instance();
    std::vector<double> array;
    array.push_back(1.2);
    array.push_back(2.3);
    array.push_back(3.4);
    python.callFunction("functions", "take_array", array);
    std::vector<double> result = python.callReturnFunction(
        "functions", "produce_array");
    for(auto r = result.begin(); r != result.end(); r++)
        std::cout << *r << ", ";
    std::cout << std::endl;
}