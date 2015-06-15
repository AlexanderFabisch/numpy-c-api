#include <python_interpreter.hpp>
#include <vector>
#include <iostream>


int main()
{
    const PythonInterpreter& python = PythonInterpreter::instance();

    std::vector<double> array{1.2, 2.3, 3.4};
    python.import("functions")->function("take_array").pass1dArray().call(&array);

    auto result = python.import("functions")->function("produce_array").call().return1dArray();
    for(double& r: *result)
        std::cout << r << ", ";
    std::cout << std::endl;

    result = python.import("functions")->function("produce_list").call().return1dArray();
    for(double& r: *result)
        std::cout << r << ", ";
    std::cout << std::endl;

    result = python.import("functions")->function("produce_tuple").call().return1dArray();
    for(double& r: *result)
        std::cout << r << ", ";
    std::cout << std::endl;

    result = python.import("functions")->function("multiple_io").passDouble().passInt().call(2.0, 1).return1dArray();
    for(double& r: *result)
        std::cout << r << ", ";
    std::cout << std::endl;
}
