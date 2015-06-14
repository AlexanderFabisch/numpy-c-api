#include <python_interpreter.hpp>
#include <vector>
#include <iostream>


int main()
{
    const PythonInterpreter& python = PythonInterpreter::instance();
    std::vector<double> array{1.2, 2.3, 3.4};
    python.callFunction("functions", "take_array", array);

    std::vector<double> result = python.callReturnFunction(
        "functions", "produce_array");
    for(double& r: result)
        std::cout << r << ", ";
    std::cout << std::endl;

    result = python.callReturnFunction(
        "functions", "produce_list");
    for(double& r: result)
        std::cout << r << ", ";
    std::cout << std::endl;

    result = python.callReturnFunction(
        "functions", "produce_tuple");
    for(double& r: result)
        std::cout << r << ", ";
    std::cout << std::endl;

    // Goal:
    //std::vector<double> result2 = python.import("function").function("multiple_io").passArray().passInt().call(array, 2).returnArray();
    //Object object = pyhon.import("classes").clazz("Test").init().passDouble().call(3.0);
    //object.method("print").passString().call("test");
}