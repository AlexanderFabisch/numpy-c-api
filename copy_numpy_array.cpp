#include <python_interpreter.hpp>
#include <vector>
#include <string>
#include <iostream>


int main()
{
    const PythonInterpreter& python = PythonInterpreter::instance();

    int intResult = python.import("functions")->function("produce_int").call().returnObject()->asInt();
    std::cout << "int should be 9: " << intResult << std::endl;

    double doubleResult = python.import("functions")->function("produce_double").call().returnObject()->asDouble();
    std::cout << "double should be 8: " << doubleResult << std::endl;

    bool boolResult = python.import("functions")->function("produce_bool").call().returnObject()->asBool();
    std::cout << "bool should be 1: " << boolResult << std::endl;

    std::string stringResult = python.import("functions")->function("produce_string").call().returnObject()->asString();
    std::cout << "string should be 'Test string': " << stringResult << std::endl;

    auto result = python.import("functions")->function("produce_array").call().returnObject()->as1dArray();
    for(double& r: *result)
        std::cout << r << ", ";
    std::cout << std::endl;

    result = python.import("functions")->function("produce_list").call().returnObject()->as1dArray();
    for(double& r: *result)
        std::cout << r << ", ";
    std::cout << std::endl;

    result = python.import("functions")->function("produce_tuple").call().returnObject()->as1dArray();
    for(double& r: *result)
        std::cout << r << ", ";
    std::cout << std::endl;

    python.import("functions")->function("take_int").pass(INT).call(intResult);
    python.import("functions")->function("take_double").pass(DOUBLE).call(doubleResult);
    python.import("functions")->function("take_bool").pass(BOOL).call(boolResult);
    python.import("functions")->function("take_string").pass(STRING).call(&stringResult);

    std::vector<double> array{1.2, 2.3, 3.4};
    python.import("functions")->function("take_array").pass(ONEDARRAY).call(&array);

    result = python.import("functions")->function("multiple_io").pass(DOUBLE).pass(INT).call(2.0, 1).returnObject()->as1dArray();
    for(double& r: *result)
        std::cout << r << ", ";
    std::cout << std::endl;
}
