#include <python_interpreter.hpp>
#include <vector>


int main()
{
    const PythonInterpreter& python = PythonInterpreter::instance();
    std::vector<double> array;
    array.push_back(1.2);
    array.push_back(2.3);
    array.push_back(3.4);
    python.callFunction("functions", "take_array", array);
}