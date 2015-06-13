#include <string>
#include <vector>


class PythonInterpreter
{
    PythonInterpreter();
    ~PythonInterpreter();
public:
    static const PythonInterpreter& instance();

    void callFunction(
        const std::string& module, const std::string& function,
        std::vector<double>& array) const;
    void callFunction(
        const std::string& module, const std::string& function) const;
    std::vector<double> callReturnFunction(
        const std::string& module, const std::string& function) const;
};