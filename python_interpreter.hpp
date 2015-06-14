#include <memory>
#include <string>

#include <vector>


struct FunctionState;
struct ModuleState;
class Function;
class Module;

class Function
{
    std::shared_ptr<FunctionState> state;
public:
    Function(ModuleState& module, const std::string& name);
    Function& call();
};

class Module
{
    std::shared_ptr<ModuleState> state;
public:
    Module(const std::string& name);
    Function& function(const std::string& name);
};

class PythonInterpreter
{
    std::shared_ptr<Module> currentModule;

    PythonInterpreter();
    ~PythonInterpreter();
public:
    static const PythonInterpreter& instance();

    std::shared_ptr<Module> import(const std::string& name) const;

    void callFunction(
        const std::string& module, const std::string& function,
        std::vector<double>& array) const;
    std::vector<double> callReturnFunction(
        const std::string& module, const std::string& function) const;
};