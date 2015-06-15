#include <memory>
#include <string>

#include <vector>


struct ObjectState;
struct FunctionState;
struct MethodState;
struct ModuleState;
class Object;
class Function;
class Method;
class Module;

enum CppType
{
    INT, DOUBLE, BOOL, STRING, ONEDARRAY
};

class Object
{
    std::shared_ptr<ObjectState> state;
public:
    Object(std::shared_ptr<ObjectState> state);
    Method& method(const std::string& name);
    std::shared_ptr<std::vector<double> > as1dArray();
    double asDouble();
    int asInt();
    bool asBool();
    std::string asString();
};

// TODO Function and Method may contain duplicate code
class Function
{
    std::shared_ptr<FunctionState> state;
public:
    Function(ModuleState& module, const std::string& name);
    Function& pass(CppType type);
    Function& call(...);
    std::shared_ptr<Object> returnObject();
};

class Method
{
    std::shared_ptr<MethodState> state;
public:
    Method(ObjectState& object, const std::string& name);
    Method& pass(CppType type);
    Method& call(...);
    std::shared_ptr<Object> returnObject();
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
};
