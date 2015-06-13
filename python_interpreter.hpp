#include <memory>
#include <string>
#include <vector>


// Forward declare PyObject as suggested on the python mailing list
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

/**
 * A managed pointer to PyObject which takes care about
 * memory management and reference counting.
 *
 * \note Reference counting only works if makePyObjectPtr() is used to create
 *       the pointer. Therefore you should always use makePyObjectPtr() to
 *       create new PyObjectPtrs.
 *
 * \note This type should only be used to encapsulate PyObjects that are
 *       'new references'. Wrapping a 'borrowed reference' will break Python.
 */
typedef std::shared_ptr<PyObject> PyObjectPtr;


class PythonInterpreter
{
    PythonInterpreter();
    ~PythonInterpreter();
public:
    static const PythonInterpreter& instance();

    PyObjectPtr create1dBuffer(double* array, unsigned size) const;
    PyObjectPtr createPyString(const std::string& str) const;
    PyObjectPtr importModule(const PyObjectPtr& module) const;
    PyObjectPtr getAttribute(PyObjectPtr obj,
                             const std::string attribute) const;
    void callFunction(const std::string& module,
                      const std::string& function,
                      std::vector<double>& array) const;
    void callFunction(const std::string& module,
                      const std::string& function) const;
};