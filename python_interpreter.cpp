// Since Python may define some pre-processor definitions which affect the
// standard headers on some systems, you must include Python.h before any
// standard headers are included.
// (See https://docs.python.org/2.7/c-api/intro.html#include-files)
#include <Python.h>
#include <numpy/arrayobject.h>

#include <python_interpreter.hpp>
#include <stdexcept>

/**
 * This deleter should be used when managing PyObject with
 * std::shared_ptr
 */
struct PyObjectDeleter
{
    void operator()(PyObject* p) const
    {
        Py_XDECREF(p);
    }
};

PyObjectPtr makePyObjectPtr(PyObject* p)
{
    return PyObjectPtr(p, PyObjectDeleter());
}

PythonInterpreter::PythonInterpreter()
{
    if(!Py_IsInitialized())
        Py_Initialize();
    import_array();
}

PythonInterpreter::~PythonInterpreter()
{
    if(Py_IsInitialized())
        Py_Finalize();
}

const PythonInterpreter& PythonInterpreter::instance()
{
    static PythonInterpreter pythonInterpreter;
    return pythonInterpreter;
}

PyObjectPtr PythonInterpreter::create1dBuffer(
    double* array, unsigned size) const
{
  //FIXME Python expects sizeof(double) == 8.
  npy_intp dims[1] = {size};
  //FIXME not sure if dims should be stack allocated
  return makePyObjectPtr(PyArray_SimpleNewFromData(1, &dims[0], NPY_DOUBLE, (void*) array));
}

PyObjectPtr PythonInterpreter::createPyString(const std::string& str) const
{
  PyObjectPtr pyStr = makePyObjectPtr(PyString_FromString(str.c_str()));
  if(!pyStr)
  {
    PyErr_Print();
    throw std::runtime_error("unable to create PyString");
  }
  return pyStr;
}

PyObjectPtr PythonInterpreter::importModule(
    const PyObjectPtr& module) const
{
  PyObjectPtr pyModule = makePyObjectPtr(PyImport_Import(module.get()));
  if(!pyModule)
  {
    PyErr_Print();
    throw std::runtime_error("unable to load module");
  }
  return pyModule;
}

PyObjectPtr PythonInterpreter::getAttribute(
    PyObjectPtr obj, const std::string attribute) const
{
    PyObjectPtr pyAttr = makePyObjectPtr(PyObject_GetAttrString(
        obj.get(), attribute.c_str()));
    if(!pyAttr)
    {
        PyErr_Print();
        throw std::runtime_error("unable to load python attribute");
    }
    return pyAttr;
}

void PythonInterpreter::callFunction(
    const std::string& module, const std::string& function,
    std::vector<double>& array) const
{
    PyObjectPtr pyModuleString = createPyString(module);
    PyObjectPtr pyModule = importModule(pyModuleString);
    PyObjectPtr pyFunc = getAttribute(pyModule, function);

    PyObjectPtr memView = create1dBuffer(&array[0], array.size());

    PyObject* result = PyObject_CallFunction(
        pyFunc.get(), (char*)"O", memView.get());

    if(PyErr_Occurred()) {
        PyErr_Print();
        throw std::runtime_error("Error calling " + function);
    }

    Py_XDECREF(result);
}

void PythonInterpreter::callFunction(
    const std::string& module, const std::string& function) const
{
    PyObjectPtr pyModuleString = createPyString(module);
    PyObjectPtr pyModule = importModule(pyModuleString);
    PyObjectPtr pyFunc = getAttribute(pyModule, function);

    PyObject* result = PyObject_CallFunction(pyFunc.get(), (char*)"");

    if(PyErr_Occurred()) {
        PyErr_Print();
        throw std::runtime_error("Error calling " + function);
    }

    Py_XDECREF(result);
}
