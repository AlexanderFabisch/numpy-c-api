#include <Python.h> // Must be included before anything else to avoid warnings
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

PyObjectPtr PythonInterpreter::makePyObjectPtr(PyObject* p) const
{
    return PyObjectPtr(p, PyObjectDeleter());
}

PyObjectPtr PythonInterpreter::getAttribute(
    PyObject* obj, const std::string attribute) const
{
    PyObjectPtr pyAttr = makePyObjectPtr(PyObject_GetAttrString(obj, attribute.c_str()));
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
    PyObjectPtr pyFunc = getAttribute(pyModule.get(), function);

    PyObjectPtr memView = create1dBuffer(&array[0], array.size());

    PyObjectPtr result = PyEval_CallFunction(
        pyFunc.get(), (char*)"O", memView.get());
    // TODO

    /*if(PyErr_Occurred()) {
        PyErr_Print();
        throw std::runtime_error("Error calling " + function);
    }

    Py_XDECREF(result);*/
}