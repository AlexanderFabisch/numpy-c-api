// Since Python may define some pre-processor definitions which affect the
// standard headers on some systems, you must include Python.h before any
// standard headers are included.
// (See https://docs.python.org/2.7/c-api/intro.html#include-files)
#include <Python.h>
#include <numpy/arrayobject.h>

#include <python_interpreter.hpp>
#include <memory>
#include <stdexcept>

// Memory management: we use shared pointers for PyObjects

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

/**
 * Make a managed PyObject that will be automatically deleted with the last
 * reference.
 */
PyObjectPtr makePyObjectPtr(PyObject* p)
{
    return PyObjectPtr(p, PyObjectDeleter());
}

PyObjectPtr create1dBuffer(double* array, unsigned size)
{
  //FIXME Python expects sizeof(double) == 8.
  npy_intp dims[1] = {size};
  //FIXME not sure if dims should be stack allocated
  return makePyObjectPtr(
      PyArray_SimpleNewFromData(1, &dims[0], NPY_DOUBLE, (void*) array));
}

PyObjectPtr createPyString(const std::string& str)
{
  PyObjectPtr pyStr = makePyObjectPtr(PyString_FromString(str.c_str()));
  if(!pyStr)
  {
    PyErr_Print();
    throw std::runtime_error("unable to create PyString");
  }
  return pyStr;
}

PyObjectPtr importModule(const PyObjectPtr& module)
{
  PyObjectPtr pyModule = makePyObjectPtr(PyImport_Import(module.get()));
  if(!pyModule)
  {
    PyErr_Print();
    throw std::runtime_error("unable to load module");
  }
  return pyModule;
}

PyObjectPtr getAttribute(PyObjectPtr obj, const std::string attribute)
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

void PythonInterpreter::callFunction(
    const std::string& module, const std::string& function,
    std::vector<double>& array) const
{
    PyObjectPtr pyModuleString = createPyString(module);
    PyObjectPtr pyModule = importModule(pyModuleString);
    PyObjectPtr pyFunc = getAttribute(pyModule, function);

    PyObjectPtr memView = create1dBuffer(&array[0], array.size());

    PyObjectPtr result = makePyObjectPtr(
        PyObject_CallFunction(pyFunc.get(), (char*)"O", memView.get()));

    if(PyErr_Occurred()) {
        PyErr_Print();
        throw std::runtime_error("Error calling " + function);
    }
}

void PythonInterpreter::callFunction(
    const std::string& module, const std::string& function) const
{
    PyObjectPtr pyModuleString = createPyString(module);
    PyObjectPtr pyModule = importModule(pyModuleString);
    PyObjectPtr pyFunc = getAttribute(pyModule, function);

    PyObjectPtr result = makePyObjectPtr(
        PyObject_CallFunction(pyFunc.get(), (char*)""));

    if(PyErr_Occurred()) {
        PyErr_Print();
        throw std::runtime_error("Error calling " + function);
    }
}

std::vector<double> PythonInterpreter::callReturnFunction(
    const std::string& module, const std::string& function) const
{
    PyObjectPtr pyModuleString = createPyString(module);
    PyObjectPtr pyModule = importModule(pyModuleString);
    PyObjectPtr pyFunc = getAttribute(pyModule, function);

    PyObjectPtr result = makePyObjectPtr(
        PyObject_CallFunction(pyFunc.get(), (char*)""));

    // TODO check whether result is a numpy array or list

    int size = PyArray_SIZE(result.get());
    std::vector<double> array(size);

    // TODO we are not sure whether the data is contiguous, use iterator:
    // http://docs.scipy.org/doc/numpy/reference/c-api.array.html#data-access
    double* data = (double *)PyArray_DATA(result.get());
    for(unsigned i = 0; i < size; i++)
        array[i] = data[i];

    if(PyErr_Occurred()) {
        PyErr_Print();
        throw std::runtime_error("Error calling " + function);
    }

    return array;
}
