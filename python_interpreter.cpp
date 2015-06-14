// Since Python may define some pre-processor definitions which affect the
// standard headers on some systems, you must include Python.h before any
// standard headers are included.
// (See https://docs.python.org/2.7/c-api/intro.html#include-files)
#include <Python.h>
#include <numpy/arrayobject.h>

#include <python_interpreter.hpp>
#include <memory>
#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////
//////////////////////// Memory management /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
//////////////////////// Python object allocation //////////////////////////////
////////////////////////////////////////////////////////////////////////////////

PyObjectPtr new1dArray(double* array, unsigned size)
{
  //FIXME Python expects sizeof(double) == 8.
  npy_intp dims[1] = {size};
  //FIXME not sure if dims should be stack allocated
  return makePyObjectPtr(
      PyArray_SimpleNewFromData(1, &dims[0], NPY_DOUBLE, (void*) array));
}

PyObjectPtr newString(const std::string& str)
{
  PyObjectPtr pyStr = makePyObjectPtr(PyString_FromString(str.c_str()));
  if(!pyStr)
  {
    PyErr_Print();
    throw std::runtime_error("unable to create PyString");
  }
  return pyStr;
}

////////////////////////////////////////////////////////////////////////////////
//////////////////////// Helper functions //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void throwPythonException()
{
  PyObject* error = PyErr_Occurred(); // Borrowed reference
  if(error != NULL)
  {
    PyObject* ptype, * pvalue, * ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    PyObjectPtr pyexception = makePyObjectPtr(PyObject_GetAttrString(
        ptype, (char*)"__name__"));
    std::string type = PyString_AsString(pyexception.get());

    PyObjectPtr pymessage = makePyObjectPtr(PyObject_Str(pvalue));
    std::string message = PyString_AsString(pymessage.get());

    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptraceback);

    throw std::runtime_error("Python exception (" + type + "): " + message);
  }
}

PyObjectPtr import(const std::string& module)
{
  PyObjectPtr pyModuleName = newString(module);
  PyObjectPtr pyModule = makePyObjectPtr(PyImport_Import(pyModuleName.get()));
  throwPythonException();
  return pyModule;
}

PyObjectPtr getAttribute(PyObjectPtr obj, const std::string attribute)
{
    PyObjectPtr pyAttr = makePyObjectPtr(PyObject_GetAttrString(
        obj.get(), attribute.c_str()));
    throwPythonException();
    return pyAttr;
}

////////////////////////////////////////////////////////////////////////////////
//////////////////////// Type conversions //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

struct NdArray
{
    PyObjectPtr obj;
    static const bool check(PyObjectPtr obj) { return PyArray_Check(obj.get()); }
    const unsigned size() { return PyArray_Size(obj.get()); }
    const int ndim() { return PyArray_NDIM(obj.get()); }
    const bool isDouble() { return PyArray_TYPE(obj.get()) == NPY_DOUBLE; }
    const double get(unsigned i) { return *((double*)PyArray_GETPTR1(obj.get(), (npy_intp)i)); }
};

struct List
{
    PyObjectPtr obj;
    static const bool check(PyObjectPtr obj) { return PyList_Check(obj.get()); }
    const unsigned size() { return PyList_Size(obj.get()); }
    const bool isDouble(unsigned i) { return PyFloat_Check(PyList_GetItem(obj.get(), i)); }
    const double get(unsigned i) { return PyFloat_AsDouble(PyList_GetItem(obj.get(), (Py_ssize_t)i)); }
};

struct Tuple
{
    PyObjectPtr obj;
    static const bool check(PyObjectPtr obj) { return PyTuple_Check(obj.get()); }
    const unsigned size() { return PyTuple_Size(obj.get()); }
    const bool isDouble(unsigned i) { return PyFloat_Check(PyTuple_GetItem(obj.get(), i)); }
    const double get(unsigned i) { return PyFloat_AsDouble(PyTuple_GetItem(obj.get(), (Py_ssize_t)i)); }
};

bool toVector(PyObjectPtr obj, std::vector<double>& result)
{
    bool knownType = true;

    if(NdArray::check(obj))
    {
        NdArray ndarray = {obj};
        const unsigned size = ndarray.size();
        result.resize(size);

        const int ndim = ndarray.ndim();
        if(ndim != 1)
            throw std::runtime_error("Array object has " + std::to_string(ndim)
                                     + " dimensions, expected 1");
        if(!ndarray.isDouble())
            throw std::runtime_error("Array object does not contain doubles");

        for(unsigned i = 0; i < size; i++)
            result[i] = ndarray.get(i);
    }
    else if(List::check(obj))
    {
        List list = {obj};
        const unsigned size = list.size();
        result.resize(size);

        for(unsigned i = 0; i < size; i++)
        {
            if(!list.isDouble(i))
                throw std::runtime_error(
                    "List object contains item that is not a double at index "
                    + std::to_string(i));
            result[i] = list.get(i);
            throwPythonException();
        }
    }
    else if(Tuple::check(obj))
    {
        Tuple tuple = {obj};
        const unsigned size = tuple.size();
        result.resize(size);

        for(unsigned i = 0; i < size; i++)
        {
            if(!tuple.isDouble(i))
                throw std::runtime_error(
                    "Tuple object contains item that is not a double at index "
                    + std::to_string(i));
            result[i] = tuple.get(i);
            throwPythonException();
        }
    }
    else
    {
        knownType = false;
    }

    return knownType;
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
    PyObjectPtr pyModule = import(module);
    PyObjectPtr pyFunc = getAttribute(pyModule, function);

    PyObjectPtr memView = new1dArray(&array[0], array.size());

    PyObjectPtr result = makePyObjectPtr(
        PyObject_CallFunction(pyFunc.get(), (char*)"O", memView.get()));

    throwPythonException();
}

void PythonInterpreter::callFunction(
    const std::string& module, const std::string& function) const
{
    PyObjectPtr pyModule = import(module);
    PyObjectPtr pyFunc = getAttribute(pyModule, function);

    PyObjectPtr result = makePyObjectPtr(
        PyObject_CallFunction(pyFunc.get(), (char*)""));

    throwPythonException();
}

std::vector<double> PythonInterpreter::callReturnFunction(
    const std::string& module, const std::string& function) const
{
    PyObjectPtr pyModule = import(module);
    PyObjectPtr pyFunc = getAttribute(pyModule, function);

    PyObjectPtr result = makePyObjectPtr(
        PyObject_CallFunction(pyFunc.get(), (char*)""));

    throwPythonException();

    std::vector<double> array;
    const bool knownType = toVector(result, array);
    if(!knownType)
        throw std::runtime_error(function + " does not return a sequence of "
                                 "doubles");

    return array;
}
