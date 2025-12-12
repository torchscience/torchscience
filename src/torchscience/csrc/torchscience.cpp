#include <Python.h>
#include <torch/torch.h>

extern "C" {
  /* Creates a dummy empty _csrc module that can be imported from Python.
    The import from Python will load the .so consisting of this file
    in this extension, so that the TORCH_LIBRARY static initializers
    below are run. */
  PyObject* PyInit__csrc(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_csrc",   /* name of module */
          NULL,      /* module documentation, may be NULL */
          -1,        /* size of per-interpreter state of the module,
                        or -1 if the module keeps state in global variables. */
          NULL,      /* methods */
      };
      return PyModule_Create(&module_def);
  }
}
