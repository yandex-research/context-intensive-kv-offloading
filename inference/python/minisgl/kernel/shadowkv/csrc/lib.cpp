#include <torch/extension.h>

void init_higgs_lib(py::module &);
void init_gather_lib(py::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    init_higgs_lib(m);
    init_gather_lib(m);
}
