#include <iostream>

#include "mobilenet.h"

int main(int args, const char **argv) {
  assert(argc == 2 && "Expected VMFB filepath to follow the executable file");
  const char *filepath = argv[1];

  auto mobileNetDriver = shortfin::cpp::MobileNetImpl();
  const auto prog_module = mobileNetDriver.loadProgramModule(filepath);

  std::vector<std::string> exports = prog_module.exports();
  for (auto exp : exports) {
    std::cerr << exp << "\n";
  }

  return 0;
}
