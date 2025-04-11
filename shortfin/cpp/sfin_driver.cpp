#include "mobilenet.h"

int main(int args, const char **argv) {
  assert(argc == 3 &&
         "Expected VMFB and IRPA filepaths to follow the executable file");
  const char *filepath = argv[1];
  const char *irpa_path = argv[2];

  auto mobileNetDriver = shortfin::cpp::MobileNetImpl();
  mobileNetDriver.loadMobileNetProgram(filepath, irpa_path);

  return 0;
}
