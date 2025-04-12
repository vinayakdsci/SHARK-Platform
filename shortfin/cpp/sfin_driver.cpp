#include <cstdio>
#include <fstream>

#include "mobilenet.h"

namespace {
std::vector<float> readImageBinary(std::filesystem::path path) {
  std::ifstream in(path, std::ios_base::in | std::ios_base::binary);
  if (!in.is_open()) {
    std::cerr << "Could not open file " << path << "\n";
    exit(1);
  }
  std::vector<float> raw_bin_data;
  uint32_t val;
  while (in.read(reinterpret_cast<char *>(&val), sizeof(val))) {
    raw_bin_data.push_back(static_cast<float>(val));
  }
  return raw_bin_data;
}

}  // namespace

namespace fs = std::filesystem;

int main(int argc, const char **argv) {
  if (argc != 4) {
    std::cerr << "Expected VMFB, IRPA and image binary filepaths to follow the "
                 "executable file"
              << std::endl;
    exit(1);
  }
  const char *filepath = argv[1];
  const char *irpa_path = argv[2];
  const char *image_bin_path = argv[3];

  auto mobileNetDriver = shortfin::cpp::MobileNetService();
  mobileNetDriver.Run(readImageBinary(fs::path(image_bin_path)), filepath,
                      irpa_path);
  // auto device_arr = shortfin::array::device_array::for_device(
  //     mobileNetDriver.device(), std::to_array<size_t>({1, 3, 224, 224}),
  //     shortfin::array::DType::float32());

  // auto host_arr = device_arr.for_transfer();
  // auto image_bin = readImageBinary(fs::path(image_bin_path));

  // {
  //   auto map = host_arr.typed_data_w<float>();
  //   std::copy(image_bin.begin(), image_bin.end(), map.begin());
  // }

  // device_arr.copy_from(host_arr);

  // int count = 0;
  // while (count < 2000) {
  //   mobileNetDriver.loadMobileNetProgram(filepath, irpa_path);
  //   count++;
  // }

  return 0;
}
