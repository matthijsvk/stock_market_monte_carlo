#include "stock_market_monte_carlo/helpers.h"

#include <fmt/core.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

void print_vector(std::vector<float> &v) {
  fmt::print("v = [ ");
  for (auto vi : v) {
    fmt::print("{:6.3f} ", vi);
  }
  fmt::print(" ]\n");
}

void write_vector_file(const std::string fname, std::vector<float> &v) {
  std::ofstream outFile(fname);
  for (const auto &e : v) outFile << e << ",";
}

void write_data_file(const std::string fname,
                     std::vector<float> &returns,
                     std::vector<float> &values) {
  fmt::print("Writing data to csv file outputs/{}\n", fname);
  std::string dir = "./outputs/";
  std::filesystem::create_directory(dir);
  std::ofstream outFile(fmt::format("output/{}", fname));
  outFile << "Returns,,";
  for (const auto &e : returns) outFile << e << ",";
  outFile << "\nValues,";
  for (const auto &e : values) outFile << e << ",";
}