#include <iostream>
#include <vector>

//#######################
// utility functions
//#######################

void print_vector(std::vector<float> &v);
void write_vector_file(std::string fname, std::vector<float> &v);
void write_data_file(std::string fname,
                     std::vector<float> &returns,
                     std::vector<float> &values);