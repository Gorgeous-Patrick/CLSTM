#include <iostream>
#include <vector>

int main() {
  // Generate a large random vector
  std::vector<float> v(1000000);
  for (int i = 0; i < v.size(); i++) {
    v[i] = rand() % 1000 / 1000.0;
  }
  // Sum it up
  float sum = 0;
  for (int j = 0; j < 1000; j++) {
  for (int i = 0; i < v.size(); i++) {
    sum += v[i];
  }
  }
  std::cout << "Sum: " << sum << std::endl;
}