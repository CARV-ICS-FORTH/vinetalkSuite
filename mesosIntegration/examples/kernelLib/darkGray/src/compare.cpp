/*
 * Copyright 2018 Foundation for Research and Technology - Hellas
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at 
 * http://www.apache.org/licenses/LICENSE-2.0 [1] [1] 
 * 
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
 * See the License for the specific language governing permissions and 
 * limitations under the License. 
 * 
 * Links: 
 * 
 * [1] http://www.apache.org/licenses/LICENSE-2.0 [1] 
 */
#include "../include/CImg.h"
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <string>

using cimg_library::CImg;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;
using std::string;

const unsigned int THRESHOLD = 0;

int main(int argc, char *argv[]) {
  long long unsigned int pixelsAboveThreshold = 0;

  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " <image1> <image2>" << endl;
    return 1;
  }

  CImg<unsigned char> imageOne = CImg<unsigned char>(argv[1]);
  CImg<unsigned char> imageTwo = CImg<unsigned char>(argv[2]);

  if (imageOne.width() != imageTwo.width()) {
    cerr << "The two images have different width." << endl;
    return 1;
  }
  if (imageOne.height() != imageTwo.height()) {
    cerr << "The two images have different height." << endl;
    return 1;
  }
  if (imageOne.spectrum() != imageTwo.spectrum()) {
    cerr << "The two images have different spectrum." << endl;
    return 1;
  }

  CImg<unsigned char> differenceImage = CImg<unsigned char>(
      imageOne.width(), imageOne.height(), 1, imageOne.spectrum());

  for (int y = 0; y < differenceImage.height(); y++) {
    for (int x = 0; x < differenceImage.width(); x++) {
      for (int c = 0; c < differenceImage.spectrum(); c++) {
        differenceImage(x, y, 0, c) =
            abs(imageOne(x, y, 0, c) - imageTwo(x, y, 0, c));

        if (differenceImage(x, y, 0, c) > THRESHOLD) {
          pixelsAboveThreshold++;
        }
      }
    }
  }

  //        differenceImage.save((string(argv[1]) + "_" + string(argv[2]) +
  //        "diff.bmp").c_str());
  cout << fixed << setprecision(2)
       << "Pixels above threshold: " << pixelsAboveThreshold << " ("
       << (100.0 * pixelsAboveThreshold) / differenceImage.size() << "\%)."
       << endl;

  return 0;
}
