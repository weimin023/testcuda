# opencv
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build && cd build
cmake -G Ninja ..
ninja -j 16
sudo ninja install
cd ..
cd ..
rm -rf opencv