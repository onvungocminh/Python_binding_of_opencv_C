# pybind11_opencv_numpy

An example of pybind11 for cv::Mat <-> np.array

```bash
/project folder
├── build
├── example
│   ├── exemple.so  # generate with make
│   └── example.cpython-36m-x86_64-linux-gnu.so  # generate with setup.py
├── CMakeLists.txt
├── setup.py
└── ...
```

## Generation with make

clone pybind11 repository

git clone https://github.com/pybind/pybind11.git

### Compile

```bash
mkdir build
cd build
# configure make with vcpkg toolchain
cmake .. 
# generate the example.so library
make

```

### Run
```bash
python3 test.py
```


