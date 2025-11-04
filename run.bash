# yamelize euroc dataset
bash ./scripts/euroc/yamelize.bash -p /root/euroc
# build
mkdir -p build && cd build
cmake .. && make -j$(nproc)

# run stereoVIO on euroc
bash ./scripts/stereoVIOEuroc.bash -p /root/euroc/MH_01_easy