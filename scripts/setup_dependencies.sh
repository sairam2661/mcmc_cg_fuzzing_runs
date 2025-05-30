#!/usr/bin/env bash

DATA_DIR="${DATA_DIR}"

sudo apt-get update
sudo apt-get install -y build-essential git clang llvm llvm-dev libclang-dev \
                       automake autoconf libtool pkg-config libc++-dev libc++abi-dev

mkdir -p "$DATA_DIR"

if [ ! -d "$DATA_DIR/AFLplusplus" ]; then
  git clone https://github.com/AFLplusplus/AFLplusplus.git "$DATA_DIR/AFLplusplus"
fi

cd "$DATA_DIR/AFLplusplus"
make -j"$(nproc)"

export PATH="$PWD:$PATH"
export AFLPP_DRV_LIB="$DATA_DIR/AFLplusplus/utils/aflpp_driver/libAFLDriver.a"
