#!/usr/bin/env bash

DATA_DIR="${DATA_DIR:-/data/saiva}"
LIBXML2_DIR="${LIBXML2_DIR:-$DATA_DIR/libxml2}"
SQLITE_DIR="${SQLITE_DIR:-$DATA_DIR/sqlite}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_DIR/experiment_data}"
HARNESS_DIR="${HARNESS_DIR:-$DATA_DIR/harness}"

command -v afl-fuzz >/dev/null
mkdir -p "$HARNESS_DIR"

setup_xml() {
  BASE_DIR="$OUTPUT_ROOT/xml_seeds/mcmc_mix_0-5_2"
  mkdir -p "$BASE_DIR/input" "$BASE_DIR/llm_seeds" "$BASE_DIR/output_sync" "$BASE_DIR/coverage/sync"
  if [ ! -d "$LIBXML2_DIR" ]; then
    git clone https://github.com/GNOME/libxml2.git "$LIBXML2_DIR"
  fi
  cd "$LIBXML2_DIR"
  ./autogen.sh
  ./configure --without-python
  make -j"$(nproc)"
  make clean
  CC=afl-clang-fast CXX=afl-clang-fast++ ./autogen.sh
  CC=afl-clang-fast CXX=afl-clang-fast++ ./configure --enable-shared=no --without-python
  make -j"$(nproc)" CC=afl-clang-fast CXX=afl-clang-fast++
  ln -sf "$LIBXML2_DIR/xmllint" "$HARNESS_DIR"
}

setup_sql() {
  if [ ! -d "$SQLITE_DIR" ]; then
    git clone https://github.com/sqlite/sqlite.git "$SQLITE_DIR"
  fi
  export CC=afl-clang-fast
  export CXX=afl-clang-fast++
  export AS=llvm-as
  export CXXFLAGS="${CXXFLAGS:-} -stdlib=libc++"
  export CFLAGS="${CFLAGS:-} -DSQLITE_MAX_LENGTH=128000000 \
-DSQLITE_MAX_SQL_LENGTH=128000000 \
-DSQLITE_MAX_MEMORY=25000000 \
-DSQLITE_PRINTF_PRECISION_LIMIT=1048576 \
-DSQLITE_DEBUG=1 \
-DSQLITE_MAX_PAGE_COUNT=16384"
  cd "$SQLITE_DIR"
  ./configure --disable-shared --enable-rtree
  export OUT_DIR="$DATA_DIR/out"
  mkdir -p "$OUT_DIR"
  export LDFLAGS="${LDFLAGS:-} -L$OUT_DIR"
  export AFLPP_DRV_LIB="$DATA_DIR/AFLplusplus/utils/aflpp_driver/libAFLDriver.a"
  export LIBS="${LIBS:-} -lc++ -lc++abi $AFLPP_DRV_LIB"
  make clean
  make -j"$(nproc)" 
  make sqlite3.c
  $CC $CFLAGS -I. test/ossfuzz.c ./sqlite3.o -o sqlite_fuzz $LDFLAGS $LIBS -pthread -ldl -lm
  ln -sf "$SQLITE_DIR/sqlite_fuzz" "$HARNESS_DIR"
}

case "${1:-all}" in
  xml) setup_xml ;;
  sql) setup_sql ;;
  all) setup_xml; setup_sql ;;
  *) exit 1 ;;
esac
