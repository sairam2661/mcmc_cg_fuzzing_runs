#!/usr/bin/env bash

DATA_DIR="${DATA_DIR}"

# Coverage outputs go under here
COVERAGE_DIR="${COVERAGE_DIR:-$DATA_DIR/coverage}"
HARNESS_COV_DIR="${HARNESS_COV_DIR:-$DATA_DIR/harness_cov}"

# Source repos / originals
LIBXML2_SRC="${LIBXML2_SRC:-$DATA_DIR/libxml2}"
SQLITE_SRC="${SQLITE_SRC:-$DATA_DIR/sqlite}"

# Where to place each build
XML_COV_BUILD="${XML_COV_BUILD:-$COVERAGE_DIR/libxml2}"
SQL_COV_BUILD="${SQL_COV_BUILD:-$COVERAGE_DIR/sqlite}"

mkdir -p "$HARNESS_COV_DIR"

setup_xml_harness() {
  rm -rf "$XML_COV_BUILD"
  git clone "$LIBXML2_SRC" "$XML_COV_BUILD"
  cd "$XML_COV_BUILD"

  export CC=clang
  export CXX=clang++
  export CFLAGS="-fprofile-instr-generate -fcoverage-mapping"
  export CXXFLAGS="$CFLAGS"

  ./autogen.sh \
    --with-http=no \
    --with-python=no \
    --with-lzma=yes \
    --with-threads=no \
    --disable-shared

  make -j"$(nproc)" clean
  make -j"$(nproc)" all

  ln -sf "$XML_COV_BUILD/xmllint" "$HARNESS_COV_DIR/xmllint_cov"
  echo "XML coverage harness ready: $HARNESS_COV_DIR/xmllint_cov"
}

setup_sql_harness() {
  rm -rf "$SQL_COV_BUILD"
  mkdir -p "$SQL_COV_BUILD"
  cp -r "$SQLITE_SRC/." "$SQL_COV_BUILD/"
  cd "$SQL_COV_BUILD"

  # clear any AFL stuff
  unset CC
  unset CXX
  unset CFLAGS
  unset CXXFLAGS
  unset LDFLAGS

  export CC=/usr/bin/clang
  export CXX=/usr/bin/clang++
  export CFLAGS="-fprofile-instr-generate -fcoverage-mapping -fsanitize=fuzzer -g -O0"
  export CXXFLAGS="$CFLAGS"
  export LDFLAGS="-fprofile-instr-generate -fsanitize=fuzzer"

  unset AFL_USE_ASAN
  unset AFL_USE_MSAN

  ./configure --disable-shared --disable-tcl
  make clean
  make -j"$(nproc)"

  # compile the amalgamation
  $CC $CFLAGS -c sqlite3.c -o sqlite3.o

  $CC $CFLAGS -I. \
    test/ossfuzz.c ./sqlite3.o \
    -o sqlite_fuzz_cov \
    $LDFLAGS -pthread -ldl -lm

  ln -sf "$SQL_COV_BUILD/sqlite_fuzz_cov" "$HARNESS_COV_DIR/sqlite_fuzz_cov"
  echo "SQL coverage harness ready: $HARNESS_COV_DIR/sqlite_fuzz_cov"
}

case "${1:-all}" in
  xml) setup_xml_harness ;;
  sql) setup_sql_harness ;;
  all) setup_xml_harness; setup_sql_harness ;;
  *)
    echo "Usage: $0 [xml|sql|all]" >&2
    exit 1
    ;;
esac
