#!/bin/sh
# Install the ppc64le cross toolchain + qemu-user on Debian/Ubuntu.
set -e
sudo apt-get update
sudo apt-get install -y g++-14-powerpc64le-linux-gnu qemu-user
echo "Build with: make CXX=powerpc64le-linux-gnu-g++-14 test"
