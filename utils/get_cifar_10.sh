#!/bin/bash

script_dir=$(dirname "$0")
demo_dir="$script_dir/../src/demo"
resource_dir="$script_dir/../resources"
#out="$demo_dir/cifar-10-binary.tar.gz"
dest="$resource_dir/cifar-10-binary.tar.gz"
echo "Downloading CIFAR-10 Dataset..."
curl -# -o "$dest" http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz 2>&1
echo "Extracting files..."
cd "$resource_dir" && tar xvzf cifar-10-binary.tar.gz
rm cifar-10-binary.tar.gz
