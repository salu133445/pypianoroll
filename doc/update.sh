#!/bin/bash
# Script for building and updating the current documentation
if [ ! -d source ]; then
    echo "Source directory not found. Probably in the wrong working directory."
    exit 1
fi
rm -r ./build
make html
rm -r ../docs/*
rm -r ../docs/.[!.]*
cp -r ./build/html/* ../docs
cp -r ./build/html/.[!.]* ../docs
