#!/bin/bash
# Script for building and updating the current documentation
rm -r ./build
make html
rm -r ../docs/*
rm -r ../docs/.[!.]*
cp -r ./build/html/* ../docs
cp -r ./build/html/.[!.]* ../docs
