#!/bin/bash
find ./tasks -name '*.cpp' -o -name '*.hpp'  -o -name '*.h'| xargs clang-format -i
find ./test -name '*.cpp' -o -name '*.hpp'  -o -name '*.h'| xargs clang-format -i
find ./ros2 -name '*.cpp' -o -name '*.hpp'  -o -name '*.cu' -o -name '*.h'| xargs clang-format -i
find ./cuda_infer -name '*.cpp' -o -name '*.hpp'  -o -name '*.cu' -o -name '*.h'| xargs clang-format -i
find ./src -name '*.cpp' -o -name '*.hpp'  -o -name '*.cu' -o -name '*.h'| xargs clang-format -i
find . -name '*.yaml' -o -name '*.yml' | xargs prettier --write
find . -name '*.py' | xargs black
find . -name '*.html' | xargs prettier --write
