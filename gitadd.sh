#!/bin/bash
STYLEARG="-style=file"  # 或 -style=LLVM 等

for FILE in $(git diff --cached --name-only HEAD | grep -E '\.(c|cpp|h|hpp)$')
do
  clang-format -i ${STYLEARG} "$FILE"
  git add "$FILE"
done

exit 0
