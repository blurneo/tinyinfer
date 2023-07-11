#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/ssc/Desktop/workspace/git_repos/tinyinfer/scripts/build-macos
  make -f /Users/ssc/Desktop/workspace/git_repos/tinyinfer/scripts/build-macos/CMakeScripts/ReRunCMake.make
fi

