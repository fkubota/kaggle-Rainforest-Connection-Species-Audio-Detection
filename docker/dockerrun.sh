#!/bin/sh

if [ $# = 0 ]; then
    echo 引数を入れてください
    exit 1
fi

while getopts "a:b:c" OPT
  do  
    case $OPT in
      a ) FLAG_A=$OPTARG;;
      b ) FLAG_B=$OPTARG;;
      c ) FLAG_C=$OPTARG;;
      \? ) usage ;;
    esac
  done 

echo "$FLAG_A $FLAG_B"
