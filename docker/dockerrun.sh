#!/bin/sh

if [ $# = 0 ]; then
    echo 引数を入れてください
    exit 1
fi

while getopts "p:g:" OPT
  do  
    case $OPT in
      p ) PORT=$OPTARG;;
      g ) GPU=$OPTARG;;
      \? ) usage ;;
    esac
  done 
sudo docker run --rm -it --gpus $GPU -p $PORT:$PORT -v /home/fkubota/Git/:/home/user/work/ fkubota/rfcx bash

