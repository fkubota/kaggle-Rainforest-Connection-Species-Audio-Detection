#!/bin/sh

# 引数チェック
if [ $# = 0 ]; then
    echo 引数を入れてください
    exit 1
fi

# 引数取得p, g
while getopts "p:g:" OPT
  do  
    case $OPT in
      p ) PORT=$OPTARG;;
      g ) GPU=$OPTARG;;
      \? ) usage ;;
    esac
  done 

# g=-1 なら--gpusオプションを削除
if [ $GPU = -1 ]; then
	sudo docker run --rm -it --shm-size=8g -p $PORT:$PORT -v /etc/localtime:/etc/localtime:ro -v /home/fkubota/Git/:/home/user/work/ fkubota/rfcx bash
else
	sudo docker run --rm -it --shm-size=8g --gpus $GPU -p $PORT:$PORT -v /etc/localtime:/etc/localtime:ro -v /home/fkubota/Git/:/home/user/work/ fkubota/rfcx bash
fi
