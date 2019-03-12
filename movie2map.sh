#!/bin/sh

# movie2map - 動画からマップを作成するツール ver0.1 by pensil 2019.02.26

# 前回の作業ファイルの削除
rm -f work/*.png

# 静止画の解析と結合
python movie2map.py $*
