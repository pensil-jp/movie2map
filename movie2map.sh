#!/bin/sh

echo movie2map - 動画からマップを作成するツール ver0.1 by pensil 2019.02.26
echo

# 引数が足りなければメッセージを出して終了
if [ $# -ne 1 ]; then
  echo "usage : movie2map.sh [FILENAME(.mp4)]"
  exit 1
fi

# 前回の作業ファイルの削除
rm -f work/*.png

# ffmpegのパラメータについて
#  -r : 1秒あたり何枚までフレーム分解するか。移動量の誤検知があるときは値を大きくしてください。
#       デフォルト5、最大30です。あまり大きくすると解析に時間がかかります。
# -ss : 開始時間(秒)
#  -t : 終了時間(秒)

# ffmpegによる動画→静止画分解
ffmpeg -i $1 -ss 0 -r 5 -f image2 work/%06d.png

# 静止画の解析と結合
python movie2map.py

# 出力ファイルのリネーム
mv map.png $1.png
