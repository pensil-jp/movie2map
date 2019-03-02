# movie2map
movie2mapは、2Dムービーから2Dマップを生成するためのシンプルなツールです。

システム要件：
* ffmpeg - 動画から静止画のを生成するために使用
* python - ライブラリとして numpy と pillow(PIL) が必要
* Windows もしくは Linux OS

インストール手順：
1) FFMPEG、PYTHON、NUMPY、PILLOWをインストールします。
   既にインストール済みの場合は不要です。
2) https://github.com/pensil-jp/movie2map を開き、
   [Clone or download]をクリックして、[Download ZIP] をクリックして、
   movie2map-XXXXX.zip をダウンロードします。
3) movie2map-XXXXX.zipをコンピュータに解凍します。
4) Windowsの場合：
   メモ帳などで movie2map.bat を開き、ffmpeg.exeとpython.exeの場所を入力します。
   環境変数PATHに含まれている場合、この手順は不要です。
   Linuxの場合：
   chmod + x movie2map.sh を実行して実行権限を付与します。

使い方：
* Windows  
     movie2map.bat [ファイル名]
* Linux  
     ./movie2map.sh [ファイル名]

   しばらくお待ちください。結構かかります。
   うまくいけば、静止画像（[ファイル名].png）が完成します。
   Enjoy!
