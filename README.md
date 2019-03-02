# movie2map
movie2map is a simple tool for generating 2D maps from 2D movies.

System Requirement:  
* ffmpeg for generating screenshots from movies.
* python with numpy and pillow(PIL)
* Windows or Linux OS

How to install:
1) Install FFMPEG, PYTHON, NUMPY and PILLOW.
   If you already have it, that step is unnecessary.
2) Visit https://github.com/pensil-jp/movie2map
   and click 'Clone or download' and click 'Download ZIP'
   and Download movie2map-XXXXX.zip
3) Extract movie2map-XXXXX.zip to your computer.
4) Windows:
   Open movie2map.bat with Notepad.exe and enter location for ffmpeg.exe and python.exe
   This step is unnecessary if they are included in the environment variable PATH.
   Linux:
   chmod +x movie2map.sh

How to use:
* Windows  
    movie2map.bat [Filename]
* Linux  
    ./movie2map.sh [Filename]

  Please wait a long time. Hopefully, the still image([Filename].png) is completed. Enjoy!
