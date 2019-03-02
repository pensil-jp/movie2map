@ECHO OFF
SETLOCAL

@ECHO movie2map - a simple tool for generating 2D maps from 2D movies  ver0.1 by pensil 2019.02.26
@ECHO.
@REM (使い方)
@REM ffmpeg.exeにパスを通すか、下の環境変数 FFMPEG にコマンドのフルパスを記載してください
@REM python.exeもパスを通すか、下の環境変数 PYTHON にフルパスを記載してください

set FFMPEG=%USERPROFILE%\Downloads\ffmpeg\bin\ffmpeg.exe
set PYTHON=%USERPROFILE%\AppData\Local\Programs\Python\Python37\python.exe

@REM 引数が空の場合は使い方説明へ
if "%1"=="" goto USAGE

@REM カレントディレクトリを自身の位置へ
cd /d %~dp0

@REM ffmpeg.exeの存在チェック
IF EXIST "%FFMPEG%" GOTO FFMPEGOK
SET FFMPEG=ffmpeg.exe
WHERE /Q %FFMPEG%
IF NOT ERRORLEVEL 1 GOTO FFMPEGOK
SET CMD=WHERE /R "%USERPROFILE%\Downloads" ffmpeg.exe
FOR /f "usebackq tokens=*" %%a in (`%CMD%`) DO @SET FFMPEG=%%a
IF EXIST "%FFMPEG%" GOTO FFMPEGOK
ECHO ffmpeg.exe が見つかりません。バッチファイルにフルパスを記載するかパスを通してください。
GOTO ENDOFBATCH
:FFMPEGOK

@REM python.exeの存在チェック
IF EXIST "%PYTHON%" GOTO PYTHONOK
SET PYTHON=python.exe
WHERE /Q %PYTHON%
IF NOT ERRORLEVEL 1 GOTO PYTHONOK
ECHO python.exe が見つかりません。バッチファイルにフルパスを記載するか、パスを通してください。
GOTO ENDOFBATCH
:PYTHONOK

@REM 前回の作業ファイルの削除
IF NOT EXIST "work\000001.png" GOTO START
DEL /F /Q work\*.png

:START
@REM ffmpegのパラメータについて
@REM  -r : 1秒あたり何枚までフレーム分解するか。移動量の誤検知があるときは値を大きくしてください。
@REM       デフォルト5、最大30です。あまり大きくすると解析に時間がかかります。
@REM -ss : 開始時間(秒)
@REM  -t : 終了時間(秒)

@REM ffmpegによる動画→静止画分解
%FFMPEG% -i %1 -ss 0 -r 5 -f image2 work\%%06d.png

@REM 静止画の解析と結合
%PYTHON% movie2map.py

@REM 出力ファイルのリネーム
RENAME map.png %~nx1.png

GOTO ENDOFBATCH

:USAGE
ECHO Usage) : movie2map.bat [FILENAME]

:ENDOFBATCH
ENDLOCAL
