
@ECHO OFF
SETLOCAL

@REM movie2map - a simple tool for generating 2D maps from 2D movies
@REM
@REM 使い方) python.exeにパスを通すか、下の環境変数 PYTHON にフルパスを記載してください

set PYTHON=%USERPROFILE%\AppData\Local\Programs\Python\Python37\python.exe

@REM カレントディレクトリを自身の位置へ
cd /d %~dp0

@REM python.exeの存在チェック
IF EXIST "%PYTHON%" GOTO PYTHONOK
SET PYTHON=python.exe
WHERE /Q %PYTHON%
IF NOT ERRORLEVEL 1 GOTO PYTHONOK
ECHO python.exe が見つかりません。バッチファイルにフルパスを記載するか、パスを通してください。
GOTO ENDOFBATCH
:PYTHONOK

%PYTHON% movie2map.py -test %*
PAUSE

:ENDOFBATCH
ENDLOCAL
