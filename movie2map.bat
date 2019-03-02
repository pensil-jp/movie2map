@ECHO OFF
SETLOCAL

@ECHO movie2map - ���悩��}�b�v���쐬����c�[�� ver0.1 by pensil 2019.02.26
@ECHO.
@REM (�g����)
@REM ffmpeg.exe�Ƀp�X��ʂ����A���̊��ϐ� FFMPEG �ɃR�}���h�̃t���p�X���L�ڂ��Ă�������
@REM python.exe���p�X��ʂ����A���̊��ϐ� PYTHON �Ƀt���p�X���L�ڂ��Ă�������

set FFMPEG=%USERPROFILE%\Downloads\ffmpeg\bin\ffmpeg.exe
set PYTHON=%USERPROFILE%\AppData\Local\Programs\Python\Python37\python.exe

@REM ��������̏ꍇ�͎g����������
if "%1"=="" goto USAGE

@REM �J�����g�f�B���N�g�������g�̈ʒu��
cd /d %~dp0

@REM ffmpeg.exe�̑��݃`�F�b�N
IF EXIST "%FFMPEG%" GOTO FFMPEGOK
SET FFMPEG=ffmpeg.exe
WHERE /Q %FFMPEG%
IF NOT ERRORLEVEL 1 GOTO FFMPEGOK
SET CMD=WHERE /R "%USERPROFILE%\Downloads" ffmpeg.exe
FOR /f "usebackq tokens=*" %%a in (`%CMD%`) DO @SET FFMPEG=%%a
IF EXIST "%FFMPEG%" GOTO FFMPEGOK
ECHO ffmpeg.exe ��������܂���B�o�b�`�t�@�C���Ƀt���p�X���L�ڂ��邩�p�X��ʂ��Ă��������B
GOTO ENDOFBATCH
:FFMPEGOK

@REM python.exe�̑��݃`�F�b�N
IF EXIST "%PYTHON%" GOTO PYTHONOK
SET PYTHON=python.exe
WHERE /Q %PYTHON%
IF NOT ERRORLEVEL 1 GOTO PYTHONOK
ECHO python.exe ��������܂���B�o�b�`�t�@�C���Ƀt���p�X���L�ڂ��邩�A�p�X��ʂ��Ă��������B
GOTO ENDOFBATCH
:PYTHONOK

@REM �O��̍�ƃt�@�C���̍폜
IF NOT EXIST "work\000001.png" GOTO START
DEL /F /Q work\*.png

:START
@REM ffmpeg�̃p�����[�^�ɂ���
@REM  -r : 1�b�����艽���܂Ńt���[���������邩�B�ړ��ʂ̌댟�m������Ƃ��͒l��傫�����Ă��������B
@REM       �f�t�H���g5�A�ő�30�ł��B���܂�傫������Ɖ�͂Ɏ��Ԃ�������܂��B
@REM -ss : �J�n����(�b)
@REM  -t : �I������(�b)

@REM ffmpeg�ɂ�铮�恨�Î~�敪��
%FFMPEG% -i %1 -ss 0 -r 5 -f image2 work\%%06d.png

@REM �Î~��̉�͂ƌ���
%PYTHON% movie2map.py

@REM �o�̓t�@�C���̃��l�[��
RENAME "map.png" "%1.png"

GOTO ENDOFBATCH

:USAGE
ECHO.
ECHO �g����(Usage) : movie2map.bat [FILENAME]

:ENDOFBATCH
ENDLOCAL
