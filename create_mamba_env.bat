@echo off

set MAMBAPATH=%USERPROFILE%\miniforge3
if not exist "%MAMBAPATH%\" (
    echo Error: Directory %MAMBAPATH% does not exist.
    exit /b 1
)

for /f "tokens=2 delims=:" %%a in ('findstr /r "^name:" environment.yml') do set ENVNAME=%%a
set ENVNAME=%ENVNAME:~1%

call %MAMBAPATH%\Scripts\activate.bat
call mamba env create --prefix ..\%ENVNAME% --file environment.yml -y 
call mamba run --prefix "..\%ENVNAME%" cmd.exe /k

pause