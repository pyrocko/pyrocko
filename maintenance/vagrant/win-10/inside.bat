@echo off
if not exist "C:\Users\vagrant\miniconda3.exe" (
    echo Downloading conda installer...
    call curl "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -o "C:\Users\vagrant\miniconda3.exe" -s
) else (
    echo Conda installer already downloaded.
)

if not exist "%UserProfile%\miniconda3" (
    echo Installing conda...
    call "C:\Users\vagrant\miniconda3.exe" /InstallationType=JustMe /S /D="%UserProfile%\miniconda3"

    echo Installing conda modules needed by Pyrocko...
    call "%UserProfile%\miniconda3\Scripts\activate.bat"
    call conda install -y m2-libiconv m2-libintl m2-vim m2-bash m2-patch git 
    call conda install -y setuptools numpy scipy matplotlib pyqt pyyaml progressbar2 requests jinja2 nose
) else (
    echo Conda is already installed, activating...
    call "%UserProfile%\miniconda3\Scripts\activate.bat"
)

if exist "%UserProfile%\pyrocko" (
    rmdir /s /q "%UserProfile%\pyrocko"
)

call git clone -b %1 "C:\vagrant\pyrocko.git" "%UserProfile%\pyrocko"

cd pyrocko

mklink /d "test\data" "C:\vagrant\pyrocko-test-data"
mklink /d "test\example_run_dir" "C:\vagrant\example_run_dir"

call python setup.py install
call python -m pyrocko.print_version deps > "C:\vagrant\test-%1.py3.out"
call python maintenance\run_tests_windows.py %2 >> "C:\vagrant\test-%1.py3.out" 2>&1

exit 0
