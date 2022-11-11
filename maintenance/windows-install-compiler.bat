@echo off
cd "%UserProfile%"

echo Get VisualStudio build tools installer
curl -SL --output vs_buildtools.exe https://aka.ms/vs/15/release/vs_buildtools.exe

echo Install VisualStudio build tools...
start /w vs_buildtools.exe --quiet --wait --norestart --nocache ^
    --installPath "%ProgramFiles(x86)%\Microsoft Visual Studio\2017\BuildTools" ^
    --add Microsoft.VisualStudio.Workload.MSBuildTools ^
    --add Microsoft.VisualStudio.Workload.VCTools ^
    --includeRecommended

echo Install VisualStudio build tools (done)
cd "%UserProfile%"
