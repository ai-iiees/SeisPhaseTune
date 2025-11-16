@echo off
::setlocal enabledelayedexpansion

set "outdir=X:\Local_Network_Data-Raw\Qeshm\qeshm1_gcf\bagh\1"
:: Replace
set "substring_to_replace=X:\Local_Network_Data-Raw\Qeshm\qeshm1_gcf"
set "replacement_string=E:\Kahbasi-PhD\Qeshm"
call set "outdir=%%outdir:!substring_to_replace!=!replacement_string!%%"

echo %outdir%

::endlocal
