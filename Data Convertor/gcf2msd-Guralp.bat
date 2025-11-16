@echo off
setlocal enabledelayedexpansion

set "ini=X:\Local_Network_Data-Raw\gcf2msd\Qeshm_EXPORTINFO-GCF2MSD.ini"
set "root=X:\Local_Network_Data-Raw\Qeshm\qeshm1_gcf\gole"

set "substring_to_replace=X:\Local_Network_Data-Raw\Qeshm\qeshm1_gcf"
set "replacement_string=E:\Kahbasi-PhD\Qeshm"

for /r "%root%" %%d in (.) do (
    echo Processing directory: %%d
    for %%f in ("%%d\*.gcf") do (
        set "outdir=%%~dpf"
		:: Replace
		call set "outdir=%%outdir:!substring_to_replace!=!replacement_string!%%"
        :: setlocal enabledelayedexpansion
        :: set "outdir=!outdir:raw-gcf-data=MSEED!"
        :: endlocal & set "outdir=%outdir%"
        :: if not exist "%outdir%" mkdir "%outdir%"
		
		::echo "%%f"  "!outdir!"
        gcf2msd "%%f" -o:"!outdir!" -i:"%ini%"
		echo "%%f"  "!outdir!"
    )
)

endlocal
