num,code or command,comment
1,get_GCF-SYSTID-STRII.AM.py,to find SYSTID and STREAM to make EXPORTINFO.ini file
1.1,making Kaki_EXPORTINFO-GCF2MSD.ini manually,
2,gcf2msd-Guralp.sh,
3,scmssort -u -E [file.mseed] > [sorted.mseed],
4,scart -I [sorted.mseed] [target/directory or leave to default],
5,rename 's/[target-pattern]/[raplace-with-it]/' [finename.xxx],if data name starts with . (there is not network code). so rename files.
