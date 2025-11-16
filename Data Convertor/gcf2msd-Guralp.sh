ini=/mnt/x/Local_Network_Data-Raw/gcf2msd/Ahar_EXPORTINFO-GCF2MSD.ini
#root=/media/ekarkooti/ext4/Kahbasi-PhD/Kaki/raw-gcf-data
root=/mnt/x/Local_Network_Data-Raw/Ahar_Varzaghan/All_Data/t6180-namarvar
for dir in $(find $root -type d)
do
	echo $dir 
    for input in $(ls ${dir}/*.gcf)
    do
		echo $input
        outdir="${input/raw-gcf-data/MSEED}"
        outdir=$(dirname $outdir)
        #echo $outdir
        /home/ikahbasi/Applications/gcf2msd_Guralp/gcf2msd ${input} -o:$outdir -i:$ini
    done
done
