root_dir=/mnt/e/Kahbasi-PhD/Qeshm/

dir1=/mnt/x/Local_Network_Data-Raw/Qeshm/sorted_msd
dir2=/mnt/x/Local_Network_Data-SeisComP/Qeshm
find "$root_dir" -mindepth 1 -maxdepth 1 -type d | while IFS= read -r dir; do
    echo "$dir"
    dir_name=$(basename "$dir")
	
	if [[ "$dir_name" == *bagh* ]]; then
        echo "Skipping directory: $dir_name"
        continue
    fi
	if [[ "$dir_name" == *FDGH* ]]; then
        echo "Skipping directory: $dir_name"
        continue
    fi
	
	echo "$dir_name"
    scmssort -v -u -E ${dir}/*/*.msd > ${dir1}/${dir_name}.msd
    scart -I ${dir1}/${dir_name}.msd  ${dir2}
done
