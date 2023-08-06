for num_row in 3 5 10
do
    for i in {0..41}
    do
        source_path="andrija@andrija-pc:~/FairFewshot/results/result_"$num_row"_"$i"_overlap"
        destination_path="andrija_pc_results/all_overlap_cleveland/result_"$num_row"_"$i"_overlap"
        mkdir -p andrija_pc_results/all_overlap_cleveland/
        scp -r  $source_path $destination_path
    done
done