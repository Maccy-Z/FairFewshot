for num_row in 2 6 10
do
    for i in {0..9}
    do  
        source_path="andrija@andrija-pc:~/FairFewshot/results/result_iwata_kshot_v2_"$i"_row_"$num_row""
        destination_path="andrija_pc_results/iwata_kshot_v2/result_iwata_kshot_v2_"$i"_row_"$num_row""
        mkdir -p andrija_pc_results/iwata_kshot_v2
        scp -r  $source_path $destination_path
    done
done