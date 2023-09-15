for num_row in 1 3 5 10 15
do
    for i in {0..9}
    do 
        source_path="andrija@andrija-pc:~/FairFewshot/results/result_iwata_"$i"_row_"$num_row""
        destination_path="andrija_pc_results/iwata/result_iwata_"$i"_row_"$num_row""
        mkdir -p andrija_pc_results/iwata
        scp -r  $source_path $destination_path
    done
done