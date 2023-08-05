# for i in {0..4}
# do
#     python Fewshot/config.py --ds-group $i
#     python Fewshot/main.py
# done

for n_overlap in 12 10 8 4 6 2 0
do
    for seed in {0..5}
    do
        i=$((n_overlap * 10 + seed))
        python Fewshot/config.py --ds-group $i
        python Fewshot/main.py
    done
done

python Fewshot/comparison2.py

