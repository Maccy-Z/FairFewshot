for i in {0..9}
do
    python Fewshot/config.py --ds-group $i --num_rows 5 --num_targets 5
    python Fewshot/main.py
done

for i in {0..9}
do
    python Fewshot/config.py --ds-group $i --num_rows 10 --num_targets 10
    python Fewshot/main.py
done

for i in {0..9}
do
    python Fewshot/config.py --ds-group $i --num_rows 5 --num_targets 5
    python Fewshot/main.py
done