for i in 0 1 2 3 4
do
    python Fewshot/config.py --ds-group $i
    python Fewshot/main.py
done