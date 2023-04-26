for i in 0 1 2 3 4 5 6 7
do
    python Fewshot/config.py --ds-group $i
    python Fewshot/main.py
done