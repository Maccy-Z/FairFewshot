for i in {0..7}
do
    python Fewshot/config.py --ds-group $i
    python Fewshot/main.py
done