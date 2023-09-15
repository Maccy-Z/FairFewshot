for i in {0..9}
do
    python Fewshot/config.py --ds-group $i
    python Fewshot/ds_base.py
done
python Fewshot/comparison2.py

# python Fewshot/config.py --ds-group 0
# python Fewshot/main.py
# python Fewshot/comparison2.py

