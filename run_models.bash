# for i in {0..9}
# do
#     python Fewshot/config.py --ds-group $i
#     python Fewshot/main.py
# done
python Fewshot/config.py 
python Fewshot/main.py
python Fewshot/comparison2.py