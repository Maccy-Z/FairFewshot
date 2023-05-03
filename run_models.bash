# for i in {0..9}
# do
#     python Fewshot/config.py --ds-group $i --num-rows 5 --num-targets 5
#     python Fewshot/main.py
# done

for i in {0..9}
do
    python Fewshot/config.py --ds-group $i --num-rows 10 --num-targets 10
    python Fewshot/main.py
done
python Fewshot/comparison2.py

for i in {0..9}
do
    python Fewshot/config.py --ds-group $i --num-rows 5 --num-targets 5
    python Fewshot/main.py
done
python Fewshot/comparison2.py