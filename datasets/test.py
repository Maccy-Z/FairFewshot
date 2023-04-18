import requests
import os

datanames = os.listdir("/mnt/storage_ssd/FairFewshot/datasets/data")



for file in datanames:

    url = f'https://archive.ics.uci.edu/ml//machine-learning-databases/{file}/'

    response = requests.get(url)
    if response.status_code==200:
        print("OK", file)
        dataset = response.content

        #print(dataset)

    #     with open(f"/mnt/storage_ssd/FairFewshot/datasets/dataset_repaired/{file}.data", "wb") as f:
    #         f.write(response.content)
    #
    # else:
    #     print()
    #     print("Failed", file)




