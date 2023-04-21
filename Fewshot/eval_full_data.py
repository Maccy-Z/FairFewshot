from evaluate_real_data import main as eval_main
from tabulate import tabulate

if __name__ == "__main__":
    data = {}
    for ds_group in range(-1, 6, 1):
        row_accs = eval_main(save_no=-1, ds_group=ds_group, print_result=False)
        transposed_dict = {
            col_name: {row_name: row_data[col_name] for row_name, row_data in row_accs.items()}
            for col_name in row_accs[1]
        }

        col_data = []
        for num_cols, acc_batch in row_accs.items():
            for model_name, acc in acc_batch.items():
                col_data.append(f'{acc:.3f}')
            col_data.append(" ")


        data[ds_group] = col_data
        print(col_data)

    row_headers = []
    for _ in range(10):
        row_headers += ["fewshot", "LR", "SVC", ""]


    table = tabulate(data, headers="keys", showindex=row_headers)
    print(table)