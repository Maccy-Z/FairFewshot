import os
import torch
import pickle
import shutil


# Save into file. Automatically make a new folder for every new save.
class SaveHolder:
    def __init__(self, base_dir, nametag=None):
        dir_path = f'../{base_dir}/saves'
        files = [f for f in os.listdir(dir_path) if os.path.isdir(f'{dir_path}/{f}')]
        existing_saves = sorted([int(f[5:]) for f in files if f.startswith("save")])  # format: save_{number}

        if existing_saves:
            save_no = existing_saves[-1] + 1
        else:
            save_no = 0

        self.save_dir = f'../{base_dir}/saves/save_{save_no}'

        print("Making new save folder at: ")
        print(self.save_dir)
        os.mkdir(self.save_dir)

        shutil.copy(f'../{base_dir}/Fewshot/defaults.toml', f'{self.save_dir}/defaults.toml')
        if nametag is not None:
            with open(f'{self.save_dir}/tag.txt', 'w') as f:
                f.write(nametag)

    def save_model(self, model: torch.nn.Module, optim, epoch=0):
        # latest model
        torch.save({"model_state_dict": model.state_dict(),
                    "optim_state_dict": optim.state_dict()}, f'{self.save_dir}/model.pt')

    def save_history(self, hist_dict: dict):
        with open(f'{self.save_dir}/history.pkl', 'wb') as f:
            pickle.dump(hist_dict, f)

