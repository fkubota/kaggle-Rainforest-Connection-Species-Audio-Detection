def save_model_architecture(dir_save, model):
    with open(f'{dir_save}/model.txt', 'w') as f:
        f.write(str(model.__repr__))
