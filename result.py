def save_config(config):
    with open(rf"{config['save_path']}\config.txt", 'w', newline='', encoding='utf-8') as f:
        f.write(f'Dataset : {config["data_name"]}')
        f.write(f'')