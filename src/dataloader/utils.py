def parse_csv(csv_path):
    with open(csv_path, 'r') as f:
        text = f.readlines()
    text = [t.split(',') for t in text]
    return text[0], text[1:]