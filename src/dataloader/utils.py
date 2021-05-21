def parse_csv(csv_path):
    with open(csv_path, 'r') as f:
        text = f.readlines()
    text = [t.split(',') for t in text]
    header = text[0]
    header = [h.split('\n')[0] for h in header]
    return header, text[1:]