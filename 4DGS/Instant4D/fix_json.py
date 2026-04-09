import json

for split in ['train', 'test']:
    json_path = f'example/bike_3d_result/transforms_{split}.json'
    with open(json_path) as f:
        data = json.load(f)
    for frame in data['frames']:
        # Extract just the filename and make it relative
        fname = frame['file_path'].split('/')[-1]
        frame['file_path'] = f'images/{fname}'
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'Fixed {json_path}')