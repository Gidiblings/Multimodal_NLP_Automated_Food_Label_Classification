import json
data = json.load(open('extracted_data.json'))
print('Keys:', list(data[0].keys()) if data else 'No data')
if data:
    print('Sample product_name:', repr(data[0].get('product_name')))
    print('Sample ingredient_list:', repr(data[0].get('ingredient_list')[:100] if data[0].get('ingredient_list') else 'None'))
    print('Sample nutrition:', data[0].get('nutrition'))