from step3_nova_classification import NOVAClassifier

clf = NOVAClassifier(ground_truth_path='ground_truth.xlsx')
tests = {
    'sugar, salt': 'Processed Culinary Ingredients',
    'sugar, salt, cinnamon': 'Processed',
    'olive oil': 'Processed Culinary Ingredients',
    'olive oil, salt, pepper': 'Processed',
    'vitamin a, sugar': 'Processed',
    'sodium benzoate, water': 'Ultra Processed',
    'fresh milk': 'Minimally Processed',
    'fish, lemon': 'Minimally Processed'
}

for ingredients, expected in tests.items():
    print(f"{ingredients} => {clf.classify_nova(ingredients)} (expected {expected})")
