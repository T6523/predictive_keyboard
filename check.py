import pandas as pd

# read CSV file into pandas DataFrame
df = pd.read_csv('data/test_set_no_answer_final.csv')

# read text file with one word per line
with open('weights/test_set_pred.txt') as f:
    words = f.readlines()
    num_lines = len(words)

# compare number of rows in DataFrame with number of lines in text file
num_rows = len(df)
if num_rows == num_lines:
    print('The number of rows in the DataFrame matches the number of lines in the text file.')
else:
    print('The number of rows in the DataFrame does not match the number of lines in the text file.')

# check every prediction starts with the required first letter
mismatches = []
for i, (required, pred) in enumerate(zip(df['first letter'], words)):
    pred = pred.rstrip('\n')
    if not pred or pred[0].lower() != str(required).lower():
        mismatches.append((i, required, pred))

if not mismatches:
    print('All predictions start with the required first letter.')
else:
    print(f'{len(mismatches)} predictions do NOT start with the required first letter:')
    for i, required, pred in mismatches[:20]:
        print(f'  row {i}: required {required!r}, got {pred!r}')
