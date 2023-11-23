# list

```python
my_list = [1, 2, 3, 4, 5]
assert all(my_list[i] <= my_list[i+1] for i in range(len(my_list)-1))
```

# function

```python
def add(a, b):
    return a + b

assert add(2, 3) == 5
```

# columns in csv

```python
with open('train_extended.csv') as f:
    reader = csv.DictReader(f)

    # Check that the data contains the expected columns
    expected_columns = ['id', 'vendor', , 'count', 'year', 'season']
    assert reader.fieldnames == expected_columns, f"Expected columns: {expected_columns}, but got {reader.fieldnames}"
```

# row count

```python
for row in reader:
    # Check that count is a positive integer
    assert int(row['count']) >= 0, f"invalid count: {row['count']}"
```

