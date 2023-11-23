# assert python

## list

```python
my_list = [1, 2, 3, 4, 5]
assert all(my_list[i] <= my_list[i+1] for i in range(len(my_list)-1))
```

## function

```python
def add(a, b):
    return a + b

assert add(2, 3) == 5
```

## columns in csv

```python
with open('train.csv') as f:
    reader = csv.DictReader(f)

    # Check that the data contains the expected columns
    expected_columns = ['id', 'vendor', , 'count', 'year', 'season']
    assert reader.fieldnames == expected_columns, f"Expected columns: {expected_columns}, but got {reader.fieldnames}"
```

## row count

```python
for row in reader:
    # Check that count is a positive integer
    assert int(row['count']) >= 0, f"invalid count: {row['count']}"
```

# assert pandas

[testing docs](https://pandas.pydata.org/pandas-docs/stable/reference/testing.html)

```python
import pandas as pd
import pandas.testing as pdt
```

## assert index

```python
idx1 = pd.Index([1, 2, 3])
idx2 = pd.Index([1, 2, 3])
pdt.assert_index_equal(idx1, idx2)
```

### index order

```python
idx1 = pd.Index([1, 2, 3])
idx2 = pd.Index([3, 2, 1])

try:
    pdt.assert_index_equal(idx1, idx2)
except AssertionError:
    print("Indexes are not equal by default")

# assert indexes are equal when order not checked
pdt.assert_index_equal(idx1, idx2, check_order=False)
```

### tolerance

```python
idx1 = pd.Index([1.0, 2.0, 3.0])
idx2 = pd.Index([1.01, 2.02, 3.03])

try:
    pdt.assert_index_equal(idx1, idx2)
except AssertionError:
    print("indexes are not equal using default tolerances")

# assert indexes are equal using tolerance of 0.1 for both rtol and atol
pdt.assert_index_equal(idx1, idx2, check_exact=False, rtol=0.1)
pdt.assert_index_equal(idx1, idx2, check_exact=False, rtol=0.1, atol=0.1)
```

### compare 

```python
idx1 = pd.Index(data['pickup_time'].dt.date)
print(idx1[:4])

idx2 = pd.Index(data['dropoff_time'].dt.date)
print(idx2[:4])

# compare idx1 to idx2 using pandas.testing.assert_index_equal
try:
    pdt.assert_index_equal(idx1, idx2, check_exact=True, check_names=False)
    print("the indexes are equal")
except AssertionError:
    print("The two indexes are not equal.")
    # Find the differences between the two indexes
    diff = (idx1 != idx2).sum()
    print(f"Number of differences: {diff}")
    diff_rows = data[idx1 != idx2][['pickup_time', 'dropoff_time']].head(7)
    print("\ntop 7 rows where idx1 and idx2 are different:")
    print(diff_rows)
```

### check idx with np.where

```python
import numpy as np
# find indices of rows where pickup and dropoff dates are different
diff_indices = np.where(idx1 != idx2)[0]

# drop rows with these indices from df
data = data.drop(diff_indices).reset_index(drop=True)

# reset index of df
data.reset_index(drop=True, inplace=True)

# reindex idx1 and idx2 to match new df
idx1 = pd.Index(data['pickup_time'].dt.date)
idx2 = pd.Index(data['dropoff_time'].dt.date)
```

## assert series

```python
s1 = pd.Series([1, 2, 3], name='series_1')
s2 = pd.Series([1, 2, 3], name='series_2')

pdt.assert_series_equal(s1, s2)
pdt.assert_series_equal(s1, s2, check_names=False)
```


### diff data types

```python
s1 = pd.Series([1, 2, 3])
s2 = pd.Series(['1', '2', '3'])

pdt.assert_series_equal(s1, s2)
pdt.assert_series_equal(s1, s2, check_dtype=False)
pdt.assert_series_equal(s1, s2, check_dtype=True)
```

### compare

```python
series_1 = data['pickup_datetime'].dt.date
series_2 = data['dropoff_datetime'].dt.date

try:
    pdt.assert_series_equal(series_1, series_2, check_exact=True, check_names=False)
    print("series are equal")
except AssertionError:
    print("series are not equal")
    # Find the differences
    diff = (series_1 != series_2).sum()
    print(f"number of differences: {diff}")
    diff_rows = data[series_1 != series_2][['pickup_time', 'dropoff_time']].head(7)
    print(diff_rows)
```

### drop rows with diff index

```python
# id indices of rows where series_1 and series_2 diff
diff_indices = series_1.index[series_1 != series_2]

# drop rows with these indices from df
data.drop(diff_indices, inplace=True)
data.reset_index(drop=True, inplace=True)

# reindex series_1 and series_2 to match the new DataFrame
series_1 = data['pickup_datetime'].dt.date
series_2 = data['dropoff_datetime'].dt.date
```

## assert df

```python
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

pdt.assert_frame_equal(df1, df2)
```
### check likeness

```python

# different column names but same data and shape
df1 = pd.DataFrame({'B': [3, 3, 3], 'A': [1, 1, 1]})
df2 = pd.DataFrame({'A': [1, 1, 1], 'B': [3, 3, 3]})

# assert that dfs are alike, same data/shape but maybe not same column names
pdt.assert_frame_equal(df1, df2)
pdt.assert_frame_equal(df1, df2, check_like=True)
```

### compare

```python
data = pd.read_csv('train/train.csv', parse_dates=['pickup_time', 'dropoff_time'], nrows=1000)

df1 = data.copy()
df1['date'] = df1['pickup_time'].dt.date

df2 = data.copy()
df2['date'] = df2['dropoff_time'].dt.date


try:
    pdt.assert_frame_equal(df1[['date']], df2[['date']], check_exact=True, check_names=False)
    print("dfs are equal")
except AssertionError:
    print("dfs are not equal")
    diff = (df_1['date'] != df_2['date']).sum()
    diff_rows = data[df1['date'] != df2['date']][['pickup_time', 'dropoff_time']].head(7)
    print(diff_rows)
```

### drop mismatched rows

```python
date_col1 = df1['date']
date_col2 = df2['date']

mismatched_rows = df1.index[date_col1.ne(date_col2)]

df1.drop(mismatched_rows, inplace=True)
df2.drop(mismatched_rows, inplace=True)
```

## assert numpy

```python
import numpy as np
import numpy.testing as npt
```

```python
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])

npt.assert_array_equal(a, b)
npt.assert_array_equal(a, b, err_msg='data mismatch')
```

```python
string1 = "hello world"
string2 = "HELLO WORLD"

npt.assert_string_equal(string1, string2)
```

```python
a = np.array([0.1, 0.2, 0.3])
b = np.array([0.11, 0.19, 0.31])

npt.assert_allclose(a, b, rtol=0.1)
```

```python
a = np.array([1, 2, 3])
b = np.array([5, 5, 5])

# values in a are less than the values in b, so should pass
npt.assert_array_less(a, b)
npt.assert_array_less(b, a)
```

```python
data = pd.read_csv('train/train.csv', parse_dates=['pickup_time', 'dropoff_time'], nrows=2000)

idx1 = pd.Index(data['pickup_time'].dt.date)
idx2 = pd.Index(data['dropoff_time'].dt.date)

try:
    npt.assert_array_equal(idx1, idx2)
    print("indexes are equal")
except AssertionError:
    print("indexes are not equal")
    diff = (idx1 != idx2).sum()
    diff_rows = data[idx1 != idx2][['pickup_datetime', 'dropoff_datetime']].head(7)
    print(diff_rows)
```

```python
import numpy as np
diff_indices = np.where(idx1 != idx2)[0]

# drop diff rows from df
data = data.drop(diff_indices).reset_index(drop=True)

data.reset_index(drop=True, inplace=True)

# reindex
idx1 = pd.Index(data['pickup_time'].dt.date)
idx2 = pd.Index(data['dropoff_time'].dt.date)
```

## assert data integrity

```python
import math

train = pd.read_csv('train.csv')

def test_missing_data(df):
    # sum up number of null values in df
    assert df.isnull().sum().sum() == 0, 'df contains missing values'
    return True
    
def test_data_types(df, columns):
    # check columns are of numerical data type
    for col in columns:
        assert df[col].dtype == 'int64' or df[col].dtype == 'float64', f'{col} has non-numerical data type'
    return True

def test_out_of_range_values(df, columns):
    # check numerical columns dont contain infinite or NaN values
    for col in columns:
        assert df[col].dtype == 'int64' or df[col].dtype == 'float64', f'{col} has non-numerical data type'
        assert df[col].max() <= math.inf, f'{col} contains infinite values'
        assert df[col].min() >= -math.inf, f'{col} contains infinite values'
        assert not np.isnan(df[col]).any(), f'{col} contains NaN values'
        assert not np.isinf(df[col]).any(), f'{col} contains infinite values'
    return True

# columns to test
num_cols = ['passenger_count', 'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude', 'trip_duration',
            'pickup_geonumber', 'dropoff_geonumber', 'pickup_hour']

# test training set
assert test_missing_data(train)
assert test_data_types(train, num_cols)
assert test_out_of_range_values(train, num_cols)
```

## assert data logic

```python
import math

train = pd.read_csv('train.csv')

def test_logical_errors(df):
    # dropoff time is after pickup time
    assert all(df['dropoff_time'] > df['pickup_time']), 'dropoff time is before pickup time'

    # no negative trip durations
    assert (df['trip_duration'] >= 0).all(), 'negative trip duration detected'

    # check pickup and dropoff coordinates are not the same
    assert not all(df['pickup_longitude'] == df['dropoff_longitude']) \
	and not all(df['pickup_latitude'] == df['dropoff_latitude']), \
	'pickup and dropoff coordinates are same!'

    # check passenger count is positive
    assert (df['passenger_count'] > 0).all(), 'passenger count must be positive'
    return True


assert test_logical_errors(train)
```
