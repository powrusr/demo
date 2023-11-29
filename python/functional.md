# functional programming
## map

[map docs](https://docs.python.org/3/library/functions.html#map)

- apply *function* to every item of *iterable* creating a new iterable
  `new_iterable = map(function, iterable)`

```python
def square(number):
    return number ** 2

numbers = [1, 2, 3, 4, 5]

squared = map(square, numbers)

list(squared)
[1, 4, 9, 16, 25]

--

numbers = [-2, -1, 0, 1, 2]

abs_values = list(map(abs, numbers))
abs_values
[2, 1, 0, 1, 2]


list(map(float, numbers))
[-2.0, -1.0, 0.0, 1.0, 2.0]

words = ["Welcome", "to", "Real", "Python"]

list(map(len, words))
[7, 2, 4, 6]
```

## itertools

### itertools functions

One type of function it produces is **infinite iterators**

| function | description |
|---|:---|
| Count | **counts up** infinitely from a value|
| Cycle | *infinitely iterates through an iterable* (for instance a list or string)|
| Repeat| **repeats an object**, either infinitely or a specific number of times|
| Takewhile| takes items from iterable while predicate function remains True |
| Chain | **combines** iterables |
| Accumulate |returns a **running total** of values in an iterable|


#### count

```python
from itertools import count

for i in count(3): # counts up starting from 3
	print(i)
	if i>= 11:
		break

""" 3  4  5  6  7  8  9  10  11 """

```

#### accumulate

```python
from itertools import accumulate, takewhile


nums = list(accumulate(range(8)))
print(nums)  # [0,  1,   3,   6,  10,   15,   21, 28]
             # [0, 0+1, 1+2, 3+3, 6+4, 10+5, 15+6, 21+7]

```

#### takewhile

```python
print(list(takewhile(lambda x: x<=6, nums)))  # [0, 1, 3, 6]

# takewhile stops as soon as predicate == FALSE!
nums = [2, 4, 6, 7, 9, 8]  # will stop returning at hitting value 7

print(list(takewhile(lambda x: x%2==0, nums))) 
# [2, 4, 6]
```

#### combinatoric functions & permutations

```python

from itertools import product, permutations

letters = ("A", "B")

list(product(letters, range(2)))
# [('A', 0), ('A', 1), ('B', 0), ('B', 1)]

list(permutations(letters))
# [('A', 'B'), ('B', 'A')]


letters = ("A", "B", "C")
list(permutations(letters))
"""
[('A', 'B', 'C'), ('A', 'C', 'B'), ('B', 'A', 'C'),
 ('B', 'C', 'A'), ('C', 'A', 'B'), ('C', 'B', 'A')]
"""

a={1, 2}
len(list(product(range(3), a))) # 6
list(product(range(3), a))
# [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)]
```

## filter

- apply boolean function to an iterable to generate a new iterable

```python

```

## reduce

- apply reduction function to an iterable to produce single cumulative value

```python

