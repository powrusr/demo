# fundamentals

## range

### range(number+1)

```python

   for count_25 in range(4+1):  # note the +1 to indicate 4 is not included
       print(count_25)
   for count_25 in range(4+1):
       print(count_25)
   0
   1
   2
   3
   4
```

```python

   # loop over all possible counts for each coin (summing up to <= 1$);
   # if the total amount is exactly $1, add current counts to "combinations"
   
   combinations = []
   
   for count_100 in range(1+1):
       for count_50 in range(2+1):
           for count_25 in range(4+1):
               for count_10 in range(10+1):
                   for count_5 in range(20+1):
                       for count_1 in range(100+1):
                           if 100*count_100 + 50*count_50 + 25*count_25 + 10*count_10 + 5*count_5 + count_1 == 100:
                               combinations.append([count_100, count_50, count_25, count_10, count_5, count_1])

   combinations
   [[0, 0, 0, 0, 0, 100],
    [0, 0, 0, 0, 1, 95],
    [0, 0, 0, 0, 2, 90],
    [0, 0, 0, 0, 3, 85],
    [0, 0, 0, 0, 4, 80],
    [0, 0, 0, 0, 5, 75],
    [0, 0, 0, 0, 6, 70],
    [0, 0, 0, 0, 7, 65],
    [0, 0, 0, 0, 8, 60],
    [0, 0, 0, 0, 9, 55], # cut here ..
    [0, 1, 1, 0, 2, 15],
    [0, 1, 1, 0, 3, 10],
    [0, 1, 1, 0, 4, 5],
    [0, 1, 1, 0, 5, 0],
    [0, 1, 1, 1, 0, 15],
    [0, 1, 1, 1, 1, 10],
    [0, 1, 1, 1, 2, 5],
    [0, 1, 1, 1, 3, 0],
    [0, 1, 1, 2, 0, 5],
    [0, 1, 1, 2, 1, 0],
    [0, 1, 2, 0, 0, 0],
    [0, 2, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0]]
    len(combinations)
    293
```

## underscore usage

* Underscore _ is considered as "I don't Care" or "Throwaway" variable in Python
* ipython stores the last expression value to the special variable called _
* underscore _ is also used for ignoring the specific values. If the values are not used, just assign the values to underscore
* particular function implementation doesn't need all of the parameters

```python

   10
   10

   _
   10

   _ * 3
   30

   # Ignore a value when unpacking

   x, _, y = (1, 2, 3)

   x
   1

   y
   3

   # Ignore the index

   for _ in range(10):
       do_something()

   # not all params in def/lambda are needed

   def callback(_):
       return True

   lambda _: 1.0  # don't require argument
```

## list indexing

```python

   x = ["first", "second", "third", "fourth"]
   y = [1, 2, 3, 4]

.. list-table:: indices
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - x = [
     - "first",
     - "second",
     - "third",
     - "fourth"     ]
   * - positive index
     - 0
     - 1
     - 2
     - 3
   * - negative index
     - -4
     - -3
     - -2
     - -1

```python

x[1:-1]
["second", "third"]

x[0:3]
# or
x[:3]
["first", "second", "third"]

x[2:]
["third", "fourth"]

# append list to end of list
x[len(x):] = [5, 6, 7]

# make copy you can modify
y = x[:]

# add list to front of list (prepend)
x[:0] # up to first element
x[:0] = [-1, 0]
x
[-1, 0, 1, 2, 3, 4, 5, 6, 7]

# delete part of list
x[1:-1] = []  # 2nd element up to last (0 -> 7) not including 7
x
[-1, 7]

# glue one list to another
x.extend(y)

# x.append(y) will add y list as an element, not extend
[1, 2, 3, 4, [5, 6, 7]]

del x[4]
[1, 2, 3, 4]

del x[:3]
[4]

z = [None] * 4
z
[None, None, None, None]
```


