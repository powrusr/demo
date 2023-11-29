# dataclasses

- creates init, repr, eq automatically order=false by default
- comparing two instances w equal properties will now return true
- order=True implements (__lt__, __le_, __gt__, __ge__) special methods
- frozen=True implements immutable properties (cannot be changed)
- unsafe_hash=False (hash(obj) of immutable objects are equal
  setting this to True will put resp w you of hash checking

```python
@dataclass(cls=None,init=True,repr=True,eq=True,order=False,unsafe_hash=False,frozen=False)
class Person:
	name: str
	age: int
	city: str
		
# hash(p1) throws TypeError bc of mutable properties, unless unsafehash=True
# will work if frozen is set to True
```

## fields

```python
"""
data class’s fields are required to have type annotations
Under the hood, the dataclass decorator generates fields based on the annotations of your class,
which can be retrieved using the __annotations__ special method
"""
p = Person('Joe', 'Berlin', 30)
p.__dataclass_fields__  # returns all fields
p.__dataclass_fields__['age']  # more info on that field

from dataclasses import dataclass, field
field(*,default=MISSING,default_factory=MISSING,init=True,repr=True,hash=None,compare=True,metadata=None)

age: int = field(default=25)
age: int = field(default_factory=get_default_age) # function with no args!
	
def get_default_age():
	ages = [12,34,56,34,12]
	return sum(ages) // len(ages)  # 29

# using init=False

# don't initialize city
city: str = field(init=False, default='Brugge')
p = Person('Joe', 44)

# don't include in object representation
city: str = field(repr=False)

# don't include property in comparing 2 objects
city: str = field(compare=False)

# metadata = dictionary w extra info on field
age: int =field(metadata={'format': 'year'})

p.__dataclass_fields_['age'] # will return metadata info, this can be useful for other functions to know
p.__dataclass_fields_['age'].metadata['format']
# returns 'year'
```

## postinit

```python
@dataclass
class Person:
	name: str
	city: str
	age: int
	is_senior: bool = field(init=False) # not passed as param, use logic
		
	def __post_init__(self):
		if self.age >= 60:
			self.is_senior = True
		else:
			self.is_senior = False
			
p = Person('the dude', 'Brugge', 62)
p.is_senior  # True
```

## inheritance
```python
@dataclass
class Student(Person):
	grade: int
	subjects: list

# what is init signature?
s = Student(name: str, city: str, age: int, subjects: list)
s = Student('Joe', 'Berlin', 30, 10, ['maths', 'physics'])
# common params will be overwritten by Student in this example
```

## asdict, astuple

```python
@dataclass
class Address:
	lat: float
	long: float
	city: str
	country: str

@dataclass
class Person:
	name: str
    addr: Address
    age: int
		
a = Address(28.75, 77.11, 'Delhi', 'India')
p = Person('Joe', a, 29)
p
Person(name='Joe', addr=Address(lat28.75, long=77.11, city='Delhi', country='India'), age=29)

# convert p to api friendly json by using asdict, astuple..
from dataclasses import dataclass, asdict
asdict(p)
{'name': 'Joe',
 'addr': {'lat': 28.75, 'long': 77.11, 'city': 'Delhi', 'country': 'India'},
 'age': 29}
import json
json.dumps(asdict(p))  # dumps as string
'{'name': 'Joe', 'addr': {'lat': 28.75, 'long': 77.11, 'city': 'Delhi', 'country': 'India'}, 'age': 29}'

```

## default_factory & InitVar

```python
"""
default_factory:
"""
@dataclass
class IncorrectBill:
    costs_by_dish: list = []
"""
ValueError: mutable default <class 'list'> for field costs_by_dish is not allowed: use default_factory
"""


"""
InitVar:

This lets you specify a field that will be passed to __init__ and then to __post_init__, but won’t be stored in the class instance.

Setting a field’s type to InitVar (with its subtype being the actual field type) signals to @dataclass to not make that field into a dataclass field, but to pass the data along to __post_init__ as an argument.
"""

ingredients: List = field(default_factory=['dow', 'tomatoes'])  # <- wrong!
ingredients: List = field(default_factory=lambda: ['dow', 'tomatoes'])  # correct
# error to specify both default and default_factory.

from dataclasses import dataclass, field, InitVar
from typing import List

@dataclass
class Book:
	name: str
	weight: float = field(default=0.0, repr=False)
	shelf_id: int = field(init=False) # logic
	chapters: List[str] = field(default_factory=list)
	condition: InitVar[str] = None


	def __post_init__(self, condition):
		if condition == "Discarded":
			self.shelf_id = None
		else:
			self.shelf_id = 0
		self.shelf_name = f"Shelf #{self.shelf_id}"
		
my_book = Book("War And Peace", weight=700.0, condition="Discarded")
```
