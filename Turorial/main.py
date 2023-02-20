#%%
import torch
from other import Person, Male, print_person, People


#%%

tom = Person(70, "female")
print(tom)
#%%
people = People([Person(30, "male"), Person(70, "female"), Person(55, "male")])
len(people)
#%%
# print(tom)
a = [1, 2, 3, 4]
print(a[1:3])
print(a.__getitem__(slice(1, 3)))
#%%
print_person(tom)
# print(f"age: {tom.age}, gender: {tom.gender}, is_grumpy: {tom.is_grumpy}")
# %%
print(tom)
# %%
tim = Male(70)
multiple_age = tim(5)
print(multiple_age)
# %%
if __name__ == "__main__":
    print(5)
