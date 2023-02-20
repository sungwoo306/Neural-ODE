#%%
class Person:
    def __init__(self, age: int | float, gender: str):
        self.age = age
        self.gender = gender
        # self.is_grumpy = self.determine_grumpy()

    def determine_grumpy(self) -> bool:
        if self.gender == "male" and self.age > 50:
            return True
        return False

    @property
    def is_grumpy(self) -> bool:
        if self.gender == "male" and self.age > 50:
            return True
        return False

    def __repr__(self) -> str:
        return f"age: {self.age}, gender: {self.gender}, is_grumpy: {self.is_grumpy}"


class Male(Person):
    def __init__(self, age):
        super().__init__(age, "male")

    def __repr__(self):
        return f"male, age = {self.age}"

    def __call__(self, multiply_factor):
        return multiply_factor * self.age


class People2:
    def __init__(self, person_list: list[Person]):
        self.person_list = person_list

    def __repr__(self):
        return self.person_list.__repr__()

    def num_of_person(self) -> int:
        return len(self.person_list)

    def ith_person(self, i) -> Person:
        return self.person_list[i]


class People:
    def __init__(self, person_list: list[Person]):
        self.person_list = person_list

    def __repr__(self):
        return self.person_list.__repr__()

    ## Immutable sequence protocol
    def __len__(self) -> int:
        return len(self.person_list)

    def __getitem__(self, idx) -> Person:
        return self.person_list[idx]

    ## Mutable sequence protocol
    def __setitem__(self, idx, value: Person) -> None:
        self.person_list[idx] = value


def print_person(person: Person) -> None:
    print(f"age: {person.age}, gender: {person.gender}, is_grumpy: {person.is_grumpy}")


if __name__ == "__main__":
    print("I have been run")
# print(__name__)
print("I have been run2")

# %%
