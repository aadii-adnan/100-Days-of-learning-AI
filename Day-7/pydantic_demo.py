from pydantic import BaseModel

class Student(BaseModel):
    name : str


new_student = {
    'name': 'John Doe'
}

student = Student(**new_student)
print(type(student))