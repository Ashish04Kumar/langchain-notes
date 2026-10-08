from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    # name: str
    name: str = 'kumar'
    age: Optional[int] = None
    # validation
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description="A decimal value representing cgpa of student")


new_student = {'name': 'Ashish',
               'age': 34, 
               'email' : 'abc@gmail.com', 
               'cgpa': 9
               }

# new_student = {}
# new_student = {'name': 'Ashish'}
student = Student(**new_student)
print(student)

# pydantic objectio to python dic and json
student_dict = dict(student) 
print(student_dict['age'])
student_json = student_dict.model_dump_json(student)