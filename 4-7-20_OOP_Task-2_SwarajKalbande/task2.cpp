#include <iostream>
#include<string>
using namespace std;


class Student
{
    public:
    string name;
    int roll,year;
    float cgpa;

    Student(){
        name = "none";
        roll = 0;
        year = 1;
        cgpa = 0;
    }

    void giveInput()
    {
        cout << "Name: ";
        cin >> name;
        cout << "roll: ";
        cin >> roll;
        cout << "year: ";
        cin >> year;
        cout << "CGPA: ";
        cin >> cgpa;
    
    }

    void giveDetail(){
        cout << "Name: " << name << endl;
        cout << "Roll: " << roll << endl;
        cout << "year: " << year << endl;
        cout << "cgpa: " << cgpa << endl;
    }

};

void showRank(int a){
    cout << "The rank is: " << a << endl;
}

void showRank(Student &x){
    x.giveDetail();
    cout << "No rank" << endl;
}

void showRank(int a, Student &x){
    x.giveDetail();
    cout << "The rank is: " << a;
}

int main(){
    Student stud;
    stud.giveInput();
    showRank(3);
    showRank(stud);
    showRank(4,stud);
}