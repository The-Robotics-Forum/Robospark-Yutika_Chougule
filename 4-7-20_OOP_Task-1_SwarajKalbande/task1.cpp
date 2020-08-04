#include<iostream>
#include<string>
using namespace std;


class Student
{
    public:
    string name;
    int roll,year;
    float cgpa;
    int *mrks = new int[5];

    Student(){
        name = "none";
        roll = 0;
        year = 1;
        cgpa = 0;
    }

    void giveInput()
    {
        cout << "Name: ";
        //cin.sync();
        //cin<<;
        cin >> name;
        cout << "roll: ";
        cin >> roll;
        cout << "year: ";
        cin >> year;
        cout << "Enter marks out of 100: " << endl;
        for (int i=0; i<5; i++){
            cin >> mrks[i];
        }
    
    }

    void giveDetail(){
        cout << "Name: " << name << endl;
        cout << "Roll: " << roll << endl;
        cout << "year: " << year << endl;
        cout << "cgpa: " << cgpa << endl;
        cout << "marks are: " << endl;
        for (int i=0; i<5; i++){
            cout << mrks[i] << endl;
        }
    }

    int totalMarks(int *mrks){
    int total=0;
    for (int i=0;i<5;i++){
        total += mrks[i];
    }
    return total;
}

    void getCGPA(){
        int n = totalMarks(mrks);
        cgpa = (n/5)/(9.5);
}

};

/*
int totalMarks(int *arr){
    int total=0;
    for (int i=0;i<5;i++){
        total += arr[i];
    }
    return total;
}
void getCGPA(Student &x){
    int n = totalMarks(x.mrks);
    x.cgpa = (n/5)/(9.5);
    cout << x.cgpa; 
}
*/


int main(){
    Student s;
    s.giveInput();
    s.getCGPA();
    //int n = totalMarks(s.mrks);
    //cout << "total: " << n;
    s.giveDetail();
}