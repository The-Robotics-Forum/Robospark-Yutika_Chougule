/*Task for today - Create your own String class.
Implement String class similar to the strings in Python and C++. (The way we implemented the List class).
Overload+ operator to concatenate two string.
Overload* operator to multiply string. (Ex: "yash"*2 returns "yashyash"). 2*"yash" should also return "yashyash". 
Overload the input and output operator. 
Try to overload as many operators as you can. (edited) 

*/
#include <iostream>
#include<string>
using namespace std;

class String
{
    public:
    string str;
    String()
    {
        str = "none";
    }
    String(string str)
    {
        this->str = str; 
    }
    void getInput(){
        cin >> str;
    }

    void display();
    
    String operator+(String s);
   
    void operator*(int n);
   
    friend void operator*(int n, String &s);

    friend istream& operator>>(istream& cin,String &s);

    friend ostream& operator<<(ostream& cout,String &s);
    
};

void String :: display(){
    cout << str << endl;
}

String String :: operator+(String s){
    String res;
    res.str = this->str + s.str;
    return res; 
}

void String :: operator*(int n){
    string temp = this->str;
    for (int i=1; i<n; i++){
        this->str = this->str + temp;
    }
}

void operator*(int n, String &s){
    string temp = s.str;
    for (int i=1; i<n; i++){
        s.str = s.str + temp;
    }
}

istream& operator>>(istream& cin, String &s){
    cin >> s.str;
    return cin;
}

ostream& operator<<(ostream& cout, String &s){
    cout << s.str;
    return cout;
}

int main(){
    String s("name");
    String s1("get");
    String s2;
    s2 = s + s1;
    //s3.display();
    s*3;
    3*s1;
    s.display();
    s1.display();
    s2.display();
    String s3,s4,s5;
    cin >> s3 >> s4 >> s5;
    cout << s3 << s4 << s5;
}