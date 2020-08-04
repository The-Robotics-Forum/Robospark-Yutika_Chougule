#include <stdio.h>

struct record {
    int ID;
    char name[3];
    int year;
    float CGPA;
    
};

int main()
{
   
    struct record student1;
    struct record student2;
    struct record student3;
    
    printf("\nEnter details of student 1 :");
    printf("\nEnter ID:\n");
    scanf("%d",& student1.ID);
    
    printf("Enter name of student:\n");
    scanf("%s", student1.name);
    
    printf("Enter year of study:\n"); 
    scanf("%d",& student1.year);
    
    printf("Enter CGPA of student:\n");
    scanf("%f", & student1.CGPA);
    
       printf("\nEnter details of student 2 :");
    printf("\nEnter ID:\n");
    scanf("%d",& student2.ID);
    
    printf("Enter name of student:\n");
    scanf("%s", student2.name);
    
    printf("Enter year of study:\n"); 
    scanf("%d",& student2.year);
    
    printf("Enter CGPA of student:\n");
    scanf("%f", & student2.CGPA);
    
    
       printf("\nEnter details of student 3 :");
    printf("\nEnter ID:\n");
    scanf("%d",& student3.ID);
    
    printf("Enter name of student:\n");
    scanf("%s", student3.name);
    
    printf("Enter year of study:\n"); 
    scanf("%d",& student3.year);
    
    printf("Enter CGPA of student:\n");
    scanf("%f", & student3.CGPA);
    
    return 0;
    
    
   

}