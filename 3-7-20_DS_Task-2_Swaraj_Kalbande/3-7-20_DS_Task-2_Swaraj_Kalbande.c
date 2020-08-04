#include<stdio.h>
#include <stdlib.h>

struct node{
    int data; 
    struct node *right_child;
    struct node *left_child; 
};

struct node* search(struct node *root, int x){  
    if(root==NULL || root->data==x){
        return root;
    }
    else if(x>root->data){
        return search(root->right_child, x);
    }
    else{
        return search(root->left_child,x);
    }
}

struct node* new_node(int x)
{
    struct node *p = (struct node *)malloc(sizeof(struct node));
    p->data = x;
    p->left_child = NULL;
    p->right_child = NULL;

    return p;
}

struct node* insert(struct node *root, int x)
{
    
    if(root==NULL)
        return new_node(x);
    else if(x>root->data) 
        root->right_child = insert(root->right_child, x);
    else 
        root->left_child = insert(root->left_child,x);
    return root;
}

void postorder(struct node *root)
{   
    if(root!=NULL) 
    {
        postorder(root->left_child); 
        postorder(root->right_child); 
        printf(" %d ", root->data); 
    }
}

int main()
{
    struct node *root;
    int arr[100];

    int count;
    printf("\nEnter the total number of elements : ");
    scanf("%d", &count);

    printf("\nEnter the elements: \n");
    for (int i = 0; i < count; i++)
    {
        scanf("%d", &arr[i]);
    }

    for (int j = 0; j < count; j++)
    {
        if (j==0)
        {   
            root = new_node(arr[j]);
        }
        else
        {
            insert(root, arr[j]);
        }   
    }

    postorder(root);
    printf("\n");

    int a;
    printf("searching : ");
    scanf("%d", &a);

    if (search(root, a)==0)
    {
        printf("The number %d is NOT there", a);
    }
    else
    {
        printf("YES the number %d is there", a);
    }

    return 0;
} 