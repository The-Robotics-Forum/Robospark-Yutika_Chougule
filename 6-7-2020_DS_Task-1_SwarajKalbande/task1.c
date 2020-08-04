/*Task for today -
Create a linked list with 4 nodes and two functions-
1. Addition/Insertion of a Node
2. Deletion of a Node
*/

#include<stdio.h>
#include<stdlib.h>

struct node
{
    int data;
    struct node *next;
};
struct node *head= NULL;

void insert_head(struct node* head_ref, int data)
{
    struct node* new_node= (struct node*)malloc(sizeof(struct node));
    new_node ->data = data;
    new_node->next = head_ref; //setting current head as next of new_node
    head= new_node; //setting new_node as head
}

void insert_after(struct node* prev, int data)
{
    struct node* new_node=(struct node*)malloc(sizeof(struct node));
    new_node->data= data;
    new_node ->next = prev ->next; //next of new_node is next of previous node 
    prev->next= new_node;   //next of previous node becomes new_node
}

void insert_end(struct node *head_ref, int data)
{
    struct node* new_node= (struct node*)malloc(sizeof(struct node));
    struct node *last= head_ref;
    new_node ->data = data;
    new_node->next=NULL; //new_node added at the end so next is NULL

    while (last->next != NULL)
    {
        last=last->next;
    }
    last->next =new_node; //next of current last becomes new_node
    
}
void delete_node (struct node *head, int key)
{
    struct node* temp= head;
    struct node *prev;

    if(temp != NULL && temp->data == key) //deleting head if key present at head
    {
        head= temp->next;
        free(temp);
        return;
    }

    while (temp!= NULL && temp->data != key)
    {
        prev = temp;
        temp= temp->next;
    }
    if (temp == NULL) 
        printf("KEY NOT FOUND"); //if key not found
    prev->next = temp->next;
    free(temp);
}


void print_data(struct node *node)
{
    while(node != NULL)
    {
        printf(" %d", node->data);
        node = node -> next;
    }
}


int main()
{
    struct node *prev, *p, *temp;
    int x,a,b,c;
    printf(" Enter the elements: ");
    for(int i=0; i<4; i++)
    {
        p= (struct node*)malloc(sizeof(struct node));
        scanf("%d",&(p->data));
        p->next = NULL;
        if(head==NULL)
            head=p;
        else
        {
            prev->next=p;
        }
        prev = p;
    }
    printf("\n Elements entered are: ");
    print_data(head);
    
    printf("\n Enter data to be deleted: ");
    scanf("%d", &x);
    delete_node(head,x);
    printf("\n Linked list after deletion: ");
    print_data(head);

    printf("\n Enter element to be inserted in the beginning: ");
    scanf("%d",&a);
    insert_head(head,a);
    printf("\n Inserting head: ");
    print_data(head);

    printf("\n Enter element to be inserted at  the end: ");
    scanf("%d",&b);
    insert_end(head,b);
    printf("\n Inserting end: ");
    print_data(head);

    printf("\n Enter element to be inserted: ");
    scanf("%d",&c);
    insert_after(head->next->next->next,c);
    printf("\n Inserting in between");
    print_data(head);
    
    return 0;
}