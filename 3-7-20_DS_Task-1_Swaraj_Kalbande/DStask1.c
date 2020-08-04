#include<stdio.h>
#include<stdlib.h>

typedef struct Queue
{
    char arr[10];
    int front,rear;
}Queue;

void initialize(Queue* q){
    q->front=0;
    q->rear=0;

}

void enqueue(Queue* q,char x){
    q->arr[q->rear++]=x;

}

int dequeue(Queue* q){
    return q->arr[q->front++];
}

int isEmpty(Queue* q){
    return q->front==q->rear;
}

void print(Queue* q){
    if(isEmpty(q)){
        printf("Queue is empty\n");
        return;
    }

    printf("front ");
    for(int i=q->front;i<q->rear;i++){
        printf("%d ",q->arr[i]);
    }
    printf("Rear\n");
}


int main(){
    Queue* q1=(Queue*)malloc(sizeof(Queue));
    initialize(q1);
    
    printf("\nEnter number of students: ");
    int n;
    int temp;
    char temps;
    scanf("%d",&n);
    char names[ n];
    int rolls[ n];
    
    printf("\nEnter roll_nos details:\n ");
    for (int i=0; i<n; i++){
        scanf("%d",&rolls[i]);
    }

   // for (int i=0; i<n; i++){
   //     printf("\nEnter names details:\n ");
   //     fflush(stdin);
   //     scanf("%s",names[i]);
        
    //}
    
    for (int i=0; i<n; i++){
        if (rolls[i]%2 == 0){
            enqueue(q1, rolls[i]);
        }
    }

    print(q1);
}





