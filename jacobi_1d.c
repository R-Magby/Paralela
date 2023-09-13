#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <string.h>
#include <math.h>

//Guardar datos.
void guardar_datos(float* phi,int M, int N){

    FILE *fp = fopen("jacobi.dat", "w");
    for (int i=0; i<M; i++) {
        for (int j=0; j<N; j++) {
            fprintf(fp, "%g\n", phi[i*N+j]);
        }
    }
    fclose(fp);
}

//Posiciones de los potenciales.
static void llenar_rho(float* rho, int filas,int columnas, float L, float dx,float dy ,float a){

    for (int i = 0; i < filas; i++){
        for (int j = 0; j < columnas; j++){
            float x = j*dx - 3.0*L/4.0;
            float y = i*dy - 2.0*L/4.0;
            float r = sqrt(x*x + y*y);

            float x2 = j*dx - 2.0*L/4.0;
            float y2 = i*dy - L/4.0;
            float r2 = sqrt(x2*x2 + y2*y2);
            rho[i*columnas+j] = 1.0/(pow(r*r + a*a, 3.0/2.0))+1.0/(pow(r2*r2 + a*a, 3.0/2.0));
        }
    }
}

//Paralelizacion de la ecuacion de Poisson.
static void Worker(float*,float* ,int,int , int  ,int , int ,float,float,float );



int main (int argc, char *argv[]){
    MPI_Init(&argc, &argv);

    int numP;
    MPI_Comm_size(MPI_COMM_WORLD, &numP);

    int myID;
    MPI_Comm_rank(MPI_COMM_WORLD, &myID);
    if(argc < 5){
        // Only first process prints message
        if(myID == 0){
            printf("Program should be called as ./jacobi rows cols errThreshold MaxInteritions\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

//Elegir cantidad de procesadores talque tengan su raiz cuadrada sea un valor entero.

//Argumentos filas, columnas, tolerancia
    int rows,colums;
    rows=atoi(argv[1]);
    colums=atoi(argv[2]);
    float errThres = atof(argv[3]);
//Definimos un maximo de interacion, en caso de que la funcion no converga.
    int Nmax = atoi(argv[4]);

//Valores para el potencial.
    float L=10.0;
    float dx=L/((float)colums);
    float dy=L/((float)rows);
//Array donde se guardaran los datos finales.
    float *data=NULL;
    float *rho=NULL;
    MPI_Barrier(MPI_COMM_WORLD);

//Inicio del cronometro.
    double start = MPI_Wtime();

//requisitos de los argumentos.
    if ((rows < 1) || (colums < 1) ){
        if (myID == 0){
            printf("Error: 'm', 'k' and 'n' must be greater than 0.");
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }



//Declaramos los array de los procesos 
    float *myData;
    float *myrho;

//Llenamos con valores los array del proceso cero, ya que, a este se le mandaran los resultados optenidos 
//por los demas procesos
    if(myID==0){
        data = (float*) malloc(rows*colums*sizeof(float));
        memset(data, 0, sizeof(float)*rows*colums);
        rho= (float*) malloc(rows*colums*sizeof(float));
        llenar_rho(rho, rows,colums, L, dx, dy,1.0);
    }


//Declaramos las filas y columnas de los procesos
    int myRown = rows/numP;

//Definimos un datatype para selecionar bloques del array "data" y "rho" para enviarlos a "myData" y "myRho".
    MPI_Datatype fila_1_D;
    MPI_Type_contiguous(myRown*colums, MPI_FLOAT, &fila_1_D);
    MPI_Type_commit(&fila_1_D);
    MPI_Request reqData,reqRho,reqEnvio;


    if (myID == 0){
        for (int i=0; i < numP; i++){
            MPI_Isend(&data[i*colums*myRown], 1, fila_1_D, i, 0, MPI_COMM_WORLD,&reqData);
            MPI_Isend(&rho[i*colums*myRown], 1, fila_1_D, i, 0, MPI_COMM_WORLD,&reqRho);
        }
        
    }

    myData = (float*) malloc(myRown*colums*sizeof(float));
    MPI_Recv(myData,myRown*colums , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    myrho = (float*) malloc(myRown*colums*sizeof(float));
    MPI_Recv(myrho, myRown*colums, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);



//Comenzamos el calculo. 
    Worker(myData , myrho , myID, numP, myRown, colums, Nmax, errThres,dx,dy);


//Recibimos los datos de "myData" y los guardamos en "data".
    /*if (myID == 0){
        for (int i=0; i < numP; i++){
            MPI_Irecv(&data[i*colums*myRown], 1, fila_1_D, i, 0, MPI_COMM_WORLD, &reqEnvio);
        }
        
    }*/
    MPI_Gather(myData, myRown*colums, MPI_FLOAT, data,  myRown*colums, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

//Fin del cronometro.
    double end = MPI_Wtime();

//Guardamos los datos
    if (myID == 0){
        printf("Time with %d processes: %f seconds.\n",numP,end-start);
        guardar_datos(data, rows,colums);
        free(data);
    }

    MPI_Finalize();

}



static void Worker(float* myData, float* myrho, int myID, int numP, int myRows,
                   int cols, int Nmax, float errThres, float dx,float dy) {
//Array temporal.
    float* buff = (float*) malloc(myRows*cols*sizeof(float));
    memcpy(buff, myData, myRows*cols*sizeof(float));

//Definimos un error incial
    float error = errThres + 1.0;
    float myError;

//Tendremos que traspasar informacion de los demas procesos.
//Cada procesos esta trabajndo en una seccion de dimensiones "blockRows*blockCols"
//Por lo que, tendremos que enviar informacion en 4 dirreciones, pensando como
//si fuera un cuadrado.
    float* topRow = (float*) malloc(cols*sizeof(float));
    float* downRow = (float*) malloc(cols*sizeof(float));

//Declaramo un datatype para enviar columnas, servirá para enviar informacion a los laterales.
    MPI_Datatype envioColumna;
    MPI_Type_vector(myRows,1,cols, MPI_FLOAT, &envioColumna);
    MPI_Type_commit(&envioColumna);


//resquest
    MPI_Request req0,req1,req2,req3,req4,req5,req6,req7,req8;
    MPI_Request reqmyData;

    //MPI_Status status0,status1,status2,status3;

//Declaramos los procesos vecinos.
    int down = myID+1, top = myID - 1;


//Inicio del calculo.
    printf("Worker %d initialized;  top is worker %d and down is worker %d\n", myID,top,down);
    int it=0;
    while (it<Nmax && error > errThres){


        if(myID>0){
            MPI_Isend(myData, cols, MPI_FLOAT,top, 0, MPI_COMM_WORLD, &req4);
	        MPI_Recv(topRow, cols, MPI_FLOAT,top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }

        if(myID<numP-1){
            MPI_Isend(&myData[(myRows-1)*cols], cols, MPI_FLOAT, down, 0, MPI_COMM_WORLD, &req6);
	        MPI_Recv(downRow, cols, MPI_FLOAT, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }

//Actualizacion de datos
        if (myID > 0){
            if (myRows > 1){
                for (int j=1; j < cols-1; j++){
                    buff[j] = 0.25f*(myData[cols+j]+myData[j-1]+myData[j+1]+topRow[j]-myrho[j]*dy*dx);
                }
            }
        }
        for (int i=1; i < myRows-1; i++){
            for (int j=1; j < cols-1; j++){
                buff[i*cols+j] = 0.25f*(myData[(i-1)*cols+j] + myData[(i+1)*cols+j] +
                myData[i*cols+j-1] + myData[i*cols+j+1]-myrho[i*cols+j]*dy*dx);
            }
        }

        if (myID < numP-1){
            if (myRows > 1){
                for (int j=1; j < cols-1; j++){
                    buff[(myRows-1)*cols+j] = 0.25f*(downRow[j]+myData[(myRows-1)*cols+j-1]+
                    myData[(myRows-1)*cols+j+1]+myData[(myRows-2)*cols+j]-myrho[(myRows-1)*cols+j]*dy*dx);
                }
            }
        }
    


//Calculo del error
        myError = 0.0;
        for (int i=1; i < myRows-1; i++){
            for (int j=1; j < cols-1; j++){
                myError += fabs(myData[i*cols+j]-buff[i*cols+j]);
            }
        }

        memcpy(myData, buff, myRows*cols*sizeof(float));

//Suma de errores.
        MPI_Allreduce(&myError, &error, 1, MPI_FLOAT, MPI_SUM,MPI_COMM_WORLD);
        it++;

    }
    //Errores de cada proceso y la cantidad de interación.
    printf("Error of worker %d is %f in %d \n", myID, myError,it);

//Enviamos los "myData" al proceso raiz (0), para ser guardados
 //   MPI_Isend(myData, cols*myRows, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,&reqmyData);
  
}

