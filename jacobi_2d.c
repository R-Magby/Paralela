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

//Posiciones de los potenciales se agregaron el dy y las filas.
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
static void Worker(float*,float* ,int,int , int , int,int , int ,float,float,float );



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
    int gridDim = sqrt(numP);

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

    if ((rows%gridDim) || (colums%gridDim)){
        if (myID == 0){
            printf("Error: 'm' and 'n' must be multiples of the grid dimensions.");
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
    int blockRows = rows/gridDim;
    int blockCols = colums/gridDim;

//Definimos un datatype para selecionar bloques del array "data" y "rho" para enviarlos a "myData" y "myRho".
    MPI_Datatype bloque;
    MPI_Type_vector(blockRows, blockCols, colums, MPI_FLOAT, &bloque);
    MPI_Type_commit(&bloque);
    MPI_Request reqData,reqRho,reqEnvio;


    if (myID == 0){
        for(int j=0; j<gridDim;j++){
            for (int i=0; i < gridDim; i++){
                MPI_Isend(&data[j*blockRows*colums+i*blockCols], 1, bloque, j*gridDim+i, 0, MPI_COMM_WORLD,&reqData);
                MPI_Isend(&rho[j*blockRows*colums+i*blockCols], 1, bloque, j*gridDim+i, 0, MPI_COMM_WORLD,&reqRho);
            }
        }
    }

    myData = (float*) malloc(blockRows*blockCols*sizeof(float));
    MPI_Recv(myData, blockRows*blockCols , MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    myrho = (float*) malloc(blockRows*blockCols*sizeof(float));
    MPI_Recv(myrho, blockRows*blockCols, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);



//Comenzamos el calculo. 
    Worker(myData , myrho , myID, numP, gridDim, blockRows, blockCols, Nmax, errThres,dx,dy);


//Recibimos los datos de "myData" y los guardamos en "data".
    if (myID == 0){
        for(int j=0; j<gridDim;j++){
            for (int i=0; i < gridDim; i++){
                MPI_Irecv(&data[j*blockRows*colums+i*blockCols], 1, bloque, j*gridDim+i, 0,MPI_COMM_WORLD, &reqEnvio);
            }
        }
    }
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

static void Worker(float* myData, float* myrho, int myID, int numP, int gridDim, int myRows,
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
    float* leftRow = (float*) malloc(myRows*sizeof(float));
    float* rightRow = (float*) malloc(myRows*sizeof(float));
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
    int rigth = myID+1, left = myID - 1, top = myID - gridDim , down = myID + gridDim;

//Limites, son necesarios para los procesos que estan a los costados y no deben traspasar
//informacion a otro extremo.
    int limite_rigth=(numP-1)%gridDim;
    int limite_left=0;

//Inicio del calculo.
    printf("Worker %d initialized; left is worker %d, right is worker %d, top is worker %d and down is worker %d\n", myID, left, rigth,top,down);
    int it=0;
    while (it<Nmax && error > errThres){
	    if (myID % gridDim > limite_left){
	        MPI_Isend(myData, 1, envioColumna, left, 0, MPI_COMM_WORLD, &req0);
	        MPI_Recv(leftRow, myRows, MPI_FLOAT, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }

	    if (myID%gridDim < limite_rigth ){
	        MPI_Isend(&myData[cols-1], 1, envioColumna, rigth, 0, MPI_COMM_WORLD, &req2);
	        MPI_Recv(rightRow, myRows, MPI_FLOAT, rigth, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }

	    if (top>=0){
	        MPI_Isend(myData, cols, MPI_FLOAT, top, 0, MPI_COMM_WORLD, &req4);
	        MPI_Recv(topRow, cols, MPI_FLOAT,top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }

	    if (down<=numP-1){
	        MPI_Isend(&myData[(myRows-1)*cols], cols, MPI_FLOAT, down, 0, MPI_COMM_WORLD, &req6);
	        MPI_Recv(downRow, cols, MPI_FLOAT, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }

//Actualizacion de datos
	    // Update the top colums
        if (top>=0){
            if (myRows > 1){
                for (int j=1; j < cols-1; j++){
                    buff[j] = 0.25f*(myData[cols+j]+myData[j-1]+myData[j+1]+topRow[j]-myrho[j]*dx*dy);
                }
                }
        }
        //update the down colums
        if (down<=numP-1){
            if (myRows > 1){
                for (int j=1; j < cols-1; j++){
                    buff[(myRows-1)*cols+j] = 0.25f*(downRow[j]+myData[(myRows-1)*cols+j-1]+
                    myData[(myRows-1)*cols+j+1]+myData[(myRows-2)*cols+j]-myrho[(myRows-1)*cols+j]*dx*dy);
                }
            }
        }
        //update the rigth row
        if (myID % gridDim < limite_rigth ){
            if (myRows > 1){
                for (int i=1; i < myRows-1; i++){
                    buff[i*cols+cols-1] = 0.25f*(myData[i*cols+cols-2]+myData[(i-1)*cols+cols-1]+myData[(i+1)*cols+cols-1]+rightRow[i]-myrho[i*cols+cols-1]*dx*dy);
                }
            }
        }
        //update the left row
        if (myID % gridDim>limite_left ){
            if (myRows > 1){
                for (int i=1; i < myRows-1; i++){
                    buff[i*cols] = 0.25f*(myData[i*cols+1]+myData[(i-1)*cols]+myData[(i+1)*cols]+leftRow[i]-myrho[i*cols]*dx*dy);
                }
            }
        }
        //update the main block
        for (int i=1; i < myRows-1; i++){
            for (int j=1; j < cols-1; j++){
                buff[i*cols+j] = 0.25f*(myData[(i-1)*cols+j] + myData[(i+1)*cols+j] +
                myData[i*cols+j-1] + myData[i*cols+j+1]-myrho[i*cols+j]*dx*dy);
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
    MPI_Isend(myData, cols*myRows, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,&reqmyData);
  
}

//COMENTARIOS FINALES

//El envio de los datos tuvo que ser por medio de Isend y Irecv para que el proceso 0
//se pueda madnar un mensaje a si mismo.

//Con la funcion gather desconosco si habia una forma de guardarlos con algun datatype
//Guardaba los datos de forma desordenada.

//En este codigo, al graficar los datos, aparecen unos puntos que no se calcularon correctamente, 
// que representarian las esquinas de las caras internas de la caja 2d

//Para la parte c)
//se usaron esstos parametros ">mpiexec -n 9 jacobi_1d 180 270 0.01 1000"
// Para mi casi el mejor tiempo fue el de una dimension, con un tiempo aproximadamente de 5.826980 segundos
// Mientras que el de 2d tardaba un poco mas de 7.778545 segundos, es posible que el envio de mensajes 
// perjudique, lo mas probable es que, haya algo mal escrito, tambien considerando que es posible que
//se hayan ejecutado mal las esquinas de cuadro. Aun asi dan el mismo grafico

//En la funcion de 1d es posbile el gather, con esta funcion, arroja un tiempo parecido al 1d.

//tambien decir que como los procesos no van sincronizadamente, puede se que no arrojen los mismos valores
//o aparezcan zonas sin calcular en el grafico.

//Agregar que un Nmax mayor a 10000 o una tolerancia mas baja, puede que el programa no converga.