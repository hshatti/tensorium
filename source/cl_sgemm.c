/*
inline float mapX(const float x){
  return x*3-2.1F;
}
// Same purpose as mapX
// [0, 1] -> [-1.25, 1.25]
inline float mapY(const float y){
  return y*3 - 1.5F;
}

#define max_iteration  10000
#define _max           4.0f


__kernel void mandel(__global uchar *buf, const int w, const int h){

  const float lnxp1_max_iteration = log1p((float)max_iteration);

  int y = get_global_id(0);
  int x = get_global_id(1);
  float xx = mapX(x/(float)w);
  float yy = mapY(y/(float)h);

  y *= w * sizeof(uint);
  x *= sizeof(uint);

  float x0 = 0.0f; float y0 = 0.0f;
  int iteration = 0;
  float oldAbs = 0.0f;
  float coverageNum = max_iteration;
  buf += y;
  while (iteration < max_iteration) {
      float xtemp = x0 * x0 - y0 * y0;
      y0 = 2 * x0 * y0;
      x0 = xtemp;
      x0 = x0 + xx;
      y0 = y0 + yy;
      float currentAbs = x0*x0 + y0*y0;
      if (currentAbs>4.0f){
         float diffToLast  = currentAbs - oldAbs;
         float diffToMax   =       _max - oldAbs;
         coverageNum = iteration + diffToMax/diffToLast;
         break;
      }
      oldAbs = currentAbs;
      iteration++;
  }
  if (iteration == max_iteration)
#if defined(__MACH__)
  {
      buf[x] = 0xff;
      buf[x+1] = 0;
      buf[x+2] = 0;
      buf[x+3] = 0;
  } else
  {
      uchar c = 0xff * log1p(coverageNum)/lnxp1_max_iteration;
      buf[x+0] = 0xff;
      buf[x+1] = c;
      buf[x+2] = c;
      buf[x+3] = c;
   }
#else
  {
      buf[x] = 0;
      buf[x+1] = 0;
      buf[x+2] = 0;
      buf[x+3] = 0xff;
  } else
  {
      uchar c = 0xff * log1p(coverageNum)/lnxp1_max_iteration;
      buf[x+0] = c;
      buf[x+1] = c;
      buf[x+2] = c;
      buf[x+3] = 0xff;
  }
#endif
}
*/


  //     K          N          N
  //   [...]      [...]      [...]
  // M [.A.]  X K [.B.] => M [.C.]
  //   [...]      [...]      [...]


void axpy(const int N, const float ALPHA, __global const float* X, __global float* Y){
   for (int i=0 ;i<N; i++)
       Y[i] += ALPHA*X[i];
}


#define WIDTH 4

__kernel void sgemm1_nn(const long M, const long N, const long K, const float ALPHA ,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    const long globalRow = get_global_id(0); // Row ID of C (0..M)
    const long globalCol = get_global_id(1); // Col ID of C (0..N)

    A += globalRow*K ;
    B += globalCol;
    C += globalRow*N + globalCol;

    long k=0;
    float acc =0;

    #pragma unroll 8
    for (; k<K; k++)
      acc += A[k]*B[k*N];
    *C = acc;
}

__kernel void sgemm2_nn(const long M, const long N, const long K, const float ALPHA ,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    const long globalRow = get_global_id(1); // Row ID of C (0..M)
    const long globalCol = get_global_id(0); // Col ID of C (0..N)

    A += globalRow*K ;
    B += globalCol;
    C += globalRow*N + globalCol;

    long k=0;
    float acc =0;

    #pragma unroll 8
    for (; k<K; k++)
      acc += A[k]*B[k*N];
    *C = acc;
}

__kernel void sgemm1_nt(const long M, const long N, const long K, const float ALPHA ,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    const long globalRow = get_global_id(0); // Row ID of C (0..M)
    const long globalCol = get_global_id(1); // Col ID of C (0..N)

    A += globalRow*K ;
    B += globalCol;
    C += globalRow*N + globalCol;

    long k=0;
    float acc =0;

    //for (int i=0; i<M; i++)
    //    for (int j=0; j<N; j++)
    //        {    // todo optimize nt
                //sum = 0;
                #pragma unroll 8
                for (long k=0; k<K; k++)
                    acc = acc + A[k] * B[k];
                *C = *C +  ALPHA * acc;
            //}
}

__kernel void sgemm1_tn(const long M, const long N, const long K, const float ALPHA ,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    const long globalRow = get_global_id(0); // Row ID of C (0..M)
    const long globalCol = get_global_id(1); // Col ID of C (0..N)

    A += globalRow*K ;
    B += globalCol;
    C += globalRow*N + globalCol;

    long k=0;
    float acc =0;

    //for (long i=0; i<M; i++)
    //  for (long j=0; j<N; j++)

        for (long k=0; k<K; k++)
            *C = *C + ALPHA * A[k * M] * B[k * N];
}

