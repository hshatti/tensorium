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

typedef enum {
    acLOGISTIC, acRELU, acRELU6, acRELIE, acLINEAR, acRAMP, acTANH, acPLSE,
    acREVLEAKY, acLEAKY, acELU, acLOGGY, acSTAIR, acHARDTAN, acLHTAN, acSELU, acSOFTMAX,
    acGELU, acSWISH, acMISH, acHARD_MISH, acNORM_CHAN, acNORM_CHAN_SOFTMAX,
    acNORM_CHAN_SOFTMAX_MAXVAL
  } ActivationType;


float sumv(const long N, __global const float* v, const long stride){
  float sum=0;
  #pragma unroll 8
  for (long i=0;i<N; i++)
    sum += v[i*stride];
  return sum;
}

float maxv(const long N, __global const float* v, const long stride){
  float m=v[0];
  #pragma unroll 8
  for (long i=1;i<N; i++)
    m = fmax(m, v[i*stride]);
  return m;
}

float minv(const long N, __global const float* v, const long stride){
  float m=v[0];
  #pragma unroll 8
  for (long i=1;i<N; i++)
    m = fmin(m, v[i*stride]);
  return m;
}

#define VW 4
float sumv_simd(const long N, __global float* v){
  float4 sum4 = 0;
  float sum = 0;
  long K = N / VW;
  #pragma unroll 8
  for (long i=0;i<K; i++)
    sum4 += vload4(i, v);
  v += K*4;
  for (long i=0; i<N%VW;i++)
    sum4[i] += v[i];
  return sum4.x + sum4.y + sum4.z + sum4.w;
}

float dotv(const long N, __global float* a, const long inca,  __global float* b, const long incb){

  float d = 0;
  #pragma unroll 8
  for (long i=0; i<N;i++)
    d += a[i*inca]*b[i*incb];
  return d;
}

float dotv_simd(const long N, __global float* a,  __global float* b){

  float4 d = 0;
  long K = N / VW;
  #pragma unroll 8
  for (long i=0; i<K;i++)
    d += vload4(i, a) * vload4(i, b);
  a += K*4;
  b += K*4;
  for (long i=0; i<N%VW;i++)
    d.x += a[i]*b[i];
  return d.x + d.y + d.z + d.w;
}

#define WIDTH 4
              // naive GEMM with unrolling for now
__kernel void sgemm1_nn(const long K, const float ALPHA ,
                      const __global float* A, const long aOffset, const long lda,
                      const __global float* B, const long bOffset, const long ldb,
                      const float BETA, __global float* C, const long cOffset, const long ldc) {

    const long globalRow = get_global_id(0); // Row ID of C (0..M)
    const long globalCol = get_global_id(1); // Col ID of C (0..N)

    A += globalRow*lda +aOffset ;
    B += globalCol + bOffset;
    C += globalRow*ldc + globalCol + cOffset;
    *C *= BETA;

    float acc =0;

    #pragma unroll 8
    for (long k=0; k<K; k++)
      acc += A[k]*B[k*ldb];
    *C += acc * ALPHA;
    //*C += dotv(K, A, 1, B, ldb) * ALPHA;
}

__kernel void sgemm2_nn(const long K, const float ALPHA ,
                      const __global float* A, const long aOffset, const long lda,
                      const __global float* B, const long bOffset, const long ldb,
                      const float BETA, __global float* C, const long cOffset, const long ldc) {

    const long globalRow = get_global_id(1); // Row ID of C (0..M)
    const long globalCol = get_global_id(0); // Col ID of C (0..N)

    A += globalRow*lda + aOffset ;
    B += globalCol + bOffset;
    C += globalRow*ldc + globalCol + cOffset;

    *C *= BETA;
    float acc =0;

    #pragma unroll 8
    for (long k=0; k<K; k++)
        acc += A[k]*B[k*ldb];
    *C += acc * ALPHA;
    //*C += dotv(K, A, 1, B, ldb) * ALPHA;
}

__kernel void sgemm1_nt(const long K, const float ALPHA ,
                      const __global float* A, const long aOffset, const long lda,
                      const __global float* B, const long bOffset, const long ldb,
                      const float BETA, __global float* C, const long cOffset, const long ldc) {

    const long globalRow = get_global_id(0);    // M
    const long globalCol = get_global_id(1);    // N

    A += globalRow*lda + aOffset;
    B += globalCol*ldb + bOffset;
    C += globalRow*ldc + globalCol + cOffset;
    *C *= BETA;

    float acc =0;

    #pragma unroll 8
    for (long k=0; k<K; k++)
        acc += A[k] * B[k];
    *C += acc * ALPHA;
    //*C += dotv(K, A, 1, B, 1) * ALPHA;
            //}
}

__kernel void sgemm1_tn(const long K, const float ALPHA ,
                      const __global float* A, const long aOffset, const long lda,
                      const __global float* B, const long bOffset, const long ldb,
                      const float BETA, __global float* C, const long cOffset, const long ldc) {

    const long row = get_global_id(0); // Row ID of C (0..M)
    const long col = get_global_id(1); // Col ID of C (0..N)

    A += row           +  aOffset;
    B += col           +  bOffset;
    C += row*ldc + col +  cOffset;

    *C *= BETA;
    float acc = 0;

    #pragma unroll 8
    for (long k=0; k<K; k++)
        acc += A[k * lda] * B[k * ldb] ;
    *C += acc * ALPHA ;
    //*C += dotv(K, A, lda, B, 1) * ALPHA ;
}

__kernel void forward_bias(__global float* a,  const long aOffset, /*const long blockSize, */__global float* b, const long bOffset, const long incb)
{

  const long N = get_global_size(0);
  const long blockSize = get_global_size(2);
  const long i = get_global_id(0);
  const long k = get_global_id(1);
  const long j = get_global_id(2);

  //if (i==0 && k==0){
    //printf("        N = %ld\n", N);
    //printf("blockSize = %ld\n", blockSize);
  //}
  //for (i = 0; i<N; i++)
  //  for (k = 0; k<batch; k++){
  a += (k*N + i)*blockSize;
  float bb = b[i * incb];
  //#pragma unroll 8
  //    for (long j=0; j<blockSize; j++)
        a[j] += bb;
  //}
}

float stair_activate(const float x)
{
  long n = floor(x);
  if (n % 2 == 0) return floor(x/ 2);
  return (x - n) + floor(x/2);
}

float hardtan_activate(const float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return (x);
}

float linear_activate(const float x)
{
  return x;
}

float logistic_activate(const float x)
{
  //result := 1/(1 + exp(EnsureRange(-x, minSingleExp, maxSingleExp)))
  return 1/(1 + exp(-x));
}

float loggy_activate(const float x)
{
  //result := 2/(1 + exp(EnsureRange(-x, minSingleExp, maxSingleExp))) - 1;
  return 2/(1 + exp(-x)) - 1;
}

float relu_activate(const float x)
{
  //return x*long(x>0);
  if (x<0) return 0;
  return x;
}

float relu6_activate(const float x)
{
  //min_val_cmp(max_val_cmp(x, 0), 6)
  //result := EnsureRange(x,0,6);
  return  x*(x>0) * (x<=6);
}

float elu_activate(const float x)
{
  return (x >= 0)*x + (x < 0)*(exp(x)-1);
}

float selu_activate(const float x)
{
  return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(exp(x)-1);
}

float gelu_activate(const float x)
{
  return 0.5f*x*(1 + tanh(0.797885f*x + 0.035677f*pow(x, 3)));
}

float relie_activate(const float x)
{
  if (x>0) return x;
  else return 0.01f*x;
}

float ramp_activate(const float x)
{
  return  x*(x>0)+0.1f*x;
}

float leaky_activate(const float x)
{
  if (x>0) return  x;
  else return  0.1f*x;
}

float tanh_activate(const float x)
{
  //result := 2 / (1 + exp(ensureRange(-2 * x, minSingleExp, maxSingleExp))) - 1
  return  (exp(2*x)-1)/(exp(2*x)+1);
}

float softplus_activate(const float x, const float threshold)
{
    if (x > threshold)
      return (x);                // too large
    else if (x < -threshold)
      return (exp(x));    // too small
    //return (log(exp(x) + 1));
    return log1p(exp(x));
}

float plse_activate(const float x)
{
    if (x < -4 ) return( 0.01f * (x + 4));
    if (x > 4 ) return( 0.01f * (x - 4) + 1);
    return  0.125f*x + 0.5f;
}

float lhtan_activate(const float x)
{
    if(x < 0) return (0.001f*x);
    if(x > 1) return (0.001f*(x-1) + 1);
    return  x;
}

float silu_activate(const float x)
{
    return x * logistic_activate(x) ;
}
//void softmax_activate(const N:SizeInt; const x: PSingle);
//{
//  long i;
//  float mx := TSingleTensor.maxv(N, Pointer(x), 1);//MaxValue(x, N);
//  for i:=0 to N-1 do
//    //x[i] := Exp(EnsureRange(x[i]-mx, minSingleExp, maxSingleExp));
//    x[i] := Exp(x[i]-mx);
//
//  mx := TSingleTensor.Sumv(N, pointer(x), 1);
//  //r:=copy(x);
//  //r.Exp();
//  for i :=0 to N-1 do
//    x[i] := x[i] / mx
//}


__kernel void activate_array( __global float* x, const ActivationType a)
{
      long i = get_global_id(0);
      //int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);

      switch (a) {
          case acLOGISTIC:
            //for (i = 0; i< N; i++)
                x[i] = logistic_activate(x[i]);
            break;
          case acRELU:
            //for (i = 0; i< N; i++)
                x[i] = relu_activate(x[i]);
            break;
          case acRELU6:
            //for (i = 0; i< N; i++)
                x[i] = relu6_activate(x[i]);
            break;
          case acRELIE:
            //for (i = 0; i< N; i++)
                x[i] = relie_activate(x[i]);
            break;
          case acLINEAR:
            //for (i = 0; i< N; i++)
            //    x[i] = linear_activate(x[i])
            break;
          case acRAMP:
            //for (i = 0; i< N; i++)
                x[i] = ramp_activate(x[i]);
            break;
          case acTANH:
            //for (i = 0; i< N; i++)
                x[i] = tanh_activate(x[i]);
            break;
          case acPLSE:
            //for (i = 0; i< N; i++)
                x[i] = plse_activate(x[i]);
            break;
          case acREVLEAKY: case acLEAKY:
            //for (i = 0; i< N; i++)
             if (x[i]<0) x[i] = 0.1f*x[i];
              //x[i] = leaky_activate(x[i]);
            break;
          case acELU:
            //for (i = 0; i< N; i++)
                x[i] = elu_activate(x[i]);
            break;
          case acLOGGY:
            //for (i = 0; i< N; i++)
                x[i] = loggy_activate(x[i]);
            break;
          case acSTAIR:
            //for (i = 0; i< N; i++)
                x[i] = stair_activate(x[i]);
            break;
          case acHARDTAN:
            //for (i = 0; i< N; i++)
                x[i] = hardtan_activate(x[i]);
            break;
          case acLHTAN:
            //for (i = 0; i< N; i++)
                x[i] = lhtan_activate(x[i]);
            break;
          case acSELU:
            //for (i = 0; i< N; i++)
                x[i] = selu_activate(x[i]);
            break;
          case acGELU:
            //for (i = 0; i< N; i++)
                x[i] = gelu_activate(x[i]);
            break;
          case acSWISH:
                x[i] = silu_activate(x[i]);
            break;
          //case acSOFTMAX:
            //softmax_activate(N, x);
            //break
          default:
            printf("[Activation] : not Implemented");

      }
   //printf("%ld, ", i);

}

__kernel void array_avtivate_swish(__global float* x, __global float* output, __global float* output2)
{
    long i = get_global_id(0);
    float x_val       = x[i];
    float sigmoid     = logistic_activate(x_val);
    output[i]         = sigmoid;
    output2[i]        = x_val * sigmoid;
}




float lhtan_gradient(const float x)
{
    if ((x > 0) &&  (x < 1))
      return 1;
    return 0.001f;
}


float hardtan_gradient(const float x)
{
    if ((x > -1) && (x < 1))
      return 1;
    return 0;
}

float linear_gradient(const float x)
{
    return 1;
}

float logistic_gradient(const float x)
{
    return (1-x)*x;
}

float loggy_gradient(const float x)
{
    float y = (x+1.0f)/2.0f;
    return 2.0f*(1.0f-y)*y;
}

float stair_gradient(const float x)
{
    if (floor(x) == x) return( 0);
    return 1;
}

float relu_gradient(const float x)
{
    return (x>0?1:0);
}

float relu6_gradient(const float x)
{
    return ((x>0) && (x<6)?1:0);
}

float elu_gradient(const float x)
{
    return (x >= 0?1:0) + (x < 0?1:0)*(x + 1);
}

float selu_gradient(const float x)
{
    return (x >= 0?1:0)*1.0507f + (x < 0?1:0)*(x + 1.0507f*1.6732f);
}

float relie_gradient(const float x)
{
    if (x>0) return 1;
    else return 0.01f;
}

float ramp_gradient(const float x)
{
    return (x>0?1:0) + 0.1f;
}

float leaky_gradient(const float x)
{
    if (x>0) return 1;
    else return 0.1f;
}

float tanh_gradient(const float x)
{
    return 1-x*x;
}

float sech(const float x)
{
    return 2.0f / (exp(x) + exp(-x));
}

float gelu_gradient(const float x)
{
    float x3 = pow(x,3);
    return 0.5f*tanh(0.0356774f*x3 + 0.797885f*x) + (0.0535161f*x3 + 0.398942f*x) * pow(sech(0.0356774f*x3 + 0.797885f*x), 2.0f) + 0.5f ;
}

float plse_gradient(const float x)
{

  if ((x < 0) || (x > 1))
    return  0.01f;
  else
    return 0.125f;
}

__kernel void gradient_array(__global const float* x, const ActivationType a, __global float* delta)
{
    long i = get_global_id(0);

    switch (a) {
        case acLOGISTIC:
          //for (i = 0; i<N;i++)
              delta[i] *= logistic_gradient(x[i]);
              break;
        case acRELU:
          //for (i = 0; i<N;i++)
              delta[i] *= x[i]>0?1:0;//relu_gradient(x[i]);
              break;
        case acRELU6:
          //for (i = 0; i<N;i++)
              delta[i] *= relu6_gradient(x[i]);
              break;
        case acRELIE:
          //for (i = 0; i<N;i++)
              delta[i] *= relie_gradient(x[i]);
              break;
        case acLINEAR:
          //////for (i = 0; i<N;i++)
          //    delta[i] *= linear_gradient(x[i])
          //;
              break;
        case acRAMP:
          //for (i = 0; i<N;i++)
              delta[i] *= ramp_gradient(x[i]);
              break;
        case acTANH:
          //for (i = 0; i<N;i++)
              delta[i] *= tanh_gradient(x[i]);
              break;
        case acPLSE:
          //for (i = 0; i<N;i++)
              delta[i] *= plse_gradient(x[i]);
              break;
        case acREVLEAKY: case acLEAKY:
          //for (i = 0; i<N;i++)
              delta[i] *= leaky_gradient(x[i]);
              break;
        case acELU:
          //for (i = 0; i<N;i++)
              delta[i] *= elu_gradient(x[i]);
              break;
        case acLOGGY:
          //for (i = 0; i<N;i++)
              delta[i] *= loggy_gradient(x[i]);
              break;
        case acSTAIR:
          //for (i = 0; i<N;i++)
              delta[i] *= stair_gradient(x[i]);
              break;
        case acHARDTAN:
          //for (i = 0; i<N;i++)
              delta[i] *= hardtan_gradient(x[i]);
              break;
        case acLHTAN:
          //for (i = 0; i<N;i++)
              delta[i] *= lhtan_gradient(x[i]);
              break;
        case acSELU:
          //for (i = 0; i<N;i++)
              delta[i] *= selu_gradient(x[i]);
              break;
        case acGELU:
          //for (i = 0; i<N;i++)
              delta[i] *= gelu_gradient(x[i]);
              break;
    //   case acSWISH:
    //               ;
    //
    //   case acMISH:
    //               ;
    //
    //   case acHARD_MISH:
    //               ;
    //
    //   case acNORM_CHAN:
    //               ;
    //
    //   case acNORM_CHAN_SOFTMAX:
    //               ;
    //
    //   case acNORM_CHAN_SOFTMAX_MAXVAL:
    //
        default:
            printf("[Gradient] : not Implemented %d", a);
          ;
    }

}
__kernel void backward_bias(__global float* a, const long blockSize, __global float* bias, const long batch, const long N)
{

    const long i = get_global_id(0);//if (i==0) printf("long %ull\n", sizeof(long));

    //for (long i=0 ; i<N ;i++) {
      float sum = 0;
      bias += i * blockSize;
      const long incbias = N*blockSize;
      #pragma unroll 8
      for (long j=0; j<batch; j++){
        sum += sumv(blockSize, bias, 1);
        bias += incbias;
      }
      a[i] +=sum;
}

__kernel void addv( __global float* dst, __global const float* src){

   const long i = get_global_id(0);
   dst[i] += src[i];
}

__kernel void subv( __global float* dst, __global const float* src){

   const long i = get_global_id(0);
   dst[i] -= src[i];
}

__kernel void axpy(const float a, __global const float* x, const long incx, __global float* y, const long incy){

   const long i = get_global_id(0);
   y[i*incy] += a*x[i*incx];

}

__kernel void scale(const float a, __global float* x, const long incx){

   const long i = get_global_id(0);
   x[i*incx] *= a;

}

#define sEPSILON 0.000001f

__kernel void crossEntropyLogistics(__global const float* pred, __global const float* truth, __global float* delta, __global float* error){

  const long i = get_global_id(0);
  float t = truth[i];
  float p = pred[i];
  error[i] = -t*log(fmax(p, sEPSILON)) - (1-t) * log(fmax(1 - p, sEPSILON));
  delta[i] = t - p;
   //printf("%ld, ", i);

}

__kernel void fill(__global float* x, const float val, const long stride){

   const long i = get_global_id(0);
   x[i*stride] = val;
}

// naive copy for now
__kernel void copy(
    __global float* a, const long aOffset, const long inca
  , __global float* b, const long bOffset, const long incb){

   const long i = get_global_id(0);
   a += aOffset; b += bOffset;
   b[i*incb] = a[i*inca];
}

__kernel void forwardMaxpool(
     __global float* input
     , const long c, const long h, const long w
     , const long stride_x, const long stride_y, const long padding, const long kernelSize
     , __global long* indexes, __global float* output){

  const long w_offset = -padding / 2;
  const long h_offset = -padding / 2;

  //const long outC = get_global_size(0);
  const long outH = get_global_size(1);
  const long outW = get_global_size(2);
  long k = get_global_id(0);
  long y = get_global_id(1);
  long x = get_global_id(2);

  long out_index = x + outW*(y + outH*k) ;//+ outW*outH*outC*b;
  float max = -FLT_MAX;
  long max_i = -1;
  #pragma unroll 8
  for (long n=0; n<kernelSize; n++)
      #pragma unroll 8
      for (long m=0; m<kernelSize; m++){
          long cur_h = h_offset+y * stride_y+n;
          long cur_w = w_offset+x * stride_x+m;
          long index = cur_w + w*(cur_h + h*k) ;//+ w*h*outC*b;
          float val = (cur_h >= 0) && (cur_h < h) && (cur_w >= 0) && (cur_w < w)? input[index]: -FLT_MAX;
          if (val > max){
            max_i = index;
            max = val;
          }
      }
  output[out_index] = max;
  if (indexes)
      indexes[out_index] = max_i;
}

__kernel void backwardMaxPool( __global float* output, __global const long* indexes, __global const float* delta){
        const long i = get_global_id(0);
        const long j = get_global_id(1);
        const long id = i*get_global_size(1) + j;

        const long index = indexes[id];
        output[index] += delta[id];
}


void softmax(const long n, __global float* input, const long stride, const float temp, __global float* output){

  float largest = maxv(n, input, stride);
  float sum = 0;

  #pragma unroll 8
  for (long i=0;i<n;i++) {
      float e = exp((input[i*stride] - largest)/temp);
      sum += e;
      output[i*stride] = e;
  }

  #pragma unroll 8
  for (long i=0; i<n; i++)
      output[i*stride]/=sum;
}

__kernel void softmaxBatch(__global float* input, const long iOffset, const long n
  , const long batch_size, const long group_size, const long stride
  , const float temp, __global float* output, const long oOffset){

  const long b = get_global_id(0);
  const long g = get_global_id(1);

  softmax(n
  , input + iOffset + b*batch_size + g*group_size
  , stride
  , temp
  , output + oOffset + b*batch_size + g*group_size);

}

void move(const __global float* src, __global float* dst , const long count){
  #pragma unroll 8
  for (long i=0 ; i<count; i++) dst[i] = src[i];
}

#define sEPSILON 0.000001f
__kernel void softmaxCrossEntropy(const __global float* pred, const __global float* truth, __global float* delta, __global float* error){

  const long i = get_global_id(0);

  float t = truth[i];
  float p = pred[i];
  if (t!=0)
      error[i] = -log(max(p , sEPSILON));
  else
      error[i] = 0;
  delta[i] = t - p;

}

__kernel void im2col(const long aHeight, const long aWidth
  , const long kernelHeight, const long kernelWidth, const long padHeight, const long padWidth
  , const long strideY, const long strideX, const long dilationY, const long dilationX
  , __global float* im , const long imOffset
  , __global float* col, const long colOffset, const long batch){

  long aChannels = get_global_size(0);
  long chan = get_global_id(0);
  long k = get_global_id(1);
  const long kernelSize = kernelHeight*kernelWidth;

  const long outWidth = (aWidth + 2 * padWidth - (dilationX * (kernelWidth - 1) + 1)) / strideX + 1;
  const long outHeight = (aHeight + 2 * padHeight - (dilationY * (kernelHeight - 1) + 1)) / strideY + 1;
  const long outSize = outWidth * outHeight;
  const long inSize = aWidth * aHeight;
  const long sizeX = outWidth - 2 * padWidth;

    //for (long k=0 ; k<kernelWidth*kernelHeight; k++)
    {
      long kernelRow = k / kernelWidth;
      long kernelCol = k % kernelWidth;
      //const long kernelRow = get_global_id(1);
      //const long kernelCol = get_global_id(2);

      #pragma unroll 8
      for (long b=0 ; b<batch; b++)
      {
        long i = (b*aChannels + chan)*inSize + aWidth*(kernelRow*dilationY - padHeight) + kernelCol*dilationX - padWidth;
        __global float* im1 = imOffset + im + i;
        __global float* col1 = colOffset + col + padWidth * outWidth + outSize*kernelSize*(chan + b*aChannels) + outSize*(kernelRow * kernelWidth + kernelCol) ;
        #pragma unroll 8
        for (long outRow=padHeight ; outRow<outHeight - padHeight ; outRow++)
        {
          long j = outRow * aWidth * strideY + padWidth * strideX ;
          if (strideX == 1)
            move(im1 + j, col1 + padWidth, sizeX);
          else
            #pragma unroll 8
            for (long outCol=padWidth ;  outCol<outWidth - padWidth; outCol++)
            {
              //j := outRow * aWidth * strideY + outCol * strideX;
              col1[outCol] = im1[j];
              j += strideX;
            }
          col1 += outWidth;
        }
      }
    }
}

void fills(__global float* dst, const long N, const float val){
  #pragma unroll 8
  for (long i=0 ; i<N; i++)
    dst[i] = val;
}


//col2im :
//https://github.com/CNugteren/CLBlast/blob/master/src/kernels/levelx/col2im.opencl

// Work-group size parameters re-used from the 'copy' kernel
#ifndef COPY_DIMX
  #define COPY_DIMX 8      // Local workgroup size in the first dimension (w)
#endif
#ifndef COPY_DIMY
  #define COPY_DIMY 8      // Local workgroup size in the second dimension (h)
#endif

// =================================================================================================

long grid_ceil(const long x, const long step) {
  return x > 0 ? ((x - 1) / step + 1) * step : x / step * step;
}

void xim2col(const long input_h, const long input_w, const long channels,
                         const long output_h, const long output_w,
                         const long kernel_h, const long kernel_w,
                         const long pad_h, const long pad_w,
                         const long stride_h, const long stride_w,
                         const long dilation_h, const long dilation_w,
                         const bool kernel_flip,
                         const __global float* restrict im_buffer, const long im_offset,
                         __global float* col_buffer, const long col_offset) {

  // Thread IDs
  const long w_id = get_global_id(0); // image width, max 'output_w'
  const long h_id = ((long)get_global_id(1)) % output_h; // image height, max 'output_h'
  const long c_id = ((long)get_global_id(1)) / output_h; // input channels
  if (h_id < output_h && w_id < output_w && c_id < channels) {
    #pragma unroll 8
    for (long kh_id = 0; kh_id < kernel_h; ++kh_id) { // kernel height
      #pragma unroll 8
      for (long kw_id = 0; kw_id < kernel_w; ++kw_id) { // kernel width

        // Retrieves the input value
        const long h_index = -pad_h + kh_id * dilation_h + stride_h * h_id;
        const long w_index = -pad_w + kw_id * dilation_w + stride_w * w_id;
        float val;
        if (h_index >= 0 && h_index < input_h &&
            w_index >= 0 && w_index < input_w) {
          const long input_index = w_index + input_w * (h_index + input_h * c_id);
          val = im_buffer[input_index + im_offset];
        }
        else {
          val = 0;
        }

        // Sets the output value
        const long kernel_index = (kernel_flip)
                               ? kernel_h * kernel_w - kw_id - kernel_w * kh_id - 1
                               : kw_id + kernel_w * kh_id;
        const long patch_index = w_id + output_w * h_id;
        const long output_index = patch_index + kernel_index * output_w * output_h +
                                  c_id * output_w * output_h * kernel_h * kernel_w;
        col_buffer[output_index + col_offset] = val;
      }
    }
  }
}

// =================================================================================================

// Kernel flip version of the Xim2col kernel (for convolution)
//#if RELAX_WORKGROUP_SIZE == 1
  __kernel
//#else
//  __kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
//#endif
void Xim2colKernelFlip(const long input_h, const long input_w, const long channels,
                       const long output_h, const long output_w,
                       const long kernel_h, const long kernel_w,
                       const long pad_h, const long pad_w,
                       const long stride_h, const long stride_w,
                       const long dilation_h, const long dilation_w,
                       const __global float* restrict im_buffer, const long im_offset,
                       __global float* col_buffer, const long col_offset) {
  const bool kernel_flip = true;
  xim2col(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          kernel_flip,
          im_buffer, im_offset, col_buffer, col_offset);
}

// Normal version of the Xim2col kernel (for cross-correlation)
//#if RELAX_WORKGROUP_SIZE == 1
  __kernel
//#else
//  __kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
//#endif
void Xim2colKernelNormal(const long input_h, const long input_w, const long channels,
                         const long output_h, const long output_w,
                         const long kernel_h, const long kernel_w,
                         const long pad_h, const long pad_w,
                         const long stride_h, const long stride_w,
                         const long dilation_h, const long dilation_w,
                         const __global float* restrict im_buffer, const long im_offset,
                         __global float* col_buffer, const long col_offset) {
  const bool kernel_flip = false;
  xim2col(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          kernel_flip,
          im_buffer, im_offset, col_buffer, col_offset);
}

void xcol2im(const long input_h, const long input_w, const long channels,
                         const long output_h, const long output_w,
                         const long kernel_h, const long kernel_w,
                         const long pad_h, const long pad_w,
                         const long stride_h, const long stride_w,
                         const long dilation_h, const long dilation_w,
                         const long stride_bez_h, const long stride_bez_w,
                         const long dilation_bez_h, const long dilation_bez_w,
                         const long gcd_h, const long gcd_w,
                         const bool kernel_flip,
                         const __global float* restrict col_buffer, const long col_offset,
                         __global float* im_buffer, const long im_offset) {

  const long input_h_scaled = (input_h - 1) / gcd_h + 1;

  // Thread IDs
  const long gcd_scale_w = get_global_id(0) + (pad_w - 1) / gcd_w + 1;
  const long gcd_scale_h = ((long) get_global_id(1)) % input_h_scaled + (pad_h - 1) / gcd_h + 1;
  const long c_id = ((long) get_global_id(1)) / input_h_scaled;

  const long w_index = gcd_scale_w * gcd_w - pad_w;
  const long h_index = gcd_scale_h * gcd_h - pad_h;
  const long th_step = stride_h * dilation_h / gcd_h;
  const long th_begin = grid_ceil(max(-stride_bez_h * gcd_scale_h * stride_h,
                                     (dilation_bez_h * gcd_scale_h - kernel_h + 1) * dilation_h),
                                 th_step);
  const long th_end = min((output_h - stride_bez_h * gcd_scale_h) * stride_h,
                         (dilation_bez_h * gcd_scale_h + 1) * dilation_h);
  const long tw_step = stride_w * dilation_w / gcd_w;
  const long tw_begin = grid_ceil(max(-stride_bez_w * gcd_scale_w * stride_w,
                                     (dilation_bez_w * gcd_scale_w - kernel_w + 1) * dilation_w),
                                 tw_step);
  const long tw_end = min((output_w - stride_bez_w * gcd_scale_w) * stride_w,
                         (dilation_bez_w * gcd_scale_w + 1) * dilation_w);
  if (w_index < input_w && c_id < channels) {
    float val = 0;
    #pragma unroll 8
    for (long th = th_begin; th < th_end; th += th_step) {
      #pragma unroll 8
      for (long tw = tw_begin; tw < tw_end; tw += tw_step) {
        const long kh_id = -th / dilation_h + dilation_bez_h * gcd_scale_h;
        const long kw_id = -tw / dilation_w + dilation_bez_w * gcd_scale_w;
        const long h_id = th / stride_h + stride_bez_h * gcd_scale_h;
        const long w_id = tw / stride_w + stride_bez_w * gcd_scale_w;
        const long kernel_index = (kernel_flip)
                               ? kernel_h * kernel_w - kw_id - kernel_w * kh_id - 1
                               : kw_id + kernel_w * kh_id;
        const long patch_index = w_id + output_w * h_id;
        const long output_index = patch_index + kernel_index * output_w * output_h +
                                 c_id * output_w * output_h * kernel_h * kernel_w;
        val += col_buffer[output_index + col_offset];
      }
    }

    // Accumulates the resulting value with the existing im-buffer (+= val)
    const long input_index = w_index + input_w * (h_index + input_h * c_id);
    float im_buffer_value = im_buffer[input_index + im_offset];
    im_buffer[input_index + im_offset] = im_buffer_value + val;
  }
}

// =================================================================================================

// Kernel flip version of the Xcol2im kernel (for convolution)
//#if RELAX_WORKGROUP_SIZE == 1
  __kernel
//#else
  //__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
//#endif
void Xcol2imKernelFlip(const long input_h, const long input_w, const long channels,
                       const long output_h, const long output_w,
                       const long kernel_h, const long kernel_w,
                       const long pad_h, const long pad_w,
                       const long stride_h, const long stride_w,
                       const long dilation_h, const long dilation_w,
                       const long stride_bez_h, const long stride_bez_w,
                       const long dilation_bez_h, const long dilation_bez_w,
                       const long gcd_h, const long gcd_w,
                       const __global float* restrict col_buffer, const long col_offset,
                       __global float* im_buffer, const long im_offset) {
  const bool kernel_flip = true;
  xcol2im(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          stride_bez_h, stride_bez_w, dilation_bez_h, dilation_bez_w, gcd_h, gcd_w,
          kernel_flip,
          col_buffer, col_offset, im_buffer, im_offset);
}

// Normal version of the Xcol2im kernel (for cross-correlation)
//#if RELAX_WORKGROUP_SIZE == 1
  __kernel
//#else
  //__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
//#endif
void Xcol2imKernelNormal(const long input_h, const long input_w, const long channels,
                         const long output_h, const long output_w,
                         const long kernel_h, const long kernel_w,
                         const long pad_h, const long pad_w,
                         const long stride_h, const long stride_w,
                         const long dilation_h, const long dilation_w,
                         const long stride_bez_h, const long stride_bez_w,
                         const long dilation_bez_h, const long dilation_bez_w,
                         const long gcd_h, const long gcd_w,
                         const __global float* restrict col_buffer, const long col_offset,
                         __global float* im_buffer, const long im_offset) {
  const bool kernel_flip = false;
  xcol2im(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          stride_bez_h, stride_bez_w, dilation_bez_h, dilation_bez_w, gcd_h, gcd_w,
          kernel_flip,
          col_buffer, col_offset, im_buffer, im_offset);
}



