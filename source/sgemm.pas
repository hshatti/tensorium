// Copyright 2024 Mozilla Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
//                   _   _          ___ _      _   ___
//                  | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                  |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                   \__|_|_||_\_, |___/____/_/ \_\___/
//                             |__/
//
//                    BASIC LINEAR ALGEBRA SUBPROGRAMS
//
//
// This file implements multithreaded CPU matrix multiplication for the
// common contiguous use case C = Aᵀ * B. These kernels are designed to
// have excellent performance[1] for matrices that fit in the CPU cache
// without imposing any overhead such as cache filling or malloc calls.
//
// This implementation does not guarantee any upper bound with rounding
// errors, which grow along with k. Our goal's to maximally exploit the
// hardware for performance, and then use whatever resources remain for
// improving numerical accuracy.
//
// [1] J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
//     Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].

unit sgemm;
{$ifdef FPC}
  {$mode delphi}
  {$ModeSwitch advancedrecords}
  {$asmmode intel}
{$endif}
{$pointermath on}

interface

{$if defined(USE_AVX512)}
const VECTOR_REGISTERS = 32 ;
{$else}
const VECTOR_REGISTERS = 16 ;
{$endif}

{.$define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)}

type
  fp16    = smallint;
  ggml_fp16_t = fp16;


  m128  = psingle;
  m128i = plongint;
  m256  = packed record a:array[0..7] of single end;
  m256i = plongint;

  Pm128   = ^m128   ;
  Pm128i  = ^m128i  ;
  Pm256   = ^m256   ;
  Pm256i  = ^m256i  ;

  PPm128  = ^Pm128  ;
  PPm128i = ^Pm128i ;
  PPm256  = ^Pm256  ;
  PPm256i = ^Pm256i ;

  SizeInt = int64;

  { tinyBLAS }

  tinyBLAS = record
  const
    KN = 8;
  type
     TArray4_3 = array[0..2] of array[0..3] of m256;
     TArray3_4 = array[0..3] of array[0..2] of m256;
     TArray5_2 = array[0..1] of array[0..4] of m256;
     TArray3_3 = array[0..2] of array[0..2] of m256;
     TArray2_5 = array[0..4] of array[0..1] of m256;
     TArray4_2 = array[0..1] of array[0..3] of m256;
     TArray2_4 = array[0..3] of array[0..1] of m256;
     TArray3_2 = array[0..1] of array[0..2] of m256;
     TArray2_3 = array[0..2] of array[0..1] of m256;
     TArray5_1 = array[0..0] of array[0..4] of m256;
     TArray4_1 = array[0..0] of array[0..3] of m256;
     TArray2_2 = array[0..1] of array[0..1] of m256;
     TArray1_5 = array[0..4] of array[0..0] of m256;
     TArray1_4 = array[0..3] of array[0..0] of m256;
     TArray3_1 = array[0..0] of array[0..2] of m256;
     TArray1_3 = array[0..2] of array[0..0] of m256;
     TArray2_1 = array[0..0] of array[0..1] of m256;
     TArray1_2 = array[0..1] of array[0..0] of m256;
     TArray1_1 = array[0..0] of array[0..0] of m256;

  public
    k:SizeInt;
    A, B, C:PSingle;
    lda, ldb, ldc: SizeInt;
    ith, nth :integer;
    constructor Create(
         const k:SizeInt ;
         const A:PSingle; const lda:SizeInt ;
         const B:PSingle; const ldb:SizeInt ;
         const C:PSingle; const ldc:SizeInt ;
         const ith, nth:integer);
    procedure matmul(const m, n :SizeInt);
    procedure mnpack(const m0, m, n0, n:SizeInt);
    procedure gemm<KR>(const m0, m, n0, n:SizeInt);inline;
  end;

//function add(const x, y): m256;assembler;inline;
//function sub(const x, y:m256): m256;assembler;inline;
//function mul(const x, y:m256): m256;assembler;inline;
function madd(const a, b, c:psingle): m256;assembler;inline;
function hsum(const x : m256):single;assembler;inline;
//function load(const x:psingle):m256;assembler;inline;

function tinysgemm(const m, n, k :SizeInt; const  A:PSingle; const lda: SizeInt; const B:PSingle; const ldb:SizeInt;
                const C:PSingle; const  ldc: SizeInt; const ith, nth:integer):boolean;
implementation
uses math;

function unhalf(const d: ggml_fp16_t):single; inline;
begin
  //  result := GGML_FP16_TO_FP32(d);
end;

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED ARITHMETIC OPERATIONS

function add(const x, y): m256;assembler;inline;
asm
  vmovups          ymm0 , yword [x]
  vaddps           ymm0 , ymm0, [y]
  vmovups          yword [result] , ymm0
end;

function sub(const x, y:m256): m256;assembler;inline;
asm
  vmovups          ymm0 , yword [x]
  vsubps           ymm0 , ymm0, [y]
  vmovups          yword [result] , ymm0
end;

function mul(const x, y:m256): m256;assembler;inline;
asm
  vmovups          ymm0 , yword [x]
  vmulps           ymm0 , ymm0, [y]
  vmovups          yword [result] , ymm0
end;



////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED FUSED MULTIPLY ADD

(**
 * Computes a * b + c.
 *)

function madd(const a, b, c: psingle): m256;
asm
  vmovups          ymm0 , yword [c]
  vmovups          ymm1 , yword [b]
  vfmadd231ps      ymm0 , ymm1, yword [a]
  vmovups          yword [result] , ymm0
end;

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED HORIZONTAL SUM

function hsum(const x : m256):single;assembler;inline;
asm
  vmovups        xmm0,  oword [x + 4*4]    //   4,5,6,7 -> xmm0
  vaddps         xmm0,  xmm0,  oword [x]   // + 0,1,2,3 -> xmm0
  vhaddps        xmm0,  xmm0, xmm0               //   0+1->0 2+3->1
  vhaddps        xmm0,  xmm0, xmm0               //   0+1->0
  vmovss         dword result, xmm0
  //_mm_add_ps(_mm256_extractf128_ps(x, 1),
  //                         _mm256_castps256_ps128(x)));
end;

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED MEMORY LOADING

function load(const x:psingle):m256;assembler;inline;
asm
  vmovups  ymm0    , yword[x]
  vmovups  yword[x], ymm0
end;


////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION


constructor tinyBLAS.Create(const k: SizeInt; const A: PSingle;
  const lda: SizeInt; const B: PSingle; const ldb: SizeInt; const C: PSingle;
  const ldc: SizeInt; const ith, nth: integer);
begin
   self.A := A;
   self.B := B;
   self.C := C;
   self.k := k;
   self.lda := lda;
   self.ldb := ldb;
   self.ldc := ldc;
   self.ith := ith;
   self.nth := nth ;
end;

procedure tinyBLAS.matmul(const m, n :SizeInt) ;
begin
        mnpack(0, m, 0, n);
end;

procedure tinyBLAS.mnpack(const m0, m, n0, n:SizeInt);
var mc, nc, mp, np: SizeInt;
begin
     case (MIN(m - m0, 5) shl 4) or MIN(n - n0, 5) of
{$if VECTOR_REGISTERS = 32}
        $55:
        begin
            mc := 5;
            nc := 5;
            gemm<TArray5_5>(m0, m, n0, n);
        end;
        $45:
        begin
            mc := 4;
            nc := 5;
            gemm<TArray4_5>(m0, m, n0, n);
        end;
        $54:
        begin
            mc := 5;
            nc := 4;
            gemm<TArray5_4>(m0, m, n0, n);
        end;
        $44:
        begin
            mc := 4;
            nc := 4;
            gemm<TArray4_4>(m0, m, n0, n);
        end;
        $53:
        begin
            mc := 5;
            nc := 3;
            gemm<TArray5_3>(m0, m, n0, n);
        end;
        $35:
        begin
            mc := 3;
            nc := 5;
            gemm<TArray3_5>(m0, m, n0, n);
        end;
        $43:
        begin
            mc := 4;
            nc := 3;
            gemm<TArray4_3>(m0, m, n0, n);
        end;
{$else}
        $55,
        $54,
        $53,
        $45,
        $44,
        $43:
          begin
            mc := 4;
            nc := 3;
            gemm<TArray4_3>(m0, m, n0, n);             // array[0..2] of array[0..3] of single;
          end;
        $35,
{$endif}
        $34:
          begin
            mc := 3;
            nc := 4;
            gemm<TArray3_4>(m0, m, n0, n);            // array[0..3] of array[0..2] of single;
          end;
        $52:
          begin
            mc := 5;
            nc := 2;
            gemm<TArray5_2>(m0, m, n0, n);           // array[0..1] of array[0..4] of single;
          end;
        $33:
          begin
            mc := 3;
            nc := 3;
            gemm<TArray3_3>(m0, m, n0, n);           // array[0..2] of array[0..2] of single;
          end;
        $25:
          begin
            mc := 2;
            nc := 5;
            gemm<TArray2_5>(m0, m, n0, n);           // array[0..4] of array[0..1] of single;
          end;
        $42:
          begin
            mc := 4;
            nc := 2;
            gemm<TArray4_2>(m0, m, n0, n);           // array[0..1] of array[0..3] of single;
          end;
        $24:
          begin
            mc := 2;
            nc := 4;
            gemm<TArray2_4>(m0, m, n0, n);          // array[0..3] of array[0..1] of single;
          end;
        $32:
          begin
            mc := 3;
            nc := 2;
            gemm<TArray3_2>(m0, m, n0, n);         // array[0..1] of array[0..2] of single;
          end;
        $23:
          begin
            mc := 2;
            nc := 3;
            gemm<TArray2_3>(m0, m, n0, n);        // array[0..2] of array[0..1] of single;
          end;
        $51:
          begin
            mc := 5;
            nc := 1;
            gemm<TArray5_1>(m0, m, n0, n);        // array[0..0] of array[0..4] of single;
          end;
        $41:
          begin
            mc := 4;
            nc := 1;
            gemm<TArray4_1>(m0, m, n0, n);        // array[0..0] of array[0..3] of single;
          end;
        $22:
          begin
            mc := 2;
            nc := 2;
            gemm<TArray2_2>(m0, m, n0, n);       // array[0..1] of array[0..1] of single;
          end;
        $15:
          begin
            mc := 1;
            nc := 5;
            gemm<TArray1_5>(m0, m, n0, n);       // array[0..4] of array[0..0] of single;
          end;
        $14:
          begin
            mc := 1;
            nc := 4;
            gemm<TArray1_4>(m0, m, n0, n);      // array[0..3] of array[0..0] of single;
          end;
        $31:
          begin
            mc := 3;
            nc := 1;
            gemm<TArray3_1>(m0, m, n0, n);     // array[0..0] of array[0..2] of single;
          end;
        $13:
          begin
            mc := 1;
            nc := 3;
            gemm<TArray1_3>(m0, m, n0, n);        // array[0..2] of array[0..0] of single;
          end;
        $21:
          begin
            mc := 2;
            nc := 1;
            gemm<TArray2_1>(m0, m, n0, n);        // array[0..0] of array[0..1] of single;
          end;
        $12:
          begin
            mc := 1;
            nc := 2;
            gemm<TArray1_2>(m0, m, n0, n);        // array[0..1] of array[0..0] of single;
          end;
       $11:
         begin
            mc := 1;
            nc := 1;
            gemm<TArray1_1>(m0, m, n0, n);       // array[0..0] of array[0..0] of single;
         end;
       else
         exit;
     end;
     mp := m0 + (m - m0) div mc * mc;
     np := n0 + (n - n0) div nc * nc;
     mnpack(mp, m, n0, np);
     mnpack(m0, m, np, n);
end;


procedure tinyBLAS.gemm<KR>(const m0, m, n0, n:SizeInt);
var
  Cv : KR;
  Cvv : PPm256;
  ytiles, xtiles, tiles, duty, start, &end, job, ii, jj, i, j, l:SizeInt;
  RN,RM:Integer;
begin
        RN := length(Cv);
        RM := length(Cv[0]);
        Cvv := pointer(@Cv[0]);
        ytiles := (m - m0) div RM;
        xtiles := (n - n0) div RN;
        tiles := xtiles * ytiles;
        duty := (tiles + nth - 1) div nth;
        start := duty * ith;
        &end := start + duty;
        if &end > tiles then
            &end := tiles;
        for job := start to &end-1 do begin
            ii := m0 + job div xtiles * RM;
            jj := n0 + job mod xtiles * RN;
            //Cv := default(Cv);
            l := 0;
            while l < k do begin
                for j := 0 to RN-1 do
                    for i := 0 to RM-1 do
                        Cvv[j][i] := madd(A + lda * (ii + i) + l,
                                        B + ldb * (jj + j) + l,
                                        @Cvv[j][i].a[0]);
                inc(l, KN)
            end;
            for j := 0 to RN-1 do
                for i := 0 to RM-1 do
                    C[ldc * (jj + j) + (ii + i)] := hsum(Cvv[j][i]);
        end
end;

//////////////////////////////////////////////////////////////////////////////////////////
// QUANT ZERO MATRIX MULTIPLICATION

(*
{$if defined(__AVX2__) or defined(__AVX512F__) or defined(__AVX__)}
template <typename TA, typename TB, typename TC>
class tinyBLAS_Q0_AVX {
  public:
    tinyBLAS_Q0_AVX(SizeInt k,
                    const TA *A, SizeInt lda,
                    const TB *B, SizeInt ldb,
                    TC *C, SizeInt ldc,
                    int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(SizeInt m, SizeInt n) {
        mnpack(0, m, 0, n);
    }

  private:
    void mnpack(SizeInt m0, SizeInt m, SizeInt n0, SizeInt n) {
        SizeInt mc, nc, mp, np;
        switch ((MIN(m - m0, 4) << 4) | MIN(n - n0, 4)) {
{$if VECTOR_REGISTERS == 32}
        case 0x44:
            mc = 4;
            nc = 4;
            gemm<4, 4>(m0, m, n0, n);
            break;
        case 0x43:
            mc = 4;
            nc = 3;
            gemm<4, 3>(m0, m, n0, n);
            break;
        case 0x34:
            mc = 3;
            nc = 4;
            gemm<3, 4>(m0, m, n0, n);
            break;
        case 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            break;
        case 0x42:
            mc = 4;
            nc = 2;
            gemm<4, 2>(m0, m, n0, n);
            break;
        case 0x24:
            mc = 2;
            nc = 4;
            gemm<2, 4>(m0, m, n0, n);
            break;
{$else}
        case 0x44:
        case 0x43:
        case 0x42:
            mc = 4;
            nc = 2;
            gemm<4, 2>(m0, m, n0, n);
            break;
        case 0x34:
        case 0x24:
            mc = 2;
            nc = 4;
            gemm<2, 4>(m0, m, n0, n);
            break;
        case 0x33:
{$endif}
        case 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            break;
        case 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            break;
        case 0x41:
            mc = 4;
            nc = 1;
            gemm<4, 1>(m0, m, n0, n);
            break;
        case 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            break;
        case 0x14:
            mc = 1;
            nc = 4;
            gemm<1, 4>(m0, m, n0, n);
            break;
        case 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            break;
        case 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            break;
        case 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            break;
        case 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            break;
        case 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            break;
        default:
            return;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN>
    NOINLINE void gemm(SizeInt m0, SizeInt m, SizeInt n0, SizeInt n) {
        SizeInt ytiles = (m - m0) / RM;
        SizeInt xtiles = (n - n0) / RN;
        SizeInt tiles = xtiles * ytiles;
        SizeInt duty = (tiles + nth - 1) / nth;
        SizeInt start = duty * ith;
        SizeInt end = start + duty;
        if (end > tiles)
            end = tiles;
        for (SizeInt job = start; job < end; ++job) {
            SizeInt ii = m0 + job / xtiles * RM;
            SizeInt jj = n0 + job % xtiles * RN;
            __m256 Cv[RN][RM] = {};
            for (SizeInt l = 0; l < k; ++l)
                for (SizeInt j = 0; j < RN; ++j)
                    for (SizeInt i = 0; i < RM; ++i) {
{$if defined(__AVX2__)}
                        __m256 udTmp = updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                              load(A + lda * (ii + i) + l)),
                                             _mm256_sign_epi8(load(B + ldb * (jj + j) + l),
                                                              load(A + lda * (ii + i) + l)));
{$else}
                        __m128i ali0 = load0(A + lda * (ii + i) + l);
                        __m128i ali1 = load1(A + lda * (ii + i) + l);
                        __m128i blj0 = load0(B + ldb * (jj + j) + l);
                        __m128i blj1 = load1(B + ldb * (jj + j) + l);

                        __m128i sepAA0 = _mm_sign_epi8(ali0, ali0);
                        __m128i sepAA1 = _mm_sign_epi8(ali1, ali1);
                        __m128i sepBA0 = _mm_sign_epi8(blj0, ali0);
                        __m128i sepBA1 = _mm_sign_epi8(blj1, ali1);

                        // updot
                        const __m128i oneFill = _mm_set1_epi16(1);
                        __m128i mad0 = _mm_maddubs_epi16(sepAA0, sepBA0);
                        __m128i mad1 = _mm_maddubs_epi16(sepAA1, sepBA1);
                        __m256 udTmp = _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_madd_epi16(oneFill, mad1), _mm_madd_epi16(oneFill, mad0)));
{$endif}
                        Cv[j][i] = madd(_mm256_set1_ps(unhalf(A[lda * (ii + i) + l].d) *
                                                       unhalf(B[ldb * (jj + j) + l].d)),
                                                       udTmp,
                                                       Cv[j][i]);
                    }
            for (SizeInt j = 0; j < RN; ++j)
                for (SizeInt i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    inline __m256i load(const block_q8_0 *b) {
        return _mm256_loadu_si256((const __m256i * )b->qs);
    }

    inline __m128i load0(const block_q8_0 *b) {
        return _mm_loadu_si128((const __m128i * )b->qs);
    }

    inline __m128i load1(const block_q8_0 *b) {
        return _mm_loadu_si128(((const __m128i * )b->qs) + 1);
    }

    inline __m256i load(const block_q4_0 *b) {
        return _mm256_sub_epi8(denibble(b->qs), _mm256_set1_epi8(8));
    }

    inline __m128i load0(const block_q4_0 *b) {
        const __m128i x = _mm_loadu_si128((const __m128i * )(b->qs));
        return _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), x), _mm_set1_epi8(8));
    }

    inline __m128i load1(const block_q4_0 *b) {
        const __m128i x = _mm_loadu_si128((const __m128i * )(b->qs));
        return _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)), _mm_set1_epi8(8));
    }

    inline __m256 updot(__m256i u, __m256i s) {
        __m256i res;
{$if defined(__AVXVNNI__) or (defined(__AVX512VNNI__) and defined(__AVX512VL__))}
        res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
{$else}
        res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
{$endif}
        return _mm256_cvtepi32_ps(res);
    }

    static inline __m256i denibble(const uint8_t *p) {
        __m128i x = _mm_loadu_si128((const __m128i * )p);
        return _mm256_and_si256(_mm256_set1_epi8(15),
                                _mm256_insertf128_si256(_mm256_castsi128_si256(x),
                                                        _mm_srli_epi16(x, 4), 1));
    }

    const TA *const A;
    const TB *const B;
    TC *const C;
    const SizeInt k;
    const SizeInt lda;
    const SizeInt ldb;
    const SizeInt ldc;
    const int ith;
    const int nth;
};
{$endif // __AVX__}

} // namespace

{ *
 * Performs optimized matrix multiplication on CPU.
 *
 * This subroutine may compute C = Aᵀ * B with column major ordering.
 * Despite its name, this isn't a generalized implementation. Work is
 * only performed when a handwritten kernel is written and available.
 * Otherwise the caller should fall back to a general matmul routine.
 *
 * For example, for single-threaded single-precision GEMM you can say
 *
 *     llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc,
 *                     0, 1,
 *                     GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32);
 *
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param A is first input matrix (always transposed)
 * @param lda is row stride of `A`
 * @param B is second input matrix (never transposed)
 * @param ldb is row stride of `B`
 * @param C is input/output array of output matrices
 * @param ldc is row stride of `C`
 * @param ith is thread id (must be less than `nth`)
 * @param nth is number of threads (must be greater than zero)
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @return true if this function was able to service the matmul request
 * }
*)
function tinysgemm(const m, n, k: SizeInt; const A: PSingle;
  const lda: SizeInt; const B: PSingle; const ldb: SizeInt; const C: PSingle;
  const ldc: SizeInt; const ith, nth: integer): boolean;
var tb : tinyBLAS;
begin
     assert(m >= 0);
     assert(n >= 0);
     assert(k >= 0);
     //assert(lda >= k);
     //assert(ldb >= m);
     //assert(ldc >= n);
     assert(nth > 0);
     assert(ith < nth);

     if boolean(k mod 8) then
            exit(false);
     tb.k:=k;
     tb.A:=A;
     tb.lda:=lda;
     tb.B:=B;
     tb.ldb:=ldb;
     tb.C:=C;
     tb.ldc:=ldc;
     tb.ith:=ith;
     tb.nth:=nth;
     tb.matmul(m, n);
     exit(true);

end;

(*

bool llamafile_sgemm(SizeInt m, SizeInt n, SizeInt k, const void *A, SizeInt lda, const void *B, SizeInt ldb, void *C,
                     SizeInt ldc, int ith, int nth, int Atype, int Btype, int Ctype) {


    if (Ctype != GGML_TYPE_F32)
        return false;

    switch (Atype) {

    case GGML_TYPE_F32: {
        if (Btype != GGML_TYPE_F32)
            return false;
{$if defined(__AVX512F__)}
        if (k % 16)
            return false;
        tinyBLAS<16, __m512, __m512, float, float, float> tb{
            k, (const float * )A, lda,
            (const float * )B, ldb,
            (float * )C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
{$elif defined(__AVX__) or defined(__AVX2__)}
        if (k % 8)
            return false;
        tinyBLAS<8, __m256, __m256, float, float, float> tb{
            k, (const float * )A, lda,
            (const float * )B, ldb,
            (float * )C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
{$elif defined(__ARM_NEON)}
        if (n < 4)
            return false;
        if (k % 4)
            return false;
        tinyBLAS<4, float32x4_t, float32x4_t, float, float, float> tb{
            k, (const float * )A, lda,
            (const float * )B, ldb,
            (float * )C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
{$else}
        return false;
{$endif}
    }

    case GGML_TYPE_F16: {
{$if defined(__AVX512F__)}
        if (k % 16)
            return false;
        if (Btype != GGML_TYPE_F32)
            return false;
        tinyBLAS<16, __m512, __m512, ggml_fp16_t, float, float> tb{
            k, (const ggml_fp16_t * )A, lda,
            (const float * )B, ldb,
            (float * )C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
{$elif (defined(__AVX__) or defined(__AVX2__)) and defined(__F16C__)}
        if (k % 8)
            return false;
        if (Btype != GGML_TYPE_F32)
            return false;
        tinyBLAS<8, __m256, __m256, ggml_fp16_t, float, float> tb{
            k, (const ggml_fp16_t * )A, lda,
            (const float * )B, ldb,
            (float * )C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
{$elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) and not(defined(_MSC_VER))}
        if (n < 8)
            return false;
        if (k % 8)
            return false;
        if (Btype != GGML_TYPE_F16)
            return false;
        tinyBLAS<8, float16x8_t, float16x8_t, ggml_fp16_t, ggml_fp16_t, float> tb{
            k, (const ggml_fp16_t * )A, lda,
            (const ggml_fp16_t * )B, ldb,
            (float * )C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
{$elif defined(__ARM_NEON) and not(defined(_MSC_VER))}
        if (k % 4)
            return false;
        if (Btype != GGML_TYPE_F32)
            return false;
        tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, float, float> tb{
            k, (const ggml_fp16_t * )A, lda,
            (const float * )B, ldb,
            (float * )C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
{$else}
        return false;
{$endif}
    }

    case GGML_TYPE_Q8_0: {
        if (Btype != GGML_TYPE_Q8_0)
           return false;
{$if defined(__AVX2__) or defined(__AVX512F__) or defined(__AVX__)}
        tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float> tb{
            k, (const block_q8_0 * )A, lda,
            (const block_q8_0 * )B, ldb,
            (float * )C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
{$elif defined(__ARM_FEATURE_DOTPROD)}
        tinyBLAS_Q0_ARM<block_q8_0> tb{
            k, (const block_q8_0 * )A, lda,
            (const block_q8_0 * )B, ldb,
            (float * )C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
{$else}
        return false;
{$endif}
    }

    case GGML_TYPE_Q4_0: {
        if (Btype != GGML_TYPE_Q8_0)
            return false;
{$if defined(__AVX2__) or defined(__AVX512F__) or defined(__AVX__)}
        tinyBLAS_Q0_AVX<block_q4_0, block_q8_0, float> tb{
            k, (const block_q4_0 * )A, lda,
            (const block_q8_0 * )B, ldb,
            (float * )C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
{$elif defined(__ARM_FEATURE_DOTPROD)}
        tinyBLAS_Q0_ARM<block_q4_0> tb{
            k, (const block_q4_0 * )A, lda,
            (const block_q8_0 * )B, ldb,
            (float * )C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
{$else}
        return false;
{$endif}
    }

    default:
        return false;
    }

    (void)m;
    (void)n;
    (void)k;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)C;
    (void)ldc;
    (void)ith;
    (void)nth;
    (void)Atype;
    (void)Btype;
    (void)Ctype;
}

*)

end.
