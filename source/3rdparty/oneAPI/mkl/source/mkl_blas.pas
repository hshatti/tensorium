unit mkl_blas;
interface

uses types, mkl_types;

{$include mkl.inc}

  {******************************************************************************
  * Copyright 1999-2021 Intel Corporation.
  *
  * This software and the related documents are Intel copyrighted  materials,  and
  * your use of  them is  governed by the  express license  under which  they were
  * provided to you (License).  Unless the License provides otherwise, you may not
  * use, modify, copy, publish, distribute,  disclose or transmit this software or
  * the related documents without Intel's prior written permission.
  *
  * This software and the related documents  are provided as  is,  with no express
  * or implied  warranties,  other  than those  that are  expressly stated  in the
  * License.
  ****************************************************************************** }
  {
  !  Content:
  !      Intel(R) oneAPI Math Kernel Library (oneMKL) interface for BLAS routines
  !***************************************************************************** }
{.$include <stddef.h>}
{.$include "mkl_types.h"}

{ C++ extern C conditionnal removed }
  { __cplusplus  }
  { Upper case declaration  }
{$ifdef UPPERCASE_DECL}
  procedure XERBLA(const srname:Pchar; const info:Plongint; const lsrname:longint);winapi; external {$ifdef libmkl} libmkl{$endif};





  function LSAME(const ca:Pchar; const cb:Pchar; const lca:MKL_INT; const lcb:MKL_INT):longint;winapi; external {$ifdef libmkl} libmkl{$endif};

  { BLAS Level1  }

  function SCABS1(const c:PMKL_Complex8):single;winapi; external {$ifdef libmkl} libmkl{$endif};




  function SASUM(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT):single;winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure SAXPY(const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure SAXPBY(const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const beta:Psingle; const 
              y:Psingle; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure SAXPYI(const nz:PMKL_INT; const a:Psingle; const x:Psingle; const indx:PMKL_INT; const y:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};




  function SCASUM(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT):single;winapi; external {$ifdef libmkl} libmkl{$endif};




  function SCNRM2(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT):single;winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure SCOPY(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};






  function SDOT(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const incy:PMKL_INT):single;winapi; external {$ifdef libmkl} libmkl{$endif};







  function SDSDOT(const n:PMKL_INT; const sb:Psingle; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const 
             incy:PMKL_INT):single;winapi; external {$ifdef libmkl} libmkl{$endif};





  function SDOTI(const nz:PMKL_INT; const x:Psingle; const indx:PMKL_INT; const y:Psingle):single;winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure SGTHR(const nz:PMKL_INT; const y:Psingle; const x:Psingle; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};



  procedure SGTHRZ(const nz:PMKL_INT; const y:Psingle; const x:Psingle; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  function SNRM2(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT):single;winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure SROT(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const incy:PMKL_INT; const 
              c:Psingle; const s:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure SROTG(const a:Psingle; const b:Psingle; const c:Psingle; const s:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure SROTI(const nz:PMKL_INT; const x:Psingle; const indx:PMKL_INT; const y:Psingle; const c:Psingle; const 
              s:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure SROTM(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const incy:PMKL_INT; const 
              param:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};


  procedure SROTMG(const d1:Psingle; const d2:Psingle; const x1:Psingle; const y1:Psingle; const param:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure SSCAL(const n:PMKL_INT; const a:Psingle; const x:Psingle; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure SSCTR(const nz:PMKL_INT; const x:Psingle; const indx:PMKL_INT; const y:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure SSWAP(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  function ISAMAX(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};




  function ISAMIN(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure CAXPY(const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT; const y:PMKL_Complex8; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure CAXPBY(const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT; const beta:PMKL_Complex8; const 
              y:PMKL_Complex8; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure CAXPYI(const nz:PMKL_INT; const a:PMKL_Complex8; const x:PMKL_Complex8; const indx:PMKL_INT; const y:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure CCOPY(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const y:PMKL_Complex8; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure CDOTC(const pres:PMKL_Complex8; const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const y:PMKL_Complex8; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure CDOTCI(const pres:PMKL_Complex8; const nz:PMKL_INT; const x:PMKL_Complex8; const indx:PMKL_INT; const y:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure CDOTU(const pres:PMKL_Complex8; const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const y:PMKL_Complex8; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure CDOTUI(const pres:PMKL_Complex8; const nz:PMKL_INT; const x:PMKL_Complex8; const indx:PMKL_INT; const y:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure CGTHR(const nz:PMKL_INT; const y:PMKL_Complex8; const x:PMKL_Complex8; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};



  procedure CGTHRZ(const nz:PMKL_INT; const y:PMKL_Complex8; const x:PMKL_Complex8; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};


  procedure CROTG(const a:PMKL_Complex8; const b:PMKL_Complex8; const c:Psingle; const s:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure CSCAL(const n:PMKL_INT; const a:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure CSCTR(const nz:PMKL_INT; const x:PMKL_Complex8; const indx:PMKL_INT; const y:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure CSROT(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const y:PMKL_Complex8; const incy:PMKL_INT; const 
              c:Psingle; const s:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure CSSCAL(const n:PMKL_INT; const a:Psingle; const x:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure CSWAP(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const y:PMKL_Complex8; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  function ICAMAX(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};




  function ICAMIN(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};


  function DCABS1(const z:PMKL_Complex16):double;winapi; external {$ifdef libmkl} libmkl{$endif};




  function DASUM(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT):double;winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure DAXPY(const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const y:Pdouble; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure DAXPBY(const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const beta:Pdouble; const 
              y:Pdouble; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure DAXPYI(const nz:PMKL_INT; const a:Pdouble; const x:Pdouble; const indx:PMKL_INT; const y:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure DCOPY(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const y:Pdouble; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};






  function DDOT(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const y:Pdouble; const incy:PMKL_INT):double;winapi; external {$ifdef libmkl} libmkl{$endif};






  function DSDOT(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const incy:PMKL_INT):double;winapi; external {$ifdef libmkl} libmkl{$endif};





  function DDOTI(const nz:PMKL_INT; const x:Pdouble; const indx:PMKL_INT; const y:Pdouble):double;winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure DGTHR(const nz:PMKL_INT; const y:Pdouble; const x:Pdouble; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};



  procedure DGTHRZ(const nz:PMKL_INT; const y:Pdouble; const x:Pdouble; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  function DNRM2(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT):double;winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure DROT(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const y:Pdouble; const incy:PMKL_INT; const 
              c:Pdouble; const s:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure DROTG(const a:Pdouble; const b:Pdouble; const c:Pdouble; const s:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure DROTI(const nz:PMKL_INT; const x:Pdouble; const indx:PMKL_INT; const y:Pdouble; const c:Pdouble; const 
              s:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure DROTM(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const y:Pdouble; const incy:PMKL_INT; const 
              param:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};


  procedure DROTMG(const d1:Pdouble; const d2:Pdouble; const x1:Pdouble; const y1:Pdouble; const param:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure DSCAL(const n:PMKL_INT; const a:Pdouble; const x:Pdouble; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure DSCTR(const nz:PMKL_INT; const x:Pdouble; const indx:PMKL_INT; const y:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure DSWAP(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const y:Pdouble; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  function DZASUM(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT):double;winapi; external {$ifdef libmkl} libmkl{$endif};




  function DZNRM2(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT):double;winapi; external {$ifdef libmkl} libmkl{$endif};




  function IDAMAX(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};




  function IDAMIN(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure ZAXPY(const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT; const y:PMKL_Complex16; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure ZAXPBY(const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT; const beta:PMKL_Complex16; const 
              y:PMKL_Complex16; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure ZAXPYI(const nz:PMKL_INT; const a:PMKL_Complex16; const x:PMKL_Complex16; const indx:PMKL_INT; const y:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure ZCOPY(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const y:PMKL_Complex16; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure ZDOTC(const pres:PMKL_Complex16; const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const y:PMKL_Complex16; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure ZDOTCI(const pres:PMKL_Complex16; const nz:PMKL_INT; const x:PMKL_Complex16; const indx:PMKL_INT; const y:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure ZDOTU(const pres:PMKL_Complex16; const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const y:PMKL_Complex16; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure ZDOTUI(const pres:PMKL_Complex16; const nz:PMKL_INT; const x:PMKL_Complex16; const indx:PMKL_INT; const y:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure ZDROT(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const y:PMKL_Complex16; const incy:PMKL_INT; const 
              c:Pdouble; const s:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure ZDSCAL(const n:PMKL_INT; const a:Pdouble; const x:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure ZGTHR(const nz:PMKL_INT; const y:PMKL_Complex16; const x:PMKL_Complex16; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};



  procedure ZGTHRZ(const nz:PMKL_INT; const y:PMKL_Complex16; const x:PMKL_Complex16; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};


  procedure ZROTG(const a:PMKL_Complex16; const b:PMKL_Complex16; const c:Pdouble; const s:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure ZSCAL(const n:PMKL_INT; const a:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure ZSCTR(const nz:PMKL_INT; const x:PMKL_Complex16; const indx:PMKL_INT; const y:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure ZSWAP(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const y:PMKL_Complex16; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  function IZAMAX(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};




  function IZAMIN(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};

  { BLAS Level2  }












  procedure SGBMV(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const kl:PMKL_INT; const ku:PMKL_INT; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const 
              beta:Psingle; const y:Psingle; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure SGEMV(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Psingle; const a:Psingle; const 
              lda:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const beta:Psingle; const y:Psingle; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure SGER(const m:PMKL_INT; const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const 
              y:Psingle; const incy:PMKL_INT; const a:Psingle; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure SSBMV(const uplo:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Psingle; const a:Psingle; const 
              lda:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const beta:Psingle; const y:Psingle; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure SSPMV(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const ap:Psingle; const x:Psingle; const 
              incx:PMKL_INT; const beta:Psingle; const y:Psingle; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure SSPR(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const 
              ap:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};








  procedure SSPR2(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const 
              y:Psingle; const incy:PMKL_INT; const ap:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};










  procedure SSYMV(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const 
              x:Psingle; const incx:PMKL_INT; const beta:Psingle; const y:Psingle; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure SSYR(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const 
              a:Psingle; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure SSYR2(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const 
              y:Psingle; const incy:PMKL_INT; const a:Psingle; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure STBMV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:Psingle; const lda:PMKL_INT; const x:Psingle; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure STBSV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:Psingle; const lda:PMKL_INT; const x:Psingle; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure STPMV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:Psingle; const 
              x:Psingle; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure STPSV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:Psingle; const 
              x:Psingle; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};








  procedure STRMV(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT; const a:Psingle; const 
              lda:PMKL_INT; const b:Psingle; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};








  procedure STRSV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const a:Psingle; const 
              lda:PMKL_INT; const x:Psingle; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure SGEM2VU(const m:PMKL_INT; const n:PMKL_INT; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const 
              x1:Psingle; const incx1:PMKL_INT; const x2:Psingle; const incx2:PMKL_INT; const beta:Psingle; const 
              y1:Psingle; const incy1:PMKL_INT; const y2:Psingle; const incy2:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure CGBMV(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const kl:PMKL_INT; const ku:PMKL_INT; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const 
              beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure CGEMV(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const 
              lda:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const beta:PMKL_Complex8; const y:PMKL_Complex8; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure CGERC(const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT; const 
              y:PMKL_Complex8; const incy:PMKL_INT; const a:PMKL_Complex8; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure CGERU(const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT; const 
              y:PMKL_Complex8; const incy:PMKL_INT; const a:PMKL_Complex8; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure CHBMV(const uplo:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const 
              lda:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const beta:PMKL_Complex8; const y:PMKL_Complex8; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};










  procedure CHEMV(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const 
              x:PMKL_Complex8; const incx:PMKL_INT; const beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure CHER(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const x:PMKL_Complex8; const incx:PMKL_INT; const 
              a:PMKL_Complex8; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure CHER2(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT; const 
              y:PMKL_Complex8; const incy:PMKL_INT; const a:PMKL_Complex8; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure CHPMV(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex8; const ap:PMKL_Complex8; const x:PMKL_Complex8; const 
              incx:PMKL_INT; const beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure CHPR(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const x:PMKL_Complex8; const incx:PMKL_INT; const 
              ap:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};








  procedure CHPR2(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT; const 
              y:PMKL_Complex8; const incy:PMKL_INT; const ap:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure CTBMV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure CTBSV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure CTPMV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:PMKL_Complex8; const 
              x:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure CTPSV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:PMKL_Complex8; const 
              x:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};








  procedure CTRMV(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT; const a:PMKL_Complex8; const 
              lda:PMKL_INT; const b:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};








  procedure CTRSV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const a:PMKL_Complex8; const 
              lda:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure CGEM2VC(const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const 
              x1:PMKL_Complex8; const incx1:PMKL_INT; const x2:PMKL_Complex8; const incx2:PMKL_INT; const beta:PMKL_Complex8; const 
              y1:PMKL_Complex8; const incy1:PMKL_INT; const y2:PMKL_Complex8; const incy2:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure SCGEMV(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const a:Psingle; const 
              lda:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const beta:PMKL_Complex8; const y:PMKL_Complex8; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure DGBMV(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const kl:PMKL_INT; const ku:PMKL_INT; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const 
              beta:Pdouble; const y:Pdouble; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure DGEMV(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const 
              lda:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const beta:Pdouble; const y:Pdouble; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure DGER(const m:PMKL_INT; const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const 
              y:Pdouble; const incy:PMKL_INT; const a:Pdouble; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure DSBMV(const uplo:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const 
              lda:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const beta:Pdouble; const y:Pdouble; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure DSPMV(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const ap:Pdouble; const x:Pdouble; const 
              incx:PMKL_INT; const beta:Pdouble; const y:Pdouble; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure DSPR(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const 
              ap:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};








  procedure DSPR2(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const 
              y:Pdouble; const incy:PMKL_INT; const ap:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};










  procedure DSYMV(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const 
              x:Pdouble; const incx:PMKL_INT; const beta:Pdouble; const y:Pdouble; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure DSYR(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const 
              a:Pdouble; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure DSYR2(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const 
              y:Pdouble; const incy:PMKL_INT; const a:Pdouble; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure DTBMV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:Pdouble; const lda:PMKL_INT; const x:Pdouble; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure DTBSV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:Pdouble; const lda:PMKL_INT; const x:Pdouble; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure DTPMV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:Pdouble; const 
              x:Pdouble; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure DTPSV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:Pdouble; const 
              x:Pdouble; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};








  procedure DTRMV(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT; const a:Pdouble; const 
              lda:PMKL_INT; const b:Pdouble; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};








  procedure DTRSV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const a:Pdouble; const 
              lda:PMKL_INT; const x:Pdouble; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure DGEM2VU(const m:PMKL_INT; const n:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const 
              x1:Pdouble; const incx1:PMKL_INT; const x2:Pdouble; const incx2:PMKL_INT; const beta:Pdouble; const 
              y1:Pdouble; const incy1:PMKL_INT; const y2:Pdouble; const incy2:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure ZGBMV(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const kl:PMKL_INT; const ku:PMKL_INT; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const 
              beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure ZGEMV(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const 
              lda:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const beta:PMKL_Complex16; const y:PMKL_Complex16; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure ZGERC(const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT; const 
              y:PMKL_Complex16; const incy:PMKL_INT; const a:PMKL_Complex16; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure ZGERU(const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT; const 
              y:PMKL_Complex16; const incy:PMKL_INT; const a:PMKL_Complex16; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure ZHBMV(const uplo:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const 
              lda:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const beta:PMKL_Complex16; const y:PMKL_Complex16; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};










  procedure ZHEMV(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const 
              x:PMKL_Complex16; const incx:PMKL_INT; const beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure ZHER(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const x:PMKL_Complex16; const incx:PMKL_INT; const 
              a:PMKL_Complex16; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure ZHER2(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT; const 
              y:PMKL_Complex16; const incy:PMKL_INT; const a:PMKL_Complex16; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure ZHPMV(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex16; const ap:PMKL_Complex16; const x:PMKL_Complex16; const 
              incx:PMKL_INT; const beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure ZHPR(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const x:PMKL_Complex16; const incx:PMKL_INT; const 
              ap:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};








  procedure ZHPR2(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT; const 
              y:PMKL_Complex16; const incy:PMKL_INT; const ap:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure ZTBMV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure ZTBSV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure ZTPMV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:PMKL_Complex16; const 
              x:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure ZTPSV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:PMKL_Complex16; const 
              x:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};








  procedure ZTRMV(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT; const a:PMKL_Complex16; const 
              lda:PMKL_INT; const b:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};








  procedure ZTRSV(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const a:PMKL_Complex16; const 
              lda:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure ZGEM2VC(const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const 
              x1:PMKL_Complex16; const incx1:PMKL_INT; const x2:PMKL_Complex16; const incx2:PMKL_INT; const beta:PMKL_Complex16; const 
              y1:PMKL_Complex16; const incy1:PMKL_INT; const y2:PMKL_Complex16; const incy2:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure DZGEMV(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const a:Pdouble; const 
              lda:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const beta:PMKL_Complex16; const y:PMKL_Complex16; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  { BLAS Level3  }












  procedure SGEMM(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const b:Psingle; const ldb:PMKL_INT; const 
              beta:Psingle; const c:Psingle; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  function SGEMM_PACK_GET_SIZE(const identifier:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT):size_t;winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure SGEMM_PACK(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Psingle; const src:Psingle; const ld:PMKL_INT; const dest:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};












  procedure SGEMM_COMPUTE(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              a:Psingle; const lda:PMKL_INT; const b:Psingle; const ldb:PMKL_INT; const beta:Psingle; const 
              c:Psingle; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};















  procedure SGEMM_BATCH(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT; const n_array:PMKL_INT; const k_array:PMKL_INT; const 
              alpha_array:Psingle; const a_array:PPsingle; const lda_array:PMKL_INT; const b_array:PPsingle; const ldb_array:PMKL_INT; const 
              beta_array:Psingle; const c_array:PPsingle; const ldc_array:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

















  procedure SGEMM_BATCH_STRIDED(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const stridea:PMKL_INT; const b:Psingle; const 
              ldb:PMKL_INT; const strideb:PMKL_INT; const beta:Psingle; const c:Psingle; const ldc:PMKL_INT; const 
              stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure SGEMMT(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const b:Psingle; const ldb:PMKL_INT; const 
              beta:Psingle; const c:Psingle; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};












  procedure SSYMM(const side:Pchar; const uplo:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Psingle; const 
              a:Psingle; const lda:PMKL_INT; const b:Psingle; const ldb:PMKL_INT; const beta:Psingle; const 
              c:Psingle; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};












  procedure SSYR2K(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Psingle; const 
              a:Psingle; const lda:PMKL_INT; const b:Psingle; const ldb:PMKL_INT; const beta:Psingle; const 
              c:Psingle; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};










  procedure SSYRK(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Psingle; const 
              a:Psingle; const lda:PMKL_INT; const beta:Psingle; const c:Psingle; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure STRMM(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const b:Psingle; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure STRSM(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const b:Psingle; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure STRSM_BATCH(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT; const 
              n_array:PMKL_INT; const alpha_array:Psingle; const a_array:PPsingle; const lda_array:PMKL_INT; const b_array:PPsingle; const 
              ldb:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};














  procedure STRSM_BATCH_STRIDED(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const stridea:PMKL_INT; const 
              b:Psingle; const ldb:PMKL_INT; const strideb:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure CGEMM(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};















  procedure CGEMM_BATCH(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT; const n_array:PMKL_INT; const k_array:PMKL_INT; const 
              alpha_array:PMKL_Complex8; const a_array:PPMKL_Complex8; const lda_array:PMKL_INT; const b_array:PPMKL_Complex8; const ldb_array:PMKL_INT; const 
              beta_array:PMKL_Complex8; const c_array:PPMKL_Complex8; const ldc_array:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

















  procedure CGEMM_BATCH_STRIDED(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const stridea:PMKL_INT; const b:PMKL_Complex8; const 
              ldb:PMKL_INT; const strideb:PMKL_INT; const beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT; const 
              stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure SCGEMM(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex8; const a:Psingle; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure CGEMM3M(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};















  procedure CGEMM3M_BATCH(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT; const n_array:PMKL_INT; const k_array:PMKL_INT; const 
              alpha_array:PMKL_Complex8; const a_array:PPMKL_Complex8; const lda_array:PMKL_INT; const b_array:PPMKL_Complex8; const ldb_array:PMKL_INT; const 
              beta_array:PMKL_Complex8; const c_array:PPMKL_Complex8; const ldc_array:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure CGEMMT(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure CTRSM_BATCH(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT; const 
              n_array:PMKL_INT; const alpha_array:PMKL_Complex8; const a_array:PPMKL_Complex8; const lda_array:PMKL_INT; const b_array:PPMKL_Complex8; const 
              ldb:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};














  procedure CTRSM_BATCH_STRIDED(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const stridea:PMKL_INT; const 
              b:PMKL_Complex8; const ldb:PMKL_INT; const strideb:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};












  procedure CHEMM(const side:Pchar; const uplo:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const beta:PMKL_Complex8; const 
              c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};












  procedure CHER2K(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const beta:Psingle; const 
              c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};










  procedure CHERK(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Psingle; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const beta:Psingle; const c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};












  procedure CSYMM(const side:Pchar; const uplo:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const beta:PMKL_Complex8; const 
              c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};












  procedure CSYR2K(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const beta:PMKL_Complex8; const 
              c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};










  procedure CSYRK(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure CTRMM(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure CTRSM(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure DGEMM(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const b:Pdouble; const ldb:PMKL_INT; const 
              beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  function DGEMM_PACK_GET_SIZE(const identifier:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT):size_t;winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure DGEMM_PACK(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Pdouble; const src:Pdouble; const ld:PMKL_INT; const dest:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};












  procedure DGEMM_COMPUTE(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              a:Pdouble; const lda:PMKL_INT; const b:Pdouble; const ldb:PMKL_INT; const beta:Pdouble; const 
              c:Pdouble; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};















  procedure DGEMM_BATCH(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT; const n_array:PMKL_INT; const k_array:PMKL_INT; const 
              alpha_array:Pdouble; const a_array:PPdouble; const lda_array:PMKL_INT; const b_array:PPdouble; const ldb_array:PMKL_INT; const 
              beta_array:Pdouble; const c_array:PPdouble; const ldc_array:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

















  procedure DGEMM_BATCH_STRIDED(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const stridea:PMKL_INT; const b:Pdouble; const 
              ldb:PMKL_INT; const strideb:PMKL_INT; const beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT; const 
              stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure DGEMMT(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const b:Pdouble; const ldb:PMKL_INT; const 
              beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};












  procedure DSYMM(const side:Pchar; const uplo:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Pdouble; const 
              a:Pdouble; const lda:PMKL_INT; const b:Pdouble; const ldb:PMKL_INT; const beta:Pdouble; const 
              c:Pdouble; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};












  procedure DSYR2K(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Pdouble; const 
              a:Pdouble; const lda:PMKL_INT; const b:Pdouble; const ldb:PMKL_INT; const beta:Pdouble; const 
              c:Pdouble; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};










  procedure DSYRK(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Pdouble; const 
              a:Pdouble; const lda:PMKL_INT; const beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure DTRMM(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const b:Pdouble; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure DTRSM(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const b:Pdouble; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure DTRSM_BATCH(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT; const 
              n_array:PMKL_INT; const alpha_array:Pdouble; const a_array:PPdouble; const lda_array:PMKL_INT; const b_array:PPdouble; const 
              ldb:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};














  procedure DTRSM_BATCH_STRIDED(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const stridea:PMKL_INT; const 
              b:Pdouble; const ldb:PMKL_INT; const strideb:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure ZGEMM(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};















  procedure ZGEMM_BATCH(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT; const n_array:PMKL_INT; const k_array:PMKL_INT; const 
              alpha_array:PMKL_Complex16; const a_array:PPMKL_Complex16; const lda_array:PMKL_INT; const b_array:PPMKL_Complex16; const ldb_array:PMKL_INT; const 
              beta_array:PMKL_Complex16; const c_array:PPMKL_Complex16; const ldc_array:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

















  procedure ZGEMM_BATCH_STRIDED(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const stridea:PMKL_INT; const b:PMKL_Complex16; const 
              ldb:PMKL_INT; const strideb:PMKL_INT; const beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT; const 
              stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure DZGEMM(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex16; const a:Pdouble; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure ZGEMM3M(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};















  procedure ZGEMM3M_BATCH(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT; const n_array:PMKL_INT; const k_array:PMKL_INT; const 
              alpha_array:PMKL_Complex16; const a_array:PPMKL_Complex16; const lda_array:PMKL_INT; const b_array:PPMKL_Complex16; const ldb_array:PMKL_INT; const 
              beta_array:PMKL_Complex16; const c_array:PPMKL_Complex16; const ldc_array:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure ZGEMMT(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};












  procedure ZHEMM(const side:Pchar; const uplo:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const beta:PMKL_Complex16; const 
              c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};












  procedure ZHER2K(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const beta:Pdouble; const 
              c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};










  procedure ZHERK(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Pdouble; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const beta:Pdouble; const c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};












  procedure ZSYMM(const side:Pchar; const uplo:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const beta:PMKL_Complex16; const 
              c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};












  procedure ZSYR2K(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const beta:PMKL_Complex16; const 
              c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};










  procedure ZSYRK(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure ZTRMM(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure ZTRSM(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure ZTRSM_BATCH(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT; const 
              n_array:PMKL_INT; const alpha_array:PMKL_Complex16; const a_array:PPMKL_Complex16; const lda_array:PMKL_INT; const b_array:PPMKL_Complex16; const 
              ldb:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};














  procedure ZTRSM_BATCH_STRIDED(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const stridea:PMKL_INT; const 
              b:PMKL_Complex16; const ldb:PMKL_INT; const strideb:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

















  procedure GEMM_S8U8S32(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT; const n:PMKL_INT; const 
              k:PMKL_INT; const alpha:Psingle; const a:PMKL_INT8; const lda:PMKL_INT; const ao:PMKL_INT8; const 
              b:PMKL_UINT8; const ldb:PMKL_INT; const bo:PMKL_INT8; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT; const co:PMKL_INT32);winapi; external {$ifdef libmkl} libmkl{$endif};

















  procedure GEMM_S16S16S32(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT; const n:PMKL_INT; const 
              k:PMKL_INT; const alpha:Psingle; const a:PMKL_INT16; const lda:PMKL_INT; const ao:PMKL_INT16; const 
              b:PMKL_INT16; const ldb:PMKL_INT; const bo:PMKL_INT16; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT; const co:PMKL_INT32);winapi; external {$ifdef libmkl} libmkl{$endif};





  function GEMM_S8U8S32_PACK_GET_SIZE(const identifier:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT):size_t;winapi; external {$ifdef libmkl} libmkl{$endif};





  function GEMM_S16S16S32_PACK_GET_SIZE(const identifier:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT):size_t;winapi; external {$ifdef libmkl} libmkl{$endif};








  procedure GEMM_S8U8S32_PACK(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              src:pointer; const ld:PMKL_INT; const dest:pointer);winapi; external {$ifdef libmkl} libmkl{$endif};








  procedure GEMM_S16S16S32_PACK(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              src:PMKL_INT16; const ld:PMKL_INT; const dest:PMKL_INT16);winapi; external {$ifdef libmkl} libmkl{$endif};

















  procedure GEMM_S8U8S32_COMPUTE(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT; const n:PMKL_INT; const 
              k:PMKL_INT; const alpha:single; const a:PMKL_INT8; const lda:PMKL_INT; const ao:PMKL_INT8; const 
              b:PMKL_UINT8; const ldb:PMKL_INT; const bo:PMKL_INT8; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT; const co:PMKL_INT32);winapi; external {$ifdef libmkl} libmkl{$endif};

















  procedure GEMM_S16S16S32_COMPUTE(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT; const n:PMKL_INT; const 
              k:PMKL_INT; const alpha:single; const a:PMKL_INT16; const lda:PMKL_INT; const ao:PMKL_INT16; const 
              b:PMKL_INT16; const ldb:PMKL_INT; const bo:PMKL_INT16; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT; const co:PMKL_INT32);winapi; external {$ifdef libmkl} libmkl{$endif};













  procedure HGEMM(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_F16; const a:PMKL_F16; const lda:PMKL_INT; const b:PMKL_F16; const ldb:PMKL_INT; const 
              beta:PMKL_F16; const c:PMKL_F16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  function HGEMM_PACK_GET_SIZE(const identifier:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT):size_t;winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure HGEMM_PACK(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_F16; const src:PMKL_F16; const ld:PMKL_INT; const dest:PMKL_F16);winapi; external {$ifdef libmkl} libmkl{$endif};












  procedure HGEMM_COMPUTE(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              a:PMKL_F16; const lda:PMKL_INT; const b:PMKL_F16; const ldb:PMKL_INT; const beta:PMKL_F16; const 
              c:PMKL_F16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

{$endif}
  { Lower case declaration  }

{$ifdef LOWERCASE_DECL}

  procedure xerbla(const srname:Pchar; const info:Plongint; const lsrname:longint);winapi; external {$ifdef libmkl} libmkl{$endif};





  function lsame(const ca:Pchar; const cb:Pchar; const lca:MKL_INT; const lcb:MKL_INT):longint;winapi; external {$ifdef libmkl} libmkl{$endif};

  { BLAS Level1  }

  function scabs1(const c:PMKL_Complex8):single;winapi; external {$ifdef libmkl} libmkl{$endif};




  function sasum(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT):single;winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure saxpy(const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure saxpby(const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const beta:Psingle; const 
              y:Psingle; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure saxpyi(const nz:PMKL_INT; const a:Psingle; const x:Psingle; const indx:PMKL_INT; const y:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};




  function scasum(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT):single;winapi; external {$ifdef libmkl} libmkl{$endif};




  function scnrm2(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT):single;winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure scopy(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};






  function sdot(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const incy:PMKL_INT):single;winapi; external {$ifdef libmkl} libmkl{$endif};





  function sdoti(const nz:PMKL_INT; const x:Psingle; const indx:PMKL_INT; const y:Psingle):single;winapi; external {$ifdef libmkl} libmkl{$endif};







  function sdsdot(const n:PMKL_INT; const sb:Psingle; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const 
             incy:PMKL_INT):single;winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure sgthr(const nz:PMKL_INT; const y:Psingle; const x:Psingle; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};



  procedure sgthrz(const nz:PMKL_INT; const y:Psingle; const x:Psingle; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  function snrm2(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT):single;winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure srot(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const incy:PMKL_INT; const 
              c:Psingle; const s:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure srotg(const a:Psingle; const b:Psingle; const c:Psingle; const s:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure sroti(const nz:PMKL_INT; const x:Psingle; const indx:PMKL_INT; const y:Psingle; const c:Psingle; const 
              s:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure srotm(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const incy:PMKL_INT; const 
              param:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};


  procedure srotmg(const d1:Psingle; const d2:Psingle; const x1:Psingle; const y1:Psingle; const param:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure sscal(const n:PMKL_INT; const a:Psingle; const x:Psingle; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure ssctr(const nz:PMKL_INT; const x:Psingle; const indx:PMKL_INT; const y:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure sswap(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  function isamax(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};




  function isamin(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure caxpy(const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT; const y:PMKL_Complex8; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure caxpby(const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT; const beta:PMKL_Complex8; const 
              y:PMKL_Complex8; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure caxpyi(const nz:PMKL_INT; const a:PMKL_Complex8; const x:PMKL_Complex8; const indx:PMKL_INT; const y:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure ccopy(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const y:PMKL_Complex8; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure cdotc(const pres:PMKL_Complex8; const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const y:PMKL_Complex8; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure cdotci(const pres:PMKL_Complex8; const nz:PMKL_INT; const x:PMKL_Complex8; const indx:PMKL_INT; const y:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure cdotu(const pres:PMKL_Complex8; const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const y:PMKL_Complex8; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure cdotui(const pres:PMKL_Complex8; const nz:PMKL_INT; const x:PMKL_Complex8; const indx:PMKL_INT; const y:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure cgthr(const nz:PMKL_INT; const y:PMKL_Complex8; const x:PMKL_Complex8; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};



  procedure cgthrz(const nz:PMKL_INT; const y:PMKL_Complex8; const x:PMKL_Complex8; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};


  procedure crotg(const a:PMKL_Complex8; const b:PMKL_Complex8; const c:Psingle; const s:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure cscal(const n:PMKL_INT; const a:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure csctr(const nz:PMKL_INT; const x:PMKL_Complex8; const indx:PMKL_INT; const y:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure csrot(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const y:PMKL_Complex8; const incy:PMKL_INT; const 
              c:Psingle; const s:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure csscal(const n:PMKL_INT; const a:Psingle; const x:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure cswap(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const y:PMKL_Complex8; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  function icamax(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};




  function icamin(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};


  function dcabs1(const z:PMKL_Complex16):double;winapi; external {$ifdef libmkl} libmkl{$endif};




  function dasum(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT):double;winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure daxpy(const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const y:Pdouble; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure daxpby(const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const beta:Pdouble; const 
              y:Pdouble; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure daxpyi(const nz:PMKL_INT; const a:Pdouble; const x:Pdouble; const indx:PMKL_INT; const y:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure dcopy(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const y:Pdouble; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};






  function ddot(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const y:Pdouble; const incy:PMKL_INT):double;winapi; external {$ifdef libmkl} libmkl{$endif};






  function dsdot(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const y:Psingle; const incy:PMKL_INT):double;winapi; external {$ifdef libmkl} libmkl{$endif};





  function ddoti(const nz:PMKL_INT; const x:Pdouble; const indx:PMKL_INT; const y:Pdouble):double;winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure dgthr(const nz:PMKL_INT; const y:Pdouble; const x:Pdouble; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};



  procedure dgthrz(const nz:PMKL_INT; const y:Pdouble; const x:Pdouble; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  function dnrm2(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT):double;winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure drot(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const y:Pdouble; const incy:PMKL_INT; const 
              c:Pdouble; const s:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure drotg(const a:Pdouble; const b:Pdouble; const c:Pdouble; const s:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure droti(const nz:PMKL_INT; const x:Pdouble; const indx:PMKL_INT; const y:Pdouble; const c:Pdouble; const 
              s:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure drotm(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const y:Pdouble; const incy:PMKL_INT; const 
              param:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};


  procedure drotmg(const d1:Pdouble; const d2:Pdouble; const x1:Pdouble; const y1:Pdouble; const param:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure dscal(const n:PMKL_INT; const a:Pdouble; const x:Pdouble; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure dsctr(const nz:PMKL_INT; const x:Pdouble; const indx:PMKL_INT; const y:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure dswap(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const y:Pdouble; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  function dzasum(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT):double;winapi; external {$ifdef libmkl} libmkl{$endif};




  function dznrm2(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT):double;winapi; external {$ifdef libmkl} libmkl{$endif};




  function idamax(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};




  function idamin(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure zaxpy(const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT; const y:PMKL_Complex16; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};







  procedure zaxpby(const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT; const beta:PMKL_Complex16; const 
              y:PMKL_Complex16; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure zaxpyi(const nz:PMKL_INT; const a:PMKL_Complex16; const x:PMKL_Complex16; const indx:PMKL_INT; const y:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure zcopy(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const y:PMKL_Complex16; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure zdotc(const pres:PMKL_Complex16; const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const y:PMKL_Complex16; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure zdotci(const pres:PMKL_Complex16; const nz:PMKL_INT; const x:PMKL_Complex16; const indx:PMKL_INT; const y:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure zdotu(const pres:PMKL_Complex16; const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const y:PMKL_Complex16; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};





  procedure zdotui(const pres:PMKL_Complex16; const nz:PMKL_INT; const x:PMKL_Complex16; const indx:PMKL_INT; const y:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure zdrot(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const y:PMKL_Complex16; const incy:PMKL_INT; const 
              c:Pdouble; const s:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure zdscal(const n:PMKL_INT; const a:Pdouble; const x:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure zgthr(const nz:PMKL_INT; const y:PMKL_Complex16; const x:PMKL_Complex16; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};



  procedure zgthrz(const nz:PMKL_INT; const y:PMKL_Complex16; const x:PMKL_Complex16; const indx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};


  procedure zrotg(const a:PMKL_Complex16; const b:PMKL_Complex16; const c:Pdouble; const s:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure zscal(const n:PMKL_INT; const a:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure zsctr(const nz:PMKL_INT; const x:PMKL_Complex16; const indx:PMKL_INT; const y:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};




  procedure zswap(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const y:PMKL_Complex16; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};




  function izamax(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};




  function izamin(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT):MKL_INT;winapi; external {$ifdef libmkl} libmkl{$endif};

  { blas level2  }












  procedure sgbmv(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const kl:PMKL_INT; const ku:PMKL_INT; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const 
              beta:Psingle; const y:Psingle; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure sgemv(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Psingle; const a:Psingle; const 
              lda:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const beta:Psingle; const y:Psingle; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure sger(const m:PMKL_INT; const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const 
              y:Psingle; const incy:PMKL_INT; const a:Psingle; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};











  procedure ssbmv(const uplo:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Psingle; const a:Psingle; const 
              lda:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const beta:Psingle; const y:Psingle; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};









  procedure sspmv(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const ap:Psingle; const x:Psingle; const 
              incx:PMKL_INT; const beta:Psingle; const y:Psingle; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};






  procedure sspr(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const 
              ap:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};








  procedure sspr2(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const 
              y:Psingle; const incy:PMKL_INT; const ap:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ssymv(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const 
              x:Psingle; const incx:PMKL_INT; const beta:Psingle; const y:Psingle; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ssyr(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const 
              a:Psingle; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ssyr2(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const
              y:Psingle; const incy:PMKL_INT; const a:Psingle; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure stbmv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:Psingle; const lda:PMKL_INT; const x:Psingle; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};


  procedure stbsv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:Psingle; const lda:PMKL_INT; const x:Psingle; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure stpmv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:Psingle; const 
              x:Psingle; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure stpsv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:Psingle; const 
              x:Psingle; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure strmv(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT; const a:Psingle; const 
              lda:PMKL_INT; const b:Psingle; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure strsv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const a:Psingle; const 
              lda:PMKL_INT; const x:Psingle; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure sgem2vu(const m:PMKL_INT; const n:PMKL_INT; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const 
              x1:Psingle; const incx1:PMKL_INT; const x2:Psingle; const incx2:PMKL_INT; const beta:Psingle; const 
              y1:Psingle; const incy1:PMKL_INT; const y2:Psingle; const incy2:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cgbmv(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const kl:PMKL_INT; const ku:PMKL_INT; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const 
              beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cgemv(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const 
              lda:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const beta:PMKL_Complex8; const y:PMKL_Complex8; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cgerc(const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT; const 
              y:PMKL_Complex8; const incy:PMKL_INT; const a:PMKL_Complex8; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cgeru(const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT; const 
              y:PMKL_Complex8; const incy:PMKL_INT; const a:PMKL_Complex8; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure chbmv(const uplo:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const 
              lda:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const beta:PMKL_Complex8; const y:PMKL_Complex8; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure chemv(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const 
              x:PMKL_Complex8; const incx:PMKL_INT; const beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cher(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const x:PMKL_Complex8; const incx:PMKL_INT; const 
              a:PMKL_Complex8; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cher2(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT; const 
              y:PMKL_Complex8; const incy:PMKL_INT; const a:PMKL_Complex8; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure chpmv(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex8; const ap:PMKL_Complex8; const x:PMKL_Complex8; const 
              incx:PMKL_INT; const beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure chpr(const uplo:Pchar; const n:PMKL_INT; const alpha:Psingle; const x:PMKL_Complex8; const incx:PMKL_INT; const 
              ap:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure chpr2(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT; const 
              y:PMKL_Complex8; const incy:PMKL_INT; const ap:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ctbmv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ctbsv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ctpmv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:PMKL_Complex8; const 
              x:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ctpsv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:PMKL_Complex8; const 
              x:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ctrmv(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT; const a:PMKL_Complex8; const 
              lda:PMKL_INT; const b:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ctrsv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const a:PMKL_Complex8; const 
              lda:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cgem2vc(const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const 
              x1:PMKL_Complex8; const incx1:PMKL_INT; const x2:PMKL_Complex8; const incx2:PMKL_INT; const beta:PMKL_Complex8; const 
              y1:PMKL_Complex8; const incy1:PMKL_INT; const y2:PMKL_Complex8; const incy2:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure scgemv(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const a:Psingle; const 
              lda:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const beta:PMKL_Complex8; const y:PMKL_Complex8; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dgbmv(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const kl:PMKL_INT; const ku:PMKL_INT; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const 
              beta:Pdouble; const y:Pdouble; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dgemv(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const
              lda:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const beta:Pdouble; const y:Pdouble; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dger(const m:PMKL_INT; const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const 
              y:Pdouble; const incy:PMKL_INT; const a:Pdouble; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dsbmv(const uplo:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const 
              lda:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const beta:Pdouble; const y:Pdouble; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dspmv(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const ap:Pdouble; const x:Pdouble; const
              incx:PMKL_INT; const beta:Pdouble; const y:Pdouble; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dspr(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const 
              ap:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dspr2(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const 
              y:Pdouble; const incy:PMKL_INT; const ap:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dsymv(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const 
              x:Pdouble; const incx:PMKL_INT; const beta:Pdouble; const y:Pdouble; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dsyr(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const 
              a:Pdouble; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dsyr2(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const 
              y:Pdouble; const incy:PMKL_INT; const a:Pdouble; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dtbmv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:Pdouble; const lda:PMKL_INT; const x:Pdouble; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dtbsv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:Pdouble; const lda:PMKL_INT; const x:Pdouble; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dtpmv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:Pdouble; const 
              x:Pdouble; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dtpsv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:Pdouble; const 
              x:Pdouble; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dtrmv(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT; const a:Pdouble; const 
              lda:PMKL_INT; const b:Pdouble; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dtrsv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const a:Pdouble; const 
              lda:PMKL_INT; const x:Pdouble; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dgem2vu(const m:PMKL_INT; const n:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const 
              x1:Pdouble; const incx1:PMKL_INT; const x2:Pdouble; const incx2:PMKL_INT; const beta:Pdouble; const 
              y1:Pdouble; const incy1:PMKL_INT; const y2:Pdouble; const incy2:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zgbmv(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const kl:PMKL_INT; const ku:PMKL_INT; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const 
              beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zgemv(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const 
              lda:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const beta:PMKL_Complex16; const y:PMKL_Complex16; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zgerc(const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT; const 
              y:PMKL_Complex16; const incy:PMKL_INT; const a:PMKL_Complex16; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zgeru(const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT; const 
              y:PMKL_Complex16; const incy:PMKL_INT; const a:PMKL_Complex16; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zhbmv(const uplo:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const 
              lda:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const beta:PMKL_Complex16; const y:PMKL_Complex16; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zhemv(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const 
              x:PMKL_Complex16; const incx:PMKL_INT; const beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zher(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const x:PMKL_Complex16; const incx:PMKL_INT; const 
              a:PMKL_Complex16; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zher2(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT; const 
              y:PMKL_Complex16; const incy:PMKL_INT; const a:PMKL_Complex16; const lda:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zhpmv(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex16; const ap:PMKL_Complex16; const x:PMKL_Complex16; const 
              incx:PMKL_INT; const beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zhpr(const uplo:Pchar; const n:PMKL_INT; const alpha:Pdouble; const x:PMKL_Complex16; const incx:PMKL_INT; const 
              ap:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zhpr2(const uplo:Pchar; const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT; const 
              y:PMKL_Complex16; const incy:PMKL_INT; const ap:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ztbmv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ztbsv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ztpmv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:PMKL_Complex16; const 
              x:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ztpsv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const ap:PMKL_Complex16; const 
              x:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ztrmv(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT; const a:PMKL_Complex16; const 
              lda:PMKL_INT; const b:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ztrsv(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT; const a:PMKL_Complex16; const 
              lda:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zgem2vc(const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const 
              x1:PMKL_Complex16; const incx1:PMKL_INT; const x2:PMKL_Complex16; const incx2:PMKL_INT; const beta:PMKL_Complex16; const 
              y1:PMKL_Complex16; const incy1:PMKL_INT; const y2:PMKL_Complex16; const incy2:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dzgemv(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const a:Pdouble; const 
              lda:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const beta:PMKL_Complex16; const y:PMKL_Complex16; const 
              incy:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  { blas level3  }

  procedure sgemm(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const b:Psingle; const ldb:PMKL_INT; const 
              beta:Psingle; const c:Psingle; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  function sgemm_pack_get_size(const identifier:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT):size_t;winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure sgemm_pack(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Psingle; const src:Psingle; const ld:PMKL_INT; const dest:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure sgemm_compute(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              a:Psingle; const lda:PMKL_INT; const b:Psingle; const ldb:PMKL_INT; const beta:Psingle; const 
              c:Psingle; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure sgemm_batch(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT; const n_array:PMKL_INT; const k_array:PMKL_INT; const 
              alpha_array:Psingle; const a_array:PPsingle; const lda_array:PMKL_INT; const b_array:PPsingle; const ldb_array:PMKL_INT; const 
              beta_array:Psingle; const c_array:PPsingle; const ldc_array:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure sgemm_batch_strided(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const stridea:PMKL_INT; const b:Psingle; const 
              ldb:PMKL_INT; const strideb:PMKL_INT; const beta:Psingle; const c:Psingle; const ldc:PMKL_INT; const 
              stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure sgemmt(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const b:Psingle; const ldb:PMKL_INT; const 
              beta:Psingle; const c:Psingle; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ssymm(const side:Pchar; const uplo:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Psingle; const 
              a:Psingle; const lda:PMKL_INT; const b:Psingle; const ldb:PMKL_INT; const beta:Psingle; const 
              c:Psingle; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ssyr2k(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Psingle; const 
              a:Psingle; const lda:PMKL_INT; const b:Psingle; const ldb:PMKL_INT; const beta:Psingle; const 
              c:Psingle; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ssyrk(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Psingle; const 
              a:Psingle; const lda:PMKL_INT; const beta:Psingle; const c:Psingle; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure strmm(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const b:Psingle; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure strsm(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const b:Psingle; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure strsm_batch(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT; const 
              n_array:PMKL_INT; const alpha_array:Psingle; const a_array:PPsingle; const lda_array:PMKL_INT; const b_array:PPsingle; const 
              ldb:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure strsm_batch_strided(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT; const stridea:PMKL_INT; const 
              b:Psingle; const ldb:PMKL_INT; const strideb:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cgemm(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cgemm_batch(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT; const n_array:PMKL_INT; const k_array:PMKL_INT; const 
              alpha_array:PMKL_Complex8; const a_array:PPMKL_Complex8; const lda_array:PMKL_INT; const b_array:PPMKL_Complex8; const ldb_array:PMKL_INT; const 
              beta_array:PMKL_Complex8; const c_array:PPMKL_Complex8; const ldc_array:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cgemm_batch_strided(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const stridea:PMKL_INT; const b:PMKL_Complex8; const 
              ldb:PMKL_INT; const strideb:PMKL_INT; const beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT; const 
              stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure scgemm(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex8; const a:Psingle; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cgemm3m(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
 
  procedure cgemm3m_batch(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT; const n_array:PMKL_INT; const k_array:PMKL_INT; const 
              alpha_array:PMKL_Complex8; const a_array:PPMKL_Complex8; const lda_array:PMKL_INT; const b_array:PPMKL_Complex8; const ldb_array:PMKL_INT; const 
              beta_array:PMKL_Complex8; const c_array:PPMKL_Complex8; const ldc_array:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cgemmt(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure chemm(const side:Pchar; const uplo:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const beta:PMKL_Complex8; const 
              c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cher2k(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const beta:Psingle; const 
              c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cherk(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Psingle; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const beta:Psingle; const c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure csymm(const side:Pchar; const uplo:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const beta:PMKL_Complex8; const 
              c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure csyr2k(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const ldb:PMKL_INT; const beta:PMKL_Complex8; const 
              c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure csyrk(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT; const beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ctrmm(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ctrsm(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const b:PMKL_Complex8; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ctrsm_batch(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT; const 
              n_array:PMKL_INT; const alpha_array:PMKL_Complex8; const a_array:PPMKL_Complex8; const lda_array:PMKL_INT; const b_array:PPMKL_Complex8; const 
              ldb:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ctrsm_batch_strided(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT; const stridea:PMKL_INT; const 
              b:PMKL_Complex8; const ldb:PMKL_INT; const strideb:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dgemm(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const b:Pdouble; const ldb:PMKL_INT; const 
              beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  function dgemm_pack_get_size(const identifier:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT):size_t;winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dgemm_pack(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Pdouble; const src:Pdouble; const ld:PMKL_INT; const dest:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dgemm_compute(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              a:Pdouble; const lda:PMKL_INT; const b:Pdouble; const ldb:PMKL_INT; const beta:Pdouble; const 
              c:Pdouble; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dgemm_batch(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT; const n_array:PMKL_INT; const k_array:PMKL_INT; const 
              alpha_array:Pdouble; const a_array:PPdouble; const lda_array:PMKL_INT; const b_array:PPdouble; const ldb_array:PMKL_INT; const 
              beta_array:Pdouble; const c_array:PPdouble; const ldc_array:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dgemm_batch_strided(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const stridea:PMKL_INT; const b:Pdouble; const 
              ldb:PMKL_INT; const strideb:PMKL_INT; const beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT; const 
              stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dgemmt(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const b:Pdouble; const ldb:PMKL_INT; const 
              beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dsymm(const side:Pchar; const uplo:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Pdouble; const 
              a:Pdouble; const lda:PMKL_INT; const b:Pdouble; const ldb:PMKL_INT; const beta:Pdouble; const 
              c:Pdouble; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
 
  procedure dsyr2k(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Pdouble; const 
              a:Pdouble; const lda:PMKL_INT; const b:Pdouble; const ldb:PMKL_INT; const beta:Pdouble; const 
              c:Pdouble; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dsyrk(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Pdouble; const 
              a:Pdouble; const lda:PMKL_INT; const beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dtrmm(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const b:Pdouble; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dtrsm(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const b:Pdouble; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dtrsm_batch(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT; const 
              n_array:PMKL_INT; const alpha_array:Pdouble; const a_array:PPdouble; const lda_array:PMKL_INT; const b_array:PPdouble; const 
              ldb:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dtrsm_batch_strided(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT; const stridea:PMKL_INT; const 
              b:Pdouble; const ldb:PMKL_INT; const strideb:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zgemm(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zgemm_batch(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT; const n_array:PMKL_INT; const k_array:PMKL_INT; const 
              alpha_array:PMKL_Complex16; const a_array:PPMKL_Complex16; const lda_array:PMKL_INT; const b_array:PPMKL_Complex16; const ldb_array:PMKL_INT; const 
              beta_array:PMKL_Complex16; const c_array:PPMKL_Complex16; const ldc_array:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zgemm_batch_strided(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const stridea:PMKL_INT; const b:PMKL_Complex16; const 
              ldb:PMKL_INT; const strideb:PMKL_INT; const beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT; const 
              stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dzgemm(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex16; const a:Pdouble; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zgemm3m(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zgemm3m_batch(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT; const n_array:PMKL_INT; const k_array:PMKL_INT; const 
              alpha_array:PMKL_Complex16; const a_array:PPMKL_Complex16; const lda_array:PMKL_INT; const b_array:PPMKL_Complex16; const ldb_array:PMKL_INT; const 
              beta_array:PMKL_Complex16; const c_array:PPMKL_Complex16; const ldc_array:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zgemmt(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zhemm(const side:Pchar; const uplo:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const beta:PMKL_Complex16; const 
              c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zher2k(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const beta:Pdouble; const 
              c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zherk(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:Pdouble; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const beta:Pdouble; const c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zsymm(const side:Pchar; const uplo:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const beta:PMKL_Complex16; const 
              c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zsyr2k(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const ldb:PMKL_INT; const beta:PMKL_Complex16; const 
              c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zsyrk(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT; const k:PMKL_INT; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT; const beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ztrmm(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ztrsm(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const b:PMKL_Complex16; const 
              ldb:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ztrsm_batch(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT; const 
              n_array:PMKL_INT; const alpha_array:PMKL_Complex16; const a_array:PPMKL_Complex16; const lda_array:PMKL_INT; const b_array:PPMKL_Complex16; const 
              ldb:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ztrsm_batch_strided(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT; const 
              n:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT; const stridea:PMKL_INT; const 
              b:PMKL_Complex16; const ldb:PMKL_INT; const strideb:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
 
  procedure gemm_s16s16s32(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT; const n:PMKL_INT; const 
              k:PMKL_INT; const alpha:Psingle; const a:PMKL_INT16; const lda:PMKL_INT; const ao:PMKL_INT16; const 
              b:PMKL_INT16; const ldb:PMKL_INT; const bo:PMKL_INT16; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT; const co:PMKL_INT32);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure gemm_s8u8s32(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT; const n:PMKL_INT; const 
              k:PMKL_INT; const alpha:Psingle; const a:PMKL_INT8; const lda:PMKL_INT; const ao:PMKL_INT8; const 
              b:PMKL_UINT8; const ldb:PMKL_INT; const bo:PMKL_INT8; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT; const co:PMKL_INT32);winapi; external {$ifdef libmkl} libmkl{$endif};

  function gemm_s8u8s32_pack_get_size(const identifier:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT):size_t;winapi; external {$ifdef libmkl} libmkl{$endif};

  function gemm_s16s16s32_pack_get_size(const identifier:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT):size_t;winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure gemm_s8u8s32_pack(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              src:pointer; const ld:PMKL_INT; const dest:pointer);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure gemm_s16s16s32_pack(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              src:PMKL_INT16; const ld:PMKL_INT; const dest:PMKL_INT16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure gemm_s8u8s32_compute(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT; const n:PMKL_INT; const 
              k:PMKL_INT; const alpha:Psingle; const a:PMKL_INT8; const lda:PMKL_INT; const ao:PMKL_INT8; const 
              b:PMKL_UINT8; const ldb:PMKL_INT; const bo:PMKL_INT8; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT; const co:PMKL_INT32);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure gemm_s16s16s32_compute(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT; const n:PMKL_INT; const 
              k:PMKL_INT; const alpha:Psingle; const a:PMKL_INT16; const lda:PMKL_INT; const ao:PMKL_INT16; const 
              b:PMKL_INT16; const ldb:PMKL_INT; const bo:PMKL_INT16; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT; const co:PMKL_INT32);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure hgemm(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_F16; const a:PMKL_F16; const lda:PMKL_INT; const b:PMKL_F16; const ldb:PMKL_INT; const 
              beta:PMKL_F16; const c:PMKL_F16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  function hgemm_pack_get_size(const identifier:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT):size_t;winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure hgemm_pack(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              alpha:PMKL_F16; const src:PMKL_F16; const ld:PMKL_INT; const dest:PMKL_F16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure hgemm_compute(const transa:Pchar; const transb:Pchar; const m:PMKL_INT; const n:PMKL_INT; const k:PMKL_INT; const 
              a:PMKL_F16; const lda:PMKL_INT; const b:PMKL_F16; const ldb:PMKL_INT; const beta:PMKL_F16; const 
              c:PMKL_F16; const ldc:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  {
   * Jit routines
    }
{$ifndef mkl_jit_create_dgemm}
  {$define mkl_jit_create_dgemm := mkl_cblas_jit_create_dgemm}    
{$endif}

  function mkl_cblas_jit_create_dgemm(const jitter:Ppointer; const layout:TMKL_LAYOUT; const transa:TMKL_TRANSPOSE; const transb:TMKL_TRANSPOSE; const m:MKL_INT; const 
             n:MKL_INT; const k:MKL_INT; const alpha:double; const lda:MKL_INT; const ldb:MKL_INT; const 
             beta:double; const ldc:MKL_INT):Tmkl_jit_status_t;winapi; external {$ifdef libmkl} libmkl{$endif};

{$ifndef mkl_jit_create_sgemm}
  {$define mkl_jit_create_sgemm := mkl_cblas_jit_create_sgemm}    
{$endif}

  function mkl_cblas_jit_create_sgemm(const jitter:Ppointer; const layout:TMKL_LAYOUT; const transa:TMKL_TRANSPOSE; const transb:TMKL_TRANSPOSE; const m:MKL_INT; const 
             n:MKL_INT; const k:MKL_INT; const alpha:single; const lda:MKL_INT; const ldb:MKL_INT; const 
             beta:single; const ldc:MKL_INT):Tmkl_jit_status_t;winapi; external {$ifdef libmkl} libmkl{$endif};

{$ifndef mkl_jit_create_cgemm}
  {$define mkl_jit_create_cgemm := mkl_cblas_jit_create_cgemm}    
{$endif}

  function mkl_cblas_jit_create_cgemm(const jitter:Ppointer; const layout:TMKL_LAYOUT; const transa:TMKL_TRANSPOSE; const transb:TMKL_TRANSPOSE; const m:MKL_INT; const 
             n:MKL_INT; const k:MKL_INT; const alpha:pointer; const lda:MKL_INT; const ldb:MKL_INT; const 
             beta:pointer; const ldc:MKL_INT):Tmkl_jit_status_t;winapi; external {$ifdef libmkl} libmkl{$endif};

{$ifndef mkl_jit_create_zgemm}
  {$define mkl_jit_create_zgemm := mkl_cblas_jit_create_zgemm}    
{$endif}

  function mkl_cblas_jit_create_zgemm(const jitter:Ppointer; const layout:TMKL_LAYOUT; const transa:TMKL_TRANSPOSE; const transb:TMKL_TRANSPOSE; const m:MKL_INT; const 
             n:MKL_INT; const k:MKL_INT; const alpha:pointer; const lda:MKL_INT; const ldb:MKL_INT; const 
             beta:pointer; const ldc:MKL_INT):Tmkl_jit_status_t;winapi; external {$ifdef libmkl} libmkl{$endif};


  function mkl_jit_get_dgemm_ptr(const jitter:pointer):Tdgemm_jit_kernel_t;winapi; external {$ifdef libmkl} libmkl{$endif};


  function mkl_jit_get_sgemm_ptr(const jitter:pointer):Tsgemm_jit_kernel_t;winapi; external {$ifdef libmkl} libmkl{$endif};


  function mkl_jit_get_cgemm_ptr(const jitter:pointer):Tcgemm_jit_kernel_t;winapi; external {$ifdef libmkl} libmkl{$endif};


  function mkl_jit_get_zgemm_ptr(const jitter:pointer):Tzgemm_jit_kernel_t;winapi; external {$ifdef libmkl} libmkl{$endif};

  function mkl_jit_destroy(const jitter:pointer):Tmkl_jit_status_t;winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif LOWERCASE_DECL}
  { Level1 BLAS batch API  }

{$ifdef UPPERCASE_DECL}
  procedure SAXPY_BATCH(const n:PMKL_INT; const alpha:Psingle; const x:PPsingle; const incx:PMKL_INT; const y:PPsingle; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}

{$ifdef LOWERCASE_DECL}
  procedure saxpy_batch(const n:PMKL_INT; const alpha:Psingle; const x:PPsingle; const incx:PMKL_INT; const y:PPsingle; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure DAXPY_BATCH(const n:PMKL_INT; const alpha:Pdouble; const x:PPdouble; const incx:PMKL_INT; const y:PPdouble; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{ifdef LOWERCASE_DECL}
  procedure daxpy_batch(const n:PMKL_INT; const alpha:Pdouble; const x:PPdouble; const incx:PMKL_INT; const y:PPdouble; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure CAXPY_BATCH(const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PPMKL_Complex8; const incx:PMKL_INT; const y:PPMKL_Complex8; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef LOWERCASE_DECL}
  procedure caxpy_batch(const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PPMKL_Complex8; const incx:PMKL_INT; const y:PPMKL_Complex8; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure ZAXPY_BATCH(const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PPMKL_Complex16; const incx:PMKL_INT; const y:PPMKL_Complex16; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef LOWERCASE_DECL}
  procedure zaxpy_batch(const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PPMKL_Complex16; const incx:PMKL_INT; const y:PPMKL_Complex16; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure SCOPY_BATCH(const n:PMKL_INT; const x:PPsingle; const incx:PMKL_INT; const y:PPsingle; const incy:PMKL_INT; const 
              group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef LOWERCASE_DECL}
  procedure scopy_batch(const n:PMKL_INT; const x:PPsingle; const incx:PMKL_INT; const y:PPsingle; const incy:PMKL_INT; const 
              group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure DCOPY_BATCH(const n:PMKL_INT; const x:PPdouble; const incx:PMKL_INT; const y:PPdouble; const incy:PMKL_INT; const 
              group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef LOWERCASE_DECL}
  procedure dcopy_batch(const n:PMKL_INT; const x:PPdouble; const incx:PMKL_INT; const y:PPdouble; const incy:PMKL_INT; const 
              group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure CCOPY_BATCH(const n:PMKL_INT; const x:PPMKL_Complex8; const incx:PMKL_INT; const y:PPMKL_Complex8; const incy:PMKL_INT; const 
              group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef LOWERCASE_DECL}
  procedure ccopy_batch(const n:PMKL_INT; const x:PPMKL_Complex8; const incx:PMKL_INT; const y:PPMKL_Complex8; const incy:PMKL_INT; const 
              group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure ZCOPY_BATCH(const n:PMKL_INT; const x:PPMKL_Complex16; const incx:PMKL_INT; const y:PPMKL_Complex16; const incy:PMKL_INT; const 
              group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef LOWERCASE_DECL}
  procedure zcopy_batch(const n:PMKL_INT; const x:PPMKL_Complex16; const incx:PMKL_INT; const y:PPMKL_Complex16; const incy:PMKL_INT; const 
              group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure SAXPY_BATCH_STRIDED(const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const stridex:PMKL_INT; const 
              y:Psingle; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef LOWERCASE_DECL}
  procedure saxpy_batch_strided(const n:PMKL_INT; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT; const stridex:PMKL_INT; const
              y:Psingle; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure DAXPY_BATCH_STRIDED(const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const stridex:PMKL_INT; const 
              y:Pdouble; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef LOWERCASE_DECL}
  procedure daxpy_batch_strided(const n:PMKL_INT; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT; const stridex:PMKL_INT; const 
              y:Pdouble; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure CAXPY_BATCH_STRIDED(const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT; const stridex:PMKL_INT; const 
              y:PMKL_Complex8; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef LOWERCASE_DECL}
  procedure caxpy_batch_strided(const n:PMKL_INT; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT; const stridex:PMKL_INT; const
              y:PMKL_Complex8; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure ZAXPY_BATCH_STRIDED(const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT; const stridex:PMKL_INT; const 
              y:PMKL_Complex16; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef LOWERCASE_DECL}
  procedure zaxpy_batch_strided(const n:PMKL_INT; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT; const stridex:PMKL_INT; const 
              y:PMKL_Complex16; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure SCOPY_BATCH_STRIDED(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const stridex:PMKL_INT; const y:Psingle; const 
              incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef LOWERCASE_DECL}
  procedure scopy_batch_strided(const n:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const stridex:PMKL_INT; const y:Psingle; const 
              incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure DCOPY_BATCH_STRIDED(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const stridex:PMKL_INT; const y:Pdouble; const 
              incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef LOWERCASE_DECL}
  procedure dcopy_batch_strided(const n:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const stridex:PMKL_INT; const y:Pdouble; const 
              incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure CCOPY_BATCH_STRIDED(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const stridex:PMKL_INT; const y:PMKL_Complex8; const 
              incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef LOWERCASE_DECL}
  procedure ccopy_batch_strided(const n:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const stridex:PMKL_INT; const y:PMKL_Complex8; const 
              incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure ZCOPY_BATCH_STRIDED(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const stridex:PMKL_INT; const y:PMKL_Complex16; const 
              incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}

{$ifdef LOWERCASE_DECL}
  procedure zcopy_batch_strided(const n:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const stridex:PMKL_INT; const y:PMKL_Complex16; const
              incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  { Level2 BLAS batch API  }

  procedure sgemv_batch(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Psingle; const a:PPsingle; const 
              lda:PMKL_INT; const x:PPsingle; const incx:PMKL_INT; const beta:Psingle; const y:PPsingle; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure sgemv_batch_strided(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Psingle; const a:Psingle; const 
              lda:PMKL_INT; const stridea:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const stridex:PMKL_INT; const 
              beta:Psingle; const y:Psingle; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dgemv_batch(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Pdouble; const a:PPdouble; const 
              lda:PMKL_INT; const x:PPdouble; const incx:PMKL_INT; const beta:Pdouble; const y:PPdouble; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure dgemv_batch_strided(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const 
              lda:PMKL_INT; const stridea:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const stridex:PMKL_INT; const 
              beta:Pdouble; const y:Pdouble; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cgemv_batch(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const a:PPMKL_Complex8; const 
              lda:PMKL_INT; const x:PPMKL_Complex8; const incx:PMKL_INT; const beta:PMKL_Complex8; const y:PPMKL_Complex8; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cgemv_batch_strided(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const 
              lda:PMKL_INT; const stridea:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const stridex:PMKL_INT; const 
              beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zgemv_batch(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const a:PPMKL_Complex16; const 
              lda:PMKL_INT; const x:PPMKL_Complex16; const incx:PMKL_INT; const beta:PMKL_Complex16; const y:PPMKL_Complex16; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zgemv_batch_strided(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const 
              lda:PMKL_INT; const stridea:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const stridex:PMKL_INT; const 
              beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}

{$ifdef UPPERCASE_DECL}
  procedure SGEMV_BATCH(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Psingle; const a:PPsingle; const 
              lda:PMKL_INT; const x:PPsingle; const incx:PMKL_INT; const beta:Psingle; const y:PPsingle; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure SGEMV_BATCH_STRIDED(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Psingle; const a:Psingle; const 
              lda:PMKL_INT; const stridea:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const stridex:PMKL_INT; const 
              beta:Psingle; const y:Psingle; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure DGEMV_BATCH(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Pdouble; const a:PPdouble; const 
              lda:PMKL_INT; const x:PPdouble; const incx:PMKL_INT; const beta:Pdouble; const y:PPdouble; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure DGEMV_BATCH_STRIDED(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:Pdouble; const a:Pdouble; const 
              lda:PMKL_INT; const stridea:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const stridex:PMKL_INT; const 
              beta:Pdouble; const y:Pdouble; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure CGEMV_BATCH(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const a:PPMKL_Complex8; const 
              lda:PMKL_INT; const x:PPMKL_Complex8; const incx:PMKL_INT; const beta:PMKL_Complex8; const y:PPMKL_Complex8; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure CGEMV_BATCH_STRIDED(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const 
              lda:PMKL_INT; const stridea:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const stridex:PMKL_INT; const 
              beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ZGEMV_BATCH(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const a:PPMKL_Complex16; const 
              lda:PMKL_INT; const x:PPMKL_Complex16; const incx:PMKL_INT; const beta:PMKL_Complex16; const y:PPMKL_Complex16; const 
              incy:PMKL_INT; const group_count:PMKL_INT; const group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ZGEMV_BATCH_STRIDED(const trans:Pchar; const m:PMKL_INT; const n:PMKL_INT; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const 
              lda:PMKL_INT; const stridea:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const stridex:PMKL_INT; const 
              beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT; const stridey:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef LOWERCASE_DECL}
  procedure sdgmm_batch(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:PPsingle; const lda:PMKL_INT; const 
              x:PPsingle; const incx:PMKL_INT; const c:PPsingle; const ldc:PMKL_INT; const group_count:PMKL_INT; const 
              group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure sdgmm_batch_strided(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:Psingle; const lda:PMKL_INT; const 
              stridea:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const stridex:PMKL_INT; const c:Psingle; const 
              ldc:PMKL_INT; const stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ddgmm_batch(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:PPdouble; const lda:PMKL_INT; const 
              x:PPdouble; const incx:PMKL_INT; const c:PPdouble; const ldc:PMKL_INT; const group_count:PMKL_INT; const 
              group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ddgmm_batch_strided(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:Pdouble; const lda:PMKL_INT; const 
              stridea:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const stridex:PMKL_INT; const c:Pdouble; const 
              ldc:PMKL_INT; const stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cdgmm_batch(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:PPMKL_Complex8; const lda:PMKL_INT; const 
              x:PPMKL_Complex8; const incx:PMKL_INT; const c:PPMKL_Complex8; const ldc:PMKL_INT; const group_count:PMKL_INT; const 
              group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure cdgmm_batch_strided(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:PMKL_Complex8; const lda:PMKL_INT; const 
              stridea:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const stridex:PMKL_INT; const c:PMKL_Complex8; const 
              ldc:PMKL_INT; const stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zdgmm_batch(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:PPMKL_Complex16; const lda:PMKL_INT; const 
              x:PPMKL_Complex16; const incx:PMKL_INT; const c:PPMKL_Complex16; const ldc:PMKL_INT; const group_count:PMKL_INT; const 
              group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure zdgmm_batch_strided(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:PMKL_Complex16; const lda:PMKL_INT; const 
              stridea:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const stridex:PMKL_INT; const c:PMKL_Complex16; const 
              ldc:PMKL_INT; const stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}
{$ifdef UPPERCASE_DECL}
  procedure SDGMM_BATCH(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:PPsingle; const lda:PMKL_INT; const 
              x:PPsingle; const incx:PMKL_INT; const c:PPsingle; const ldc:PMKL_INT; const group_count:PMKL_INT; const 
              group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure SDGMM_BATCH_STRIDED(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:Psingle; const lda:PMKL_INT; const 
              stridea:PMKL_INT; const x:Psingle; const incx:PMKL_INT; const stridex:PMKL_INT; const c:Psingle; const 
              ldc:PMKL_INT; const stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure DDGMM_BATCH(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:PPdouble; const lda:PMKL_INT; const 
              x:PPdouble; const incx:PMKL_INT; const c:PPdouble; const ldc:PMKL_INT; const group_count:PMKL_INT; const 
              group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure DDGMM_BATCH_STRIDED(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:Pdouble; const lda:PMKL_INT; const 
              stridea:PMKL_INT; const x:Pdouble; const incx:PMKL_INT; const stridex:PMKL_INT; const c:Pdouble; const 
              ldc:PMKL_INT; const stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure CDGMM_BATCH(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:PPMKL_Complex8; const lda:PMKL_INT; const 
              x:PPMKL_Complex8; const incx:PMKL_INT; const c:PPMKL_Complex8; const ldc:PMKL_INT; const group_count:PMKL_INT; const 
              group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure CDGMM_BATCH_STRIDED(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:PMKL_Complex8; const lda:PMKL_INT; const 
              stridea:PMKL_INT; const x:PMKL_Complex8; const incx:PMKL_INT; const stridex:PMKL_INT; const c:PMKL_Complex8; const 
              ldc:PMKL_INT; const stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ZDGMM_BATCH(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:PPMKL_Complex16; const lda:PMKL_INT; const 
              x:PPMKL_Complex16; const incx:PMKL_INT; const c:PPMKL_Complex16; const ldc:PMKL_INT; const group_count:PMKL_INT; const 
              group_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure ZDGMM_BATCH_STRIDED(const side:Pchar; const m:PMKL_INT; const n:PMKL_INT; const a:PMKL_Complex16; const lda:PMKL_INT; const 
              stridea:PMKL_INT; const x:PMKL_Complex16; const incx:PMKL_INT; const stridex:PMKL_INT; const c:PMKL_Complex16; const 
              ldc:PMKL_INT; const stridec:PMKL_INT; const batch_size:PMKL_INT);winapi; external {$ifdef libmkl} libmkl{$endif};
{$endif}

{ C++ end of extern C conditionnal removed }
  { __cplusplus  }
  { _MKL_BLAS_H_  }

implementation


end.
