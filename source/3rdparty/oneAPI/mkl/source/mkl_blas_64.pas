unit mkl_blas_64;
interface

uses types, mkl_types;

{$include mkl.inc}

  {******************************************************************************
  * Copyright 2021 Intel Corporation.
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
  !      Intel(R) oneAPI Math Kernel Library (oneMKL) interface for ILP64 BLAS routines
  !***************************************************************************** }

  { Upper case declaration  }
  { BLAS Level1  }

{$ifdef UPPERCASE_DECL}
  function SCABS1_64(const c:PMKL_Complex8):single;winapi; external;




  function SASUM_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64):single;winapi; external;






  procedure SAXPY_64(const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const 
              incy:PMKL_INT64);winapi; external;







  procedure SAXPBY_64(const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const beta:Psingle; const 
              y:Psingle; const incy:PMKL_INT64);winapi; external;





  procedure SAXPYI_64(const nz:PMKL_INT64; const a:Psingle; const x:Psingle; const indx:PMKL_INT64; const y:Psingle);winapi; external;




  function SCASUM_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64):single;winapi; external;




  function SCNRM2_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64):single;winapi; external;





  procedure SCOPY_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const incy:PMKL_INT64);winapi; external;






  function SDOT_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const incy:PMKL_INT64):single;winapi; external;







  function SDSDOT_64(const n:PMKL_INT64; const sb:Psingle; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const 
             incy:PMKL_INT64):single;winapi; external;





  function SDOTI_64(const nz:PMKL_INT64; const x:Psingle; const indx:PMKL_INT64; const y:Psingle):single;winapi; external;




  procedure SGTHR_64(const nz:PMKL_INT64; const y:Psingle; const x:Psingle; const indx:PMKL_INT64);winapi; external;



  procedure SGTHRZ_64(const nz:PMKL_INT64; const y:Psingle; const x:Psingle; const indx:PMKL_INT64);winapi; external;




  function SNRM2_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64):single;winapi; external;






  procedure SROT_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const incy:PMKL_INT64; const 
              c:Psingle; const s:Psingle);winapi; external;

  procedure SROTG_64(const a:Psingle; const b:Psingle; const c:Psingle; const s:Psingle);winapi; external;





  procedure SROTI_64(const nz:PMKL_INT64; const x:Psingle; const indx:PMKL_INT64; const y:Psingle; const c:Psingle; const 
              s:Psingle);winapi; external;





  procedure SROTM_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const incy:PMKL_INT64; const 
              param:Psingle);winapi; external;


  procedure SROTMG_64(const d1:Psingle; const d2:Psingle; const x1:Psingle; const y1:Psingle; const param:Psingle);winapi; external;




  procedure SSCAL_64(const n:PMKL_INT64; const a:Psingle; const x:Psingle; const incx:PMKL_INT64);winapi; external;




  procedure SSCTR_64(const nz:PMKL_INT64; const x:Psingle; const indx:PMKL_INT64; const y:Psingle);winapi; external;




  procedure SSWAP_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const incy:PMKL_INT64);winapi; external;




  function ISAMAX_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64):MKL_INT64;winapi; external;




  function ISAMIN_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64):MKL_INT64;winapi; external;






  procedure CAXPY_64(const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64; const y:PMKL_Complex8; const 
              incy:PMKL_INT64);winapi; external;







  procedure CAXPBY_64(const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64; const beta:PMKL_Complex8; const 
              y:PMKL_Complex8; const incy:PMKL_INT64);winapi; external;





  procedure CAXPYI_64(const nz:PMKL_INT64; const a:PMKL_Complex8; const x:PMKL_Complex8; const indx:PMKL_INT64; const y:PMKL_Complex8);winapi; external;





  procedure CCOPY_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const y:PMKL_Complex8; const incy:PMKL_INT64);winapi; external;






  procedure CDOTC_64(const pres:PMKL_Complex8; const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const y:PMKL_Complex8; const 
              incy:PMKL_INT64);winapi; external;





  procedure CDOTCI_64(const pres:PMKL_Complex8; const nz:PMKL_INT64; const x:PMKL_Complex8; const indx:PMKL_INT64; const y:PMKL_Complex8);winapi; external;






  procedure CDOTU_64(const pres:PMKL_Complex8; const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const y:PMKL_Complex8; const 
              incy:PMKL_INT64);winapi; external;





  procedure CDOTUI_64(const pres:PMKL_Complex8; const nz:PMKL_INT64; const x:PMKL_Complex8; const indx:PMKL_INT64; const y:PMKL_Complex8);winapi; external;




  procedure CGTHR_64(const nz:PMKL_INT64; const y:PMKL_Complex8; const x:PMKL_Complex8; const indx:PMKL_INT64);winapi; external;



  procedure CGTHRZ_64(const nz:PMKL_INT64; const y:PMKL_Complex8; const x:PMKL_Complex8; const indx:PMKL_INT64);winapi; external;


  procedure CROTG_64(const a:PMKL_Complex8; const b:PMKL_Complex8; const c:Psingle; const s:PMKL_Complex8);winapi; external;




  procedure CSCAL_64(const n:PMKL_INT64; const a:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;




  procedure CSCTR_64(const nz:PMKL_INT64; const x:PMKL_Complex8; const indx:PMKL_INT64; const y:PMKL_Complex8);winapi; external;






  procedure CSROT_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const y:PMKL_Complex8; const incy:PMKL_INT64; const 
              c:Psingle; const s:Psingle);winapi; external;




  procedure CSSCAL_64(const n:PMKL_INT64; const a:Psingle; const x:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;




  procedure CSWAP_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const y:PMKL_Complex8; const incy:PMKL_INT64);winapi; external;




  function ICAMAX_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64):MKL_INT64;winapi; external;




  function ICAMIN_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64):MKL_INT64;winapi; external;


  function DCABS1_64(const z:PMKL_Complex16):double;winapi; external;




  function DASUM_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64):double;winapi; external;






  procedure DAXPY_64(const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const y:Pdouble; const 
              incy:PMKL_INT64);winapi; external;







  procedure DAXPBY_64(const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const beta:Pdouble; const 
              y:Pdouble; const incy:PMKL_INT64);winapi; external;





  procedure DAXPYI_64(const nz:PMKL_INT64; const a:Pdouble; const x:Pdouble; const indx:PMKL_INT64; const y:Pdouble);winapi; external;





  procedure DCOPY_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const y:Pdouble; const incy:PMKL_INT64);winapi; external;






  function DDOT_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const y:Pdouble; const incy:PMKL_INT64):double;winapi; external;






  function DSDOT_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const incy:PMKL_INT64):double;winapi; external;





  function DDOTI_64(const nz:PMKL_INT64; const x:Pdouble; const indx:PMKL_INT64; const y:Pdouble):double;winapi; external;




  procedure DGTHR_64(const nz:PMKL_INT64; const y:Pdouble; const x:Pdouble; const indx:PMKL_INT64);winapi; external;



  procedure DGTHRZ_64(const nz:PMKL_INT64; const y:Pdouble; const x:Pdouble; const indx:PMKL_INT64);winapi; external;




  function DNRM2_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64):double;winapi; external;






  procedure DROT_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const y:Pdouble; const incy:PMKL_INT64; const 
              c:Pdouble; const s:Pdouble);winapi; external;

  procedure DROTG_64(const a:Pdouble; const b:Pdouble; const c:Pdouble; const s:Pdouble);winapi; external;





  procedure DROTI_64(const nz:PMKL_INT64; const x:Pdouble; const indx:PMKL_INT64; const y:Pdouble; const c:Pdouble; const 
              s:Pdouble);winapi; external;





  procedure DROTM_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const y:Pdouble; const incy:PMKL_INT64; const 
              param:Pdouble);winapi; external;


  procedure DROTMG_64(const d1:Pdouble; const d2:Pdouble; const x1:Pdouble; const y1:Pdouble; const param:Pdouble);winapi; external;




  procedure DSCAL_64(const n:PMKL_INT64; const a:Pdouble; const x:Pdouble; const incx:PMKL_INT64);winapi; external;




  procedure DSCTR_64(const nz:PMKL_INT64; const x:Pdouble; const indx:PMKL_INT64; const y:Pdouble);winapi; external;




  procedure DSWAP_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const y:Pdouble; const incy:PMKL_INT64);winapi; external;




  function DZASUM_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64):double;winapi; external;




  function DZNRM2_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64):double;winapi; external;




  function IDAMAX_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64):MKL_INT64;winapi; external;




  function IDAMIN_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64):MKL_INT64;winapi; external;






  procedure ZAXPY_64(const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64; const y:PMKL_Complex16; const 
              incy:PMKL_INT64);winapi; external;







  procedure ZAXPBY_64(const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64; const beta:PMKL_Complex16; const 
              y:PMKL_Complex16; const incy:PMKL_INT64);winapi; external;





  procedure ZAXPYI_64(const nz:PMKL_INT64; const a:PMKL_Complex16; const x:PMKL_Complex16; const indx:PMKL_INT64; const y:PMKL_Complex16);winapi; external;





  procedure ZCOPY_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const y:PMKL_Complex16; const incy:PMKL_INT64);winapi; external;






  procedure ZDOTC_64(const pres:PMKL_Complex16; const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const y:PMKL_Complex16; const 
              incy:PMKL_INT64);winapi; external;





  procedure ZDOTCI_64(const pres:PMKL_Complex16; const nz:PMKL_INT64; const x:PMKL_Complex16; const indx:PMKL_INT64; const y:PMKL_Complex16);winapi; external;






  procedure ZDOTU_64(const pres:PMKL_Complex16; const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const y:PMKL_Complex16; const 
              incy:PMKL_INT64);winapi; external;





  procedure ZDOTUI_64(const pres:PMKL_Complex16; const nz:PMKL_INT64; const x:PMKL_Complex16; const indx:PMKL_INT64; const y:PMKL_Complex16);winapi; external;






  procedure ZDROT_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const y:PMKL_Complex16; const incy:PMKL_INT64; const 
              c:Pdouble; const s:Pdouble);winapi; external;




  procedure ZDSCAL_64(const n:PMKL_INT64; const a:Pdouble; const x:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;




  procedure ZGTHR_64(const nz:PMKL_INT64; const y:PMKL_Complex16; const x:PMKL_Complex16; const indx:PMKL_INT64);winapi; external;



  procedure ZGTHRZ_64(const nz:PMKL_INT64; const y:PMKL_Complex16; const x:PMKL_Complex16; const indx:PMKL_INT64);winapi; external;


  procedure ZROTG_64(const a:PMKL_Complex16; const b:PMKL_Complex16; const c:Pdouble; const s:PMKL_Complex16);winapi; external;




  procedure ZSCAL_64(const n:PMKL_INT64; const a:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;




  procedure ZSCTR_64(const nz:PMKL_INT64; const x:PMKL_Complex16; const indx:PMKL_INT64; const y:PMKL_Complex16);winapi; external;




  procedure ZSWAP_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const y:PMKL_Complex16; const incy:PMKL_INT64);winapi; external;




  function IZAMAX_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64):MKL_INT64;winapi; external;




  function IZAMIN_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64):MKL_INT64;winapi; external;

  { BLAS Level2  }












  procedure SGBMV_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const kl:PMKL_INT64; const ku:PMKL_INT64; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const 
              beta:Psingle; const y:Psingle; const incy:PMKL_INT64);winapi; external;











  procedure SGEMV_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Psingle; const a:Psingle; const 
              lda:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const beta:Psingle; const y:Psingle; const 
              incy:PMKL_INT64);winapi; external;









  procedure SGER_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const 
              y:Psingle; const incy:PMKL_INT64; const a:Psingle; const lda:PMKL_INT64);winapi; external;











  procedure SSBMV_64(const uplo:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Psingle; const a:Psingle; const 
              lda:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const beta:Psingle; const y:Psingle; const 
              incy:PMKL_INT64);winapi; external;









  procedure SSPMV_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const ap:Psingle; const x:Psingle; const 
              incx:PMKL_INT64; const beta:Psingle; const y:Psingle; const incy:PMKL_INT64);winapi; external;






  procedure SSPR_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const 
              ap:Psingle);winapi; external;








  procedure SSPR2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const 
              y:Psingle; const incy:PMKL_INT64; const ap:Psingle);winapi; external;










  procedure SSYMV_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const 
              x:Psingle; const incx:PMKL_INT64; const beta:Psingle; const y:Psingle; const incy:PMKL_INT64);winapi; external;







  procedure SSYR_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const 
              a:Psingle; const lda:PMKL_INT64);winapi; external;









  procedure SSYR2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const 
              y:Psingle; const incy:PMKL_INT64; const a:Psingle; const lda:PMKL_INT64);winapi; external;









  procedure STBMV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:Psingle; const lda:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64);winapi; external;









  procedure STBSV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:Psingle; const lda:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64);winapi; external;







  procedure STPMV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:Psingle; const 
              x:Psingle; const incx:PMKL_INT64);winapi; external;







  procedure STPSV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:Psingle; const 
              x:Psingle; const incx:PMKL_INT64);winapi; external;








  procedure STRMV_64(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:Psingle; const 
              lda:PMKL_INT64; const b:Psingle; const incx:PMKL_INT64);winapi; external;








  procedure STRSV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:Psingle; const 
              lda:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64);winapi; external;













  procedure SGEM2VU_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const 
              x1:Psingle; const incx1:PMKL_INT64; const x2:Psingle; const incx2:PMKL_INT64; const beta:Psingle; const 
              y1:Psingle; const incy1:PMKL_INT64; const y2:Psingle; const incy2:PMKL_INT64);winapi; external;













  procedure CGBMV_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const kl:PMKL_INT64; const ku:PMKL_INT64; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const 
              beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT64);winapi; external;











  procedure CGEMV_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const 
              lda:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const beta:PMKL_Complex8; const y:PMKL_Complex8; const 
              incy:PMKL_INT64);winapi; external;









  procedure CGERC_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64; const 
              y:PMKL_Complex8; const incy:PMKL_INT64; const a:PMKL_Complex8; const lda:PMKL_INT64);winapi; external;









  procedure CGERU_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64; const 
              y:PMKL_Complex8; const incy:PMKL_INT64; const a:PMKL_Complex8; const lda:PMKL_INT64);winapi; external;











  procedure CHBMV_64(const uplo:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const 
              lda:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const beta:PMKL_Complex8; const y:PMKL_Complex8; const 
              incy:PMKL_INT64);winapi; external;










  procedure CHEMV_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const 
              x:PMKL_Complex8; const incx:PMKL_INT64; const beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT64);winapi; external;







  procedure CHER_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const x:PMKL_Complex8; const incx:PMKL_INT64; const 
              a:PMKL_Complex8; const lda:PMKL_INT64);winapi; external;









  procedure CHER2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64; const 
              y:PMKL_Complex8; const incy:PMKL_INT64; const a:PMKL_Complex8; const lda:PMKL_INT64);winapi; external;









  procedure CHPMV_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex8; const ap:PMKL_Complex8; const x:PMKL_Complex8; const 
              incx:PMKL_INT64; const beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT64);winapi; external;






  procedure CHPR_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const x:PMKL_Complex8; const incx:PMKL_INT64; const 
              ap:PMKL_Complex8);winapi; external;








  procedure CHPR2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64; const 
              y:PMKL_Complex8; const incy:PMKL_INT64; const ap:PMKL_Complex8);winapi; external;









  procedure CTBMV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;









  procedure CTBSV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;







  procedure CTPMV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:PMKL_Complex8; const 
              x:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;







  procedure CTPSV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:PMKL_Complex8; const 
              x:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;








  procedure CTRMV_64(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:PMKL_Complex8; const 
              lda:PMKL_INT64; const b:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;








  procedure CTRSV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:PMKL_Complex8; const 
              lda:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;













  procedure CGEM2VC_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const 
              x1:PMKL_Complex8; const incx1:PMKL_INT64; const x2:PMKL_Complex8; const incx2:PMKL_INT64; const beta:PMKL_Complex8; const 
              y1:PMKL_Complex8; const incy1:PMKL_INT64; const y2:PMKL_Complex8; const incy2:PMKL_INT64);winapi; external;











  procedure SCGEMV_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const a:Psingle; const 
              lda:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const beta:PMKL_Complex8; const y:PMKL_Complex8; const 
              incy:PMKL_INT64);winapi; external;













  procedure DGBMV_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const kl:PMKL_INT64; const ku:PMKL_INT64; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const 
              beta:Pdouble; const y:Pdouble; const incy:PMKL_INT64);winapi; external;











  procedure DGEMV_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const 
              lda:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const beta:Pdouble; const y:Pdouble; const 
              incy:PMKL_INT64);winapi; external;









  procedure DGER_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const 
              y:Pdouble; const incy:PMKL_INT64; const a:Pdouble; const lda:PMKL_INT64);winapi; external;











  procedure DSBMV_64(const uplo:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const 
              lda:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const beta:Pdouble; const y:Pdouble; const 
              incy:PMKL_INT64);winapi; external;









  procedure DSPMV_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const ap:Pdouble; const x:Pdouble; const 
              incx:PMKL_INT64; const beta:Pdouble; const y:Pdouble; const incy:PMKL_INT64);winapi; external;






  procedure DSPR_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const 
              ap:Pdouble);winapi; external;








  procedure DSPR2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const 
              y:Pdouble; const incy:PMKL_INT64; const ap:Pdouble);winapi; external;










  procedure DSYMV_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const 
              x:Pdouble; const incx:PMKL_INT64; const beta:Pdouble; const y:Pdouble; const incy:PMKL_INT64);winapi; external;







  procedure DSYR_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const 
              a:Pdouble; const lda:PMKL_INT64);winapi; external;









  procedure DSYR2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const 
              y:Pdouble; const incy:PMKL_INT64; const a:Pdouble; const lda:PMKL_INT64);winapi; external;









  procedure DTBMV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:Pdouble; const lda:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64);winapi; external;









  procedure DTBSV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:Pdouble; const lda:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64);winapi; external;







  procedure DTPMV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:Pdouble; const 
              x:Pdouble; const incx:PMKL_INT64);winapi; external;







  procedure DTPSV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:Pdouble; const 
              x:Pdouble; const incx:PMKL_INT64);winapi; external;








  procedure DTRMV_64(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:Pdouble; const 
              lda:PMKL_INT64; const b:Pdouble; const incx:PMKL_INT64);winapi; external;








  procedure DTRSV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:Pdouble; const 
              lda:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64);winapi; external;













  procedure DGEM2VU_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const 
              x1:Pdouble; const incx1:PMKL_INT64; const x2:Pdouble; const incx2:PMKL_INT64; const beta:Pdouble; const 
              y1:Pdouble; const incy1:PMKL_INT64; const y2:Pdouble; const incy2:PMKL_INT64);winapi; external;













  procedure ZGBMV_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const kl:PMKL_INT64; const ku:PMKL_INT64; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const 
              beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT64);winapi; external;











  procedure ZGEMV_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const 
              lda:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const beta:PMKL_Complex16; const y:PMKL_Complex16; const 
              incy:PMKL_INT64);winapi; external;









  procedure ZGERC_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64; const 
              y:PMKL_Complex16; const incy:PMKL_INT64; const a:PMKL_Complex16; const lda:PMKL_INT64);winapi; external;









  procedure ZGERU_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64; const 
              y:PMKL_Complex16; const incy:PMKL_INT64; const a:PMKL_Complex16; const lda:PMKL_INT64);winapi; external;











  procedure ZHBMV_64(const uplo:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const 
              lda:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const beta:PMKL_Complex16; const y:PMKL_Complex16; const 
              incy:PMKL_INT64);winapi; external;










  procedure ZHEMV_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const 
              x:PMKL_Complex16; const incx:PMKL_INT64; const beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT64);winapi; external;







  procedure ZHER_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const x:PMKL_Complex16; const incx:PMKL_INT64; const 
              a:PMKL_Complex16; const lda:PMKL_INT64);winapi; external;









  procedure ZHER2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64; const 
              y:PMKL_Complex16; const incy:PMKL_INT64; const a:PMKL_Complex16; const lda:PMKL_INT64);winapi; external;









  procedure ZHPMV_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex16; const ap:PMKL_Complex16; const x:PMKL_Complex16; const 
              incx:PMKL_INT64; const beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT64);winapi; external;






  procedure ZHPR_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const x:PMKL_Complex16; const incx:PMKL_INT64; const 
              ap:PMKL_Complex16);winapi; external;








  procedure ZHPR2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64; const 
              y:PMKL_Complex16; const incy:PMKL_INT64; const ap:PMKL_Complex16);winapi; external;









  procedure ZTBMV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;









  procedure ZTBSV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;







  procedure ZTPMV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:PMKL_Complex16; const 
              x:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;






  procedure ZTPSV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:PMKL_Complex16; const 
              x:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;








  procedure ZTRMV_64(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:PMKL_Complex16; const 
              lda:PMKL_INT64; const b:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;








  procedure ZTRSV_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:PMKL_Complex16; const 
              lda:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;













  procedure ZGEM2VC_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const 
              x1:PMKL_Complex16; const incx1:PMKL_INT64; const x2:PMKL_Complex16; const incx2:PMKL_INT64; const beta:PMKL_Complex16; const 
              y1:PMKL_Complex16; const incy1:PMKL_INT64; const y2:PMKL_Complex16; const incy2:PMKL_INT64);winapi; external;











  procedure DZGEMV_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const a:Pdouble; const 
              lda:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const beta:PMKL_Complex16; const y:PMKL_Complex16; const 
              incy:PMKL_INT64);winapi; external;

  { BLAS Level3  }












  procedure SGEMM_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const b:Psingle; const ldb:PMKL_INT64; const 
              beta:Psingle; const c:Psingle; const ldc:PMKL_INT64);winapi; external;





  function SGEMM_PACK_GET_SIZE_64(const identifier:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64):size_t;winapi; external;









  procedure SGEMM_PACK_64(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Psingle; const src:Psingle; const ld:PMKL_INT64; const dest:Psingle);winapi; external;












  procedure SGEMM_COMPUTE_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:Psingle; const lda:PMKL_INT64; const b:Psingle; const ldb:PMKL_INT64; const beta:Psingle; const 
              c:Psingle; const ldc:PMKL_INT64);winapi; external;















  procedure SGEMM_BATCH_64(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT64; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const 
              alpha_array:Psingle; const a_array:PPsingle; const lda_array:PMKL_INT64; const b_array:PPsingle; const ldb_array:PMKL_INT64; const 
              beta_array:Psingle; const c_array:PPsingle; const ldc_array:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;

















  procedure SGEMM_BATCH_STRIDED_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const stridea:PMKL_INT64; const b:Psingle; const 
              ldb:PMKL_INT64; const strideb:PMKL_INT64; const beta:Psingle; const c:Psingle; const ldc:PMKL_INT64; const 
              stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure SGEMMT_64(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const b:Psingle; const ldb:PMKL_INT64; const 
              beta:Psingle; const c:Psingle; const ldc:PMKL_INT64);winapi; external;












  procedure SSYMM_64(const side:Pchar; const uplo:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Psingle; const 
              a:Psingle; const lda:PMKL_INT64; const b:Psingle; const ldb:PMKL_INT64; const beta:Psingle; const 
              c:Psingle; const ldc:PMKL_INT64);winapi; external;












  procedure SSYR2K_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Psingle; const 
              a:Psingle; const lda:PMKL_INT64; const b:Psingle; const ldb:PMKL_INT64; const beta:Psingle; const 
              c:Psingle; const ldc:PMKL_INT64);winapi; external;










  procedure SSYRK_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Psingle; const 
              a:Psingle; const lda:PMKL_INT64; const beta:Psingle; const c:Psingle; const ldc:PMKL_INT64);winapi; external;













  procedure SSYRK_BATCH_STRIDED_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Psingle; const 
              a:Psingle; const lda:PMKL_INT64; const stridea:PMKL_INT64; const beta:Psingle; const c:Psingle; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;












  procedure SSYRK_BATCH_64(const uplo_array:Pchar; const trans_array:Pchar; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const alpha_array:Psingle; const 
              a_array:PPsingle; const lda_array:PMKL_INT64; const beta_array:Psingle; const c_array:PPsingle; const ldc_array:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;











  procedure STRMM_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const b:Psingle; const 
              ldb:PMKL_INT64);winapi; external;











  procedure STRSM_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const b:Psingle; const 
              ldb:PMKL_INT64);winapi; external;













  procedure STRSM_BATCH_64(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT64; const 
              n_array:PMKL_INT64; const alpha_array:Psingle; const a_array:PPsingle; const lda_array:PMKL_INT64; const b_array:PPsingle; const 
              ldb:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;














  procedure STRSM_BATCH_STRIDED_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const stridea:PMKL_INT64; const 
              b:Psingle; const ldb:PMKL_INT64; const strideb:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure CGEMM_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;















  procedure CGEMM_BATCH_64(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT64; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const 
              alpha_array:PMKL_Complex8; const a_array:PPMKL_Complex8; const lda_array:PMKL_INT64; const b_array:PPMKL_Complex8; const ldb_array:PMKL_INT64; const 
              beta_array:PMKL_Complex8; const c_array:PPMKL_Complex8; const ldc_array:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;

















  procedure CGEMM_BATCH_STRIDED_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const stridea:PMKL_INT64; const b:PMKL_Complex8; const 
              ldb:PMKL_INT64; const strideb:PMKL_INT64; const beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT64; const 
              stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure SCGEMM_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex8; const a:Psingle; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;













  procedure CGEMM3M_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;















  procedure CGEMM3M_BATCH_64(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT64; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const 
              alpha_array:PMKL_Complex8; const a_array:PPMKL_Complex8; const lda_array:PMKL_INT64; const b_array:PPMKL_Complex8; const ldb_array:PMKL_INT64; const 
              beta_array:PMKL_Complex8; const c_array:PPMKL_Complex8; const ldc_array:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;













  procedure CGEMMT_64(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;













  procedure CTRSM_BATCH_64(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT64; const 
              n_array:PMKL_INT64; const alpha_array:PMKL_Complex8; const a_array:PPMKL_Complex8; const lda_array:PMKL_INT64; const b_array:PPMKL_Complex8; const 
              ldb:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;














  procedure CTRSM_BATCH_STRIDED_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const stridea:PMKL_INT64; const 
              b:PMKL_Complex8; const ldb:PMKL_INT64; const strideb:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;












  procedure CHEMM_64(const side:Pchar; const uplo:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const beta:PMKL_Complex8; const 
              c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;












  procedure CHER2K_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const beta:Psingle; const 
              c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;










  procedure CHERK_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Psingle; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const beta:Psingle; const c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;












  procedure CSYMM_64(const side:Pchar; const uplo:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const beta:PMKL_Complex8; const 
              c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;












  procedure CSYR2K_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const beta:PMKL_Complex8; const 
              c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;










  procedure CSYRK_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;












  procedure CSYRK_BATCH_64(const uplo_array:Pchar; const trans_array:Pchar; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const alpha_array:PMKL_Complex8; const 
              a_array:PPMKL_Complex8; const lda_array:PMKL_INT64; const beta_array:PMKL_Complex8; const c_array:PPMKL_Complex8; const ldc_array:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;













  procedure CSYRK_BATCH_STRIDED_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const stridea:PMKL_INT64; const beta:PMKL_Complex8; const c:PMKL_Complex8; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;











  procedure CTRMM_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const 
              ldb:PMKL_INT64);winapi; external;











  procedure CTRSM_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const 
              ldb:PMKL_INT64);winapi; external;













  procedure DGEMM_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const b:Pdouble; const ldb:PMKL_INT64; const 
              beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT64);winapi; external;





  function DGEMM_PACK_GET_SIZE_64(const identifier:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64):size_t;winapi; external;









  procedure DGEMM_PACK_64(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Pdouble; const src:Pdouble; const ld:PMKL_INT64; const dest:Pdouble);winapi; external;












  procedure DGEMM_COMPUTE_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:Pdouble; const lda:PMKL_INT64; const b:Pdouble; const ldb:PMKL_INT64; const beta:Pdouble; const 
              c:Pdouble; const ldc:PMKL_INT64);winapi; external;















  procedure DGEMM_BATCH_64(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT64; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const 
              alpha_array:Pdouble; const a_array:PPdouble; const lda_array:PMKL_INT64; const b_array:PPdouble; const ldb_array:PMKL_INT64; const 
              beta_array:Pdouble; const c_array:PPdouble; const ldc_array:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;

















  procedure DGEMM_BATCH_STRIDED_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const stridea:PMKL_INT64; const b:Pdouble; const 
              ldb:PMKL_INT64; const strideb:PMKL_INT64; const beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT64; const 
              stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure DGEMMT_64(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const b:Pdouble; const ldb:PMKL_INT64; const 
              beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT64);winapi; external;












  procedure DSYMM_64(const side:Pchar; const uplo:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Pdouble; const 
              a:Pdouble; const lda:PMKL_INT64; const b:Pdouble; const ldb:PMKL_INT64; const beta:Pdouble; const 
              c:Pdouble; const ldc:PMKL_INT64);winapi; external;












  procedure DSYR2K_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Pdouble; const 
              a:Pdouble; const lda:PMKL_INT64; const b:Pdouble; const ldb:PMKL_INT64; const beta:Pdouble; const 
              c:Pdouble; const ldc:PMKL_INT64);winapi; external;










  procedure DSYRK_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Pdouble; const 
              a:Pdouble; const lda:PMKL_INT64; const beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT64);winapi; external;












  procedure DSYRK_BATCH_64(const uplo_array:Pchar; const trans_array:Pchar; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const alpha_array:Pdouble; const 
              a_array:PPdouble; const lda_array:PMKL_INT64; const beta_array:Pdouble; const c_array:PPdouble; const ldc_array:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;













  procedure DSYRK_BATCH_STRIDED_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Pdouble; const 
              a:Pdouble; const lda:PMKL_INT64; const stridea:PMKL_INT64; const beta:Pdouble; const c:Pdouble; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;











  procedure DTRMM_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const b:Pdouble; const 
              ldb:PMKL_INT64);winapi; external;











  procedure DTRSM_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const b:Pdouble; const 
              ldb:PMKL_INT64);winapi; external;













  procedure DTRSM_BATCH_64(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT64; const 
              n_array:PMKL_INT64; const alpha_array:Pdouble; const a_array:PPdouble; const lda_array:PMKL_INT64; const b_array:PPdouble; const 
              ldb:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;














  procedure DTRSM_BATCH_STRIDED_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const stridea:PMKL_INT64; const 
              b:Pdouble; const ldb:PMKL_INT64; const strideb:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure ZGEMM_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;















  procedure ZGEMM_BATCH_64(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT64; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const 
              alpha_array:PMKL_Complex16; const a_array:PPMKL_Complex16; const lda_array:PMKL_INT64; const b_array:PPMKL_Complex16; const ldb_array:PMKL_INT64; const 
              beta_array:PMKL_Complex16; const c_array:PPMKL_Complex16; const ldc_array:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;

















  procedure ZGEMM_BATCH_STRIDED_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const stridea:PMKL_INT64; const b:PMKL_Complex16; const 
              ldb:PMKL_INT64; const strideb:PMKL_INT64; const beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT64; const 
              stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure DZGEMM_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex16; const a:Pdouble; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;













  procedure ZGEMM3M_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;















  procedure ZGEMM3M_BATCH_64(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT64; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const 
              alpha_array:PMKL_Complex16; const a_array:PPMKL_Complex16; const lda_array:PMKL_INT64; const b_array:PPMKL_Complex16; const ldb_array:PMKL_INT64; const 
              beta_array:PMKL_Complex16; const c_array:PPMKL_Complex16; const ldc_array:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;













  procedure ZGEMMT_64(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;












  procedure ZHEMM_64(const side:Pchar; const uplo:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const beta:PMKL_Complex16; const 
              c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;












  procedure ZHER2K_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const beta:Pdouble; const 
              c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;










  procedure ZHERK_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Pdouble; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const beta:Pdouble; const c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;












  procedure ZSYMM_64(const side:Pchar; const uplo:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const beta:PMKL_Complex16; const 
              c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;












  procedure ZSYR2K_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const beta:PMKL_Complex16; const 
              c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;










  procedure ZSYRK_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;












  procedure ZSYRK_BATCH_64(const uplo_array:Pchar; const trans_array:Pchar; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const alpha_array:PMKL_Complex16; const 
              a_array:PPMKL_Complex16; const lda_array:PMKL_INT64; const beta_array:PMKL_Complex16; const c_array:PPMKL_Complex16; const ldc_array:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;













  procedure ZSYRK_BATCH_STRIDED_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const stridea:PMKL_INT64; const beta:PMKL_Complex16; const c:PMKL_Complex16; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;











  procedure ZTRMM_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const 
              ldb:PMKL_INT64);winapi; external;











  procedure ZTRSM_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const 
              ldb:PMKL_INT64);winapi; external;













  procedure ZTRSM_BATCH_64(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT64; const 
              n_array:PMKL_INT64; const alpha_array:PMKL_Complex16; const a_array:PPMKL_Complex16; const lda_array:PMKL_INT64; const b_array:PPMKL_Complex16; const 
              ldb:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;














  procedure ZTRSM_BATCH_STRIDED_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const stridea:PMKL_INT64; const 
              b:PMKL_Complex16; const ldb:PMKL_INT64; const strideb:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;

















  procedure GEMM_S8U8S32_64(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const 
              k:PMKL_INT64; const alpha:Psingle; const a:PMKL_INT8; const lda:PMKL_INT64; const ao:PMKL_INT8; const 
              b:PMKL_UINT8; const ldb:PMKL_INT64; const bo:PMKL_INT8; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT64; const co:PMKL_INT32);winapi; external;

















  procedure GEMM_S16S16S32_64(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const 
              k:PMKL_INT64; const alpha:Psingle; const a:PMKL_INT16; const lda:PMKL_INT64; const ao:PMKL_INT16; const 
              b:PMKL_INT16; const ldb:PMKL_INT64; const bo:PMKL_INT16; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT64; const co:PMKL_INT32);winapi; external;





  function GEMM_S8U8S32_PACK_GET_SIZE_64(const identifier:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64):size_t;winapi; external;





  function GEMM_S16S16S32_PACK_GET_SIZE_64(const identifier:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64):size_t;winapi; external;








  procedure GEMM_S8U8S32_PACK_64(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              src:pointer; const ld:PMKL_INT64; const dest:pointer);winapi; external;








  procedure GEMM_S16S16S32_PACK_64(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              src:PMKL_INT16; const ld:PMKL_INT64; const dest:PMKL_INT16);winapi; external;

















  procedure GEMM_S8U8S32_COMPUTE_64(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const 
              k:PMKL_INT64; const alpha:Psingle; const a:PMKL_INT8; const lda:PMKL_INT64; const ao:PMKL_INT8; const 
              b:PMKL_UINT8; const ldb:PMKL_INT64; const bo:PMKL_INT8; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT64; const co:PMKL_INT32);winapi; external;

















  procedure GEMM_S16S16S32_COMPUTE_64(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const 
              k:PMKL_INT64; const alpha:Psingle; const a:PMKL_INT16; const lda:PMKL_INT64; const ao:PMKL_INT16; const 
              b:PMKL_INT16; const ldb:PMKL_INT64; const bo:PMKL_INT16; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT64; const co:PMKL_INT32);winapi; external;













  procedure HGEMM_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_F16; const a:PMKL_F16; const lda:PMKL_INT64; const b:PMKL_F16; const ldb:PMKL_INT64; const 
              beta:PMKL_F16; const c:PMKL_F16; const ldc:PMKL_INT64);winapi; external;





  function HGEMM_PACK_GET_SIZE_64(const identifier:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64):size_t;winapi; external;









  procedure HGEMM_PACK_64(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_F16; const src:PMKL_F16; const ld:PMKL_INT64; const dest:PMKL_F16);winapi; external;












  procedure HGEMM_COMPUTE_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:PMKL_F16; const lda:PMKL_INT64; const b:PMKL_F16; const ldb:PMKL_INT64; const beta:PMKL_F16; const 
              c:PMKL_F16; const ldc:PMKL_INT64);winapi; external;
{$endif}
  { Lower case declaration  }
  { BLAS Level1  }
{$ifdef LOWERCASE_DECL}
  function scabs1_64(const c:PMKL_Complex8):single;winapi; external;




  function sasum_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64):single;winapi; external;






  procedure saxpy_64(const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const 
              incy:PMKL_INT64);winapi; external;







  procedure saxpby_64(const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const beta:Psingle; const 
              y:Psingle; const incy:PMKL_INT64);winapi; external;





  procedure saxpyi_64(const nz:PMKL_INT64; const a:Psingle; const x:Psingle; const indx:PMKL_INT64; const y:Psingle);winapi; external;




  function scasum_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64):single;winapi; external;




  function scnrm2_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64):single;winapi; external;





  procedure scopy_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const incy:PMKL_INT64);winapi; external;






  function sdot_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const incy:PMKL_INT64):single;winapi; external;





  function sdoti_64(const nz:PMKL_INT64; const x:Psingle; const indx:PMKL_INT64; const y:Psingle):single;winapi; external;







  function sdsdot_64(const n:PMKL_INT64; const sb:Psingle; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const 
             incy:PMKL_INT64):single;winapi; external;




  procedure sgthr_64(const nz:PMKL_INT64; const y:Psingle; const x:Psingle; const indx:PMKL_INT64);winapi; external;



  procedure sgthrz_64(const nz:PMKL_INT64; const y:Psingle; const x:Psingle; const indx:PMKL_INT64);winapi; external;




  function snrm2_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64):single;winapi; external;






  procedure srot_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const incy:PMKL_INT64; const 
              c:Psingle; const s:Psingle);winapi; external;

  procedure srotg_64(const a:Psingle; const b:Psingle; const c:Psingle; const s:Psingle);winapi; external;





  procedure sroti_64(const nz:PMKL_INT64; const x:Psingle; const indx:PMKL_INT64; const y:Psingle; const c:Psingle; const 
              s:Psingle);winapi; external;





  procedure srotm_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const incy:PMKL_INT64; const 
              param:Psingle);winapi; external;


  procedure srotmg_64(const d1:Psingle; const d2:Psingle; const x1:Psingle; const y1:Psingle; const param:Psingle);winapi; external;




  procedure sscal_64(const n:PMKL_INT64; const a:Psingle; const x:Psingle; const incx:PMKL_INT64);winapi; external;




  procedure ssctr_64(const nz:PMKL_INT64; const x:Psingle; const indx:PMKL_INT64; const y:Psingle);winapi; external;




  procedure sswap_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const incy:PMKL_INT64);winapi; external;




  function isamax_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64):MKL_INT64;winapi; external;




  function isamin_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64):MKL_INT64;winapi; external;






  procedure caxpy_64(const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64; const y:PMKL_Complex8; const 
              incy:PMKL_INT64);winapi; external;







  procedure caxpby_64(const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64; const beta:PMKL_Complex8; const 
              y:PMKL_Complex8; const incy:PMKL_INT64);winapi; external;





  procedure caxpyi_64(const nz:PMKL_INT64; const a:PMKL_Complex8; const x:PMKL_Complex8; const indx:PMKL_INT64; const y:PMKL_Complex8);winapi; external;





  procedure ccopy_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const y:PMKL_Complex8; const incy:PMKL_INT64);winapi; external;






  procedure cdotc_64(const pres:PMKL_Complex8; const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const y:PMKL_Complex8; const 
              incy:PMKL_INT64);winapi; external;





  procedure cdotci_64(const pres:PMKL_Complex8; const nz:PMKL_INT64; const x:PMKL_Complex8; const indx:PMKL_INT64; const y:PMKL_Complex8);winapi; external;






  procedure cdotu_64(const pres:PMKL_Complex8; const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const y:PMKL_Complex8; const 
              incy:PMKL_INT64);winapi; external;





  procedure cdotui_64(const pres:PMKL_Complex8; const nz:PMKL_INT64; const x:PMKL_Complex8; const indx:PMKL_INT64; const y:PMKL_Complex8);winapi; external;




  procedure cgthr_64(const nz:PMKL_INT64; const y:PMKL_Complex8; const x:PMKL_Complex8; const indx:PMKL_INT64);winapi; external;



  procedure cgthrz_64(const nz:PMKL_INT64; const y:PMKL_Complex8; const x:PMKL_Complex8; const indx:PMKL_INT64);winapi; external;


  procedure crotg_64(const a:PMKL_Complex8; const b:PMKL_Complex8; const c:Psingle; const s:PMKL_Complex8);winapi; external;




  procedure cscal_64(const n:PMKL_INT64; const a:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;




  procedure csctr_64(const nz:PMKL_INT64; const x:PMKL_Complex8; const indx:PMKL_INT64; const y:PMKL_Complex8);winapi; external;






  procedure csrot_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const y:PMKL_Complex8; const incy:PMKL_INT64; const 
              c:Psingle; const s:Psingle);winapi; external;




  procedure csscal_64(const n:PMKL_INT64; const a:Psingle; const x:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;




  procedure cswap_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const y:PMKL_Complex8; const incy:PMKL_INT64);winapi; external;




  function icamax_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64):MKL_INT64;winapi; external;




  function icamin_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64):MKL_INT64;winapi; external;


  function dcabs1_64(const z:PMKL_Complex16):double;winapi; external;




  function dasum_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64):double;winapi; external;






  procedure daxpy_64(const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const y:Pdouble; const 
              incy:PMKL_INT64);winapi; external;







  procedure daxpby_64(const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const beta:Pdouble; const 
              y:Pdouble; const incy:PMKL_INT64);winapi; external;





  procedure daxpyi_64(const nz:PMKL_INT64; const a:Pdouble; const x:Pdouble; const indx:PMKL_INT64; const y:Pdouble);winapi; external;





  procedure dcopy_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const y:Pdouble; const incy:PMKL_INT64);winapi; external;






  function ddot_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const y:Pdouble; const incy:PMKL_INT64):double;winapi; external;






  function dsdot_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const y:Psingle; const incy:PMKL_INT64):double;winapi; external;





  function ddoti_64(const nz:PMKL_INT64; const x:Pdouble; const indx:PMKL_INT64; const y:Pdouble):double;winapi; external;




  procedure dgthr_64(const nz:PMKL_INT64; const y:Pdouble; const x:Pdouble; const indx:PMKL_INT64);winapi; external;



  procedure dgthrz_64(const nz:PMKL_INT64; const y:Pdouble; const x:Pdouble; const indx:PMKL_INT64);winapi; external;




  function dnrm2_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64):double;winapi; external;






  procedure drot_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const y:Pdouble; const incy:PMKL_INT64; const 
              c:Pdouble; const s:Pdouble);winapi; external;

  procedure drotg_64(const a:Pdouble; const b:Pdouble; const c:Pdouble; const s:Pdouble);winapi; external;





  procedure droti_64(const nz:PMKL_INT64; const x:Pdouble; const indx:PMKL_INT64; const y:Pdouble; const c:Pdouble; const 
              s:Pdouble);winapi; external;





  procedure drotm_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const y:Pdouble; const incy:PMKL_INT64; const 
              param:Pdouble);winapi; external;


  procedure drotmg_64(const d1:Pdouble; const d2:Pdouble; const x1:Pdouble; const y1:Pdouble; const param:Pdouble);winapi; external;




  procedure dscal_64(const n:PMKL_INT64; const a:Pdouble; const x:Pdouble; const incx:PMKL_INT64);winapi; external;




  procedure dsctr_64(const nz:PMKL_INT64; const x:Pdouble; const indx:PMKL_INT64; const y:Pdouble);winapi; external;




  procedure dswap_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const y:Pdouble; const incy:PMKL_INT64);winapi; external;




  function dzasum_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64):double;winapi; external;




  function dznrm2_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64):double;winapi; external;




  function idamax_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64):MKL_INT64;winapi; external;




  function idamin_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64):MKL_INT64;winapi; external;






  procedure zaxpy_64(const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64; const y:PMKL_Complex16; const 
              incy:PMKL_INT64);winapi; external;







  procedure zaxpby_64(const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64; const beta:PMKL_Complex16; const 
              y:PMKL_Complex16; const incy:PMKL_INT64);winapi; external;





  procedure zaxpyi_64(const nz:PMKL_INT64; const a:PMKL_Complex16; const x:PMKL_Complex16; const indx:PMKL_INT64; const y:PMKL_Complex16);winapi; external;





  procedure zcopy_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const y:PMKL_Complex16; const incy:PMKL_INT64);winapi; external;






  procedure zdotc_64(const pres:PMKL_Complex16; const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const y:PMKL_Complex16; const 
              incy:PMKL_INT64);winapi; external;





  procedure zdotci_64(const pres:PMKL_Complex16; const nz:PMKL_INT64; const x:PMKL_Complex16; const indx:PMKL_INT64; const y:PMKL_Complex16);winapi; external;






  procedure zdotu_64(const pres:PMKL_Complex16; const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const y:PMKL_Complex16; const 
              incy:PMKL_INT64);winapi; external;





  procedure zdotui_64(const pres:PMKL_Complex16; const nz:PMKL_INT64; const x:PMKL_Complex16; const indx:PMKL_INT64; const y:PMKL_Complex16);winapi; external;






  procedure zdrot_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const y:PMKL_Complex16; const incy:PMKL_INT64; const 
              c:Pdouble; const s:Pdouble);winapi; external;




  procedure zdscal_64(const n:PMKL_INT64; const a:Pdouble; const x:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;




  procedure zgthr_64(const nz:PMKL_INT64; const y:PMKL_Complex16; const x:PMKL_Complex16; const indx:PMKL_INT64);winapi; external;



  procedure zgthrz_64(const nz:PMKL_INT64; const y:PMKL_Complex16; const x:PMKL_Complex16; const indx:PMKL_INT64);winapi; external;


  procedure zrotg_64(const a:PMKL_Complex16; const b:PMKL_Complex16; const c:Pdouble; const s:PMKL_Complex16);winapi; external;




  procedure zscal_64(const n:PMKL_INT64; const a:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;




  procedure zsctr_64(const nz:PMKL_INT64; const x:PMKL_Complex16; const indx:PMKL_INT64; const y:PMKL_Complex16);winapi; external;




  procedure zswap_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const y:PMKL_Complex16; const incy:PMKL_INT64);winapi; external;




  function izamax_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64):MKL_INT64;winapi; external;




  function izamin_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64):MKL_INT64;winapi; external;

  { blas level2  }












  procedure sgbmv_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const kl:PMKL_INT64; const ku:PMKL_INT64; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const 
              beta:Psingle; const y:Psingle; const incy:PMKL_INT64);winapi; external;











  procedure sgemv_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Psingle; const a:Psingle; const 
              lda:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const beta:Psingle; const y:Psingle; const 
              incy:PMKL_INT64);winapi; external;









  procedure sger_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const 
              y:Psingle; const incy:PMKL_INT64; const a:Psingle; const lda:PMKL_INT64);winapi; external;











  procedure ssbmv_64(const uplo:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Psingle; const a:Psingle; const 
              lda:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const beta:Psingle; const y:Psingle; const 
              incy:PMKL_INT64);winapi; external;









  procedure sspmv_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const ap:Psingle; const x:Psingle; const 
              incx:PMKL_INT64; const beta:Psingle; const y:Psingle; const incy:PMKL_INT64);winapi; external;






  procedure sspr_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const 
              ap:Psingle);winapi; external;








  procedure sspr2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const 
              y:Psingle; const incy:PMKL_INT64; const ap:Psingle);winapi; external;










  procedure ssymv_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const 
              x:Psingle; const incx:PMKL_INT64; const beta:Psingle; const y:Psingle; const incy:PMKL_INT64);winapi; external;







  procedure ssyr_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const 
              a:Psingle; const lda:PMKL_INT64);winapi; external;









  procedure ssyr2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const 
              y:Psingle; const incy:PMKL_INT64; const a:Psingle; const lda:PMKL_INT64);winapi; external;









  procedure stbmv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:Psingle; const lda:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64);winapi; external;









  procedure stbsv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:Psingle; const lda:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64);winapi; external;







  procedure stpmv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:Psingle; const 
              x:Psingle; const incx:PMKL_INT64);winapi; external;







  procedure stpsv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:Psingle; const 
              x:Psingle; const incx:PMKL_INT64);winapi; external;








  procedure strmv_64(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:Psingle; const 
              lda:PMKL_INT64; const b:Psingle; const incx:PMKL_INT64);winapi; external;








  procedure strsv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:Psingle; const 
              lda:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64);winapi; external;













  procedure sgem2vu_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const 
              x1:Psingle; const incx1:PMKL_INT64; const x2:Psingle; const incx2:PMKL_INT64; const beta:Psingle; const 
              y1:Psingle; const incy1:PMKL_INT64; const y2:Psingle; const incy2:PMKL_INT64);winapi; external;













  procedure cgbmv_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const kl:PMKL_INT64; const ku:PMKL_INT64; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const 
              beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT64);winapi; external;











  procedure cgemv_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const 
              lda:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const beta:PMKL_Complex8; const y:PMKL_Complex8; const 
              incy:PMKL_INT64);winapi; external;









  procedure cgerc_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64; const 
              y:PMKL_Complex8; const incy:PMKL_INT64; const a:PMKL_Complex8; const lda:PMKL_INT64);winapi; external;









  procedure cgeru_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64; const 
              y:PMKL_Complex8; const incy:PMKL_INT64; const a:PMKL_Complex8; const lda:PMKL_INT64);winapi; external;











  procedure chbmv_64(const uplo:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const 
              lda:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const beta:PMKL_Complex8; const y:PMKL_Complex8; const 
              incy:PMKL_INT64);winapi; external;










  procedure chemv_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const 
              x:PMKL_Complex8; const incx:PMKL_INT64; const beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT64);winapi; external;







  procedure cher_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const x:PMKL_Complex8; const incx:PMKL_INT64; const 
              a:PMKL_Complex8; const lda:PMKL_INT64);winapi; external;









  procedure cher2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64; const 
              y:PMKL_Complex8; const incy:PMKL_INT64; const a:PMKL_Complex8; const lda:PMKL_INT64);winapi; external;









  procedure chpmv_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex8; const ap:PMKL_Complex8; const x:PMKL_Complex8; const 
              incx:PMKL_INT64; const beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT64);winapi; external;






  procedure chpr_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Psingle; const x:PMKL_Complex8; const incx:PMKL_INT64; const 
              ap:PMKL_Complex8);winapi; external;








  procedure chpr2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64; const 
              y:PMKL_Complex8; const incy:PMKL_INT64; const ap:PMKL_Complex8);winapi; external;









  procedure ctbmv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;









  procedure ctbsv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;







  procedure ctpmv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:PMKL_Complex8; const 
              x:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;







  procedure ctpsv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:PMKL_Complex8; const 
              x:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;








  procedure ctrmv_64(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:PMKL_Complex8; const 
              lda:PMKL_INT64; const b:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;








  procedure ctrsv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:PMKL_Complex8; const 
              lda:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64);winapi; external;













  procedure cgem2vc_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const 
              x1:PMKL_Complex8; const incx1:PMKL_INT64; const x2:PMKL_Complex8; const incx2:PMKL_INT64; const beta:PMKL_Complex8; const 
              y1:PMKL_Complex8; const incy1:PMKL_INT64; const y2:PMKL_Complex8; const incy2:PMKL_INT64);winapi; external;











  procedure scgemv_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const a:Psingle; const 
              lda:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const beta:PMKL_Complex8; const y:PMKL_Complex8; const 
              incy:PMKL_INT64);winapi; external;













  procedure dgbmv_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const kl:PMKL_INT64; const ku:PMKL_INT64; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const 
              beta:Pdouble; const y:Pdouble; const incy:PMKL_INT64);winapi; external;











  procedure dgemv_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const 
              lda:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const beta:Pdouble; const y:Pdouble; const 
              incy:PMKL_INT64);winapi; external;









  procedure dger_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const 
              y:Pdouble; const incy:PMKL_INT64; const a:Pdouble; const lda:PMKL_INT64);winapi; external;











  procedure dsbmv_64(const uplo:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const 
              lda:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const beta:Pdouble; const y:Pdouble; const 
              incy:PMKL_INT64);winapi; external;









  procedure dspmv_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const ap:Pdouble; const x:Pdouble; const 
              incx:PMKL_INT64; const beta:Pdouble; const y:Pdouble; const incy:PMKL_INT64);winapi; external;






  procedure dspr_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const 
              ap:Pdouble);winapi; external;








  procedure dspr2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const 
              y:Pdouble; const incy:PMKL_INT64; const ap:Pdouble);winapi; external;










  procedure dsymv_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const 
              x:Pdouble; const incx:PMKL_INT64; const beta:Pdouble; const y:Pdouble; const incy:PMKL_INT64);winapi; external;







  procedure dsyr_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const 
              a:Pdouble; const lda:PMKL_INT64);winapi; external;









  procedure dsyr2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const 
              y:Pdouble; const incy:PMKL_INT64; const a:Pdouble; const lda:PMKL_INT64);winapi; external;









  procedure dtbmv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:Pdouble; const lda:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64);winapi; external;









  procedure dtbsv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:Pdouble; const lda:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64);winapi; external;







  procedure dtpmv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:Pdouble; const 
              x:Pdouble; const incx:PMKL_INT64);winapi; external;







  procedure dtpsv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:Pdouble; const 
              x:Pdouble; const incx:PMKL_INT64);winapi; external;








  procedure dtrmv_64(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:Pdouble; const 
              lda:PMKL_INT64; const b:Pdouble; const incx:PMKL_INT64);winapi; external;








  procedure dtrsv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:Pdouble; const 
              lda:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64);winapi; external;













  procedure dgem2vu_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const 
              x1:Pdouble; const incx1:PMKL_INT64; const x2:Pdouble; const incx2:PMKL_INT64; const beta:Pdouble; const 
              y1:Pdouble; const incy1:PMKL_INT64; const y2:Pdouble; const incy2:PMKL_INT64);winapi; external;













  procedure zgbmv_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const kl:PMKL_INT64; const ku:PMKL_INT64; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const 
              beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT64);winapi; external;











  procedure zgemv_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const 
              lda:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const beta:PMKL_Complex16; const y:PMKL_Complex16; const 
              incy:PMKL_INT64);winapi; external;









  procedure zgerc_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64; const 
              y:PMKL_Complex16; const incy:PMKL_INT64; const a:PMKL_Complex16; const lda:PMKL_INT64);winapi; external;









  procedure zgeru_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64; const 
              y:PMKL_Complex16; const incy:PMKL_INT64; const a:PMKL_Complex16; const lda:PMKL_INT64);winapi; external;











  procedure zhbmv_64(const uplo:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const 
              lda:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const beta:PMKL_Complex16; const y:PMKL_Complex16; const 
              incy:PMKL_INT64);winapi; external;










  procedure zhemv_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const 
              x:PMKL_Complex16; const incx:PMKL_INT64; const beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT64);winapi; external;







  procedure zher_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const x:PMKL_Complex16; const incx:PMKL_INT64; const 
              a:PMKL_Complex16; const lda:PMKL_INT64);winapi; external;









  procedure zher2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64; const 
              y:PMKL_Complex16; const incy:PMKL_INT64; const a:PMKL_Complex16; const lda:PMKL_INT64);winapi; external;









  procedure zhpmv_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex16; const ap:PMKL_Complex16; const x:PMKL_Complex16; const 
              incx:PMKL_INT64; const beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT64);winapi; external;






  procedure zhpr_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:Pdouble; const x:PMKL_Complex16; const incx:PMKL_INT64; const 
              ap:PMKL_Complex16);winapi; external;








  procedure zhpr2_64(const uplo:Pchar; const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64; const 
              y:PMKL_Complex16; const incy:PMKL_INT64; const ap:PMKL_Complex16);winapi; external;









  procedure ztbmv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;









  procedure ztbsv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;







  procedure ztpmv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:PMKL_Complex16; const 
              x:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;







  procedure ztpsv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const ap:PMKL_Complex16; const 
              x:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;








  procedure ztrmv_64(const uplo:Pchar; const transa:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:PMKL_Complex16; const 
              lda:PMKL_INT64; const b:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;








  procedure ztrsv_64(const uplo:Pchar; const trans:Pchar; const diag:Pchar; const n:PMKL_INT64; const a:PMKL_Complex16; const 
              lda:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64);winapi; external;













  procedure zgem2vc_64(const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const 
              x1:PMKL_Complex16; const incx1:PMKL_INT64; const x2:PMKL_Complex16; const incx2:PMKL_INT64; const beta:PMKL_Complex16; const 
              y1:PMKL_Complex16; const incy1:PMKL_INT64; const y2:PMKL_Complex16; const incy2:PMKL_INT64);winapi; external;











  procedure dzgemv_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const a:Pdouble; const 
              lda:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const beta:PMKL_Complex16; const y:PMKL_Complex16; const 
              incy:PMKL_INT64);winapi; external;

  { blas level3  }












  procedure sgemm_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const b:Psingle; const ldb:PMKL_INT64; const 
              beta:Psingle; const c:Psingle; const ldc:PMKL_INT64);winapi; external;





  function sgemm_pack_get_size_64(const identifier:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64):size_t;winapi; external;









  procedure sgemm_pack_64(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Psingle; const src:Psingle; const ld:PMKL_INT64; const dest:Psingle);winapi; external;












  procedure sgemm_compute_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:Psingle; const lda:PMKL_INT64; const b:Psingle; const ldb:PMKL_INT64; const beta:Psingle; const 
              c:Psingle; const ldc:PMKL_INT64);winapi; external;















  procedure sgemm_batch_64(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT64; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const 
              alpha_array:Psingle; const a_array:PPsingle; const lda_array:PMKL_INT64; const b_array:PPsingle; const ldb_array:PMKL_INT64; const 
              beta_array:Psingle; const c_array:PPsingle; const ldc_array:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;

















  procedure sgemm_batch_strided_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const stridea:PMKL_INT64; const b:Psingle; const 
              ldb:PMKL_INT64; const strideb:PMKL_INT64; const beta:Psingle; const c:Psingle; const ldc:PMKL_INT64; const 
              stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure sgemmt_64(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const b:Psingle; const ldb:PMKL_INT64; const 
              beta:Psingle; const c:Psingle; const ldc:PMKL_INT64);winapi; external;












  procedure ssymm_64(const side:Pchar; const uplo:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Psingle; const 
              a:Psingle; const lda:PMKL_INT64; const b:Psingle; const ldb:PMKL_INT64; const beta:Psingle; const 
              c:Psingle; const ldc:PMKL_INT64);winapi; external;












  procedure ssyr2k_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Psingle; const 
              a:Psingle; const lda:PMKL_INT64; const b:Psingle; const ldb:PMKL_INT64; const beta:Psingle; const 
              c:Psingle; const ldc:PMKL_INT64);winapi; external;










  procedure ssyrk_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Psingle; const 
              a:Psingle; const lda:PMKL_INT64; const beta:Psingle; const c:Psingle; const ldc:PMKL_INT64);winapi; external;












  procedure ssyrk_batch_64(const uplo_array:Pchar; const trans_array:Pchar; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const alpha_array:Psingle; const 
              a_array:PPsingle; const lda_array:PMKL_INT64; const beta_array:Psingle; const c_array:PPsingle; const ldc_array:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;













  procedure ssyrk_batch_strided_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Psingle; const 
              a:Psingle; const lda:PMKL_INT64; const stridea:PMKL_INT64; const beta:Psingle; const c:Psingle; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;











  procedure strmm_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const b:Psingle; const 
              ldb:PMKL_INT64);winapi; external;











  procedure strsm_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const b:Psingle; const 
              ldb:PMKL_INT64);winapi; external;













  procedure strsm_batch_64(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT64; const 
              n_array:PMKL_INT64; const alpha_array:Psingle; const a_array:PPsingle; const lda_array:PMKL_INT64; const b_array:PPsingle; const 
              ldb:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;














  procedure strsm_batch_strided_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:Psingle; const a:Psingle; const lda:PMKL_INT64; const stridea:PMKL_INT64; const 
              b:Psingle; const ldb:PMKL_INT64; const strideb:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure cgemm_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;















  procedure cgemm_batch_64(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT64; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const 
              alpha_array:PMKL_Complex8; const a_array:PPMKL_Complex8; const lda_array:PMKL_INT64; const b_array:PPMKL_Complex8; const ldb_array:PMKL_INT64; const 
              beta_array:PMKL_Complex8; const c_array:PPMKL_Complex8; const ldc_array:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;

















  procedure cgemm_batch_strided_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const stridea:PMKL_INT64; const b:PMKL_Complex8; const 
              ldb:PMKL_INT64; const strideb:PMKL_INT64; const beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT64; const 
              stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure scgemm_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex8; const a:Psingle; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;













  procedure cgemm3m_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;















  procedure cgemm3m_batch_64(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT64; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const 
              alpha_array:PMKL_Complex8; const a_array:PPMKL_Complex8; const lda_array:PMKL_INT64; const b_array:PPMKL_Complex8; const ldb_array:PMKL_INT64; const 
              beta_array:PMKL_Complex8; const c_array:PPMKL_Complex8; const ldc_array:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;













  procedure cgemmt_64(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;












  procedure chemm_64(const side:Pchar; const uplo:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const beta:PMKL_Complex8; const 
              c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;












  procedure cher2k_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const beta:Psingle; const 
              c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;










  procedure cherk_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Psingle; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const beta:Psingle; const c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;












  procedure csymm_64(const side:Pchar; const uplo:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const beta:PMKL_Complex8; const 
              c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;












  procedure csyr2k_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const ldb:PMKL_INT64; const beta:PMKL_Complex8; const 
              c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;










  procedure csyrk_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const beta:PMKL_Complex8; const c:PMKL_Complex8; const ldc:PMKL_INT64);winapi; external;












  procedure csyrk_batch_64(const uplo_array:Pchar; const trans_array:Pchar; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const alpha_array:PMKL_Complex8; const 
              a_array:PPMKL_Complex8; const lda_array:PMKL_INT64; const beta_array:PMKL_Complex8; const c_array:PPMKL_Complex8; const ldc_array:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;













  procedure csyrk_batch_strided_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex8; const 
              a:PMKL_Complex8; const lda:PMKL_INT64; const stridea:PMKL_INT64; const beta:PMKL_Complex8; const c:PMKL_Complex8; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;











  procedure ctrmm_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const 
              ldb:PMKL_INT64);winapi; external;











  procedure ctrsm_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const b:PMKL_Complex8; const 
              ldb:PMKL_INT64);winapi; external;













  procedure ctrsm_batch_64(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT64; const 
              n_array:PMKL_INT64; const alpha_array:PMKL_Complex8; const a_array:PPMKL_Complex8; const lda_array:PMKL_INT64; const b_array:PPMKL_Complex8; const 
              ldb:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;














  procedure ctrsm_batch_strided_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const lda:PMKL_INT64; const stridea:PMKL_INT64; const 
              b:PMKL_Complex8; const ldb:PMKL_INT64; const strideb:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure dgemm_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const b:Pdouble; const ldb:PMKL_INT64; const 
              beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT64);winapi; external;





  function dgemm_pack_get_size_64(const identifier:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64):size_t;winapi; external;









  procedure dgemm_pack_64(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Pdouble; const src:Pdouble; const ld:PMKL_INT64; const dest:Pdouble);winapi; external;












  procedure dgemm_compute_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:Pdouble; const lda:PMKL_INT64; const b:Pdouble; const ldb:PMKL_INT64; const beta:Pdouble; const 
              c:Pdouble; const ldc:PMKL_INT64);winapi; external;















  procedure dgemm_batch_64(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT64; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const 
              alpha_array:Pdouble; const a_array:PPdouble; const lda_array:PMKL_INT64; const b_array:PPdouble; const ldb_array:PMKL_INT64; const 
              beta_array:Pdouble; const c_array:PPdouble; const ldc_array:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;

















  procedure dgemm_batch_strided_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const stridea:PMKL_INT64; const b:Pdouble; const 
              ldb:PMKL_INT64; const strideb:PMKL_INT64; const beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT64; const 
              stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure dgemmt_64(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const b:Pdouble; const ldb:PMKL_INT64; const 
              beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT64);winapi; external;












  procedure dsymm_64(const side:Pchar; const uplo:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Pdouble; const 
              a:Pdouble; const lda:PMKL_INT64; const b:Pdouble; const ldb:PMKL_INT64; const beta:Pdouble; const 
              c:Pdouble; const ldc:PMKL_INT64);winapi; external;












  procedure dsyr2k_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Pdouble; const 
              a:Pdouble; const lda:PMKL_INT64; const b:Pdouble; const ldb:PMKL_INT64; const beta:Pdouble; const 
              c:Pdouble; const ldc:PMKL_INT64);winapi; external;










  procedure dsyrk_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Pdouble; const 
              a:Pdouble; const lda:PMKL_INT64; const beta:Pdouble; const c:Pdouble; const ldc:PMKL_INT64);winapi; external;












  procedure dsyrk_batch_64(const uplo_array:Pchar; const trans_array:Pchar; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const alpha_array:Pdouble; const 
              a_array:PPdouble; const lda_array:PMKL_INT64; const beta_array:Pdouble; const c_array:PPdouble; const ldc_array:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;













  procedure dsyrk_batch_strided_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Pdouble; const 
              a:Pdouble; const lda:PMKL_INT64; const stridea:PMKL_INT64; const beta:Pdouble; const c:Pdouble; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;











  procedure dtrmm_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const b:Pdouble; const 
              ldb:PMKL_INT64);winapi; external;











  procedure dtrsm_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const b:Pdouble; const 
              ldb:PMKL_INT64);winapi; external;













  procedure dtrsm_batch_64(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT64; const 
              n_array:PMKL_INT64; const alpha_array:Pdouble; const a_array:PPdouble; const lda_array:PMKL_INT64; const b_array:PPdouble; const 
              ldb:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;














  procedure dtrsm_batch_strided_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const lda:PMKL_INT64; const stridea:PMKL_INT64; const 
              b:Pdouble; const ldb:PMKL_INT64; const strideb:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure zgemm_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;















  procedure zgemm_batch_64(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT64; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const 
              alpha_array:PMKL_Complex16; const a_array:PPMKL_Complex16; const lda_array:PMKL_INT64; const b_array:PPMKL_Complex16; const ldb_array:PMKL_INT64; const 
              beta_array:PMKL_Complex16; const c_array:PPMKL_Complex16; const ldc_array:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;

















  procedure zgemm_batch_strided_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const stridea:PMKL_INT64; const b:PMKL_Complex16; const 
              ldb:PMKL_INT64; const strideb:PMKL_INT64; const beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT64; const 
              stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure dzgemm_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex16; const a:Pdouble; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;













  procedure zgemm3m_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;















  procedure zgemm3m_batch_64(const transa_array:Pchar; const transb_array:Pchar; const m_array:PMKL_INT64; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const 
              alpha_array:PMKL_Complex16; const a_array:PPMKL_Complex16; const lda_array:PMKL_INT64; const b_array:PPMKL_Complex16; const ldb_array:PMKL_INT64; const 
              beta_array:PMKL_Complex16; const c_array:PPMKL_Complex16; const ldc_array:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;













  procedure zgemmt_64(const uplo:Pchar; const transa:Pchar; const transb:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const 
              beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;












  procedure zhemm_64(const side:Pchar; const uplo:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const beta:PMKL_Complex16; const 
              c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;












  procedure zher2k_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const beta:Pdouble; const 
              c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;










  procedure zherk_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:Pdouble; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const beta:Pdouble; const c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;












  procedure zsymm_64(const side:Pchar; const uplo:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const beta:PMKL_Complex16; const 
              c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;












  procedure zsyr2k_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const ldb:PMKL_INT64; const beta:PMKL_Complex16; const 
              c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;










  procedure zsyrk_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const beta:PMKL_Complex16; const c:PMKL_Complex16; const ldc:PMKL_INT64);winapi; external;












  procedure zsyrk_batch_64(const uplo_array:Pchar; const trans_array:Pchar; const n_array:PMKL_INT64; const k_array:PMKL_INT64; const alpha_array:PMKL_Complex16; const 
              a_array:PPMKL_Complex16; const lda_array:PMKL_INT64; const beta_array:PMKL_Complex16; const c_array:PPMKL_Complex16; const ldc_array:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;













  procedure zsyrk_batch_strided_64(const uplo:Pchar; const trans:Pchar; const n:PMKL_INT64; const k:PMKL_INT64; const alpha:PMKL_Complex16; const 
              a:PMKL_Complex16; const lda:PMKL_INT64; const stridea:PMKL_INT64; const beta:PMKL_Complex16; const c:PMKL_Complex16; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;











  procedure ztrmm_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const 
              ldb:PMKL_INT64);winapi; external;











  procedure ztrsm_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const b:PMKL_Complex16; const 
              ldb:PMKL_INT64);winapi; external;













  procedure ztrsm_batch_64(const side_array:Pchar; const uplo_array:Pchar; const transa_array:Pchar; const diag_array:Pchar; const m_array:PMKL_INT64; const 
              n_array:PMKL_INT64; const alpha_array:PMKL_Complex16; const a_array:PPMKL_Complex16; const lda_array:PMKL_INT64; const b_array:PPMKL_Complex16; const 
              ldb:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;














  procedure ztrsm_batch_strided_64(const side:Pchar; const uplo:Pchar; const transa:Pchar; const diag:Pchar; const m:PMKL_INT64; const 
              n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const lda:PMKL_INT64; const stridea:PMKL_INT64; const 
              b:PMKL_Complex16; const ldb:PMKL_INT64; const strideb:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;

















  procedure gemm_s16s16s32_64(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const 
              k:PMKL_INT64; const alpha:Psingle; const a:PMKL_INT16; const lda:PMKL_INT64; const ao:PMKL_INT16; const 
              b:PMKL_INT16; const ldb:PMKL_INT64; const bo:PMKL_INT16; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT64; const co:PMKL_INT32);winapi; external;

















  procedure gemm_s8u8s32_64(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const 
              k:PMKL_INT64; const alpha:Psingle; const a:PMKL_INT8; const lda:PMKL_INT64; const ao:PMKL_INT8; const 
              b:PMKL_UINT8; const ldb:PMKL_INT64; const bo:PMKL_INT8; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT64; const co:PMKL_INT32);winapi; external;





  function gemm_s8u8s32_pack_get_size_64(const identifier:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64):size_t;winapi; external;





  function gemm_s16s16s32_pack_get_size_64(const identifier:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64):size_t;winapi; external;








  procedure gemm_s8u8s32_pack_64(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              src:pointer; const ld:PMKL_INT64; const dest:pointer);winapi; external;








  procedure gemm_s16s16s32_pack_64(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              src:PMKL_INT16; const ld:PMKL_INT64; const dest:PMKL_INT16);winapi; external;

















  procedure gemm_s8u8s32_compute_64(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const 
              k:PMKL_INT64; const alpha:Psingle; const a:PMKL_INT8; const lda:PMKL_INT64; const ao:PMKL_INT8; const 
              b:PMKL_UINT8; const ldb:PMKL_INT64; const bo:PMKL_INT8; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT64; const co:PMKL_INT32);winapi; external;

















  procedure gemm_s16s16s32_compute_64(const transa:Pchar; const transb:Pchar; const offsetc:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const 
              k:PMKL_INT64; const alpha:Psingle; const a:PMKL_INT16; const lda:PMKL_INT64; const ao:PMKL_INT16; const 
              b:PMKL_INT16; const ldb:PMKL_INT64; const bo:PMKL_INT16; const beta:Psingle; const c:PMKL_INT32; const 
              ldc:PMKL_INT64; const co:PMKL_INT32);winapi; external;













  procedure hgemm_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_F16; const a:PMKL_F16; const lda:PMKL_INT64; const b:PMKL_F16; const ldb:PMKL_INT64; const 
              beta:PMKL_F16; const c:PMKL_F16; const ldc:PMKL_INT64);winapi; external;





  function hgemm_pack_get_size_64(const identifier:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64):size_t;winapi; external;









  procedure hgemm_pack_64(const identifier:Pchar; const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              alpha:PMKL_F16; const src:PMKL_F16; const ld:PMKL_INT64; const dest:PMKL_F16);winapi; external;












  procedure hgemm_compute_64(const transa:Pchar; const transb:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const k:PMKL_INT64; const 
              a:PMKL_F16; const lda:PMKL_INT64; const b:PMKL_F16; const ldb:PMKL_INT64; const beta:PMKL_F16; const 
              c:PMKL_F16; const ldc:PMKL_INT64);winapi; external;

  { Level1 BLAS batch API  }

{$endif}




{$ifdef UPPERCASE_DECL}
  procedure SAXPY_BATCH_64(const n:PMKL_INT64; const alpha:Psingle; const x:PPsingle; const incx:PMKL_INT64; const y:PPsingle; const 
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
{$endif}





{$ifdef LOWERCASE_DECL}
  procedure saxpy_batch_64(const n:PMKL_INT64; const alpha:Psingle; const x:PPsingle; const incx:PMKL_INT64; const y:PPsingle; const
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
 {$endif}







{$ifdef UPPERCASE_DECL}
  procedure DAXPY_BATCH_64(const n:PMKL_INT64; const alpha:Pdouble; const x:PPdouble; const incx:PMKL_INT64; const y:PPdouble; const 
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
{$endif}






{$ifdef LOWERCASE_DECL}
  procedure daxpy_batch_64(const n:PMKL_INT64; const alpha:Pdouble; const x:PPdouble; const incx:PMKL_INT64; const y:PPdouble; const 
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
 {$endif}







{$ifdef UPPERCASE_DECL}
  procedure CAXPY_BATCH_64(const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PPMKL_Complex8; const incx:PMKL_INT64; const y:PPMKL_Complex8; const 
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
{$endif}






{$ifdef LOWERCASE_DECL}
  procedure caxpy_batch_64(const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PPMKL_Complex8; const incx:PMKL_INT64; const y:PPMKL_Complex8; const 
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
  {$endif}







{$ifdef UPPERCASE_DECL}
  procedure ZAXPY_BATCH_64(const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PPMKL_Complex16; const incx:PMKL_INT64; const y:PPMKL_Complex16; const 
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
{$endif}






{$ifdef LOWERCASE_DECL}
  procedure zaxpy_batch_64(const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PPMKL_Complex16; const incx:PMKL_INT64; const y:PPMKL_Complex16; const 
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
 {$endif}






{$ifdef UPPERCASE_DECL}
  procedure SCOPY_BATCH_64(const n:PMKL_INT64; const x:PPsingle; const incx:PMKL_INT64; const y:PPsingle; const incy:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
{$endif}





{$ifdef LOWERCASE_DECL}
  procedure scopy_batch_64(const n:PMKL_INT64; const x:PPsingle; const incx:PMKL_INT64; const y:PPsingle; const incy:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
  {$endif}






{$ifdef UPPERCASE_DECL}
  procedure DCOPY_BATCH_64(const n:PMKL_INT64; const x:PPdouble; const incx:PMKL_INT64; const y:PPdouble; const incy:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
{$endif}





{$ifdef LOWERCASE_DECL}
  procedure dcopy_batch_64(const n:PMKL_INT64; const x:PPdouble; const incx:PMKL_INT64; const y:PPdouble; const incy:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
 {$endif}






{$ifdef UPPERCASE_DECL}
  procedure CCOPY_BATCH_64(const n:PMKL_INT64; const x:PPMKL_Complex8; const incx:PMKL_INT64; const y:PPMKL_Complex8; const incy:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
{$endif}





{$ifdef LOWERCASE_DECL}
  procedure ccopy_batch_64(const n:PMKL_INT64; const x:PPMKL_Complex8; const incx:PMKL_INT64; const y:PPMKL_Complex8; const incy:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
 {$endif}






{$ifdef UPPERCASE_DECL}
  procedure ZCOPY_BATCH_64(const n:PMKL_INT64; const x:PPMKL_Complex16; const incx:PMKL_INT64; const y:PPMKL_Complex16; const incy:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
{$endif}





{$ifdef LOWERCASE_DECL}
  procedure zcopy_batch_64(const n:PMKL_INT64; const x:PPMKL_Complex16; const incx:PMKL_INT64; const y:PPMKL_Complex16; const incy:PMKL_INT64; const 
              group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;
 {$endif}








{$ifdef UPPERCASE_DECL}
  procedure SAXPY_BATCH_STRIDED_64(const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              y:Psingle; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
{$endif}







{$ifdef LOWERCASE_DECL}
  procedure saxpy_batch_strided_64(const n:PMKL_INT64; const alpha:Psingle; const x:Psingle; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              y:Psingle; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
 {$endif}








{$ifdef UPPERCASE_DECL}
  procedure DAXPY_BATCH_STRIDED_64(const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              y:Pdouble; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
{$endif}







{$ifdef LOWERCASE_DECL}
  procedure daxpy_batch_strided_64(const n:PMKL_INT64; const alpha:Pdouble; const x:Pdouble; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              y:Pdouble; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
 {$endif}








{$ifdef UPPERCASE_DECL}
  procedure CAXPY_BATCH_STRIDED_64(const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              y:PMKL_Complex8; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
{$endif}







{$ifdef LOWERCASE_DECL}
  procedure caxpy_batch_strided_64(const n:PMKL_INT64; const alpha:PMKL_Complex8; const x:PMKL_Complex8; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              y:PMKL_Complex8; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
 {$endif}








{$ifdef UPPERCASE_DECL}
  procedure ZAXPY_BATCH_STRIDED_64(const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              y:PMKL_Complex16; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
{$endif}







{$ifdef LOWERCASE_DECL}
  procedure zaxpy_batch_strided_64(const n:PMKL_INT64; const alpha:PMKL_Complex16; const x:PMKL_Complex16; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              y:PMKL_Complex16; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
 {$endif}







{$ifdef UPPERCASE_DECL}
  procedure SCOPY_BATCH_STRIDED_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const stridex:PMKL_INT64; const y:Psingle; const 
              incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
{$endif}






{$ifdef LOWERCASE_DECL}
  procedure scopy_batch_strided_64(const n:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const stridex:PMKL_INT64; const y:Psingle; const 
              incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
  {$endif}







{$ifdef UPPERCASE_DECL}
  procedure DCOPY_BATCH_STRIDED_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const stridex:PMKL_INT64; const y:Pdouble; const 
              incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
{$endif}






{$ifdef LOWERCASE_DECL}
  procedure dcopy_batch_strided_64(const n:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const stridex:PMKL_INT64; const y:Pdouble; const 
              incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
 {$endif}







{$ifdef UPPERCASE_DECL}
  procedure CCOPY_BATCH_STRIDED_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const stridex:PMKL_INT64; const y:PMKL_Complex8; const 
              incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
{$endif}






{$ifdef LOWERCASE_DECL}
  procedure ccopy_batch_strided_64(const n:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const stridex:PMKL_INT64; const y:PMKL_Complex8; const 
              incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
  {$endif}







{$ifdef UPPERCASE_DECL}
  procedure ZCOPY_BATCH_STRIDED_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const stridex:PMKL_INT64; const y:PMKL_Complex16; const 
              incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
{$endif}






{$ifdef LOWERCASE_DECL}
  procedure zcopy_batch_strided_64(const n:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const stridex:PMKL_INT64; const y:PMKL_Complex16; const 
              incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;

  { Level2 BLAS batch API  }












  procedure sgemv_batch_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Psingle; const a:PPsingle; const 
              lda:PMKL_INT64; const x:PPsingle; const incx:PMKL_INT64; const beta:Psingle; const y:PPsingle; const 
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;















  procedure sgemv_batch_strided_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Psingle; const a:Psingle; const 
              lda:PMKL_INT64; const stridea:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              beta:Psingle; const y:Psingle; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure dgemv_batch_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Pdouble; const a:PPdouble; const 
              lda:PMKL_INT64; const x:PPdouble; const incx:PMKL_INT64; const beta:Pdouble; const y:PPdouble; const 
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;















  procedure dgemv_batch_strided_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const 
              lda:PMKL_INT64; const stridea:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              beta:Pdouble; const y:Pdouble; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure cgemv_batch_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PPMKL_Complex8; const 
              lda:PMKL_INT64; const x:PPMKL_Complex8; const incx:PMKL_INT64; const beta:PMKL_Complex8; const y:PPMKL_Complex8; const 
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;















  procedure cgemv_batch_strided_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const 
              lda:PMKL_INT64; const stridea:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure zgemv_batch_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PPMKL_Complex16; const 
              lda:PMKL_INT64; const x:PPMKL_Complex16; const incx:PMKL_INT64; const beta:PMKL_Complex16; const y:PPMKL_Complex16; const 
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;















  procedure zgemv_batch_strided_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const 
              lda:PMKL_INT64; const stridea:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;









{$endif}

{$ifdef UPPERCASE_DECL}

  procedure SGEMV_BATCH_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Psingle; const a:PPsingle; const 
              lda:PMKL_INT64; const x:PPsingle; const incx:PMKL_INT64; const beta:Psingle; const y:PPsingle; const 
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;















  procedure SGEMV_BATCH_STRIDED_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Psingle; const a:Psingle; const 
              lda:PMKL_INT64; const stridea:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              beta:Psingle; const y:Psingle; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure DGEMV_BATCH_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Pdouble; const a:PPdouble; const 
              lda:PMKL_INT64; const x:PPdouble; const incx:PMKL_INT64; const beta:Pdouble; const y:PPdouble; const 
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;















  procedure DGEMV_BATCH_STRIDED_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:Pdouble; const a:Pdouble; const 
              lda:PMKL_INT64; const stridea:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              beta:Pdouble; const y:Pdouble; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure CGEMV_BATCH_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PPMKL_Complex8; const 
              lda:PMKL_INT64; const x:PPMKL_Complex8; const incx:PMKL_INT64; const beta:PMKL_Complex8; const y:PPMKL_Complex8; const 
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;















  procedure CGEMV_BATCH_STRIDED_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex8; const a:PMKL_Complex8; const 
              lda:PMKL_INT64; const stridea:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              beta:PMKL_Complex8; const y:PMKL_Complex8; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;













  procedure ZGEMV_BATCH_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PPMKL_Complex16; const 
              lda:PMKL_INT64; const x:PPMKL_Complex16; const incx:PMKL_INT64; const beta:PMKL_Complex16; const y:PPMKL_Complex16; const 
              incy:PMKL_INT64; const group_count:PMKL_INT64; const group_size:PMKL_INT64);winapi; external;















  procedure ZGEMV_BATCH_STRIDED_64(const trans:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const alpha:PMKL_Complex16; const a:PMKL_Complex16; const 
              lda:PMKL_INT64; const stridea:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const stridex:PMKL_INT64; const 
              beta:PMKL_Complex16; const y:PMKL_Complex16; const incy:PMKL_INT64; const stridey:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;

{$endif}








{$ifdef LOWERCASE_DECL}
  procedure sdgmm_batch_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:PPsingle; const lda:PMKL_INT64; const 
              x:PPsingle; const incx:PMKL_INT64; const c:PPsingle; const ldc:PMKL_INT64; const group_count:PMKL_INT64; const 
              group_size:PMKL_INT64);winapi; external;













  procedure sdgmm_batch_strided_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:Psingle; const lda:PMKL_INT64; const 
              stridea:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const stridex:PMKL_INT64; const c:Psingle; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;











  procedure ddgmm_batch_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:PPdouble; const lda:PMKL_INT64; const 
              x:PPdouble; const incx:PMKL_INT64; const c:PPdouble; const ldc:PMKL_INT64; const group_count:PMKL_INT64; const 
              group_size:PMKL_INT64);winapi; external;













  procedure ddgmm_batch_strided_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:Pdouble; const lda:PMKL_INT64; const 
              stridea:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const stridex:PMKL_INT64; const c:Pdouble; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;











  procedure cdgmm_batch_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:PPMKL_Complex8; const lda:PMKL_INT64; const 
              x:PPMKL_Complex8; const incx:PMKL_INT64; const c:PPMKL_Complex8; const ldc:PMKL_INT64; const group_count:PMKL_INT64; const 
              group_size:PMKL_INT64);winapi; external;













  procedure cdgmm_batch_strided_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:PMKL_Complex8; const lda:PMKL_INT64; const 
              stridea:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const stridex:PMKL_INT64; const c:PMKL_Complex8; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;











  procedure zdgmm_batch_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:PPMKL_Complex16; const lda:PMKL_INT64; const 
              x:PPMKL_Complex16; const incx:PMKL_INT64; const c:PPMKL_Complex16; const ldc:PMKL_INT64; const group_count:PMKL_INT64; const 
              group_size:PMKL_INT64);winapi; external;













  procedure zdgmm_batch_strided_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:PMKL_Complex16; const lda:PMKL_INT64; const 
              stridea:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const stridex:PMKL_INT64; const c:PMKL_Complex16; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;


{$endif}






 {$ifdef UPPERCASE_DECL}

  procedure SDGMM_BATCH_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:PPsingle; const lda:PMKL_INT64; const 
              x:PPsingle; const incx:PMKL_INT64; const c:PPsingle; const ldc:PMKL_INT64; const group_count:PMKL_INT64; const 
              group_size:PMKL_INT64);winapi; external;













  procedure SDGMM_BATCH_STRIDED_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:Psingle; const lda:PMKL_INT64; const 
              stridea:PMKL_INT64; const x:Psingle; const incx:PMKL_INT64; const stridex:PMKL_INT64; const c:Psingle; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;











  procedure DDGMM_BATCH_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:PPdouble; const lda:PMKL_INT64; const 
              x:PPdouble; const incx:PMKL_INT64; const c:PPdouble; const ldc:PMKL_INT64; const group_count:PMKL_INT64; const 
              group_size:PMKL_INT64);winapi; external;













  procedure DDGMM_BATCH_STRIDED_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:Pdouble; const lda:PMKL_INT64; const 
              stridea:PMKL_INT64; const x:Pdouble; const incx:PMKL_INT64; const stridex:PMKL_INT64; const c:Pdouble; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;











  procedure CDGMM_BATCH_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:PPMKL_Complex8; const lda:PMKL_INT64; const 
              x:PPMKL_Complex8; const incx:PMKL_INT64; const c:PPMKL_Complex8; const ldc:PMKL_INT64; const group_count:PMKL_INT64; const 
              group_size:PMKL_INT64);winapi; external;













  procedure CDGMM_BATCH_STRIDED_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:PMKL_Complex8; const lda:PMKL_INT64; const 
              stridea:PMKL_INT64; const x:PMKL_Complex8; const incx:PMKL_INT64; const stridex:PMKL_INT64; const c:PMKL_Complex8; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;











  procedure ZDGMM_BATCH_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:PPMKL_Complex16; const lda:PMKL_INT64; const 
              x:PPMKL_Complex16; const incx:PMKL_INT64; const c:PPMKL_Complex16; const ldc:PMKL_INT64; const group_count:PMKL_INT64; const 
              group_size:PMKL_INT64);winapi; external;













  procedure ZDGMM_BATCH_STRIDED_64(const side:Pchar; const m:PMKL_INT64; const n:PMKL_INT64; const a:PMKL_Complex16; const lda:PMKL_INT64; const 
              stridea:PMKL_INT64; const x:PMKL_Complex16; const incx:PMKL_INT64; const stridex:PMKL_INT64; const c:PMKL_Complex16; const 
              ldc:PMKL_INT64; const stridec:PMKL_INT64; const batch_size:PMKL_INT64);winapi; external;
{$endif}
implementation


end.
