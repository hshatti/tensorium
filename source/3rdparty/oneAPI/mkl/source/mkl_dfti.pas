
unit mkl_dfti;
interface
uses mkl_types;

{$include mkl.inc}
{$IFDEF FPC}
{$PACKRECORDS C}
{$ENDIF}


  {******************************************************************************
  * Copyright 2002-2021 Intel Corporation.
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
  ! Content:
  !    Intel(R) oneAPI Math Kernel Library (oneMKL)
  !    Discrete Fourier Transform Interface (DFTI)
  !*************************************************************************** }

  type
    CDFT_CONFIG_PARAM = (CDFT_LOCAL_SIZE := 1000,CDFT_LOCAL_X_START := 1001,
      CDFT_LOCAL_NX := 1002,CDFT_MPI_COMM := 1003,
      CDFT_WORKSPACE := 1004,CDFT_LOCAL_OUT_X_START := 1005,
      CDFT_LOCAL_OUT_NX := 1006);

  { Definition of handle to descriptor  }

    DFTI_DESCRIPTOR = record    end;

  const
    CDFT_MPI_ERROR = 1000;    
    CDFT_SPREAD_ERROR = 1001;    
  { Codes of parameters for DftiGetValueDM / DftiSetValueDM  }
   DFTI_NO_ERROR = 0;    
    DFTI_MEMORY_ERROR = 1;    
    DFTI_INVALID_CONFIGURATION = 2;    
    DFTI_INCONSISTENT_CONFIGURATION = 3;    
    DFTI_MULTITHREADED_ERROR = 4;    
    DFTI_BAD_DESCRIPTOR = 5;    
    DFTI_UNIMPLEMENTED = 6;    
    DFTI_MKL_INTERNAL_ERROR = 7;    
    DFTI_NUMBER_OF_THREADS_ERROR = 8;    
    DFTI_1D_LENGTH_EXCEEDS_INT32 = 9;    
    DFTI_1D_MEMORY_EXCEEDS_INT32 = 9;    
  { Maximum length of error string  }
    DFTI_MAX_MESSAGE_LENGTH = 80;    
  { DEPRECATED  }
    DFTI_MAX_NAME_LENGTH = 10;    
  { Maximum length of oneMKL version string  }
    DFTI_VERSION_LENGTH = 198;    
  { Descriptor configuration parameters [default values in brackets]  }
  { Domain for forward transform. No default value  }
  { Dimensionality, or rank. No default value  }
  { Length(s) of transform. No default value  }
  { Floating point precision. No default value  }
  { Scale factor for forward transform [1.0]  }
  { Scale factor for backward transform [1.0]  }
  { Exponent sign for forward transform [DFTI_NEGATIVE]   }
  { DFTI_FORWARD_SIGN = 6, ## NOT IMPLEMENTED  }
  { Number of data sets to be transformed [1]  }
  { Storage of finite complex-valued sequences in complex domain
         [DFTI_COMPLEX_COMPLEX]  }
  { Storage of finite real-valued sequences in real domain
         [DFTI_REAL_REAL]  }
  { Storage of finite complex-valued sequences in conjugate-even
         domain [DFTI_COMPLEX_REAL]  }
  { Placement of result [DFTI_INPLACE]  }
  { Generalized strides for input data layout [tight, row-major for
         C]  }
  { Generalized strides for output data layout [tight, row-major
         for C]  }
  { Distance between first input elements for multiple transforms
         [0]  }
  { Distance between first output elements for multiple transforms
         [0]  }
  { Effort spent in initialization [DFTI_MEDIUM]  }
  { DFTI_INITIALIZATION_EFFORT = 16, ## NOT IMPLEMENTED  }
  { Use of workspace during computation [DFTI_ALLOW]  }
  { Ordering of the result [DFTI_ORDERED]  }
  { Possible transposition of result [DFTI_NONE]  }
  { User-settable descriptor name [""]  }
  { DEPRECATED  }
  { Packing format for DFTI_COMPLEX_REAL storage of finite
         conjugate-even sequences [DFTI_CCS_FORMAT]  }
  { Commit status of the descriptor - R/O parameter  }
  { Version string for this DFTI implementation - R/O parameter  }
  { Ordering of the forward transform - R/O parameter  }
  { DFTI_FORWARD_ORDERING  = 24, ## NOT IMPLEMENTED  }
  { Ordering of the backward transform - R/O parameter  }
  { DFTI_BACKWARD_ORDERING = 25, ## NOT IMPLEMENTED  }
  { Number of user threads that share the descriptor [1]  }
  { Limit the number of threads used by this descriptor [0 = don't care]  }
  { Possible input data destruction [DFTI_AVOID = prevent input data] }
  { Distance between first input elements for multiple transforms
         [0]  }
  { Distance between first input elements for multiple transforms
         [0]  }

  type
    DFTI_CONFIG_PARAM = (DFTI_FORWARD_DOMAIN := 0,DFTI_DIMENSION := 1,
      DFTI_LENGTHS := 2,DFTI_PRECISION := 3,
      DFTI_FORWARD_SCALE := 4,DFTI_BACKWARD_SCALE := 5,
      DFTI_NUMBER_OF_TRANSFORMS := 7,DFTI_COMPLEX_STORAGE := 8,
      DFTI_REAL_STORAGE := 9,DFTI_CONJUGATE_EVEN_STORAGE := 10,
      DFTI_PLACEMENT := 11,DFTI_INPUT_STRIDES := 12,
      DFTI_OUTPUT_STRIDES := 13,DFTI_INPUT_DISTANCE := 14,
      DFTI_OUTPUT_DISTANCE := 15,DFTI_WORKSPACE := 17,
      DFTI_ORDERING := 18,DFTI_TRANSPOSE := 19,
      DFTI_DESCRIPTOR_NAME := 20,DFTI_PACKED_FORMAT := 21,
      DFTI_COMMIT_STATUS := 22,DFTI_VERSION := 23,
      DFTI_NUMBER_OF_USER_THREADS := 26,DFTI_THREAD_LIMIT := 27,
      DFTI_DESTROY_INPUT := 28,DFTI_FWD_DISTANCE := 58,
      DFTI_BWD_DISTANCE := 59);

  { Values of the descriptor configuration parameters  }
  { DFTI_COMMIT_STATUS  }
  { DFTI_FORWARD_DOMAIN  }
  { DFTI_CONJUGATE_EVEN = 34,   ## NOT IMPLEMENTED  }
  { DFTI_PRECISION  }
  { DFTI_FORWARD_SIGN  }
  { DFTI_NEGATIVE = 37,         ## NOT IMPLEMENTED  }
  { DFTI_POSITIVE = 38,         ## NOT IMPLEMENTED  }
  { DFTI_COMPLEX_STORAGE and DFTI_CONJUGATE_EVEN_STORAGE  }
  { DFTI_REAL_STORAGE  }
  { DFTI_PLACEMENT  }
  { Result overwrites input  }
  { Have another place for result  }
  { DFTI_INITIALIZATION_EFFORT  }
  { DFTI_LOW = 45,              ## NOT IMPLEMENTED  }
  { DFTI_MEDIUM = 46,           ## NOT IMPLEMENTED  }
  { DFTI_HIGH = 47,             ## NOT IMPLEMENTED  }
  { DFTI_ORDERING  }
  { DFTI_FORWARD_SCRAMBLED = 50, ## NOT IMPLEMENTED  }
  { Allow/avoid certain usages  }
  { Allow transposition or workspace  }
  { DFTI_PACKED_FORMAT (for storing congugate-even finite sequence
         in real array)  }
  { Complex conjugate-symmetric  }
  { Pack format for real DFT  }
  { Perm format for real DFT  }
  { Complex conjugate-even  }
    DFTI_CONFIG_VALUE = (DFTI_COMMITTED := 30,DFTI_UNCOMMITTED := 31,
      DFTI_COMPLEX := 32,DFTI_REAL := 33,DFTI_SINGLE := 35,
      DFTI_DOUBLE := 36,DFTI_COMPLEX_COMPLEX := 39,
      DFTI_COMPLEX_REAL := 40,DFTI_REAL_COMPLEX := 41,
      DFTI_REAL_REAL := 42,DFTI_INPLACE := 43,
      DFTI_NOT_INPLACE := 44,DFTI_ORDERED := 48,
      DFTI_BACKWARD_SCRAMBLED := 49,DFTI_ALLOW := 51,
      DFTI_AVOID := 52,DFTI_NONE := 53,DFTI_CCS_FORMAT := 54,
      DFTI_PACK_FORMAT := 55,DFTI_PERM_FORMAT := 56,
      DFTI_CCE_FORMAT := 57);

    PDFTI_DESCRIPTOR_HANDLE = ^DFTI_DESCRIPTOR_HANDLE;
     DFTI_DESCRIPTOR_HANDLE = DFTI_DESCRIPTOR;
  { deprecated  }
  { deprecated  }

  
  {$define DFTI_DFT_Desc_struct := DFTI_DESCRIPTOR}    
  { deprecated  }
  {$define  DFTI_Descriptor_struct = DFTI_DESCRIPTOR}    
  { deprecated  }
//    DFTI_Descriptor = DFTI_DESCRIPTOR;    
  { precision  }
  { domain  }

{.$define DftiCreateDescriptor}
//  function DftiCreateDescriptor(_para1:PDFTI_DESCRIPTOR_HANDLE; _para2:DFTI_CONFIG_VALUE; _para3:DFTI_CONFIG_VALUE; _para4:MKL_LONG; args:array of const):MKL_LONG;winapi;external;

  function DftiCreateDescriptor(hand:PDFTI_DESCRIPTOR_HANDLE; prec:DFTI_CONFIG_VALUE; domain:DFTI_CONFIG_VALUE; dims:MKL_LONG):MKL_LONG;varargs;winapi;external;

  { from descriptor  }
  function DftiCopyDescriptor(_para1:DFTI_DESCRIPTOR_HANDLE; _para2:PDFTI_DESCRIPTOR_HANDLE):MKL_LONG;winapi;external;

  { to descriptor  }
  function DftiCommitDescriptor(hand:DFTI_DESCRIPTOR_HANDLE):MKL_LONG;winapi;external;

//  function DftiComputeForward(_para1:DFTI_DESCRIPTOR_HANDLE; _para2:pointer; args:array of const):MKL_LONG;winapi;external;

  function DftiComputeForward(hand:DFTI_DESCRIPTOR_HANDLE; _data:pointer):MKL_LONG;varargs;winapi;external;

//  function DftiComputeBackward(_para1:DFTI_DESCRIPTOR_HANDLE; _para2:pointer; args:array of const):MKL_LONG;winapi;external;

  function DftiComputeBackward(hand:DFTI_DESCRIPTOR_HANDLE; _data:pointer):MKL_LONG;varargs;winapi;external;

//  function DftiSetValue(_para1:DFTI_DESCRIPTOR_HANDLE; _para2:DFTI_CONFIG_PARAM; args:array of const):MKL_LONG;winapi;external;

  function DftiSetValue(hand:DFTI_DESCRIPTOR_HANDLE; _para2:DFTI_CONFIG_PARAM):MKL_LONG;varargs;winapi;external;

//  function DftiGetValue(_para1:DFTI_DESCRIPTOR_HANDLE; _para2:DFTI_CONFIG_PARAM; args:array of const):MKL_LONG;winapi;external;

  function DftiGetValue(hand:DFTI_DESCRIPTOR_HANDLE; _para2:DFTI_CONFIG_PARAM):MKL_LONG;varargs;winapi;external;

  function DftiFreeDescriptor(hand:PDFTI_DESCRIPTOR_HANDLE):MKL_LONG;winapi;external;

  function DftiErrorMessage(_para1:MKL_LONG):PChar;winapi;external;

  function DftiErrorClass(_para1:MKL_LONG; _para2:MKL_LONG):MKL_LONG;winapi;external;

  {*********************************************************************
   * INTERNAL INTERFACES. These internal interfaces are not intended to
   * be called directly by oneMKL users and may change in future releases.
    }
//  { MKL_LONG onedim  }  function DftiCreateDescriptor_s_1d(_para1:PDFTI_DESCRIPTOR_HANDLE; domain:DFTI_CONFIG_VALUE; args:array of const):MKL_LONG;winapi;external;

  function DftiCreateDescriptor_s_1d(const _para1:PDFTI_DESCRIPTOR_HANDLE; const domain:DFTI_CONFIG_VALUE; const args:array of const):MKL_LONG;winapi;external;

//  { MKL_LONG *dims  }  function DftiCreateDescriptor_s_md(_para1:PDFTI_DESCRIPTOR_HANDLE; domain:DFTI_CONFIG_VALUE; many:MKL_LONG; args:array of const):MKL_LONG;winapi;external;

  function DftiCreateDescriptor_s_md(const _para1:PDFTI_DESCRIPTOR_HANDLE; const domain:DFTI_CONFIG_VALUE; const many:MKL_LONG; const args:array of const):MKL_LONG;winapi;external;

//  { MKL_LONG onedim  }  function DftiCreateDescriptor_d_1d(_para1:PDFTI_DESCRIPTOR_HANDLE; domain:DFTI_CONFIG_VALUE; args:array of const):MKL_LONG;winapi;external;

  function DftiCreateDescriptor_d_1d(const _para1:PDFTI_DESCRIPTOR_HANDLE; const domain:DFTI_CONFIG_VALUE; const args:array of const):MKL_LONG;winapi;external;

//  { MKL_LONG *dims  }  function DftiCreateDescriptor_d_md(_para1:PDFTI_DESCRIPTOR_HANDLE; domain:DFTI_CONFIG_VALUE; many:MKL_LONG; args:array of const):MKL_LONG;winapi;external;

  function DftiCreateDescriptor_d_md(const _para1:PDFTI_DESCRIPTOR_HANDLE; const domain:DFTI_CONFIG_VALUE; const many:MKL_LONG; const args:array of const):MKL_LONG;winapi;external;

  {*********************************************************************
   * Compile-time separation of specific cases
    }

{$if not defined(DftiCreateDescriptor)}

  function DftiCreateDescriptor(const desc:PDFTI_DESCRIPTOR_HANDLE;const prec:DFTI_CONFIG_VALUE; const domain:DFTI_CONFIG_VALUE; const dim: MKL_LONG; const sizes : array of const) : MKL_LONG;  inline;
{$endif}

implementation

{$if not defined(DftiCreateDescriptor)}

  function DftiCreateDescriptor(const desc:PDFTI_DESCRIPTOR_HANDLE;const prec:DFTI_CONFIG_VALUE; const domain:DFTI_CONFIG_VALUE; const dim: MKL_LONG; const sizes : array of const) : MKL_LONG;inline;
  begin
    if (prec=DFTI_SINGLE) and (dim=1) then
      DftiCreateDescriptor:=DftiCreateDescriptor_s_1d(desc,domain,sizes)
    else if prec=DFTI_SINGLE then
      DftiCreateDescriptor:=DftiCreateDescriptor_s_md(desc,domain,dim,sizes)
    else if (prec=DFTI_DOUBLE) and (dim=1) then
      DftiCreateDescriptor:=DftiCreateDescriptor_d_1d(desc,domain,sizes)
    else ifprec=DFTI_DOUBLE then
      DftiCreateDescriptor:=DftiCreateDescriptor_d_md(desc,domain,dim,sizes)
    else
      DftiCreateDescriptor:=DftiCreateDescriptor(desc,prec,domain,dim,sizes)
  end;
{$endif}

end.
