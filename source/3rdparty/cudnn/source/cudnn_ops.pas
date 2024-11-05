(*
 * Copyright 2014-2023 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 *)

(*   cudnn : Neural Networks Library  *)

unit cudnn_ops;
{$ifdef FPC}
  {$mode Delphi}
{$endif}
interface
uses cudnn_graph;

const
  CUDNN_LRN_MIN_N         = 1      ;
  CUDNN_LRN_MAX_N         = 16     ;
  CUDNN_LRN_MIN_K         = 1e-5   ;
  CUDNN_LRN_MIN_BETA      = 0.01   ;
  CUDNN_BN_MIN_EPSILON    = 0.0     ;

type
  PcudnnDeterminism_t =^cudnnDeterminism_t;
  cudnnDeterminism_t = (
    CUDNN_NON_DETERMINISTIC = 0,
    CUDNN_DETERMINISTIC     = 1
) ;

  PcudnnReduceTensorIndices_t = ^cudnnReduceTensorIndices_t;
  cudnnReduceTensorIndices_t = (
    CUDNN_REDUCE_TENSOR_NO_INDICES        = 0,
    CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = 1
)  ;

 (*
 * CUDNN tensor indices type size (all unsigned)
 * Currently not supported, default is 32 bit unsigned.
 *)
 PcudnnIndicesType_t = ^cudnnIndicesType_t;
 cudnnIndicesType_t =  (
    CUDNN_32BIT_INDICES = 0,
    CUDNN_64BIT_INDICES = 1,
    CUDNN_16BIT_INDICES = 2,
    CUDNN_8BIT_INDICES  = 3
)  ;

 PcudnnOpTensorOp_t= ^cudnnOpTensorOp_t;
 cudnnOpTensorOp_t = (
    CUDNN_OP_TENSOR_ADD  = 0,
    CUDNN_OP_TENSOR_MUL  = 1,
    CUDNN_OP_TENSOR_MIN  = 2,
    CUDNN_OP_TENSOR_MAX  = 3,
    CUDNN_OP_TENSOR_SQRT = 4,
    CUDNN_OP_TENSOR_NOT  = 5
) ;

 PcudnnFoldingDirection_t = ^cudnnFoldingDirection_t;
 cudnnFoldingDirection_t = (
    CUDNN_TRANSFORM_FOLD   = 0,
    CUDNN_TRANSFORM_UNFOLD = 1
) ;

(*
 *  softmax algorithm
 *)
 PcudnnSoftmaxAlgorithm_t = ^cudnnSoftmaxAlgorithm_t;
 cudnnSoftmaxAlgorithm_t = (
    CUDNN_SOFTMAX_FAST     = 0, (* straightforward implementation *)
    CUDNN_SOFTMAX_ACCURATE = 1, (* subtract max from every point to avoid overflow *)
    CUDNN_SOFTMAX_LOG      = 2
) ;

  PcudnnSoftmaxMode_t = ^cudnnSoftmaxMode_t;
  cudnnSoftmaxMode_t = (
    CUDNN_SOFTMAX_MODE_INSTANCE = 0, (* compute the softmax over all C, H, W for each N *)
    CUDNN_SOFTMAX_MODE_CHANNEL  = 1  (* compute the softmax over all C for each H, W, N *)
) ;

(* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" *)

(*
 *  pooling mode
 *)
 PcudnnPoolingMode_t = ^cudnnPoolingMode_t;
 cudnnPoolingMode_t = (
    CUDNN_POOLING_MAX                           = 0,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1, (* count for average includes padded values *)
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2, (* count for average does not include padded values *)
    CUDNN_POOLING_MAX_DETERMINISTIC             = 3
)  ;

  PcudnnDivNormMode_t = ^cudnnDivNormMode_t;
  cudnnDivNormMode_t = (
    CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0
) ;

 PcudnnBatchNormMode_t = ^cudnnBatchNormMode_t;
 cudnnBatchNormMode_t = (
    (* bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice) *)
    CUDNN_BATCHNORM_PER_ACTIVATION = 0,

    (* bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors) *)
    CUDNN_BATCHNORM_SPATIAL = 1,

    (*
     * bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors).
     * May be faster than CUDNN_BATCHNORM_SPATIAL but imposes some limits on the range of values
     *)
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2
)  ;

  PcudnnBatchNormOps_t = ^cudnnBatchNormOps_t;
  cudnnBatchNormOps_t = (
    CUDNN_BATCHNORM_OPS_BN                = 0, (* do batch normalization only *)
    CUDNN_BATCHNORM_OPS_BN_ACTIVATION     = 1, (* do batchNorm, then activation *)
    CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION = 2  (* do batchNorm, then elemWiseAdd, then activation *)
)  ;

  PcudnnNormMode_t = ^cudnnNormMode_t ;
  cudnnNormMode_t = (
    (* bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice) *)
    CUDNN_NORM_PER_ACTIVATION = 0,

    (* bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors) *)
    CUDNN_NORM_PER_CHANNEL = 1
)  ;

  PcudnnNormAlgo_t = ^cudnnNormAlgo_t;
  cudnnNormAlgo_t = ( CUDNN_NORM_ALGO_STANDARD = 0, CUDNN_NORM_ALGO_PERSIST = 1 )  ;

  PcudnnNormOps_t = ^cudnnNormOps_t;
  cudnnNormOps_t = (
    CUDNN_NORM_OPS_NORM                = 0, (* do normalization only *)
    CUDNN_NORM_OPS_NORM_ACTIVATION     = 1, (* do Norm, then activation *)
    CUDNN_NORM_OPS_NORM_ADD_ACTIVATION = 2  (* do Norm, then elemWiseAdd, then activation *)
)  ;

(* APIs for spatial transformer network *)
  PcudnnSamplerType_t = ^cudnnSamplerType_t;
  cudnnSamplerType_t = (
    CUDNN_SAMPLER_BILINEAR = 0
) ;


(* TODO: move these enums out to the appropriate submodule *)
  PcudnnConvolutionFwdAlgo_t =^cudnnConvolutionFwdAlgo_t;
  cudnnConvolutionFwdAlgo_t = (
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED     = 7,
    CUDNN_CONVOLUTION_FWD_ALGO_COUNT                 = 8
) ;

  PcudnnConvolutionBwdFilterAlgo_t = ^cudnnConvolutionBwdFilterAlgo_t;
  cudnnConvolutionBwdFilterAlgo_t = (
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0                 = 0, (* non-deterministic *)
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1                 = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT               = 2,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3                 = 3, (* non-deterministic *)
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD          = 4, (* not implemented *)
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING        = 6,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT             = 7
) ;

  PcudnnConvolutionBwdDataAlgo_t = ^cudnnConvolutionBwdDataAlgo_t;
  cudnnConvolutionBwdDataAlgo_t = (
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0                 = 0, (* non-deterministic *)
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1                 = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT               = 2,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING        = 3,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD          = 4,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT             = 6
) ;

  PcudnnCTCLossAlgo_t = ^cudnnCTCLossAlgo_t;
  cudnnCTCLossAlgo_t = ( CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = 0, CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC = 1 ) ;

  PcudnnLRNMode_t = ^cudnnLRNMode_t ;
  cudnnLRNMode_t = (
      CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0  (* Normalize across tensor's dimA[1] dimension *)
  ) ;


(* Data structures to represent Image/Filter and the Neural Network Layer *)
 PcudnnTensorDescriptor_t = ^cudnnTensorDescriptor_t;
 cudnnTensorDescriptor_t      = ^cudnnTensorStruct;
 cudnnTensorStruct             = record end;

 PcudnnPoolingDescriptor_t = ^cudnnPoolingDescriptor_t;
 cudnnPoolingDescriptor_t     = ^cudnnTensorStruct;
 cudnnPoolingStruct            = record end;

 PcudnnFilterDescriptor_t = ^cudnnFilterDescriptor_t;
 cudnnFilterDescriptor_t      = ^cudnnFilterStruct;
 cudnnFilterStruct             = record end;

 PcudnnLRNDescriptor_t = ^cudnnLRNDescriptor_t;
 cudnnLRNDescriptor_t         = ^cudnnLRNStruct;
 cudnnLRNStruct                = record end;

 PcudnnActivationDescriptor_t = ^cudnnActivationDescriptor_t;
 cudnnActivationDescriptor_t  = ^cudnnActivationStruct;
 cudnnActivationStruct         = record end;

 PcudnnSpatialTransformerDescriptor_t = ^cudnnSpatialTransformerDescriptor_t;
 cudnnSpatialTransformerDescriptor_t   = ^cudnnSpatialTransformerStruct;
 cudnnSpatialTransformerStruct = record end;

 PcudnnOpTensorDescriptor_t = ^cudnnOpTensorDescriptor_t;
 cudnnOpTensorDescriptor_t    = ^cudnnOpTensorStruct;
 cudnnOpTensorStruct           = record end;

 PcudnnReduceTensorDescriptor_t = ^cudnnReduceTensorDescriptor_t;
 cudnnReduceTensorDescriptor_t  = ^cudnnReduceTensorStruct;
 cudnnReduceTensorStruct       = record end;

 PcudnnCTCLossDescriptor_t  = ^cudnnCTCLossDescriptor_t;
 cudnnCTCLossDescriptor_t     = ^cudnnCTCLossStruct;
 cudnnCTCLossStruct            = record end;

 PcudnnTensorTransformDescriptor_t = ^cudnnTensorTransformDescriptor_t;
 cudnnTensorTransformDescriptor_t      = ^cudnnTensorTransformStruct;
 cudnnTensorTransformStruct    = record end;

 PcudnnDropoutDescriptor_t  = ^cudnnDropoutDescriptor_t;
 cudnnDropoutDescriptor_t    =  ^cudnnDropoutStruct;
 cudnnDropoutStruct            = record end;


 function cudnnCreateTensorDescriptor(tensorDesc: PcudnnTensorDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnSetTensor4dDescriptor(tensorDesc: cudnnTensorDescriptor_t; format: cudnnTensorFormat_t; dataType: cudnnDataType_t; n: longint; c: longint; h: longint; w: longint):cudnnStatus_t; winapi; external libname;
 function cudnnSetTensor4dDescriptorEx(tensorDesc: cudnnTensorDescriptor_t; dataType: cudnnDataType_t; n: longint; c: longint; h: longint; w: longint; nStride: longint; cStride: longint; hStride: longint; wStride: longint):cudnnStatus_t; winapi; external libname;
 function cudnnGetTensor4dDescriptor(const tensorDesc: cudnnTensorDescriptor_t; dataType: PcudnnDataType_t; n: Plongint; c: Plongint; h: Plongint; w: Plongint; nStride: Plongint; cStride: Plongint; hStride: Plongint; wStride: Plongint):cudnnStatus_t; winapi; external libname;
 function cudnnSetTensorNdDescriptor(tensorDesc: cudnnTensorDescriptor_t; dataType: cudnnDataType_t; nbDims: longint; const dimA: array of longint; const strideA: array of longint):cudnnStatus_t; winapi; external libname;
 function cudnnSetTensorNdDescriptorEx(tensorDesc: cudnnTensorDescriptor_t; format: cudnnTensorFormat_t; dataType: cudnnDataType_t; nbDims: longint; const dimA: array of longint):cudnnStatus_t; winapi; external libname;
 function cudnnGetTensorNdDescriptor(const tensorDesc: cudnnTensorDescriptor_t; nbDimsRequested: longint; dataType: PcudnnDataType_t; nbDims: Plongint; dimA: array of longint; strideA: array of longint):cudnnStatus_t; winapi; external libname;
 function cudnnGetTensorSizeInBytes(const tensorDesc: cudnnTensorDescriptor_t; size: PSizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnInitTransformDest(const transformDesc: cudnnTensorTransformDescriptor_t; const srcDesc: cudnnTensorDescriptor_t; destDesc: cudnnTensorDescriptor_t; destSizeInBytes: PSizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnCreateTensorTransformDescriptor(transformDesc: PcudnnTensorTransformDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnSetTensorTransformDescriptor(transformDesc: cudnnTensorTransformDescriptor_t; const nbDims: uint32; const destFormat: cudnnTensorFormat_t; const padBeforeA: array of int32; const padAfterA: array of int32; const foldA: array of uint32; const direction: cudnnFoldingDirection_t):cudnnStatus_t; winapi; external libname;
 function cudnnGetTensorTransformDescriptor(transformDesc: cudnnTensorTransformDescriptor_t; nbDimsRequested: uint32; destFormat: PcudnnTensorFormat_t; padBeforeA: array of int32; padAfterA: array of int32; foldA: array of uint32; direction: PcudnnFoldingDirection_t):cudnnStatus_t; winapi; external libname;
 function cudnnDestroyTensorTransformDescriptor(transformDesc: cudnnTensorTransformDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnTransformTensor(handle: cudnnHandle_t; const alpha: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const beta: Pointer; const yDesc: cudnnTensorDescriptor_t; y: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnTransformTensorEx(handle: cudnnHandle_t; const transDesc: cudnnTensorTransformDescriptor_t; const alpha: Pointer; const srcDesc: cudnnTensorDescriptor_t; const srcData: Pointer; const beta: Pointer; const destDesc: cudnnTensorDescriptor_t; destData: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnAddTensor(handle: cudnnHandle_t; const alpha: Pointer; const aDesc: cudnnTensorDescriptor_t; const A: Pointer; const beta: Pointer; const cDesc: cudnnTensorDescriptor_t; C: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnCreateOpTensorDescriptor(opTensorDesc: PcudnnOpTensorDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnSetOpTensorDescriptor(opTensorDesc: cudnnOpTensorDescriptor_t; opTensorOp: cudnnOpTensorOp_t; opTensorCompType: cudnnDataType_t; opTensorNanOpt: cudnnNanPropagation_t):cudnnStatus_t; winapi; external libname;
 function cudnnGetOpTensorDescriptor(const opTensorDesc: cudnnOpTensorDescriptor_t; opTensorOp: PcudnnOpTensorOp_t; opTensorCompType: PcudnnDataType_t; opTensorNanOpt: PcudnnNanPropagation_t):cudnnStatus_t; winapi; external libname;
 function cudnnDestroyOpTensorDescriptor(opTensorDesc: cudnnOpTensorDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnOpTensor(handle: cudnnHandle_t; const opTensorDesc: cudnnOpTensorDescriptor_t; const alpha1: Pointer; const aDesc: cudnnTensorDescriptor_t; const A: Pointer; const alpha2: Pointer; const bDesc: cudnnTensorDescriptor_t; const B: Pointer; const beta: Pointer; const cDesc: cudnnTensorDescriptor_t; C: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnCreateReduceTensorDescriptor(reduceTensorDesc: PcudnnReduceTensorDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnSetReduceTensorDescriptor(reduceTensorDesc: cudnnReduceTensorDescriptor_t; reduceTensorOp: cudnnReduceTensorOp_t; reduceTensorCompType: cudnnDataType_t; reduceTensorNanOpt: cudnnNanPropagation_t; reduceTensorIndices: cudnnReduceTensorIndices_t; reduceTensorIndicesType: cudnnIndicesType_t):cudnnStatus_t; winapi; external libname;
 function cudnnGetReduceTensorDescriptor(const reduceTensorDesc: cudnnReduceTensorDescriptor_t; reduceTensorOp: PcudnnReduceTensorOp_t; reduceTensorCompType: PcudnnDataType_t; reduceTensorNanOpt: PcudnnNanPropagation_t; reduceTensorIndices: PcudnnReduceTensorIndices_t; reduceTensorIndicesType: PcudnnIndicesType_t):cudnnStatus_t; winapi; external libname;
 function cudnnDestroyReduceTensorDescriptor(reduceTensorDesc: cudnnReduceTensorDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnGetReductionIndicesSize(handle: cudnnHandle_t; const reduceTensorDesc: cudnnReduceTensorDescriptor_t; const aDesc: cudnnTensorDescriptor_t; const cDesc: cudnnTensorDescriptor_t; sizeInBytes: PSizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnGetReductionWorkspaceSize(handle: cudnnHandle_t; const reduceTensorDesc: cudnnReduceTensorDescriptor_t; const aDesc: cudnnTensorDescriptor_t; const cDesc: cudnnTensorDescriptor_t; sizeInBytes: PSizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnReduceTensor(handle: cudnnHandle_t; const reduceTensorDesc: cudnnReduceTensorDescriptor_t; indices: Pointer; indicesSizeInBytes: SizeInt; workspace: Pointer; workspaceSizeInBytes: SizeInt; const alpha: Pointer; const aDesc: cudnnTensorDescriptor_t; const A: Pointer; const beta: Pointer; const cDesc: cudnnTensorDescriptor_t; C: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnSetTensor(handle: cudnnHandle_t; const yDesc: cudnnTensorDescriptor_t; y: Pointer; const valuePtr: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnScaleTensor(handle: cudnnHandle_t; const yDesc: cudnnTensorDescriptor_t; y: Pointer; const alpha: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnCreateFilterDescriptor(filterDesc: PcudnnFilterDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnSetFilter4dDescriptor(filterDesc: cudnnFilterDescriptor_t; dataType: cudnnDataType_t; format: cudnnTensorFormat_t; k: longint; c: longint; h: longint; w: longint):cudnnStatus_t; winapi; external libname;
 function cudnnGetFilter4dDescriptor(const filterDesc: cudnnFilterDescriptor_t; dataType: PcudnnDataType_t; format: PcudnnTensorFormat_t; k: Plongint; c: Plongint; h: Plongint; w: Plongint):cudnnStatus_t; winapi; external libname;
 function cudnnSetFilterNdDescriptor(filterDesc: cudnnFilterDescriptor_t; dataType: cudnnDataType_t; format: cudnnTensorFormat_t; nbDims: longint; const filterDimA: array of longint):cudnnStatus_t; winapi; external libname;
 function cudnnGetFilterNdDescriptor(const filterDesc: cudnnFilterDescriptor_t; nbDimsRequested: longint; dataType: PcudnnDataType_t; format: PcudnnTensorFormat_t; nbDims: Plongint; filterDimA: array of longint):cudnnStatus_t; winapi; external libname;
 function cudnnGetFilterSizeInBytes(const filterDesc: cudnnFilterDescriptor_t; size: PSizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnTransformFilter(handle: cudnnHandle_t; const transDesc: cudnnTensorTransformDescriptor_t; const alpha: Pointer; const srcDesc: cudnnFilterDescriptor_t; const srcData: Pointer; const beta: Pointer; const destDesc: cudnnFilterDescriptor_t; destData: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnDestroyFilterDescriptor(filterDesc: cudnnFilterDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnSoftmaxForward(handle: cudnnHandle_t; algo: cudnnSoftmaxAlgorithm_t; mode: cudnnSoftmaxMode_t; const alpha: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const beta: Pointer; const yDesc: cudnnTensorDescriptor_t; y: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnCreatePoolingDescriptor(poolingDesc: PcudnnPoolingDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnSetPooling2dDescriptor(poolingDesc: cudnnPoolingDescriptor_t; mode: cudnnPoolingMode_t; maxpoolingNanOpt: cudnnNanPropagation_t; windowHeight: longint; windowWidth: longint; verticalPadding: longint; horizontalPadding: longint; verticalStride: longint; horizontalStride: longint):cudnnStatus_t; winapi; external libname;
 function cudnnGetPooling2dDescriptor(const poolingDesc: cudnnPoolingDescriptor_t; mode: PcudnnPoolingMode_t; maxpoolingNanOpt: PcudnnNanPropagation_t; windowHeight: Plongint; windowWidth: Plongint; verticalPadding: Plongint; horizontalPadding: Plongint; verticalStride: Plongint; horizontalStride: Plongint):cudnnStatus_t; winapi; external libname;
 function cudnnSetPoolingNdDescriptor(poolingDesc: cudnnPoolingDescriptor_t; const mode: cudnnPoolingMode_t; const maxpoolingNanOpt: cudnnNanPropagation_t; nbDims: longint; const windowDimA: array of longint; const paddingA: array of longint; const strideA: array of longint):cudnnStatus_t; winapi; external libname;
 function cudnnGetPoolingNdDescriptor(const poolingDesc: cudnnPoolingDescriptor_t; nbDimsRequested: longint; mode: PcudnnPoolingMode_t; maxpoolingNanOpt: PcudnnNanPropagation_t; nbDims: Plongint; windowDimA: array of longint; paddingA: array of longint; strideA: array of longint):cudnnStatus_t; winapi; external libname;
 function cudnnGetPoolingNdForwardOutputDim(const poolingDesc: cudnnPoolingDescriptor_t; const inputTensorDesc: cudnnTensorDescriptor_t; nbDims: longint; outputTensorDimA: array of longint):cudnnStatus_t; winapi; external libname;
 function cudnnGetPooling2dForwardOutputDim(const poolingDesc: cudnnPoolingDescriptor_t; const inputTensorDesc: cudnnTensorDescriptor_t; n: Plongint; c: Plongint; h: Plongint; w: Plongint):cudnnStatus_t; winapi; external libname;
 function cudnnDestroyPoolingDescriptor(poolingDesc: cudnnPoolingDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnPoolingForward(handle: cudnnHandle_t; const poolingDesc: cudnnPoolingDescriptor_t; const alpha: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const beta: Pointer; const yDesc: cudnnTensorDescriptor_t; y: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnCreateActivationDescriptor(activationDesc: PcudnnActivationDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnSetActivationDescriptor(activationDesc: cudnnActivationDescriptor_t; mode: cudnnActivationMode_t; reluNanOpt: cudnnNanPropagation_t; coef: double):cudnnStatus_t; winapi; external libname;
 function cudnnGetActivationDescriptor(const activationDesc: cudnnActivationDescriptor_t; mode: PcudnnActivationMode_t; reluNanOpt: PcudnnNanPropagation_t; coef: Pdouble):cudnnStatus_t; winapi; external libname;
 function cudnnSetActivationDescriptorSwishBeta(activationDesc: cudnnActivationDescriptor_t; swish_beta: double):cudnnStatus_t; winapi; external libname;
 function cudnnGetActivationDescriptorSwishBeta(activationDesc: cudnnActivationDescriptor_t; swish_beta: Pdouble):cudnnStatus_t; winapi; external libname;
 function cudnnDestroyActivationDescriptor(activationDesc: cudnnActivationDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnActivationForward(handle: cudnnHandle_t; activationDesc: cudnnActivationDescriptor_t; const alpha: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const beta: Pointer; const yDesc: cudnnTensorDescriptor_t; y: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnCreateLRNDescriptor(normDesc: PcudnnLRNDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnSetLRNDescriptor(normDesc: cudnnLRNDescriptor_t; lrnN: longword; lrnAlpha: double; lrnBeta: double; lrnK: double):cudnnStatus_t; winapi; external libname;
 function cudnnGetLRNDescriptor(normDesc: cudnnLRNDescriptor_t; lrnN: Plongword; lrnAlpha: Pdouble; lrnBeta: Pdouble; lrnK: Pdouble):cudnnStatus_t; winapi; external libname;
 function cudnnDestroyLRNDescriptor(lrnDesc: cudnnLRNDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnLRNCrossChannelForward(handle: cudnnHandle_t; normDesc: cudnnLRNDescriptor_t; lrnMode: cudnnLRNMode_t; const alpha: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const beta: Pointer; const yDesc: cudnnTensorDescriptor_t; y: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnDivisiveNormalizationForward(handle: cudnnHandle_t; normDesc: cudnnLRNDescriptor_t; mode: cudnnDivNormMode_t; const alpha: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const means: Pointer; temp: Pointer; temp2: Pointer; const beta: Pointer; const yDesc: cudnnTensorDescriptor_t; y: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnDeriveBNTensorDescriptor(derivedBnDesc: cudnnTensorDescriptor_t; const xDesc: cudnnTensorDescriptor_t; mode: cudnnBatchNormMode_t):cudnnStatus_t; winapi; external libname;
 function cudnnBatchNormalizationForwardInference(handle: cudnnHandle_t; mode: cudnnBatchNormMode_t; const alpha: Pointer; const beta: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const yDesc: cudnnTensorDescriptor_t; y: Pointer; const bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t; const bnScale: Pointer; const bnBias: Pointer; const estimatedMean: Pointer; const estimatedVariance: Pointer; epsilon: double):cudnnStatus_t; winapi; external libname;
 function cudnnDeriveNormTensorDescriptor(derivedNormScaleBiasDesc: cudnnTensorDescriptor_t; derivedNormMeanVarDesc: cudnnTensorDescriptor_t; const xDesc: cudnnTensorDescriptor_t; mode: cudnnNormMode_t; groupCnt: longint):cudnnStatus_t; winapi; external libname;
 function cudnnNormalizationForwardInference(handle: cudnnHandle_t; mode: cudnnNormMode_t; normOps: cudnnNormOps_t; algo: cudnnNormAlgo_t; const alpha: Pointer; const beta: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const normScaleBiasDesc: cudnnTensorDescriptor_t; const normScale: Pointer; const normBias: Pointer; const normMeanVarDesc: cudnnTensorDescriptor_t; const estimatedMean: Pointer; const estimatedVariance: Pointer; const zDesc: cudnnTensorDescriptor_t; const z: Pointer; activationDesc: cudnnActivationDescriptor_t; const yDesc: cudnnTensorDescriptor_t; y: Pointer; epsilon: double; groupCnt: longint):cudnnStatus_t; winapi; external libname;
 function cudnnCreateSpatialTransformerDescriptor(stDesc: PcudnnSpatialTransformerDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnSetSpatialTransformerNdDescriptor(stDesc: cudnnSpatialTransformerDescriptor_t; samplerType: cudnnSamplerType_t; dataType: cudnnDataType_t; const nbDims: longint; const dimA: array of longint):cudnnStatus_t; winapi; external libname;
 function cudnnDestroySpatialTransformerDescriptor(stDesc: cudnnSpatialTransformerDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnSpatialTfGridGeneratorForward(handle: cudnnHandle_t; const stDesc: cudnnSpatialTransformerDescriptor_t; const theta: Pointer; grid: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnSpatialTfSamplerForward(handle: cudnnHandle_t; stDesc: cudnnSpatialTransformerDescriptor_t; const alpha: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const grid: Pointer; const beta: Pointer; yDesc: cudnnTensorDescriptor_t; y: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnCreateDropoutDescriptor(dropoutDesc: PcudnnDropoutDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnDestroyDropoutDescriptor(dropoutDesc: cudnnDropoutDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnDropoutGetStatesSize(handle: cudnnHandle_t; sizeInBytes: PSizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnDropoutGetReserveSpaceSize(xdesc: cudnnTensorDescriptor_t; sizeInBytes: PSizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnSetDropoutDescriptor(dropoutDesc: cudnnDropoutDescriptor_t; handle: cudnnHandle_t; dropout: single; states: Pointer; stateSizeInBytes: SizeInt; seed: uint64):cudnnStatus_t; winapi; external libname;
 function cudnnRestoreDropoutDescriptor(dropoutDesc: cudnnDropoutDescriptor_t; handle: cudnnHandle_t; dropout: single; states: Pointer; stateSizeInBytes: SizeInt; seed: uint64):cudnnStatus_t; winapi; external libname;
 function cudnnGetDropoutDescriptor(dropoutDesc: cudnnDropoutDescriptor_t; handle: cudnnHandle_t; dropout: Psingle; states: PPointer; seed: Puint64):cudnnStatus_t; winapi; external libname;
 function cudnnDropoutForward(handle: cudnnHandle_t; const dropoutDesc: cudnnDropoutDescriptor_t; const xdesc: cudnnTensorDescriptor_t; const x: Pointer; const ydesc: cudnnTensorDescriptor_t; y: Pointer; reserveSpace: Pointer; reserveSpaceSizeInBytes: SizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnOpsVersionCheck():cudnnStatus_t; winapi; external libname;
 function cudnnSoftmaxBackward(handle: cudnnHandle_t; algo: cudnnSoftmaxAlgorithm_t; mode: cudnnSoftmaxMode_t; const alpha: Pointer; const yDesc: cudnnTensorDescriptor_t; const y: Pointer; const dyDesc: cudnnTensorDescriptor_t; const dy: Pointer; const beta: Pointer; const dxDesc: cudnnTensorDescriptor_t; dx: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnPoolingBackward(handle: cudnnHandle_t; const poolingDesc: cudnnPoolingDescriptor_t; const alpha: Pointer; const yDesc: cudnnTensorDescriptor_t; const y: Pointer; const dyDesc: cudnnTensorDescriptor_t; const dy: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const beta: Pointer; const dxDesc: cudnnTensorDescriptor_t; dx: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnActivationBackward(handle: cudnnHandle_t; activationDesc: cudnnActivationDescriptor_t; const alpha: Pointer; const yDesc: cudnnTensorDescriptor_t; const y: Pointer; const dyDesc: cudnnTensorDescriptor_t; const dy: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const beta: Pointer; const dxDesc: cudnnTensorDescriptor_t; dx: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnLRNCrossChannelBackward(handle: cudnnHandle_t; normDesc: cudnnLRNDescriptor_t; lrnMode: cudnnLRNMode_t; const alpha: Pointer; const yDesc: cudnnTensorDescriptor_t; const y: Pointer; const dyDesc: cudnnTensorDescriptor_t; const dy: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const beta: Pointer; const dxDesc: cudnnTensorDescriptor_t; dx: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnDivisiveNormalizationBackward(handle: cudnnHandle_t; normDesc: cudnnLRNDescriptor_t; mode: cudnnDivNormMode_t; const alpha: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const means: Pointer; const dy: Pointer; temp: Pointer; temp2: Pointer; const beta: Pointer; const dXdMeansDesc: cudnnTensorDescriptor_t; dx: Pointer; dMeans: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle: cudnnHandle_t; mode: cudnnBatchNormMode_t; bnOps: cudnnBatchNormOps_t; const xDesc: cudnnTensorDescriptor_t; const zDesc: cudnnTensorDescriptor_t; const yDesc: cudnnTensorDescriptor_t; const bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t; const activationDesc: cudnnActivationDescriptor_t; sizeInBytes: PSizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle: cudnnHandle_t; mode: cudnnBatchNormMode_t; bnOps: cudnnBatchNormOps_t; const xDesc: cudnnTensorDescriptor_t; const yDesc: cudnnTensorDescriptor_t; const dyDesc: cudnnTensorDescriptor_t; const dzDesc: cudnnTensorDescriptor_t; const dxDesc: cudnnTensorDescriptor_t; const dBnScaleBiasDesc: cudnnTensorDescriptor_t; const activationDesc: cudnnActivationDescriptor_t; sizeInBytes: PSizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle: cudnnHandle_t; mode: cudnnBatchNormMode_t; bnOps: cudnnBatchNormOps_t; const activationDesc: cudnnActivationDescriptor_t; const xDesc: cudnnTensorDescriptor_t; sizeInBytes: PSizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnBatchNormalizationForwardTraining(handle: cudnnHandle_t; mode: cudnnBatchNormMode_t; const alpha: Pointer; const beta: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const yDesc: cudnnTensorDescriptor_t; y: Pointer; const bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t; const bnScale: Pointer; const bnBias: Pointer; exponentialAverageFactor: double; resultRunningMean: Pointer; resultRunningVariance: Pointer; epsilon: double; resultSaveMean: Pointer; resultSaveInvVariance: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnBatchNormalizationForwardTrainingEx(handle: cudnnHandle_t; mode: cudnnBatchNormMode_t; bnOps: cudnnBatchNormOps_t; const alpha: Pointer; const beta: Pointer; const xDesc: cudnnTensorDescriptor_t; const xData: Pointer; const zDesc: cudnnTensorDescriptor_t; const zData: Pointer; const yDesc: cudnnTensorDescriptor_t; yData: Pointer; const bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t; const bnScale: Pointer; const bnBias: Pointer; exponentialAverageFactor: double; resultRunningMean: Pointer; resultRunningVariance: Pointer; epsilon: double; resultSaveMean: Pointer; resultSaveInvVariance: Pointer; activationDesc: cudnnActivationDescriptor_t; workspace: Pointer; workSpaceSizeInBytes: SizeInt; reserveSpace: Pointer; reserveSpaceSizeInBytes: SizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnBatchNormalizationBackward(handle: cudnnHandle_t; mode: cudnnBatchNormMode_t; const alphaDataDiff: Pointer; const betaDataDiff: Pointer; const alphaParamDiff: Pointer; const betaParamDiff: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const dyDesc: cudnnTensorDescriptor_t; const dy: Pointer; const dxDesc: cudnnTensorDescriptor_t; dx: Pointer; const dBnScaleBiasDesc: cudnnTensorDescriptor_t; const bnScale: Pointer; dBnScaleResult: Pointer; dBnBiasResult: Pointer; epsilon: double; const savedMean: Pointer; const savedInvVariance: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnBatchNormalizationBackwardEx(handle: cudnnHandle_t; mode: cudnnBatchNormMode_t; bnOps: cudnnBatchNormOps_t; const alphaDataDiff: Pointer; const betaDataDiff: Pointer; const alphaParamDiff: Pointer; const betaParamDiff: Pointer; const xDesc: cudnnTensorDescriptor_t; const xData: Pointer; const yDesc: cudnnTensorDescriptor_t; const yData: Pointer; const dyDesc: cudnnTensorDescriptor_t; const dyData: Pointer; const dzDesc: cudnnTensorDescriptor_t; dzData: Pointer; const dxDesc: cudnnTensorDescriptor_t; dxData: Pointer; const dBnScaleBiasDesc: cudnnTensorDescriptor_t; const bnScaleData: Pointer; const bnBiasData: Pointer; dBnScaleData: Pointer; dBnBiasData: Pointer; epsilon: double; const savedMean: Pointer; const savedInvVariance: Pointer; activationDesc: cudnnActivationDescriptor_t; workSpace: Pointer; workSpaceSizeInBytes: SizeInt; reserveSpace: Pointer; reserveSpaceSizeInBytes: SizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnGetNormalizationForwardTrainingWorkspaceSize(handle: cudnnHandle_t; mode: cudnnNormMode_t; normOps: cudnnNormOps_t; algo: cudnnNormAlgo_t; const xDesc: cudnnTensorDescriptor_t; const zDesc: cudnnTensorDescriptor_t; const yDesc: cudnnTensorDescriptor_t; const normScaleBiasDesc: cudnnTensorDescriptor_t; const activationDesc: cudnnActivationDescriptor_t; const normMeanVarDesc: cudnnTensorDescriptor_t; sizeInBytes: PSizeInt; groupCnt: longint):cudnnStatus_t; winapi; external libname;
 function cudnnGetNormalizationBackwardWorkspaceSize(handle: cudnnHandle_t; mode: cudnnNormMode_t; normOps: cudnnNormOps_t; algo: cudnnNormAlgo_t; const xDesc: cudnnTensorDescriptor_t; const yDesc: cudnnTensorDescriptor_t; const dyDesc: cudnnTensorDescriptor_t; const dzDesc: cudnnTensorDescriptor_t; const dxDesc: cudnnTensorDescriptor_t; const dNormScaleBiasDesc: cudnnTensorDescriptor_t; const activationDesc: cudnnActivationDescriptor_t; const normMeanVarDesc: cudnnTensorDescriptor_t; sizeInBytes: PSizeInt; groupCnt: longint):cudnnStatus_t; winapi; external libname;
 function cudnnGetNormalizationTrainingReserveSpaceSize(handle: cudnnHandle_t; mode: cudnnNormMode_t; normOps: cudnnNormOps_t; algo: cudnnNormAlgo_t; const activationDesc: cudnnActivationDescriptor_t; const xDesc: cudnnTensorDescriptor_t; sizeInBytes: PSizeInt; groupCnt: longint):cudnnStatus_t; winapi; external libname;
 function cudnnNormalizationForwardTraining(handle: cudnnHandle_t; mode: cudnnNormMode_t; normOps: cudnnNormOps_t; algo: cudnnNormAlgo_t; const alpha: Pointer; const beta: Pointer; const xDesc: cudnnTensorDescriptor_t; const xData: Pointer; const normScaleBiasDesc: cudnnTensorDescriptor_t; const normScale: Pointer; const normBias: Pointer; exponentialAverageFactor: double; const normMeanVarDesc: cudnnTensorDescriptor_t; resultRunningMean: Pointer; resultRunningVariance: Pointer; epsilon: double; resultSaveMean: Pointer; resultSaveInvVariance: Pointer; activationDesc: cudnnActivationDescriptor_t; const zDesc: cudnnTensorDescriptor_t; const zData: Pointer; const yDesc: cudnnTensorDescriptor_t; yData: Pointer; workspace: Pointer; workSpaceSizeInBytes: SizeInt; reserveSpace: Pointer; reserveSpaceSizeInBytes: SizeInt; groupCnt: longint):cudnnStatus_t; winapi; external libname;
 function cudnnNormalizationBackward(handle: cudnnHandle_t; mode: cudnnNormMode_t; normOps: cudnnNormOps_t; algo: cudnnNormAlgo_t; const alphaDataDiff: Pointer; const betaDataDiff: Pointer; const alphaParamDiff: Pointer; const betaParamDiff: Pointer; const xDesc: cudnnTensorDescriptor_t; const xData: Pointer; const yDesc: cudnnTensorDescriptor_t; const yData: Pointer; const dyDesc: cudnnTensorDescriptor_t; const dyData: Pointer; const dzDesc: cudnnTensorDescriptor_t; dzData: Pointer; const dxDesc: cudnnTensorDescriptor_t; dxData: Pointer; const dNormScaleBiasDesc: cudnnTensorDescriptor_t; const normScaleData: Pointer; const normBiasData: Pointer; dNormScaleData: Pointer; dNormBiasData: Pointer; epsilon: double; const normMeanVarDesc: cudnnTensorDescriptor_t; const savedMean: Pointer; const savedInvVariance: Pointer; activationDesc: cudnnActivationDescriptor_t; workSpace: Pointer; workSpaceSizeInBytes: SizeInt; reserveSpace: Pointer; reserveSpaceSizeInBytes: SizeInt; groupCnt: longint):cudnnStatus_t; winapi; external libname;
 function cudnnSpatialTfGridGeneratorBackward(handle: cudnnHandle_t; const stDesc: cudnnSpatialTransformerDescriptor_t; const dgrid: Pointer; dtheta: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnSpatialTfSamplerBackward(handle: cudnnHandle_t; stDesc: cudnnSpatialTransformerDescriptor_t; const alpha: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const beta: Pointer; const dxDesc: cudnnTensorDescriptor_t; dx: Pointer; const alphaDgrid: Pointer; const dyDesc: cudnnTensorDescriptor_t; const dy: Pointer; const grid: Pointer; const betaDgrid: Pointer; dgrid: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnDropoutBackward(handle: cudnnHandle_t; const dropoutDesc: cudnnDropoutDescriptor_t; const dydesc: cudnnTensorDescriptor_t; const dy: Pointer; const dxdesc: cudnnTensorDescriptor_t; dx: Pointer; reserveSpace: Pointer; reserveSpaceSizeInBytes: SizeInt):cudnnStatus_t; winapi; external libname;


implementation

end.

