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

unit cudnn_cnn;

{$ifdef FPC}
{$mode Delphi}
{$endif}

interface

uses
  cudnn_graph, cudnn_ops;

type
    PcudnnFusedOps_t = ^cudnnFusedOps_t;
    cudnnFusedOps_t= (
    (* each op in [ ] can be disabled by passing NULL ptr *)
    (* [per channel scale], [per channel bias], [activation], convolution, [generate BN stats] *)
    CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS = 0,
    (* [per channel scale], [per channel bias], [activation], convolutionBackwardWeights *)
    CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD = 1,
    (* utility for BN training in BN-conv fusion *)
    (* computes the equivalent scale and bias from ySum ySqSum and learned scale, bias *)
    (* optionally update running stats and generate saved stats *)
    CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING = 2,
    (* utility for BN inference in BN-conv fusion *)
    (* computes the equivalent scale and bias from learned running stats and learned scale, bias *)
    CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE = 3,
    (* reserved for future use: convolution, [per channel scale], [per channel bias], [residual add], [activation] *)
    CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION = 4,
    (* reserved for future use: [per channel scale], [per channel bias], [residual add],  activation, bitmask *)
    CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK = 5,
    (* reserved for future use *)
    CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM = 6
) ;

  PcudnnFusedOpsConstParamLabel_t = ^cudnnFusedOpsConstParamLabel_t;
  cudnnFusedOpsConstParamLabel_t = (
    (* set XDESC: pass previously initialized cudnnTensorDescriptor_t *)
    (* get XDESC: pass previously created cudnnTensorDescriptor_t *)
    CUDNN_PARAM_XDESC = 0,
    (* set/get XDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_XDATA_PLACEHOLDER = 1,
    (* set/get BN_MODE: pass cudnnBatchNormMode_t* *)
    CUDNN_PARAM_BN_MODE = 2,
    (* set CUDNN_PARAM_BN_EQSCALEBIAS_DESC: pass previously initialized cudnnTensorDescriptor_t *)
    (* get CUDNN_PARAM_BN_EQSCALEBIAS_DESC: pass previously created cudnnTensorDescriptor_t *)
    CUDNN_PARAM_BN_EQSCALEBIAS_DESC = 3,
    (* set/get BN_EQSCALE_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER = 4,
    (* set/get BN_EQBIAS_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER = 5,
    (* set ACTIVATION_DESC: pass previously initialized cudnnActivationDescriptor_t *)
    (* get ACTIVATION_DESC: pass previously created cudnnActivationDescriptor_t *)
    CUDNN_PARAM_ACTIVATION_DESC = 6,
    (* set CONV_DESC: pass previously initialized cudnnConvolutionDescriptor_t *)
    (* get CONV_DESC: pass previously created cudnnConvolutionDescriptor_t *)
    CUDNN_PARAM_CONV_DESC = 7,
    (* set WDESC: pass previously initialized cudnnFilterDescriptor_t *)
    (* get WDESC: pass previously created cudnnFilterDescriptor_t *)
    CUDNN_PARAM_WDESC = 8,
    (* set/get WDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_WDATA_PLACEHOLDER = 9,
    (* set DWDESC: pass previously initialized cudnnFilterDescriptor_t *)
    (* get DWDESC: pass previously created cudnnFilterDescriptor_t *)
    CUDNN_PARAM_DWDESC = 10,
    (* set/get DWDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_DWDATA_PLACEHOLDER = 11,
    (* set YDESC: pass previously initialized cudnnTensorDescriptor_t *)
    (* get YDESC: pass previously created cudnnTensorDescriptor_t *)
    CUDNN_PARAM_YDESC = 12,
    (* set/get YDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_YDATA_PLACEHOLDER = 13,
    (* set DYDESC: pass previously initialized cudnnTensorDescriptor_t *)
    (* get DYDESC: pass previously created cudnnTensorDescriptor_t *)
    CUDNN_PARAM_DYDESC = 14,
    (* set/get DYDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_DYDATA_PLACEHOLDER = 15,
    (* set YSTATS_DESC: pass previously initialized cudnnTensorDescriptor_t *)
    (* get YSTATS_DESC: pass previously created cudnnTensorDescriptor_t *)
    CUDNN_PARAM_YSTATS_DESC = 16,
    (* set/get YSUM_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_YSUM_PLACEHOLDER = 17,
    (* set/get YSQSUM_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_YSQSUM_PLACEHOLDER = 18,
    (* set CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC: pass previously initialized cudnnTensorDescriptor_t *)
    (* get CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC: pass previously created cudnnTensorDescriptor_t *)
    CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC = 19,
    (* set/get CUDNN_PARAM_BN_SCALE_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_BN_SCALE_PLACEHOLDER = 20,
    (* set/get CUDNN_PARAM_BN_BIAS_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_BN_BIAS_PLACEHOLDER = 21,
    (* set/get CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER = 22,
    (* set/get CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER = 23,
    (* set/get CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER = 24,
    (* set/get CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER = 25,

    (* set ZDESC: pass previously initialized cudnnTensorDescriptor_t *)
    (* get ZDESC: pass previously created cudnnTensorDescriptor_t *)
    CUDNN_PARAM_ZDESC = 26,
    (* set/get ZDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_ZDATA_PLACEHOLDER = 27,
    (* set BN_Z_EQSCALEBIAS_DESC: pass previously initialized cudnnTensorDescriptor_t *)
    (* get BN_Z_EQSCALEBIAS_DESC: pass previously created cudnnTensorDescriptor_t *)
    CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC = 28,
    (* set/get BN_Z_EQSCALE_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER = 29,
    (* set/get BN_Z_EQBIAS_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER = 30,

    (* set ACTIVATION_BITMASK_DESC: pass previously initialized cudnnTensorDescriptor_t *)
    (* get ACTIVATION_BITMASK_DESC: pass previously created cudnnTensorDescriptor_t *)
    CUDNN_PARAM_ACTIVATION_BITMASK_DESC = 31,
    (* set/get ACTIVATION_BITMASK_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER = 32,

    (* set DXDESC: pass previously initialized cudnnTensorDescriptor_t *)
    (* get DXDESC: pass previously created cudnnTensorDescriptor_t *)
    CUDNN_PARAM_DXDESC = 33,
    (* set/get DXDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_DXDATA_PLACEHOLDER = 34,
    (* set DZDESC: pass previously initialized cudnnTensorDescriptor_t *)
    (* get DZDESC: pass previously created cudnnTensorDescriptor_t *)
    CUDNN_PARAM_DZDESC = 35,
    (* set/get DZDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_DZDATA_PLACEHOLDER = 36,
    (* set/get CUDNN_PARAM_BN_DSCALE_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_BN_DSCALE_PLACEHOLDER = 37,
    (* set/get CUDNN_PARAM_BN_DBIAS_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* *)
    CUDNN_PARAM_BN_DBIAS_PLACEHOLDER = 38
) ;

  PcudnnFusedOpsPointerPlaceHolder_t = ^cudnnFusedOpsPointerPlaceHolder_t;
  cudnnFusedOpsPointerPlaceHolder_t = (
    CUDNN_PTR_NULL         = 0,
    CUDNN_PTR_ELEM_ALIGNED = 1,
    CUDNN_PTR_16B_ALIGNED  = 2
)  ;

  PcudnnFusedOpsVariantParamLabel_t = ^cudnnFusedOpsVariantParamLabel_t;
  cudnnFusedOpsVariantParamLabel_t = (
    (* set: pass void* pointing to dev memory *)
    (* get: pass void** pointing to host memory *)
    CUDNN_PTR_XDATA              = 0,
    CUDNN_PTR_BN_EQSCALE         = 1,
    CUDNN_PTR_BN_EQBIAS          = 2,
    CUDNN_PTR_WDATA              = 3,
    CUDNN_PTR_DWDATA             = 4,
    CUDNN_PTR_YDATA              = 5,
    CUDNN_PTR_DYDATA             = 6,
    CUDNN_PTR_YSUM               = 7,
    CUDNN_PTR_YSQSUM             = 8,
    CUDNN_PTR_WORKSPACE          = 9,
    CUDNN_PTR_BN_SCALE           = 10,
    CUDNN_PTR_BN_BIAS            = 11,
    CUDNN_PTR_BN_SAVED_MEAN      = 12,
    CUDNN_PTR_BN_SAVED_INVSTD    = 13,
    CUDNN_PTR_BN_RUNNING_MEAN    = 14,
    CUDNN_PTR_BN_RUNNING_VAR     = 15,
    CUDNN_PTR_ZDATA              = 16,
    CUDNN_PTR_BN_Z_EQSCALE       = 17,
    CUDNN_PTR_BN_Z_EQBIAS        = 18,
    CUDNN_PTR_ACTIVATION_BITMASK = 19,
    CUDNN_PTR_DXDATA             = 20,
    CUDNN_PTR_DZDATA             = 21,
    CUDNN_PTR_BN_DSCALE          = 22,
    CUDNN_PTR_BN_DBIAS           = 23,

    (* set/get: pass size_t* pointing to host memory *)
    CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES = 100,
    (* set/get: pass int64_t* pointing to host memory *)
    CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT = 101,
    (* set/get: pass double* pointing to host memory *)
    CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR = 102,
    (* set/get: pass double* pointing to host memory *)
    CUDNN_SCALAR_DOUBLE_BN_EPSILON = 103
) ;

 PcudnnConvolutionDescriptor_t = ^cudnnConvolutionDescriptor_t;
 cudnnConvolutionDescriptor_t = ^cudnnConvolutionStruct;
 cudnnConvolutionStruct = record end ;

 PcudnnConvolutionFwdAlgoPerf_t = ^cudnnConvolutionFwdAlgoPerf_t;
 cudnnConvolutionFwdAlgoPerf_t = record
    algo        : cudnnConvolutionFwdAlgo_t  ;
    status      : cudnnStatus_t              ;
    time        : Single                      ;
    memory      : SizeInt                     ;
    determinism : cudnnDeterminism_t         ;
    mathType    : cudnnMathType_t            ;
    reserved    : array [0..2] of integer    ;
  end;
(* helper function to provide the convolution backward data algo that fit best the requirement *)

  PcudnnConvolutionBwdDataAlgoPerf_t = ^cudnnConvolutionBwdDataAlgoPerf_t;
  cudnnConvolutionBwdDataAlgoPerf_t = record
    algo           : cudnnConvolutionBwdDataAlgo_t;
    status         : cudnnStatus_t                ;
    time           : Single                        ;
    memory         : SizeInt                       ;
    determinism    : cudnnDeterminism_t           ;
    mathType       : cudnnMathType_t              ;
    reserved       : array [0..2] of integer      ;
  end;

(* cudnnFusedOps... *)
 PcudnnFusedOpsConstParamPack_t = ^cudnnFusedOpsConstParamPack_t ;
 cudnnFusedOpsConstParamPack_t = record end;


 PcudnnFusedOpsVariantParamPack_t = ^cudnnFusedOpsVariantParamPack_t ;
 cudnnFusedOpsVariantParamPack_t = record end;

 PcudnnFusedOpsPlan_t = ^cudnnFusedOpsPlan_t;
 cudnnFusedOpsPlan_t = record end;

(* helper function to provide the convolution backward filter algo that fit best the requirement *)

 PcudnnConvolutionBwdFilterAlgoPerf_t = ^cudnnConvolutionBwdFilterAlgoPerf_t;
 cudnnConvolutionBwdFilterAlgoPerf_t = record
   algo           : cudnnConvolutionBwdFilterAlgo_t;
   status         : cudnnStatus_t                  ;
   time           : Single                          ;
   memory         : SizeInt                         ;
   determinism    : cudnnDeterminism_t             ;
   mathType       : cudnnMathType_t                ;
   reserved       : array [0..2] of integer        ;
 end;

 function cudnnCreateConvolutionDescriptor(convDesc: PcudnnConvolutionDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnDestroyConvolutionDescriptor(convDesc: cudnnConvolutionDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnSetConvolutionMathType(convDesc: cudnnConvolutionDescriptor_t; mathType: cudnnMathType_t):cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolutionMathType(convDesc: cudnnConvolutionDescriptor_t; mathType: PcudnnMathType_t):cudnnStatus_t; winapi; external libname;
 function cudnnSetConvolutionGroupCount(convDesc: cudnnConvolutionDescriptor_t; groupCount: longint):cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolutionGroupCount(convDesc: cudnnConvolutionDescriptor_t; groupCount: Plongint):cudnnStatus_t; winapi; external libname;
 function cudnnSetConvolutionReorderType(convDesc: cudnnConvolutionDescriptor_t; reorderType: cudnnReorderType_t):cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolutionReorderType(convDesc: cudnnConvolutionDescriptor_t; reorderType: PcudnnReorderType_t):cudnnStatus_t; winapi; external libname;
 function cudnnSetConvolution2dDescriptor(convDesc: cudnnConvolutionDescriptor_t; pad_h: longint; pad_w: longint; u: longint; v: longint; dilation_h: longint; dilation_w: longint; mode: cudnnConvolutionMode_t; computeType: cudnnDataType_t):cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolution2dDescriptor(const convDesc: cudnnConvolutionDescriptor_t; pad_h: Plongint; pad_w: Plongint; u: Plongint; v: Plongint; dilation_h: Plongint; dilation_w: Plongint; mode: PcudnnConvolutionMode_t; computeType: PcudnnDataType_t):cudnnStatus_t; winapi; external libname;
 function cudnnSetConvolutionNdDescriptor(convDesc: cudnnConvolutionDescriptor_t; arrayLength: longint; const padA: array of longint; const filterStrideA: array of longint; const dilationA: array of longint; mode: cudnnConvolutionMode_t; computeType: cudnnDataType_t):cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolutionNdDescriptor(const convDesc: cudnnConvolutionDescriptor_t; arrayLengthRequested: longint; arrayLength: Plongint; padA: array of longint; strideA: array of longint; dilationA: array of longint; mode: PcudnnConvolutionMode_t; computeType: PcudnnDataType_t):cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolution2dForwardOutputDim(const convDesc: cudnnConvolutionDescriptor_t; const inputTensorDesc: cudnnTensorDescriptor_t; const filterDesc: cudnnFilterDescriptor_t; n: Plongint; c: Plongint; h: Plongint; w: Plongint):cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolutionNdForwardOutputDim(const convDesc: cudnnConvolutionDescriptor_t; const inputTensorDesc: cudnnTensorDescriptor_t; const filterDesc: cudnnFilterDescriptor_t; nbDims: longint; tensorOuputDimA: array of longint):cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolutionForwardAlgorithmMaxCount(handle: cudnnHandle_t; count: Plongint):cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolutionForwardAlgorithm_v7(handle: cudnnHandle_t; const srcDesc: cudnnTensorDescriptor_t; const filterDesc: cudnnFilterDescriptor_t; const convDesc: cudnnConvolutionDescriptor_t; const destDesc: cudnnTensorDescriptor_t; const requestedAlgoCount: longint; returnedAlgoCount: Plongint; perfResults: PcudnnConvolutionFwdAlgoPerf_t):cudnnStatus_t; winapi; external libname;
 function cudnnFindConvolutionForwardAlgorithm(handle: cudnnHandle_t; const xDesc: cudnnTensorDescriptor_t; const wDesc: cudnnFilterDescriptor_t; const convDesc: cudnnConvolutionDescriptor_t; const yDesc: cudnnTensorDescriptor_t; const requestedAlgoCount: longint; returnedAlgoCount: Plongint; perfResults: PcudnnConvolutionFwdAlgoPerf_t):cudnnStatus_t; winapi; external libname;
 function cudnnFindConvolutionForwardAlgorithmEx(handle: cudnnHandle_t; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const wDesc: cudnnFilterDescriptor_t; const w: Pointer; const convDesc: cudnnConvolutionDescriptor_t; const yDesc: cudnnTensorDescriptor_t; y: Pointer; const requestedAlgoCount: longint; returnedAlgoCount: Plongint; perfResults: PcudnnConvolutionFwdAlgoPerf_t; workSpace: Pointer; workSpaceSizeInBytes: SizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnIm2Col(handle: cudnnHandle_t; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const wDesc: cudnnFilterDescriptor_t; const convDesc: cudnnConvolutionDescriptor_t; colBuffer: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnReorderFilterAndBias(handle: cudnnHandle_t; const filterDesc: cudnnFilterDescriptor_t; reorderType: cudnnReorderType_t; const filterData: Pointer; reorderedFilterData: Pointer; reorderBias: longint; const biasData: Pointer; reorderedBiasData: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolutionForwardWorkspaceSize(handle: cudnnHandle_t; const xDesc: cudnnTensorDescriptor_t; const wDesc: cudnnFilterDescriptor_t; const convDesc: cudnnConvolutionDescriptor_t; const yDesc: cudnnTensorDescriptor_t; algo: cudnnConvolutionFwdAlgo_t; sizeInBytes : PSizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnConvolutionForward(handle: cudnnHandle_t; const alpha: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const wDesc: cudnnFilterDescriptor_t; const w: Pointer; const convDesc: cudnnConvolutionDescriptor_t; algo: cudnnConvolutionFwdAlgo_t; workSpace: Pointer; workSpaceSizeInBytes: SizeInt; const beta: Pointer; const yDesc: cudnnTensorDescriptor_t; y: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnConvolutionBiasActivationForward(handle: cudnnHandle_t; const alpha1: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const wDesc: cudnnFilterDescriptor_t; const w: Pointer; const convDesc: cudnnConvolutionDescriptor_t; algo: cudnnConvolutionFwdAlgo_t; workSpace: Pointer; workSpaceSizeInBytes: SizeInt; const alpha2: Pointer; const zDesc: cudnnTensorDescriptor_t; const z: Pointer; const biasDesc: cudnnTensorDescriptor_t; const bias: Pointer; const activationDesc: cudnnActivationDescriptor_t; const yDesc: cudnnTensorDescriptor_t; y: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle: cudnnHandle_t; count: Plongint):cudnnStatus_t; winapi; external libname;
 function cudnnFindConvolutionBackwardDataAlgorithm(handle: cudnnHandle_t; const wDesc: cudnnFilterDescriptor_t; const dyDesc: cudnnTensorDescriptor_t; const convDesc: cudnnConvolutionDescriptor_t; const dxDesc: cudnnTensorDescriptor_t; const requestedAlgoCount: longint; returnedAlgoCount: Plongint; perfResults: PcudnnConvolutionBwdDataAlgoPerf_t):cudnnStatus_t; winapi; external libname;
 function cudnnFindConvolutionBackwardDataAlgorithmEx(handle: cudnnHandle_t; const wDesc: cudnnFilterDescriptor_t; const w: Pointer; const dyDesc: cudnnTensorDescriptor_t; const dy: Pointer; const convDesc: cudnnConvolutionDescriptor_t; const dxDesc: cudnnTensorDescriptor_t; dx: Pointer; const requestedAlgoCount: longint; returnedAlgoCount: Plongint; perfResults: PcudnnConvolutionBwdDataAlgoPerf_t; workSpace: Pointer; workSpaceSizeInBytes: SizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolutionBackwardDataAlgorithm_v7(handle: cudnnHandle_t; const filterDesc: cudnnFilterDescriptor_t; const diffDesc: cudnnTensorDescriptor_t; const convDesc: cudnnConvolutionDescriptor_t; const gradDesc: cudnnTensorDescriptor_t; const requestedAlgoCount: longint; returnedAlgoCount: Plongint; perfResults: PcudnnConvolutionBwdDataAlgoPerf_t):cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolutionBackwardDataWorkspaceSize(handle: cudnnHandle_t; const wDesc: cudnnFilterDescriptor_t; const dyDesc: cudnnTensorDescriptor_t; const convDesc: cudnnConvolutionDescriptor_t; const dxDesc: cudnnTensorDescriptor_t; algo: cudnnConvolutionBwdDataAlgo_t; sizeInBytes : PSizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnConvolutionBackwardData(handle: cudnnHandle_t; const alpha: Pointer; const wDesc: cudnnFilterDescriptor_t; const w: Pointer; const dyDesc: cudnnTensorDescriptor_t; const dy: Pointer; const convDesc: cudnnConvolutionDescriptor_t; algo: cudnnConvolutionBwdDataAlgo_t; workSpace: Pointer; workSpaceSizeInBytes: SizeInt; const beta: Pointer; const dxDesc: cudnnTensorDescriptor_t; dx: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnGetFoldedConvBackwardDataDescriptors(const handle: cudnnHandle_t; const filterDesc: cudnnFilterDescriptor_t; const diffDesc: cudnnTensorDescriptor_t; const convDesc: cudnnConvolutionDescriptor_t; const gradDesc: cudnnTensorDescriptor_t; const transformFormat: cudnnTensorFormat_t; foldedFilterDesc: cudnnFilterDescriptor_t; paddedDiffDesc: cudnnTensorDescriptor_t; foldedConvDesc: cudnnConvolutionDescriptor_t; foldedGradDesc: cudnnTensorDescriptor_t; filterFoldTransDesc: cudnnTensorTransformDescriptor_t; diffPadTransDesc: cudnnTensorTransformDescriptor_t; gradFoldTransDesc: cudnnTensorTransformDescriptor_t; gradUnfoldTransDesc: cudnnTensorTransformDescriptor_t):cudnnStatus_t; winapi; external libname;
 function cudnnCnnVersionCheck():cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle: cudnnHandle_t; count: Plongint):cudnnStatus_t; winapi; external libname;
 function cudnnFindConvolutionBackwardFilterAlgorithm(handle: cudnnHandle_t; const xDesc: cudnnTensorDescriptor_t; const dyDesc: cudnnTensorDescriptor_t; const convDesc: cudnnConvolutionDescriptor_t; const dwDesc: cudnnFilterDescriptor_t; const requestedAlgoCount: longint; returnedAlgoCount: Plongint; perfResults: PcudnnConvolutionBwdFilterAlgoPerf_t):cudnnStatus_t; winapi; external libname;
 function cudnnFindConvolutionBackwardFilterAlgorithmEx(handle: cudnnHandle_t; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const dyDesc: cudnnTensorDescriptor_t; const y: Pointer; const convDesc: cudnnConvolutionDescriptor_t; const dwDesc: cudnnFilterDescriptor_t; dw: Pointer; const requestedAlgoCount: longint; returnedAlgoCount: Plongint; perfResults: PcudnnConvolutionBwdFilterAlgoPerf_t; workSpace: Pointer; workSpaceSizeInBytes: SizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle: cudnnHandle_t; const srcDesc: cudnnTensorDescriptor_t; const diffDesc: cudnnTensorDescriptor_t; const convDesc: cudnnConvolutionDescriptor_t; const gradDesc: cudnnFilterDescriptor_t; const requestedAlgoCount: longint; returnedAlgoCount: Plongint; perfResults: PcudnnConvolutionBwdFilterAlgoPerf_t):cudnnStatus_t; winapi; external libname;
 function cudnnGetConvolutionBackwardFilterWorkspaceSize(handle: cudnnHandle_t; const xDesc: cudnnTensorDescriptor_t; const dyDesc: cudnnTensorDescriptor_t; const convDesc: cudnnConvolutionDescriptor_t; const gradDesc: cudnnFilterDescriptor_t; algo: cudnnConvolutionBwdFilterAlgo_t; sizeInBytes : PSizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnConvolutionBackwardFilter(handle: cudnnHandle_t; const alpha: Pointer; const xDesc: cudnnTensorDescriptor_t; const x: Pointer; const dyDesc: cudnnTensorDescriptor_t; const dy: Pointer; const convDesc: cudnnConvolutionDescriptor_t; algo: cudnnConvolutionBwdFilterAlgo_t; workSpace: Pointer; workSpaceSizeInBytes: SizeInt; const beta: Pointer; const dwDesc: cudnnFilterDescriptor_t; dw: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnConvolutionBackwardBias(handle: cudnnHandle_t; const alpha: Pointer; const dyDesc: cudnnTensorDescriptor_t; const dy: Pointer; const beta: Pointer; const dbDesc: cudnnTensorDescriptor_t; db: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnCreateFusedOpsConstParamPack(constPack: PcudnnFusedOpsConstParamPack_t; ops: cudnnFusedOps_t):cudnnStatus_t; winapi; external libname;
 function cudnnDestroyFusedOpsConstParamPack(constPack: cudnnFusedOpsConstParamPack_t):cudnnStatus_t; winapi; external libname;
 function cudnnSetFusedOpsConstParamPackAttribute(constPack: cudnnFusedOpsConstParamPack_t; paramLabel: cudnnFusedOpsConstParamLabel_t; const param: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnGetFusedOpsConstParamPackAttribute(const constPack: cudnnFusedOpsConstParamPack_t; paramLabel: cudnnFusedOpsConstParamLabel_t; param: Pointer; isNULL: Plongint):cudnnStatus_t; winapi; external libname;
 function cudnnCreateFusedOpsVariantParamPack(varPack: PcudnnFusedOpsVariantParamPack_t; ops: cudnnFusedOps_t):cudnnStatus_t; winapi; external libname;
 function cudnnDestroyFusedOpsVariantParamPack(varPack: cudnnFusedOpsVariantParamPack_t):cudnnStatus_t; winapi; external libname;
 function cudnnSetFusedOpsVariantParamPackAttribute(varPack: cudnnFusedOpsVariantParamPack_t; paramLabel: cudnnFusedOpsVariantParamLabel_t; ptr: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnGetFusedOpsVariantParamPackAttribute(const varPack: cudnnFusedOpsVariantParamPack_t; paramLabel: cudnnFusedOpsVariantParamLabel_t; ptr: Pointer):cudnnStatus_t; winapi; external libname;
 function cudnnCreateFusedOpsPlan(plan: PcudnnFusedOpsPlan_t; ops: cudnnFusedOps_t):cudnnStatus_t; winapi; external libname;
 function cudnnDestroyFusedOpsPlan(plan: cudnnFusedOpsPlan_t):cudnnStatus_t; winapi; external libname;
 function cudnnMakeFusedOpsPlan(handle: cudnnHandle_t; plan: cudnnFusedOpsPlan_t; const constPack: cudnnFusedOpsConstParamPack_t; workspaceSizeInBytes : PSizeInt):cudnnStatus_t; winapi; external libname;
 function cudnnFusedOpsExecute(handle: cudnnHandle_t; const plan: cudnnFusedOpsPlan_t; varPack: cudnnFusedOpsVariantParamPack_t):cudnnStatus_t; winapi; external libname;



implementation

end.

