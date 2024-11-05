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

unit cudnn_adv;
{$ifdef FPC}
  {$mode delphi}
{$endif}
interface

uses cudnn_graph, cudnn_ops;

const
  CUDNN_RNN_PADDED_IO_DISABLED = 0;
  CUDNN_RNN_PADDED_IO_ENABLED = 1 shl 0;

  CUDNN_SEQDATA_DIM_COUNT = 4;

  CUDNN_ATTN_QUERYMAP_ALL_TO_ONE =  0         ;
  CUDNN_ATTN_QUERYMAP_ONE_TO_ONE =  (1 shl 0) ;

  CUDNN_ATTN_DISABLE_PROJ_BIASES =  0         ;
  CUDNN_ATTN_ENABLE_PROJ_BIASES  =  (1 shl 1) ;

  CUDNN_ATTN_WKIND_COUNT = 8;

type

  PcudnnRNNDescriptor_t = ^cudnnRNNDescriptor_t;
  cudnnRNNDescriptor_t = ^cudnnRNNStruct;
  cudnnRNNStruct = record
  end;

  PcudnnRNNDataDescriptor_t = ^cudnnRNNDataDescriptor_t;
  cudnnRNNDataDescriptor_t = ^cudnnRNNDataStruct;

  cudnnRNNDataStruct = record
  end;

  PcudnnSeqDataDescriptor_t = ^cudnnSeqDataDescriptor_t;
  cudnnSeqDataDescriptor_t = ^cudnnSeqDataStruct;
  cudnnSeqDataStruct = record
  end;

  PcudnnAttnDescriptor_t = ^cudnnAttnDescriptor_t;
  cudnnAttnDescriptor_t = ^cudnnAttnStruct;
  cudnnAttnStruct=record end;

  PcudnnRNNAlgo_t  = ^cudnnRNNAlgo_t;
  cudnnRNNAlgo_t = (CUDNN_RNN_ALGO_STANDARD = 0, CUDNN_RNN_ALGO_PERSIST_STATIC = 1, CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2, CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H = 3, CUDNN_RNN_ALGO_COUNT = 4);

  PcudnnForwardMode_t = ^cudnnForwardMode_t;
  cudnnForwardMode_t = (CUDNN_FWD_MODE_INFERENCE = 0, CUDNN_FWD_MODE_TRAINING  = 1);

  PcudnnRNNMode_t = ^cudnnRNNMode_t;
  cudnnRNNMode_t = (CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1, CUDNN_LSTM = 2, CUDNN_GRU = 3);

  PcudnnRNNBiasMode_t = ^cudnnRNNBiasMode_t;
  cudnnRNNBiasMode_t = (CUDNN_RNN_NO_BIAS = 0, CUDNN_RNN_SINGLE_INP_BIAS = 1, CUDNN_RNN_DOUBLE_BIAS = 2, CUDNN_RNN_SINGLE_REC_BIAS = 3);

  PcudnnDirectionMode_t = ^cudnnDirectionMode_t;
  cudnnDirectionMode_t = (CUDNN_UNIDIRECTIONAL = 0, CUDNN_BIDIRECTIONAL  = 1);

  PcudnnRNNInputMode_t = ^cudnnRNNInputMode_t;
  cudnnRNNInputMode_t = (CUDNN_LINEAR_INPUT = 0, CUDNN_SKIP_INPUT   = 1);

  PcudnnRNNClipMode_t = ^cudnnRNNClipMode_t;
  cudnnRNNClipMode_t = (CUDNN_RNN_CLIP_NONE = 0, CUDNN_RNN_CLIP_MINMAX = 1);

  PcudnnRNNDataLayout_t = ^cudnnRNNDataLayout_t;
  cudnnRNNDataLayout_t = (CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED   = 0, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED = 1, CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED = 2);

  PcudnnSeqDataAxis_t = ^cudnnSeqDataAxis_t;
  cudnnSeqDataAxis_t = (CUDNN_SEQDATA_BATCH_DIM = 1,  CUDNN_SEQDATA_BEAM_DIM  = 2, CUDNN_SEQDATA_VECT_DIM  = 3) ;

  PcudnnMultiHeadAttnWeightKind_t = ^cudnnMultiHeadAttnWeightKind_t;
  cudnnMultiHeadAttnWeightKind_t = (
    CUDNN_MH_ATTN_Q_WEIGHTS = 0,
    CUDNN_MH_ATTN_K_WEIGHTS = 1,
    CUDNN_MH_ATTN_V_WEIGHTS = 2,
    CUDNN_MH_ATTN_O_WEIGHTS = 3,
    CUDNN_MH_ATTN_Q_BIASES  = 4,
    CUDNN_MH_ATTN_K_BIASES  = 5,
    CUDNN_MH_ATTN_V_BIASES  = 6,
    CUDNN_MH_ATTN_O_BIASES  = 7
  ) ;

  PcudnnWgradMode_t= ^cudnnWgradMode_t;
  cudnnWgradMode_t = (
    CUDNN_WGRAD_MODE_ADD = 0,
    CUDNN_WGRAD_MODE_SET = 1
  ) ;

  PcudnnLossNormalizationMode_t = ^cudnnLossNormalizationMode_t;
  cudnnLossNormalizationMode_t = (
    CUDNN_LOSS_NORMALIZATION_NONE    = 0,
    CUDNN_LOSS_NORMALIZATION_SOFTMAX = 1
  );

  function cudnnCreateRNNDescriptor(rnnDesc: PcudnnRNNDescriptor_t):cudnnStatus_t; winapi; external libname;
  function cudnnDestroyRNNDescriptor(rnnDesc: cudnnRNNDescriptor_t):cudnnStatus_t; winapi; external libname;
  function cudnnSetRNNDescriptor_v8(rnnDesc: cudnnRNNDescriptor_t; algo: cudnnRNNAlgo_t; cellMode: cudnnRNNMode_t; biasMode: cudnnRNNBiasMode_t; dirMode: cudnnDirectionMode_t; inputMode: cudnnRNNInputMode_t; dataType: cudnnDataType_t; mathPrec: cudnnDataType_t; mathType: cudnnMathType_t; inputSize: longint; hiddenSize: longint; projSize: longint; numLayers: longint; dropoutDesc: cudnnDropoutDescriptor_t; auxFlags: longword):cudnnStatus_t; winapi; external libname;
  function cudnnGetRNNDescriptor_v8(rnnDesc: cudnnRNNDescriptor_t; algo: PcudnnRNNAlgo_t; cellMode: PcudnnRNNMode_t; biasMode: PcudnnRNNBiasMode_t; dirMode: PcudnnDirectionMode_t; inputMode: PcudnnRNNInputMode_t; dataType: PcudnnDataType_t; mathPrec: PcudnnDataType_t; mathType: PcudnnMathType_t; inputSize: Plongint; hiddenSize: Plongint; projSize: Plongint; numLayers: Plongint; dropoutDesc: PcudnnDropoutDescriptor_t; auxFlags: Plongword):cudnnStatus_t; winapi; external libname;
  function cudnnRNNSetClip_v8(rnnDesc: cudnnRNNDescriptor_t; clipMode: cudnnRNNClipMode_t; clipNanOpt: cudnnNanPropagation_t; lclip: double; rclip: double):cudnnStatus_t; winapi; external libname;
  function cudnnRNNSetClip_v9(rnnDesc: cudnnRNNDescriptor_t; clipMode: cudnnRNNClipMode_t; lclip: double; rclip: double):cudnnStatus_t; winapi; external libname;
  function cudnnRNNGetClip_v8(rnnDesc: cudnnRNNDescriptor_t; clipMode: PcudnnRNNClipMode_t; clipNanOpt: PcudnnNanPropagation_t; lclip: Pdouble; rclip: Pdouble):cudnnStatus_t; winapi; external libname;
  function cudnnRNNGetClip_v9(rnnDesc: cudnnRNNDescriptor_t; clipMode: PcudnnRNNClipMode_t; lclip: Pdouble; rclip: Pdouble):cudnnStatus_t; winapi; external libname;
  function cudnnBuildRNNDynamic(handle: cudnnHandle_t; rnnDesc: cudnnRNNDescriptor_t; miniBatch: longint):cudnnStatus_t; winapi; external libname;
  function cudnnGetRNNTempSpaceSizes(handle: cudnnHandle_t; rnnDesc: cudnnRNNDescriptor_t; fwdMode: cudnnForwardMode_t; xDesc: cudnnRNNDataDescriptor_t; workSpaceSize: PSizeInt; reserveSpaceSize: PSizeInt):cudnnStatus_t; winapi; external libname;
  function cudnnGetRNNWeightSpaceSize(handle: cudnnHandle_t; rnnDesc: cudnnRNNDescriptor_t; weightSpaceSize: PSizeInt):cudnnStatus_t; winapi; external libname;
  function cudnnGetRNNWeightParams(handle: cudnnHandle_t; rnnDesc: cudnnRNNDescriptor_t; pseudoLayer: longint; weightSpaceSize: SizeInt; const weightSpace: Pointer; linLayerID: longint; mDesc: cudnnTensorDescriptor_t; mAddr: PPointer; bDesc: cudnnTensorDescriptor_t; bAddr: PPointer):cudnnStatus_t; winapi; external libname;
  function cudnnCreateRNNDataDescriptor(rnnDataDesc: PcudnnRNNDataDescriptor_t):cudnnStatus_t; winapi; external libname;
  function cudnnDestroyRNNDataDescriptor(rnnDataDesc: cudnnRNNDataDescriptor_t):cudnnStatus_t; winapi; external libname;
  function cudnnSetRNNDataDescriptor(rnnDataDesc: cudnnRNNDataDescriptor_t; dataType: cudnnDataType_t; layout: cudnnRNNDataLayout_t; maxSeqLength: longint; batchSize: longint; vectorSize: longint; const seqLengthArray: array of longint; paddingFill: Pointer):cudnnStatus_t; winapi; external libname;
  function cudnnGetRNNDataDescriptor(rnnDataDesc: cudnnRNNDataDescriptor_t; dataType: PcudnnDataType_t; layout: PcudnnRNNDataLayout_t; maxSeqLength: Plongint; batchSize: Plongint; vectorSize: Plongint; arrayLengthRequested: longint; seqLengthArray: array of longint; paddingFill: Pointer):cudnnStatus_t; winapi; external libname;
  function cudnnRNNForward(handle: cudnnHandle_t; rnnDesc: cudnnRNNDescriptor_t; fwdMode: cudnnForwardMode_t; const devSeqLengths: array of longint; xDesc: cudnnRNNDataDescriptor_t; const x: Pointer; yDesc: cudnnRNNDataDescriptor_t; y: Pointer; hDesc: cudnnTensorDescriptor_t; const hx: Pointer; hy: Pointer; cDesc: cudnnTensorDescriptor_t; const cx: Pointer; cy: Pointer; weightSpaceSize: SizeInt; const weightSpace: Pointer; workSpaceSize: SizeInt; workSpace: Pointer; reserveSpaceSize: SizeInt; reserveSpace: Pointer):cudnnStatus_t; winapi; external libname;
  function cudnnCreateSeqDataDescriptor(seqDataDesc: PcudnnSeqDataDescriptor_t):cudnnStatus_t; winapi; external libname;
  function cudnnDestroySeqDataDescriptor(seqDataDesc: cudnnSeqDataDescriptor_t):cudnnStatus_t; winapi; external libname;
  function cudnnSetSeqDataDescriptor(seqDataDesc: cudnnSeqDataDescriptor_t; dataType: cudnnDataType_t; nbDims: longint; const dimA: array of longint; const axes: array of cudnnSeqDataAxis_t; seqLengthArraySize: SizeInt; const seqLengthArray: array of longint; paddingFill: Pointer):cudnnStatus_t; winapi; external libname;
  function cudnnGetSeqDataDescriptor(const seqDataDesc: cudnnSeqDataDescriptor_t; dataType: PcudnnDataType_t; nbDims: Plongint; nbDimsRequested: longint; dimA: array of longint; axes: array of cudnnSeqDataAxis_t; seqLengthArraySize: PSizeInt; seqLengthSizeRequested: SizeInt; seqLengthArray: array of longint; paddingFill: Pointer):cudnnStatus_t; winapi; external libname;
  function cudnnCreateAttnDescriptor(attnDesc: PcudnnAttnDescriptor_t):cudnnStatus_t; winapi; external libname;
  function cudnnDestroyAttnDescriptor(attnDesc: cudnnAttnDescriptor_t):cudnnStatus_t; winapi; external libname;
  function cudnnSetAttnDescriptor(attnDesc: cudnnAttnDescriptor_t; attnMode: longword; nHeads: longint; smScaler: double; dataType: cudnnDataType_t; computePrec: cudnnDataType_t; mathType: cudnnMathType_t; attnDropoutDesc: cudnnDropoutDescriptor_t; postDropoutDesc: cudnnDropoutDescriptor_t; qSize: longint; kSize: longint; vSize: longint; qProjSize: longint; kProjSize: longint; vProjSize: longint; oProjSize: longint; qoMaxSeqLength: longint; kvMaxSeqLength: longint; maxBatchSize: longint; maxBeamSize: longint):cudnnStatus_t; winapi; external libname;
  function cudnnGetAttnDescriptor(attnDesc: cudnnAttnDescriptor_t; attnMode: Plongword; nHeads: Plongint; smScaler: Pdouble; dataType: PcudnnDataType_t; computePrec: PcudnnDataType_t; mathType: PcudnnMathType_t; attnDropoutDesc: PcudnnDropoutDescriptor_t; postDropoutDesc: PcudnnDropoutDescriptor_t; qSize: Plongint; kSize: Plongint; vSize: Plongint; qProjSize: Plongint; kProjSize: Plongint; vProjSize: Plongint; oProjSize: Plongint; qoMaxSeqLength: Plongint; kvMaxSeqLength: Plongint; maxBatchSize: Plongint; maxBeamSize: Plongint):cudnnStatus_t; winapi; external libname;
  function cudnnGetMultiHeadAttnBuffers(handle: cudnnHandle_t; const attnDesc: cudnnAttnDescriptor_t; weightSizeInBytes: PSizeInt; workSpaceSizeInBytes: PSizeInt; reserveSpaceSizeInBytes: PSizeInt):cudnnStatus_t; winapi; external libname;
  function cudnnGetMultiHeadAttnWeights(handle: cudnnHandle_t; const attnDesc: cudnnAttnDescriptor_t; wKind: cudnnMultiHeadAttnWeightKind_t; weightSizeInBytes: SizeInt; const weights: Pointer; wDesc: cudnnTensorDescriptor_t; wAddr: PPointer):cudnnStatus_t; winapi; external libname;
  function cudnnMultiHeadAttnForward(handle: cudnnHandle_t; const attnDesc: cudnnAttnDescriptor_t; currIdx: longint; const loWinIdx: array of longint; const hiWinIdx: array of longint; const devSeqLengthsQO: array of longint; const devSeqLengthsKV: array of longint; const qDesc: cudnnSeqDataDescriptor_t; const queries: Pointer; const residuals: Pointer; const kDesc: cudnnSeqDataDescriptor_t; const keys: Pointer; const vDesc: cudnnSeqDataDescriptor_t; const values: Pointer; const oDesc: cudnnSeqDataDescriptor_t; &out: Pointer; weightSizeInBytes: SizeInt; const weights: Pointer; workSpaceSizeInBytes: SizeInt; workSpace: Pointer; reserveSpaceSizeInBytes: SizeInt; reserveSpace: Pointer):cudnnStatus_t; winapi; external libname;
  function cudnnAdvVersionCheck():cudnnStatus_t; winapi; external libname;
  function cudnnRNNBackwardData_v8(handle: cudnnHandle_t; rnnDesc: cudnnRNNDescriptor_t; const devSeqLengths: array of longint; yDesc: cudnnRNNDataDescriptor_t; const y: Pointer; const dy: Pointer; xDesc: cudnnRNNDataDescriptor_t; dx: Pointer; hDesc: cudnnTensorDescriptor_t; const hx: Pointer; const dhy: Pointer; dhx: Pointer; cDesc: cudnnTensorDescriptor_t; const cx: Pointer; const dcy: Pointer; dcx: Pointer; weightSpaceSize: SizeInt; const weightSpace: Pointer; workSpaceSize: SizeInt; workSpace: Pointer; reserveSpaceSize: SizeInt; reserveSpace: Pointer):cudnnStatus_t; winapi; external libname;
  function cudnnRNNBackwardWeights_v8(handle: cudnnHandle_t; rnnDesc: cudnnRNNDescriptor_t; addGrad: cudnnWgradMode_t; const devSeqLengths: array of longint; xDesc: cudnnRNNDataDescriptor_t; const x: Pointer; hDesc: cudnnTensorDescriptor_t; const hx: Pointer; yDesc: cudnnRNNDataDescriptor_t; const y: Pointer; weightSpaceSize: SizeInt; dweightSpace: Pointer; workSpaceSize: SizeInt; workSpace: Pointer; reserveSpaceSize: SizeInt; reserveSpace: Pointer):cudnnStatus_t; winapi; external libname;
  function cudnnMultiHeadAttnBackwardData(handle: cudnnHandle_t; const attnDesc: cudnnAttnDescriptor_t; const loWinIdx: array of longint; const hiWinIdx: array of longint; const devSeqLengthsDQDO: array of longint; const devSeqLengthsDKDV: array of longint; const doDesc: cudnnSeqDataDescriptor_t; const dout: Pointer; const dqDesc: cudnnSeqDataDescriptor_t; dqueries: Pointer; const queries: Pointer; const dkDesc: cudnnSeqDataDescriptor_t; dkeys: Pointer; const keys: Pointer; const dvDesc: cudnnSeqDataDescriptor_t; dvalues: Pointer; const values: Pointer; weightSizeInBytes: SizeInt; const weights: Pointer; workSpaceSizeInBytes: SizeInt; workSpace: Pointer; reserveSpaceSizeInBytes: SizeInt; reserveSpace: Pointer):cudnnStatus_t; winapi; external libname;
  function cudnnMultiHeadAttnBackwardWeights(handle: cudnnHandle_t; const attnDesc: cudnnAttnDescriptor_t; addGrad: cudnnWgradMode_t; const qDesc: cudnnSeqDataDescriptor_t; const queries: Pointer; const kDesc: cudnnSeqDataDescriptor_t; const keys: Pointer; const vDesc: cudnnSeqDataDescriptor_t; const values: Pointer; const doDesc: cudnnSeqDataDescriptor_t; const dout: Pointer; weightSizeInBytes: SizeInt; const weights: Pointer; dweights: Pointer; workSpaceSizeInBytes: SizeInt; workSpace: Pointer; reserveSpaceSizeInBytes: SizeInt; reserveSpace: Pointer):cudnnStatus_t; winapi; external libname;
  function cudnnCreateCTCLossDescriptor(ctcLossDesc: PcudnnCTCLossDescriptor_t):cudnnStatus_t; winapi; external libname;
  function cudnnSetCTCLossDescriptor(ctcLossDesc: cudnnCTCLossDescriptor_t; compType: cudnnDataType_t):cudnnStatus_t; winapi; external libname;
  function cudnnSetCTCLossDescriptorEx(ctcLossDesc: cudnnCTCLossDescriptor_t; compType: cudnnDataType_t; normMode: cudnnLossNormalizationMode_t; gradMode: cudnnNanPropagation_t):cudnnStatus_t; winapi; external libname;
  function cudnnSetCTCLossDescriptor_v8(ctcLossDesc: cudnnCTCLossDescriptor_t; compType: cudnnDataType_t; normMode: cudnnLossNormalizationMode_t; gradMode: cudnnNanPropagation_t; maxLabelLength: longint):cudnnStatus_t; winapi; external libname;
  function cudnnSetCTCLossDescriptor_v9(ctcLossDesc: cudnnCTCLossDescriptor_t; compType: cudnnDataType_t; normMode: cudnnLossNormalizationMode_t; ctcGradMode: cudnnCTCGradMode_t; maxLabelLength: longint):cudnnStatus_t; winapi; external libname;
  function cudnnGetCTCLossDescriptor(ctcLossDesc: cudnnCTCLossDescriptor_t; compType: PcudnnDataType_t):cudnnStatus_t; winapi; external libname;
  function cudnnGetCTCLossDescriptorEx(ctcLossDesc: cudnnCTCLossDescriptor_t; compType: PcudnnDataType_t; normMode: PcudnnLossNormalizationMode_t; gradMode: PcudnnNanPropagation_t):cudnnStatus_t; winapi; external libname;
  function cudnnGetCTCLossDescriptor_v8(ctcLossDesc: cudnnCTCLossDescriptor_t; compType: PcudnnDataType_t; normMode: PcudnnLossNormalizationMode_t; gradMode: PcudnnNanPropagation_t; maxLabelLength: Plongint):cudnnStatus_t; winapi; external libname;
  function cudnnGetCTCLossDescriptor_v9(ctcLossDesc: cudnnCTCLossDescriptor_t; compType: PcudnnDataType_t; normMode: PcudnnLossNormalizationMode_t; ctcGradMode: PcudnnCTCGradMode_t; maxLabelLength: Plongint):cudnnStatus_t; winapi; external libname;
  function cudnnDestroyCTCLossDescriptor(ctcLossDesc: cudnnCTCLossDescriptor_t):cudnnStatus_t; winapi; external libname;
  function cudnnCTCLoss(handle: cudnnHandle_t; const probsDesc: cudnnTensorDescriptor_t; const probs: Pointer; const hostLabels: array of longint; const hostLabelLengths: array of longint; const hostInputLengths: array of longint; costs: Pointer; const gradientsDesc: cudnnTensorDescriptor_t; gradients: Pointer; algo: cudnnCTCLossAlgo_t; ctcLossDesc: cudnnCTCLossDescriptor_t; workspace: Pointer; workSpaceSizeInBytes: SizeInt):cudnnStatus_t; winapi; external libname;
  function cudnnCTCLoss_v8(handle: cudnnHandle_t; algo: cudnnCTCLossAlgo_t; ctcLossDesc: cudnnCTCLossDescriptor_t; const probsDesc: cudnnTensorDescriptor_t; const probs: Pointer; const labels: array of longint; const labelLengths: array of longint; const inputLengths: array of longint; costs: Pointer; const gradientsDesc: cudnnTensorDescriptor_t; gradients: Pointer; workSpaceSizeInBytes: SizeInt; workspace: Pointer):cudnnStatus_t; winapi; external libname;
  function cudnnGetCTCLossWorkspaceSize(handle: cudnnHandle_t; const probsDesc: cudnnTensorDescriptor_t; const gradientsDesc: cudnnTensorDescriptor_t; const labels: Plongint; const labelLengths: Plongint; const inputLengths: Plongint; algo: cudnnCTCLossAlgo_t; ctcLossDesc: cudnnCTCLossDescriptor_t; sizeInBytes: PSizeInt):cudnnStatus_t; winapi; external libname;
  function cudnnGetCTCLossWorkspaceSize_v8(handle: cudnnHandle_t; algo: cudnnCTCLossAlgo_t; ctcLossDesc: cudnnCTCLossDescriptor_t; const probsDesc: cudnnTensorDescriptor_t; const gradientsDesc: cudnnTensorDescriptor_t; sizeInBytes: PSizeInt):cudnnStatus_t; winapi; external libname;


implementation
end.
