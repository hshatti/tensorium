unit nXNORConvolutionLayer;
{$ifdef FPC}
{$mode Delphi}
{$endif}

interface

uses
  SysUtils, nTensors, nTypes, nBaseLayer, nConvolutionLayer, nBatchNorm, nCol2Im, nActivation;

type

  { TConvolutionLayer }

  { TXNORConvolutionLayer }

  TXNORConvolutionLayer=class(TConvolutionLayer)
  private
    procedure setTrain(ATrain: boolean); override;
  public
    Binary, xnor                  : boolean;
    ldaAlign, bitAlign            : integer;
    binaryWeights, binaryInput    : TSingleTensor;
    binRePackedInput              : TArray<UInt32>;
    tBitInput                     : TArray<Byte>;
    meanArr                       : TArray<single>;
    alignBitWeights               : TArray<ShortInt>;
    cweights                      : TArray<byte>;

    function getWorkspaceSize32():SizeInt;
    function getWorkspaceSize16():SizeInt;
    function getWorkspaceSize():SizeInt;
    procedure swapBinary();
    constructor Create(const ABatch, {ASteps,} Aheight, Awidth, Achannels,
      Afilters: SizeInt; AGroups, AKernelSize, AStride_x, AStride_y, ADilation,
      APadding: SizeInt; const AActivation: TActivationType;
      const ABatch_normalize: boolean=false; const ABinary: boolean=false;
      const AXnor: boolean=false; const AAdam: boolean=false;
      const AUse_bin_output: boolean=false; const AIndex: SizeInt=0;
      const AAntialiasing: SizeInt=0; const AShare_layer: TConvolutionLayer=nil;
      const AAssistedExcitation: SizeInt=0; const ADeform: boolean=false;
      const ATrain: boolean=false);
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure update(const args: TUpdateArgs); override;

  end;

implementation
uses math, nnet;

{ TXNORConvolutionLayer }

function TXNORConvolutionLayer.getWorkspaceSize32(): SizeInt;
var
    most: SizeInt;
    s: SizeInt;
    re_packed_input_size: SizeInt;
begin
    if xnor then
        begin
            re_packed_input_size := c * w * h {* sizeof(single)};  // setting size of array requires element count instead of  byte size
            result := bitAlign * kernelSize * kernelSize * c {* sizeof(single)};
            if result < re_packed_input_size then
                result := re_packed_input_size;
            exit()
        end;
    result := (c div groups) * outH * outW * kernelSize * kernelSize {* sizeof(single)}
end;

function TXNORConvolutionLayer.getWorkspaceSize16(): SizeInt;
begin
  result :=0
end;

function TXNORConvolutionLayer.getWorkspaceSize(): SizeInt;
var
  workspace_size16: size_t;
begin
    result := getWorkspaceSize32();
    workspace_size16 := getWorkspaceSize16();
    if (workspace_size16 > result) then
        result := workspace_size16;
end;

procedure TXNORConvolutionLayer.swapBinary();
var
    swap: TSingleTensor;
begin
    swap.ShallowCopy(weights);
    weights.ShallowCopy(binaryWeights);
    binaryWeights.ShallowCopy(swap);
end;

procedure binarize_weights(const weights: PSingle; const n, size: SizeInt; const binary: PSingle);
var
    i: SizeInt;
    f: SizeInt;
    mean: single;
begin
    for f := 0 to n -1 do
        begin
            mean := 0;
            for i := 0 to size -1 do
                mean := mean + abs(weights[f * size+i]);
            mean := mean / size;
            for i := 0 to size -1 do
                if weights[f * size+i] > 0 then
                    binary[f * size+i] := mean
                else
                    binary[f * size+i] := -mean
        end
end;
procedure binarize_input(const input: PSingle; const n, size: SizeInt; const binary: PSingle);
var
    i, s: SizeInt;
    mean: single;
begin
    for s := 0 to size -1 do
        begin
            mean := 0;
            for i := 0 to n -1 do
                mean := mean + abs(input[i * size+s]);
            mean := mean / n;
            for i := 0 to n -1 do
                if input[i * size+s] > 0 then
                    binary[i * size+s] := mean
                else
                    binary[i * size+s] := -mean
        end
end;

procedure binarize_cpu(const input: PSingle; const n: SizeInt; const binary: PSingle);
var
    i: SizeInt;
begin
    for i := 0 to n -1 do
        if input[i] > 0 then
            binary[i] := 1
        else
            binary[i] := -1
end;

procedure repack_input(const input, re_packed_input: Psingle; const w, h, c: SizeInt);
var
    items_per_channel, chan, i, c_pack: SizeInt;
    src: single;
begin
    items_per_channel := w * h;
    chan := 0;
    while chan < c do begin
        for i := 0 to items_per_channel -1 do
            begin
                for c_pack := 0 to 32 -1 do
                    begin
                        src := input[(chan+c_pack) * items_per_channel+i];
                        re_packed_input[chan * items_per_channel+i * 32+c_pack] := src
                    end
            end;
        chan := chan + 32
    end
end;

function xnor_int64(const a,b:uint64):uint64;inline;
begin
    result := not(a xor b)
end;

{$ifndef FPC}
function popcnt(v:uint32):uint32;inline;
begin
  v := v - ((v shr 1) and $55555555);
  v := (v and $33333333) + ((v shr 2) and $33333333);
  result := ((v + (v shr 4) and $F0F0F0F) * $1010101) shr 24;
end;
{$endif}


procedure gemm_nn_custom_bin_mean_transposed(const M, N, K: SizeInt; const ALPHA_UNUSED: single; const A: PByte; const lda: SizeInt; const B: PByte; const ldb: SizeInt; const C: Psingle; const ldc: SizeInt; const mean_arr: Psingle);
var
    i, j, kk, count, tmp_count: SizeInt;
    mean_val: single;
    a_bit64, b_bit64, c_bit64: uint64;
begin
    for i := 0 to M -1 do
        begin
            mean_val := mean_arr[i];
            for j := 0 to N -1 do
                begin
                    count := 0;
                    kk := 0;
                    while kk < K do begin
                        a_bit64 :=  PUint64((A+(i * lda+kk) div 8))^;
                        b_bit64 :=  PUint64((B+(j * ldb+kk) div 8))^;
                        c_bit64 := xnor_int64(a_bit64, b_bit64);
                        tmp_count := POPCNT(c_bit64);
                        if K-kk < 64 then
                            tmp_count := tmp_count-(64-(K-kk));
                        count := count + tmp_count;
                        kk := kk + 64
                    end;
                    C[i * ldc+j] := (2 * count-K) * mean_val
                end
        end
end;

function get_bit(const src: PByte; const index: IntPtr): boolean;
var
  p: PByte;
begin
  p:=@src[index div 8];
  result:=P^ and (1 shl (index mod 8))>0
end;

procedure set_bit(const src: PByte; const index: IntPtr);
var
    p:PByte;
begin
  p:=@src[index div 8];
  p[0]:=P[0] or (1 shl (index mod 8))
end;

function im2col_get_pixel(const im: PSingle; const height, width, channels:SizeInt; row, col:SizeInt;const channel, pad: SizeInt):single;overload;
begin
    row := row - pad;
    col := col - pad;
    if (row < 0) or (col < 0) or (row >= height) or (col >= width) then
        exit(0);
    exit(im[col+width * (row+height * channel)])
end;

procedure im2col_cpu_custom_bin(const data_im: Psingle; const channels, height,
  width, ksize, stride, pad: SizeInt; const data_col: Psingle;
  const bit_align: SizeInt);
var
    c, height_col, width_col, channels_col, new_ldb, h, w, w_offset, h_offset, c_im, im_row, im_col, col_index: SizeInt;
    val: single;
begin
    height_col := (height+2 * pad-ksize) div stride+1;
    width_col := (width+2 * pad-ksize) div stride+1;
    channels_col := channels * ksize * ksize;
    if (height_col = height) and (width_col = width) and (stride = 1) and (pad = 1) then
        begin
            new_ldb := bit_align;
            for c := 0 to channels_col -1 do
                begin
                    w_offset := c mod ksize;
                    h_offset := (c div ksize) mod ksize;
                    c_im := c div ksize div ksize;
                    for h := pad to height_col-pad -1 do
                        begin
                            w := pad;
                            while w < width_col-pad-8 do begin
                                im_row := h_offset+h-pad;
                                im_col := w_offset+w-pad;
                                col_index := c * new_ldb+h * width_col+w;
                                val := data_im[im_col+width * (im_row+height * c_im)];
                                if val > 0 then
                                    set_bit(PByte(data_col), col_index);
                                w := w + 1
                            end;
                            while w < width_col-pad do begin
                                im_row := h_offset+h-pad;
                                im_col := w_offset+w-pad;
                                col_index := c * new_ldb+h * width_col+w;
                                val := data_im[im_col+width * (im_row+height * c_im)];
                                if val > 0 then
                                    set_bit(PByte(data_col), col_index);
                                inc(w)
                            end
                        end;
                    w := 0;
                    for h := 0 to height_col -1 do
                        begin
                            im_row := h_offset+h;
                            im_col := w_offset+w;
                            col_index := c * new_ldb+h * width_col+w;
                            val := im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                            if val > 0 then
                                set_bit(PByte(data_col), col_index)
                        end;
                    w := width_col-1;
                    for h := 0 to height_col -1 do
                        begin
                            im_row := h_offset+h;
                            im_col := w_offset+w;
                            col_index := c * new_ldb+h * width_col+w;
                            val := im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                            if val > 0 then
                                set_bit(PByte(data_col), col_index)
                        end;
                    h := 0;
                    for w := 0 to width_col -1 do
                        begin
                            im_row := h_offset+h;
                            im_col := w_offset+w;
                            col_index := c * new_ldb+h * width_col+w;
                            val := im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                            if val > 0 then
                                set_bit(PByte(data_col), col_index)
                        end;
                    h := height_col-1;
                    for w := 0 to width_col -1 do
                        begin
                            im_row := h_offset+h;
                            im_col := w_offset+w;
                            col_index := c * new_ldb+h * width_col+w;
                            val := im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                            if val > 0 then
                                set_bit(PByte(data_col), col_index)
                        end
                end
        end
    else
        writeln(#10' Error: is no non-optimized version '#10)
end;

procedure float_to_bit(const src:PSingle; const dst: PByte;const size:IntPtr);
var
  dst_size, i:IntPtr;
  dst_tmp : byte;
  byte_arr:TArray<byte>;
begin
    dst_size := size div 8 + 1;
    fillchar(dst[0], dst_size,0);

    setLength(byte_arr, size);
    for i := 0 to size-1 do
        if src[i] > 0 then byte_arr[i] := 1;

    //for (i = 0; i < size; ++i) {
    //    dst[i / 8] |= byte_arr[i] << (i % 8);
    //}

    for i := 0 to size-1 do begin
        dst_tmp := 0;
        dst_tmp := dst_tmp + byte_arr[i + 0] shl 0;
        dst_tmp := dst_tmp + byte_arr[i + 1] shl 1;
        dst_tmp := dst_tmp + byte_arr[i + 2] shl 2;
        dst_tmp := dst_tmp + byte_arr[i + 3] shl 3;
        dst_tmp := dst_tmp + byte_arr[i + 4] shl 4;
        dst_tmp := dst_tmp + byte_arr[i + 5] shl 5;
        dst_tmp := dst_tmp + byte_arr[i + 6] shl 6;
        dst_tmp := dst_tmp + byte_arr[i + 7] shl 7;
        dst[i div 8] := dst_tmp;
    end;
    //free(byte_arr);
end;

procedure transpose_uint32(src, dst: Puint32; src_h: SizeInt; src_w: SizeInt; src_align: SizeInt; dst_align: SizeInt);
var
    i,j: SizeInt;
begin
    i := 0;
    // todo simdfy
    while i < src_h do begin
        j := 0;
        while j < src_w do begin
            dst[j * dst_align div 32+i] := src[i * src_align+j];
            j := j + 1
        end;
        inc(i)
    end
end;

function reverse_8_bit(const a: uint8):uint8;inline;
begin
    exit(((a * $0802 and $22110) or (a * $8020 and $88440)) * $10101 shr 16)
end;

function reverse_32_bit(const a: uint32):uint32;inline;
begin
    exit(
         (reverse_8_bit(a shr 24) shl 0) or
         (reverse_8_bit(a shr 16) shl 8) or
         (reverse_8_bit(a shr 8) shl 16) or
         (reverse_8_bit(a shr 0) shl 24)
        )
end;

type TLongWordx32 = array[0..31] of uint32;

procedure transpose32_optimized(A: TLongWordx32);inline;
var
    j, k: SizeInt;
    m, t: uint32;
    tmp: uint32;
begin
    j := 16;
    m := $0000FFFF;
    k := 0;
    while k < 32 do begin
          t := (A[k] xor (A[k+j] shr j)) and m;
          A[k] := A[k] xor t;
          A[k+j] := A[k+j] xor (t shl j);
        k := (k+j+1) and not j
    end;
    j := 8;
    m := $00ff00ff;
    k := 0;
    while k < 32 do begin
        t := (A[k] xor (A[k+j] shr j)) and m;
        A[k] := A[k] xor t;
        A[k+j] := A[k+j] xor (t shl j);
        k := (k+j+1) and not j
    end;
    j := 4;
    m := $0f0f0f0f;
    k := 0;
    while k < 32 do begin
        t := (A[k] xor (A[k+j] shr j)) and m;
        A[k] := A[k] xor t;
        A[k+j] := A[k+j] xor (t shl j);
        k := (k+j+1) and not j
    end;
    j := 2;
    m := $33333333;
    k := 0;
    while k < 32 do begin
        t := (A[k] xor (A[k+j] shr j)) and m;
        A[k] := A[k] xor t;
        A[k+j] := A[k+j] xor (t shl j);
        k := (k+j+1) and not j
    end;
    j := 1;
    m := $55555555;
    k := 0;
    while k < 32 do begin
        t := (A[k] xor (A[k+j] shr j)) and m;
        A[k] := A[k] xor t;
        A[k+j] := A[k+j] xor (t shl j);
        k := (k+j+1) and not j
    end;
    for j := 0 to 16 -1 do
        begin
            tmp := A[j];
            A[j] := reverse_32_bit(A[31-j]);
            A[31-j] := reverse_32_bit(tmp)
        end
end;

procedure transpose_32x32_bits_reversed_diagonale(const A: Puint32; const B: Puint32; const m, n: SizeInt); inline;
var
    //A_tmp: array[0..31] of uint32;
    A_tmp : TLongWordx32;
    i: SizeInt;
begin
    // todo unroll optimiztion
    for i := 0 to 32 -1 do
        A_tmp[i] := A[i * m];
    transpose32_optimized(A_tmp);
    //todo unroll optimization
    for i := 0 to 32 -1 do
        B[i * n] := A_tmp[i]
end;

procedure transpose_bin(A: PUInt32; B: PUInt32; const n: SizeInt; const m: SizeInt; const lda: SizeInt; const ldb: SizeInt; const block_size: SizeInt);
var
    i, j, a_index, b_index: SizeInt;
begin
    i := 0;
    // todo SIMDIfy
    while i < n do begin
        j := 0;
        while j < m do begin
            a_index := i * lda+j;
            b_index := j * ldb+i;
            transpose_32x32_bits_reversed_diagonale( @A[a_index div 32],  @B[b_index div 32], lda div 32, ldb div 32);
            j := j + 32
        end;
        while j < m do begin
            if get_bit(PByte(A), i * lda+j) then
                set_bit(PByte(B), j * ldb+i);
            inc(j)
        end;
        i := i + 32
    end
end;

function binary_transpose_align_input(k: SizeInt; n: SizeInt; b: PSingle; t_bit_input: PPByte; ldb_align: size_t; bit_align: SizeInt):size_t;
var
    new_ldb, t_intput_size, t_bit_input_size: size_t;
begin
    new_ldb := k + (ldb_align - k mod ldb_align); // (k / 8 + 1) * 8;
    //printf("\n n = %d, bit_align = %d \n", n, bit_align);
    t_intput_size := new_ldb * bit_align;// n;
    t_bit_input_size := t_intput_size div 8;// +1;

    fillchar(t_bit_input[0][0], t_bit_input_size, 0);
    //int src_size = k * bit_align;

    // b - [bit_align, k] - [l.bit_align, l.size*l.size*l.c] = src_size
    // t_input - [bit_align, k] - [n', k]
    // t_bit_input - [new_ldb, n] - [k', n]

    //transpose_bin(t_input, *t_bit_input, k, n, bit_align, new_ldb, 8);
    transpose_bin(PUint32(b), PUInt32(t_bit_input^), k, n, bit_align, new_ldb, 8);
    exit(t_intput_size);


end;

procedure TXNORConvolutionLayer.setTrain(ATrain: boolean);
var
    total_batch:SizeInt;
begin
  if ATrain = FTrain then exit;
  FTrain := ATrain;
  // todo Check XORConvolution setTrain algo
  if FTrain then begin
      total_batch := batch{*steps};
      delta := TSingleTensor.Create([total_batch , filters, outH, outW]);
      if not assigned(shareLayer) then begin
          weight_updates := TSingleTensor.Create([c div groups, filters, kernelSize, kernelSize]);
          bias_updates := TSingleTensor.Create([filters]);
          weights_ema := TSingleTensor.Create([c div groups, filters, kernelSize, kernelSize]);
          biases_ema := TSingleTensor.Create([filters]);
          if isBatchNormalized then begin
              scales_ema := TSingleTensor.Create([filters]);
              scale_updates := TSingleTensor.Create([filters]);
              mean := TSingleTensor.Create([filters]);
              variance := TSingleTensor.Create([filters]);
              mean_delta := TSingleTensor.Create([filters]);
              variance_delta := TSingleTensor.Create([filters])
          end;
      end;
      if isBatchNormalized then begin
          x := TSingleTensor.Create([total_batch, filters, outH, outW]);
          x_norm := TSingleTensor.Create([total_batch, filters, outH, outW])
      end;

  end else begin
      delta := Default(TSingleTensor);
      if not assigned(shareLayer) then begin
          weight_updates := Default(TSingleTensor);
          bias_updates   := Default(TSingleTensor);
          weights_ema    := Default(TSingleTensor);
          biases_ema     := Default(TSingleTensor);
          if isBatchNormalized then begin
              scales_ema       := Default(TSingleTensor);
              scale_updates    :=Default(TSingleTensor);
              mean             := Default(TSingleTensor);
              variance         := Default(TSingleTensor);
              mean_delta       := Default(TSingleTensor);
              variance_delta   := Default(TSingleTensor)
          end;
      end;
      if isBatchNormalized then begin
          x                := Default(TSingleTensor);
          x_norm           := Default(TSingleTensor)
      end;
  end;

  //inherited setTrain(AValue);
end;

constructor TXNORConvolutionLayer.Create(const ABatch, Aheight, Awidth,
  Achannels, Afilters: SizeInt; AGroups, AKernelSize, AStride_x, AStride_y,
  ADilation, APadding: SizeInt; const AActivation: TActivationType;
  const ABatch_normalize: boolean; const ABinary: boolean;
  const AXnor: boolean; const AAdam: boolean; const AUse_bin_output: boolean;
  const AIndex: SizeInt; const AAntialiasing: SizeInt;
  const AShare_layer: TConvolutionLayer; const AAssistedExcitation: SizeInt;
  const ADeform: boolean; const ATrain: boolean);

var
  total_batch, _align, src_align, k,
    in_re_packed_input_size ,k_aligned,
    t_bit_input_size, blur_nweights, i, blur_size, blur_pad: SizeInt;

  _scale : single;

begin
  total_batch := aBatch {* aSteps};
  layerType := ltCONVOLUTIONAL;
  FTrain := ATrain;
  if xnor then
      groups := 1;
  if groups < 1 then
      groups := 1;
  blurStride_x := aStride_x;
  blurStride_y := aStride_y;
  antialiasing := AAntialiasing;
  if antialiasing>0 then begin
      AStride_x        := 1;
      AStride_y        := 1;
      stride_x         := 1;
      stride_y         := 1
  end;
  wait_stream_id := -1;
  deform := aDeform;
  assistedExcitation := AAssistedExcitation;
  shareLayer := Ashare_layer;
  index := Aindex;
  h := Aheight;
  w := Awidth;
  c := Achannels;
  n := Afilters; // same as filters := AFilters;
  batch := ABatch;
  inputShape := [batch, c, h, w];
  groups := AGroups;
  binary := ABinary;
  xnor := Axnor;
  //result.use_bin_output := use_bin_output;
  //steps := ASteps;
  stride_x := AStride_x;
  stride_y := AStride_y;
  dilation := ADilation;
  kernelSize := AKernelSize;
  Padding := APadding;
  isBatchNormalized := ABatch_normalize;
  learningRateScale := 1;
  nweights := (c div groups) * filters * kernelSize * kernelSize;
  if assigned(shareLayer) then
      begin
          if (kernelSize <> shareLayer.kernelSize) or (nweights <> shareLayer.nweights) or (c <> shareLayer.c) or (n <> shareLayer.n) then
              raise Exception.create('Layer KernelSize, nweights, channels or filters don''t match for the shareLayer');
          weights := shareLayer.weights;
          weight_updates := shareLayer.weight_updates;
          biases := shareLayer.biases;
          bias_updates := shareLayer.bias_updates
      end
  else
      begin
          weights := TSingleTensor.Create([c div groups, filters, kernelSize, kernelSize]);
          biases := TSingleTensor.Create([filters]);
          if train then
              begin
                  weight_updates := TSingleTensor.Create([c div groups, filters, kernelSize, kernelSize]);
                  bias_updates := TSingleTensor.Create([filters]);
                  weights_ema := TSingleTensor.Create([c div groups, filters, kernelSize, kernelSize]);
                  biases_ema := TSingleTensor.Create([filters])
              end
      end;
  _scale := sqrt(2 / (kernelSize * kernelSize * c / groups));
  if ActivationType in [acNORM_CHAN, acNORM_CHAN_SOFTMAX, acNORM_CHAN_SOFTMAX_MAXVAL] then
      weights.fill(1)
  else
      weights.UniformDistribution( - _scale, _scale);
  outW := outWidth();
  outH := outHeight();
  outC := n;
  outputs := outC * outH * outW;
  inputs := w * h * c;
  ActivationType := AActivation;
  output := TSingleTensor.Create([total_batch , filters, outH, outW], total_batch);

  if train then
      delta := TSingleTensor.Create([total_batch , filters, outH, outW, total_batch]);

  if binary then
      begin
          binaryWeights := TSingleTensor.Create([c div groups, filters, kernelSize, kernelSize]);
          setLength(cweights, nweights);
          scales := TSingleTensor.Create([filters])
      end;
  if xnor then
      begin
          binaryWeights := TSingleTensor.Create([(c div groups) , filters , kernelSize , kernelSize]);
          binaryInput := TSingleTensor.Create([batch, c, h, w], batch);
          _align := 32;
          src_align := outH * outW;
          bitAlign := src_align+(_align-src_align mod _align);
          setLength(meanArr, filters) ;
          in_re_packed_input_size := (c div 32) * h * w+1;
          setlength(binRePackedInput, in_re_packed_input_size);
          ldaAlign := 256;
          k := kernelSize * kernelSize * c;
          k_aligned := k+(ldaAlign-k mod ldaAlign);
          t_bit_input_size := k_aligned * bitAlign div 8;
          setLength(tBitInput, t_bit_input_size)
          //setLength(result.t_bit_input, t_bit_input_size);
      end;
  if isBatchNormalized then
      begin
          if assigned(shareLayer) then
              begin
                  scales := shareLayer.scales;
                  scale_updates := shareLayer.scale_updates;
                  mean := shareLayer.mean;
                  variance := shareLayer.variance;
                  mean_delta := shareLayer.mean_delta;
                  variance_delta := shareLayer.variance_delta;
                  rolling_mean := shareLayer.rolling_mean;
                  rolling_variance := shareLayer.rolling_variance
              end
          else
              begin
                  scales := TSingleTensor.Create([filters]);
                  scales.fill(1);
                  if train then
                      begin
                          scales_ema := TSingleTensor.Create([filters]);
                          scale_updates := TSingleTensor.Create([filters]);
                          mean := TSingleTensor.Create([filters]);
                          variance := TSingleTensor.Create([filters]);
                          mean_delta := TSingleTensor.Create([filters]);
                          variance_delta := TSingleTensor.Create([filters])
                      end;
                  rolling_mean := TSingleTensor.Create([filters]);
                  rolling_variance := TSingleTensor.Create([filters])
              end;
{$ifndef GPU}
          if train then
              begin
                  x := TSingleTensor.Create([total_batch, filters, outH, outW], total_batch);
                  x_norm := TSingleTensor.Create([total_batch, filters, outH, outW], total_batch)
              end
{$endif}
      end;
{$ifndef GPU}
  if ActivationType in [acSWISH, acMISH, acHARD_MISH] then
      ActivationInput := TSingleTensor.Create([total_batch, filters, outH, outW], total_batch);
{$endif}
  if adam then
      begin
          m       := TSingleTensor.Create([(c div groups), filters, kernelSize, kernelSize]);
          v       := TSingleTensor.Create([(c div groups), filters, kernelSize, kernelSize]);
          bias_m  := TSingleTensor.Create([filters]);
          scale_m := TSingleTensor.Create([filters]);
          bias_v  := TSingleTensor.Create([filters]);
          scale_v := TSingleTensor.Create([filters])
      end;


  workspaceSize := getWorkspaceSize();
  if antialiasing>0 then
      begin
          blur_size := 3;
          blur_pad := blur_size div 2;
          if antialiasing = 2 then
              begin
                  blur_size := 2;
                  blur_pad := 0
              end;
          InputLayer := TXNORConvolutionLayer.Create(batch, {steps,} outH, outW, n, n, n, blur_size, blurstride_x, blurstride_y, 1, blur_pad, acLINEAR, false, false, false, false, false, index, 0, nil, 0, false, train);
          blur_nweights := n * blur_size * blur_size;

          if blur_size = 2 then begin
              i := 0;
              while i < blur_nweights do begin
                  InputLayer.weights.data[i+0] := 1 / 4;
                  InputLayer.weights.data[i+1] := 1 / 4;
                  InputLayer.weights.data[i+2] := 1 / 4;
                  InputLayer.weights.data[i+3] := 1 / 4;
                  i := i + (blur_size * blur_size)
              end
          end
          else begin
              i := 0;
              while i < blur_nweights do begin
                  InputLayer.weights.data[i+0] := 1 / 16;
                  InputLayer.weights.data[i+1] := 2 / 16;
                  InputLayer.weights.data[i+2] := 1 / 16;
                  InputLayer.weights.data[i+3] := 2 / 16;
                  InputLayer.weights.data[i+4] := 4 / 16;
                  InputLayer.weights.data[i+5] := 2 / 16;
                  InputLayer.weights.data[i+6] := 1 / 16;
                  InputLayer.weights.data[i+7] := 2 / 16;
                  InputLayer.weights.data[i+8] := 1 / 16;
                  i := i + (blur_size * blur_size)
              end;
          end;
          inputLayer.biases.fill(0);
      end

end;

procedure TXNORConvolutionLayer.forward(var state: TNNetState);
var
    out_h, out_w, i, j, m, k, outImgSize,  ldb_align, re_packed_input_size, new_k: SizeInt;
    _A, _B, _C: PSingle;
    new_ldb, t_intput_size, new_c, in_re_packed_input_size: size_t;
    im: PSingle;
    s: TNNetState;

begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(layerType);
    {$endif}

    //outH := convolutional_out_height(l);
    //outW := convolutional_out_width(l);
    output.fill(0);
    //fill_cpu(l.outputs * l.batch, 0, l.output, 1);

    if xnor and (not assigned(alignBitWeights) or state.isTraining) then
        begin
            if not assigned(alignBitWeights) or state.isTraining then
                binarize_weights(weights, filters, nweights, binaryWeights);
            swapbinary();

            state.input.copyTo(binaryInput.Data);
            binaryInput.threshold(0, 1, -1);
            state.input := binaryInput
        end;
    m := filters div groups;
    k := kernelSize * kernelSize * c div groups;
    outImgSize := outH * outW;
    //u := 0;
    //inc(u);
    for i := 0 to batch -1 do
        for j := 0 to groups -1 do
            begin
                _A := weights.Data + j * nweights div groups;
                _B := pointer(state.workspace);
                _C := output.data + (i * groups+j) * outImgSize * m;
                if xnor and assigned(alignBitWeights) and not state.isTraining and (stride_x = stride_y) then begin
                    filldword(_B[0], bitAlign * kernelSize * kernelSize * c ,0);
                    if c mod 32 = 0 then
                        begin
                            ldb_align := ldaAlign;
                            new_ldb := k+(ldb_align-k mod ldb_align);
                            re_packed_input_size := c * w * h;
                            filldword(state.workspace[0], re_packed_input_size,0);
                            new_c := c div 32;
                            in_re_packed_input_size := new_c * w * h + 1;
                            filldword(binRePackedInput[0], in_re_packed_input_size ,0);
                            repack_input(state.input, pointer(state.workspace), w, h, c);
                            float_to_bit(pointer(state.workspace), PByte(binRePackedInput), c * w * h);
                            im2col_cpu_custom(PSingle(binRePackedInput), new_c, h, w, kernelSize, stride, Padding, pointer(state.workspace));
                            new_k := kernelSize * kernelSize * c div 32;
                            transpose_uint32(Puint32(state.workspace), Puint32(tBitInput), new_k, outImgSize, outImgSize, new_ldb);
                            gemm_nn_custom_bin_mean_transposed(m, outImgSize, k, 1, PByte(alignBitWeights), new_ldb, PByte(tBitInput), new_ldb, _C, outImgSize, Pointer(meanArr))
                        end
                    else
                        begin
                            im2col_cpu_custom_bin(state.input, c, h, w, kernelSize, stride, Padding, pointer(state.workspace), bitAlign);
                            ldb_align := ldaAlign;
                            new_ldb := k + (ldb_align - k mod ldb_align);
                            t_intput_size := binary_transpose_align_input(k, outImgSize, pointer(state.workspace), PPByte(tBitInput), ldb_align, bitAlign);

                            // 5x times faster than gemm()-float32
                            gemm_nn_custom_bin_mean_transposed(m, outImgSize, k, 1, PByte(alignBitWeights), new_ldb, PByte(tBitInput), new_ldb, _C, outImgSize, Pointer(meanArr));

                        end;
                    //add_bias(output, biases, batch, n, outH * outW);
                    output.Add(biases);

                     //todo implement conv activations
                    case ActivationType of
                       acSWISH :
                            activate_array_swish(output, outputs * batch, ActivationInput, output);
                       acMISH :
                            activate_array_mish(output, outputs * batch, ActivationInput, output);
                       acHARD_MISH :
                            activate_array_hard_mish(output, outputs * batch, ActivationInput, output);
                       acNORM_CHAN :
                            activate_array_normalize_channels(output, outputs * batch, batch, outC, outW * outH, output);
                       acNORM_CHAN_SOFTMAX, acNORM_CHAN_SOFTMAX_MAXVAL :
                            activate_array_normalize_channels_softmax(output, outputs * batch, batch, outC, outW * outH, output, activationType = acNORM_CHAN_SOFTMAX_MAXVAL);
                    else
                        //activate_array(output, m * outImgSize * batch, activation)
                        activate();
                    end;
                    {$ifdef USE_TELEMETRY}
                    if benchmark then metrics.forward.finish(layerType);
                    {$endif}
                    exit()
                end else begin
                    im := state.input.data +(i * groups+j) * (c div groups) * h * w;
                    if (kernelSize = 1) and (stride = 1) and (dilation = 1) then
                        _B := im
                    else
                        //im2col_cpu(im, l.c div l.groups, l.h, l.w, l.size, l.stride_x, l.pad, b);
                        im2col_cpu_ext(im, c div groups, h, w, kernelSize, kernelSize, Padding * dilation, Padding * dilation, stride_y, stride_x, dilation, dilation, _B);
                    TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, outImgSize, k, 1, _A, k, _B, outImgSize, 1, _C, outImgSize);
                end
            end;
    if isBatchNormalized then
        //forward_batchnorm_layer(l, state)
       batchNorm(state)
    else
        //add_bias(l.output, l.biases, l.batch, l.n, outH * outW);
        output.Add(biases);
    case ActivationType of
       acSWISH :
          activate_array_swish(output, outputs * batch, activationInput, output);
       acMISH :
          activate_array_mish(output, outputs * batch, activationInput, output);
       acHARD_MISH :
            activate_array_hard_mish(output, outputs * batch, activationInput, output);
       acNORM_CHAN :
            activate_array_normalize_channels(output, outputs * batch, batch, outC, outW * outH, output);
       acNORM_CHAN_SOFTMAX, acNORM_CHAN_SOFTMAX_MAXVAL :
            activate_array_normalize_channels_softmax(output, outputs * batch, batch, outC, outW * outH, output, activationType = acNORM_CHAN_SOFTMAX_MAXVAL);
       else
        //activate_array_cpu_custom(l.output, l.outputs * l.batch, l.activation);
            //activate_array(output, outputs * batch, activation);
           activate()
    end;
    if binary or xnor then
        swapBinary();
    if (assistedExcitation<>0) and state.isTraining then
        assistedForward(state);
    if antialiasing<>0 then
        begin
            s := default(TNNetState);
            s.isTraining := state.isTraining;
            s.workspace := state.workspace;
            //s.net := state.net;
            s.input := output;
            //forward_convolutional_layer( l.input_layer[0], @s);
            inputLayer.forward(s);
            //move(l.input_layer[0].output[0], l.output[0], l.input_layer[0].outputs * l.input_layer[0].batch * sizeof(single))
            inputLayer.output.copyTo(output.Data);
        end;

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(layerType);
    {$endif}

end;

procedure TXNORConvolutionLayer.backward(var state: TNNetState);
var
    i,j, m, n, k: SizeInt;
    _A, _B, _C, im: PSingle;
begin
      m := filters div groups;
      n := kernelSize * kernelSize * c div groups;
      k := outW * outW;

      case activationType of
        acSWISH :
          gradient_array_swish(output, outputs * batch, ActivationInput, delta) ;
        acMISH  :
          gradient_array_mish(outputs * batch, ActivationInput, delta) ;
        acHARD_MISH :
          gradient_array_hard_mish(outputs * batch, ActivationInput, delta) ;
        acNORM_CHAN_SOFTMAX, acNORM_CHAN_SOFTMAX_MAXVAL :
          gradient_array_normalize_channels_softmax(output, outputs * batch, batch, outC, outW * outW, delta) ;
        acNORM_CHAN :
          gradient_array_normalize_channels(output, outputs * batch, batch, outC, outW * outW, delta) ;
        else
          Derivative();
      end;
      if isBatchNormalized then
          batchNormBack(state)
      else
          //backward_bias(bias_updates, delta, batch, filters, k);
          bias_updates.Add(delta);
      for i := 0 to batch -1 do
          for j := 0 to groups -1 do
              begin
                  _A := delta.Data+(i * groups+j) * m * k;
                  _B := pointer(state.workspace);
                  _C := weight_updates.data  + j * nweights div groups;
                  im := state.input.data +(i * groups+j) * (c div groups) * h * w;
                  im2col_cpu_ext(im, c div groups, h, w, kernelSize, kernelSize, padding * dilation, padding * dilation, stride_y, stride_x, dilation, dilation, _B);
                  TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, _A, k, _B, k, 1, _C, n);
                  if assigned(state.delta.Data) then
                      begin
                          _A := weights.Data+j * nweights div groups;
                          _B := delta.Data + (i * groups+j) * m * k;
                          _C := pointer(state.workspace);
                          TSingleTensor.gemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, k, m, 1, _A, n, _B, k, 0, _C, k);
                          col2im_cpu_ext(pointer(state.workspace), c div groups, h, w, kernelSize, kernelSize, padding * dilation, padding * dilation, stride_y, stride_x, dilation, dilation, state.delta.data +(i * groups+j) * (c div groups) * h * w)
                      end
              end
end;

procedure TXNORConvolutionLayer.update(const args: TUpdateArgs);
var
    learning_rate: single;
begin
    learning_rate := args.learningRate * learningRateScale;
    //axpy_cpu(l.nweights, -args.decay * args.batch, l.weights, 1, l.weight_updates, 1);
    weight_updates.axpy(-args.decay * args.batch, weights);

    //axpy_cpu(l.nweights, learning_rate / args.batch, l.weight_updates, 1, l.weights, 1);
    weights.axpy(learning_rate / args.batch, weight_updates);

    //scal_cpu(l.nweights, args.momentum, l.weight_updates, 1);
    weight_updates.Multiply(args.momentum);

    //axpy_cpu(l.n, learning_rate / args.batch, l.bias_updates, 1, l.biases, 1);
    biases.axpy(learning_rate / args.batch, bias_updates);

    //scal_cpu(l.n, args.momentum, l.bias_updates, 1);
    bias_updates.multiply(args.momentum);

    if assigned(scales.Data) then
        begin
            //axpy_cpu(l.n, learning_rate / args.batch, l.scale_updates, 1, l.scales, 1);
            scales.axpy(learning_rate / args.batch, scale_updates);

            //scal_cpu(l.n, args.momentum, l.scale_updates, 1)
            scale_updates.multiply(args.momentum);
        end;
  inherited update(args);
end;

end.
