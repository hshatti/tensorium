unit nConvolutionLayer;
{$ifdef FPC}
{$mode Delphi}
{$endif}

interface

uses
  SysUtils, nTensors, nTypes, nBaseLayer, nCol2Im, nActivation
  {$ifdef MSWINDOWS)} , shellApi{$endif}
  ;

type

  { TBaseConvolutionalLayer }

  TBaseConvolutionalLayer = class(TBaseImageLayer)
    n, blurStride_x, blurStride_y : SizeInt;
    stride_x, stride_y            : SizeInt;
    kernelSize, Dilation, Padding : SizeInt;

    antialiasing                  : SizeInt;
    shareLayer                    : TBaseConvolutionalLayer;
    inputLayer                    : TBaseConvolutionalLayer;
    constructor create;
    function outWidth():SizeInt;
    function outHeight():SizeInt;
    procedure fuseBatchNorm;  override;
    function getWorkspaceSize: SizeInt; override;
    function getWorkspaceShape: TArray<SizeInt>; override;
    property Stride   : SizeInt read stride_x;
    property filters  : SizeInt read n write n;
  end;

  { TConvolutionLayer }

  TConvolutionalLayer=class(TBaseConvolutionalLayer)
  protected
    procedure setTrain(ATrain: boolean); override;
  public

    waitStreamId                  : SizeInt;
    maxBoxes, truths              : SizeInt;
    deform, Adam                  : boolean;
    assistedExcitation            : SizeInt;
    //steps,
    ActivationInput               : TSingleTensor;

    constructor Create(const ABatch, Aheight, Awidth, Achannels,
      Afilters: SizeInt; AGroups, AKernelSize:SizeInt;
      AStride_x:SizeInt=1; AStride_y:SizeInt=1; const ADilation: SizeInt=1; APadding: SizeInt=-1;
      const AActivation: TActivationType=acSIGMOID;
      const ABatch_normalize: boolean=false; const AAdam: boolean=false;
      const AIndex: SizeInt=0; const AAntialiasing: SizeInt=0; const AShare_layer: TConvolutionalLayer=nil;
      const AAssistedExcitation: SizeInt=0; const ADeform: boolean=false;
      const ATrain: boolean=false);
    procedure setBatch( ABatch: SizeInt); override;
    procedure assistedForward(var state: TNNetState);
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure update(const args: TUpdateArgs); override;
  end;

implementation
uses math, nnet;

{ TBaseConvolutionalLayer }

constructor TBaseConvolutionalLayer.create;
begin
  groups := 1
end;

function TBaseConvolutionalLayer.outWidth: SizeInt;
begin
  result := (w + 2 * Padding - kernelSize) div stride_x + 1
end;

function TBaseConvolutionalLayer.outHeight: SizeInt;
begin
  result := (h + 2 * Padding - kernelSize) div stride_y + 1
end;

procedure TBaseConvolutionalLayer.fuseBatchNorm;
var
  precomputed: single;
  filter_size, f, i, w_index: SizeInt;
begin
  if shareLayer <> nil then
    shareLayer.isBatchNormalized := false;
  if isBatchNormalized then
    begin
      for f := 0 to n -1 do
          begin
              precomputed := scales.data[f] / sqrt(max(rolling_variance.data[f], sEPSILON));
              biases.data[f] := biases.data[f] - rolling_mean.data[f] * precomputed;
              filter_size := kernelSize * kernelSize * c div groups;
              for i := 0 to filter_size -1 do
                  begin
                      w_index := f * filter_size + i;
                      weights.data[w_index] := weights.data[w_index] * precomputed
                  end
          end;
      if not assigned(shareLayer) then
          freeBatchNorm;
      isBatchNormalized := false;
    end
end;

function TBaseConvolutionalLayer.getWorkspaceSize: SizeInt;
begin
  result := batch * c * outH * outW * kernelSize * kernelSize
end;

function TBaseConvolutionalLayer.getWorkspaceShape: TArray<SizeInt>;
begin
  result := [batch, (c div groups), outH , outW , kernelSize * kernelSize]
end;

{ TConvolutionalLayer }

constructor TConvolutionalLayer.Create(const ABatch, Aheight, Awidth, Achannels,
  Afilters: SizeInt; AGroups, AKernelSize: SizeInt; AStride_x: SizeInt;
  AStride_y: SizeInt; const ADilation: SizeInt; APadding: SizeInt;
  const AActivation: TActivationType; const ABatch_normalize: boolean;
  const AAdam: boolean; const AIndex: SizeInt; const AAntialiasing: SizeInt;
  const AShare_layer: TConvolutionalLayer; const AAssistedExcitation: SizeInt;
  const ADeform: boolean; const ATrain: boolean);

var
  blur_nweights, i, blur_size, blur_pad: SizeInt;

  _scale : single;

begin
  inherited Create;
  layerType := ltCONVOLUTIONAL;
  FTrain := ATrain;
  if AGroups < 1 then
      groups := 1
  else
    groups := AGroups;
  blurStride_x := aStride_x;
  blurStride_y := aStride_y;
  antialiasing := AAntialiasing;
  if antialiasing>0 then begin
      AStride_x        := 1;
      AStride_y        := 1;
      stride_x         := 1;
      stride_y         := 1
  end;
  waitStreamId := -1;
  deform := aDeform;
  assistedExcitation := AAssistedExcitation;
  shareLayer := Ashare_layer;
  index := Aindex;
  h := Aheight;
  w := Awidth;
  c := Achannels;
  n := Afilters; // same as filters := AFilters;

  //result.use_bin_output := use_bin_output;
  batch := ABatch;
  //steps := ASteps;
  stride_x := AStride_x;
  stride_y := AStride_y;
  dilation := ADilation;
  kernelSize := AKernelSize;
  if APadding<0 then
    Padding := ADilation -1 + kernelSize div 2
  else
    Padding := APadding;
  isBatchNormalized := ABatch_normalize;
  learningRateScale := 1;
  //nweights := (c div groups) * filters * kernelSize * kernelSize;
  if assigned(shareLayer) then
      begin
          if (kernelSize <> shareLayer.kernelSize) or (weights.Size() <> shareLayer.weights.Size()) or (c <> shareLayer.c) or (n <> shareLayer.n) then
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
  outputs := filters * outH * outW;
  inputs := w * h * c;
  inputShape := [batch, c, h, w];
  ActivationType := AActivation;
  output := TSingleTensor.Create([batch , filters, outH, outW], batch);

  if FTrain then
      delta := TSingleTensor.Create([batch , filters, outH, outW], batch);

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
                  x := TSingleTensor.Create([batch, filters, outH, outW], batch);
                  x_norm := TSingleTensor.Create([batch, filters, outH, outW], batch)
              end
{$endif}
      end;
{$ifndef GPU}
  if ActivationType in [acSWISH, acMISH, acHARD_MISH] then
      ActivationInput := TSingleTensor.Create([batch, filters, outH, outW], batch);
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


  //workspaceSize := getWorkspaceSize();
  if antialiasing>0 then
      begin
          blur_size := 3;
          blur_pad := blur_size div 2;
          if antialiasing = 2 then
              begin
                  blur_size := 2;
                  blur_pad := 0
              end;
          InputLayer := TConvolutionalLayer.Create(batch, {steps,} outH, outW, n, n, n, blur_size, blurstride_x, blurstride_y, 1, blur_pad, acLINEAR, false, false, index, 0, nil, 0, false, train);
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
          end else begin
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

procedure TConvolutionalLayer.setBatch(ABatch: SizeInt);
//var total_batch : SizeInt;
begin
  if ABatch=Batch then exit();
  Batch := ABatch;
  inputShape[0] := batch;

  output.reSize([batch , filters, outH, outW], batch);
  if train then
      delta.reSize([batch , filters, outH, outW], batch);
{$ifndef GPU}
  if ActivationType in [acSWISH, acMISH, acHARD_MISH] then
      ActivationInput.resize([batch, filters, outH, outW], batch);
{$endif}
{$ifndef GPU}
  if train then
      begin
          x.reSize([batch, filters, outH, outW], batch);
          x_norm.reSize([batch, filters, outH, outW], batch)
      end;
{$endif}

  if antialiasing<>0 then
    inputLayer.setBatch(batch);
end;


procedure TConvolutionalLayer.assistedForward(var state: TNNetState);
var
    iteration_num, b, _w, _h, _c, t, left, right, top, bottom: SizeInt;
    alpha: single;
    a_avg, g: TArray<single>;
    Ps :PSingle;
    truth: TBox;
    net : TNNet;
begin
    net := TNNet(state.net);
    iteration_num := net.seen div (net.batch * net.subdivisions);
    alpha := (1+cos(nTensors.PI * iteration_num / net.maxBatches));
    if assistedExcitation > 1 then
        begin
            if iteration_num > assistedExcitation then
                alpha := 0
            else
                alpha := (1+cos(nTensors.PI * iteration_num / assistedExcitation))
        end;
    setLength(a_avg, outw * outh * batch);
    setLength(g, outW * outH * batch);
    maxboxes := net.numBoxes;
    truths := maxBoxes * (4+1);
    for b := 0 to batch -1 do
        begin
            for t := 0 to net.numBoxes -1 do
                begin
                    truth := TBox.fromFloat(Pointer(state.truth.data + t * (4+1)+b * truths), 1);
                    if truth.x=0 then
                        break;
                    left := floor((truth.x - truth.w / 2) * outW);
                    right := ceil((truth.x + truth.w / 2) * outW);
                    top := floor((truth.y - truth.h / 2) * outH);
                    bottom := ceil((truth.y + truth.h / 2) * outH);
                    for _w := left to right do
                        for _h := top to bottom -1 do
                            g[_w + outW * _h + outW * outH * b] := 1
                end
        end;
    for b := 0 to batch -1 do
        for _w := 0 to outW -1 do
            for _h := 0 to outH -1 do
                begin
                    Ps := @a_avg[w + outW * (h + outH * b)];
                    for _c := 0 to outC -1 do
                        Ps^ := Ps^ + output.data[_w + outW * (_h + outH * (_c + outC * b))];
                    Ps^ := Ps^ / outC
                end;
    for b := 0 to batch -1 do
        for _w := 0 to outW -1 do
            for _h := 0 to outH -1 do
                for _c := 0 to outC -1 do begin
                    Ps := @output.Data[_w + outW * (_h + outH * (_c + outC * b))];
                    Ps^ := Ps^ + alpha * g[_w + outW * (_h + outH * b)] * a_avg[_w + outW * (_h + outH * b)];
                end
end;

procedure TConvolutionalLayer.setTrain(ATrain: boolean);
//var
    //total_batch:SizeInt;
begin
  if ATrain = FTrain  then exit;
  FTrain := ATrain;
  if ATrain then begin
      delta := TSingleTensor.Create([batch , filters, outH, outW]);
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
          x := TSingleTensor.Create([batch, filters, outH, outW]);
          x_norm := TSingleTensor.Create([batch, filters, outH, outW])
      end;

  end else begin
      delta.free;
      if not assigned(shareLayer) then begin
          weight_updates.free;
          bias_updates.free;
          weights_ema.free;
          biases_ema.free;
          if isBatchNormalized then begin
              scales_ema.free;
              scale_updates.free;
              mean.free;
              variance.free;
              mean_delta.free;
              variance_delta.free
          end;
      end;
      if isBatchNormalized then begin
          x.free;
          x_norm.free
      end;
  end;

end;

procedure TConvolutionalLayer.forward(var state: TNNetState);
var
    s: TNNetState;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(layerType);
    {$endif}

    ////output.fill(0);


    //outH := convolutional_out_height(l);
    //outW := convolutional_out_width(l);
    // todo TConvolutionalLayer : implement forward and backward groups and groupId for multi GPU
    //k := kernelSize * kernelSize * c div groups;
    //outImgSize := outH * outW;
    //imcolSize  := outImgSize * kernelSize * kernelSize * c ;
    //m := filters div groups;
    //nweights := weights.size();
    //u := 0;
    //inc(u);
    //for i := 0 to batch -1 do
    //    for j := 0 to groups -1 do
    //    begin
    //        _A := weights.Data;
    //        im := state.input.data +(i * groups + j)*(c div groups) * h * w;
    //        _B := pointer(state.workspace.data + (i * groups + j)*(c div groups)*imcolSize);

            //im := state.input.data + i * c * h * w + j * c * h * w div groups;
            //_B := pointer(state.workspace.data + i*c*imcolSize + j*c*imcolSize div groups);

    //        _C := output.data + i * outImgSize * m;
    //        if (kernelSize = 1) and (stride = 1) and (dilation = 1) then
    //            _B := im
    //        else
    //            im2col_cpu_ext(pointer(im), c div groups, h, w, kernelSize, kernelSize, Padding * dilation, Padding * dilation, stride_y, stride_x, dilation, dilation, pointer(_B));
    //        TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, outImgSize, k, 1, _A, k, _B, outImgSize, 1, _C, outImgSize);
    //    end;

    // todo [TConvolutionLayer] implement multi GPU grouos

    state.input.Conv2D(weights, output, Padding, Padding, stride_x, stride_y, Dilation, Dilation);

    //state.input.conv2d(weights, output, Padding, padding, stride_x, stride_y, Dilation, Dilation) ;
    if isBatchNormalized then
      batchNorm(state)
    else
      output.Add(biases);

    case ActivationType of
       acSWISH :
          activate_array_swish(output, outputs * batch, ActivationInput, output);
       acMISH :
          activate_array_mish(output, outputs * batch, activationInput, output);
       acHARD_MISH :
            activate_array_hard_mish(output, outputs * batch, ActivationInput, output);
       acNORM_CHAN :
            activate_array_normalize_channels(output, outputs * batch, batch, outC, outW * outH, output);
       acNORM_CHAN_SOFTMAX, acNORM_CHAN_SOFTMAX_MAXVAL :
            activate_array_normalize_channels_softmax(output, outputs * batch, batch, outC, outW * outH, output, activationType = acNORM_CHAN_SOFTMAX_MAXVAL);
       else begin
           //output.print(true, 3);
           activate()
       end;
    end;
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

procedure TConvolutionalLayer.backward(var state: TNNetState);
var
    i,j, m, n, k, nweights, colSize: SizeInt;
    _A, _B, _C, im: Pointer;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
      m := filters div groups;
      n := kernelSize * kernelSize * c div groups;
      k := outW * outW;

      colSize := getWorkspaceSize div batch;

      case activationType of
        acSWISH :
          gradient_array_swish(output, outputs * batch, activationInput, delta) ;
        acMISH  :
          gradient_array_mish(outputs * batch, activationInput, delta) ;
        acHARD_MISH :
          gradient_array_hard_mish(outputs * batch, activationInput, delta) ;
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
          bias_updates.Add(delta);

      nweights := weights.size();
      //for i := 0 to batch -1 do
      //    for j := 0 to groups -1 do
      //        begin
      //            _A := delta.Data+(i * groups+j) * m * k;
      //            _B := pointer(state.workspace.data);
      //            _C := weight_updates.data  + j * nweights div groups;
      //            im := state.input.data +(i * groups+j) * (c div groups) * h * w;
      //            im2col_cpu_ext(im, c div groups, h, w, kernelSize, kernelSize, padding * dilation, padding * dilation, stride_y, stride_x, dilation, dilation, _B);
      //            TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, _A, k, _B, k, 1, _C, n);
      //            if assigned(state.delta.Data) then
      //                begin
      //                    _A := weights.Data+j * nweights div groups;
      //                    _B := delta.Data + (i * groups+j) * m * k;
      //                    _C := pointer(state.workspace.data);
      //                    TSingleTensor.gemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, k, m, 1, _A, n, _B, k, 0, _C, k);
      //                    col2im_cpu_ext(pointer(state.workspace.data), c div groups, h, w, kernelSize, kernelSize, padding * dilation, padding * dilation, stride_y, stride_x, dilation, dilation, pointer(state.delta.data +(i * groups+j) * (c div groups) * h * w))
      //                end
      //        end ;

      //for i:= 0 to batch -1 do begin
      //  _A := delta.Data+i * m * k;
      //  _B := pointer(state.workspace.data + i*imcolSize);
      //  _C := weight_updates.data ;
      //  im := state.input.data + c * h * w;
      //  im2col_cpu_ext(im, c div groups, h, w, kernelSize, kernelSize, padding * dilation, padding * dilation, stride_y, stride_x, dilation, dilation, _B);
      //  TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, _A, k, _B, k, 1, _C, n);
      //end;

      state.input.im2Col(kernelSize, kernelSize, padding * dilation, padding * dilation, stride_y, stride_x, dilation, dilation, state.workspace.data);

      for i:= 0 to batch -1 do
        TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, delta.data + i*m*k, k, state.workspace.data + i*ColSize, k, 1, weight_updates.data, n);

      if assigned(state.delta.data) then begin
        for i := 0 to batch -1 do begin
            _A := weights.Data;
            _B := delta.Data + i * m * k;
            _C := pointer(state.workspace.data + i*colSize);
            TSingleTensor.gemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, k, m, 1, _A, n, _B, k, 0, _C, k);
        end;

        state.delta.col2Im(kernelSize, kernelSize, padding*Dilation, padding*Dilation, stride_x, stride_y, dilation, dilation, state.workspace.data);
      end;
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;

procedure TConvolutionalLayer.update(const args: TUpdateArgs);
var
    learning_rate: single;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.start(layerType);
  {$endif}

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

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.finish(layerType);
  {$endif}
end;

initialization

end.

