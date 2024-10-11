unit nBatchNormLayer;

{$ifdef FPC}
{$mode Delphi}
{$endif}

interface

uses
  SysUtils, nTensors, nTypes, nBaseLayer;

type

  { TBatchNorm }

  TBatchNormLayer = class(TBaseImageLayer)
    constructor Create(const Abatch, AWidth, AHeight, AChannels: SizeInt; const ATrain:boolean);
    procedure setBatch(ABatch: SizeInt); override;
    procedure setTrain(ATrain: boolean); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure update(const args: TUpdateArgs); override;
  end;

implementation

{ TBatchNormLayer }

constructor TBatchNormLayer.Create(const Abatch, AWidth, AHeight,
  AChannels: SizeInt; const ATrain: boolean);
begin
  layerType        := ltBATCHNORM;
  batch            := Abatch;
  train            := Atrain;

  w                := AWidth;
  h                := AHeight;
  c                := AChannels;

  //outW             := w;
  //outH             := AHeight;
  //outC             := AChannels;

  //n                := c;

  output           := TSingleTensor.Create([batch, c ,h, w], batch);

  delta            := TSingleTensor.Create([batch, c ,h, w], batch);

  inputs           := w * h * c;
  inputShape       := [batch, c, h, w];
  outputs          := inputs;

  biases           := TSingleTensor.Create([c]);

  bias_updates     := TSingleTensor.Create([c]);

  scales           := TSingleTensor.Create([c]);
  scales.fill(1);
  scale_updates    := TSingleTensor.Create([c]);

  mean             := TSingleTensor.Create([c]);
  variance         := TSingleTensor.Create([c]);
  rolling_mean     := TSingleTensor.Create([c]);
  rolling_variance := TSingleTensor.Create([c]);
  mean_delta       := TSingleTensor.Create([c]);
  variance_delta   := TSingleTensor.Create([c]);

  x                := TSingleTensor.Create([batch, c ,h, w], batch);
  x_norm           := TSingleTensor.Create([batch, c ,h, w], batch);
end;

procedure TBatchNormLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch=Batch then exit();
  batch            := ABatch;
  inputShape[0]    := batch;
  output.resize([batch, c ,h, w], batch);
  delta.resize([batch, c ,h, w], batch);
  x.resize([batch, c ,h, w], batch);
  x_norm.resize([batch, c ,h, w], batch);
end;

procedure TBatchNormLayer.setTrain(ATrain: boolean);
begin
  if FTrain = ATrain then exit();
  FTrain := ATrain
end;

procedure TBatchNormLayer.forward(var state: TNNetState);
begin
{$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
{$endif}
  
    //if l.&type = ltBATCHNORM then
    //    copy_cpu(l.outputs * l.batch, state.input, 1, l.output, 1);
    //if l.&type = ltCONNECTED then begin
    //    l.out_c :=l.outputs;
    //    l.out_h :=1; l.out_w:=1;
    //end;
    //
    //if state.train then
    //    begin
    //        mean_cpu(@l.output[0], l.batch, l.out_c, l.out_h * l.out_w, @l.mean[0]);
    //        variance_cpu(@l.output[0], @l.mean[0], l.batch, l.out_c, l.out_h * l.out_w, @l.variance[0]);
    //
    //        scal_cpu(l.out_c, 0.9, @l.rolling_mean[0], 1);
    //        axpy_cpu(l.out_c, 0.1, @l.mean[0], 1, @l.rolling_mean[0], 1);
    //        scal_cpu(l.out_c, 0.9, @l.rolling_variance[0], 1);
    //        axpy_cpu(l.out_c, 0.1, @l.variance[0], 1, @l.rolling_variance[0], 1);
    //
    //        copy_cpu(l.outputs * l.batch, l.output, 1, l.x, 1);
    //        normalize_cpu(@l.output[0], @l.mean[0], @l.variance[0], l.batch, l.out_c, l.out_h * l.out_w);
    //        copy_cpu(l.outputs * l.batch, @l.output[0], 1, l.x_norm, 1)
    //    end
    //else
    //    normalize_cpu(@l.output[0], @l.rolling_mean[0], @l.rolling_variance[0], l.batch, l.out_c, l.out_h * l.out_w);
    //
    //scale_add_bias(@l.output[0], @l.scales[0], @l.biases[0], l.batch, l.out_c, l.out_h * l.out_w);
    //
    ////scale_bias(@l.output[0], @l.scales[0], l.batch, l.out_c, l.out_h * l.out_w);
    ////add_bias(@l.output[0], @l.biases[0], l.batch, l.out_c, l.out_h * l.out_w)
/////************************************************
(*  if LayerType = ltBATCHNORM then
      state.input.copyTo(output.data);

  //if l.&type = ltCONNECTED then begin
  //    outC := outputs;
  //    outH :=1;
  //    outW:=1;
  //end;

  if state.isTraining then begin
      output.MeansAndVars(mean, variance);
      rolling_mean.Multiply(0.9);
      rolling_mean.axpy(0.1, mean);
      rolling_variance.Multiply(0.9);
      rolling_variance.axpy(0.1, variance);
      output.CopyTo(x.Data);
      output.Normalize(mean, variance);
      output.copyTo(x_norm.Data)
  end else
      output.Normalize(rolling_mean, rolling_variance);

  output.Multiply(scales);
  output.add(biases);

{$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
{$endif}
*)
  batchNorm(state);
end;

procedure TBatchNormLayer.backward(var state: TNNetState);
begin
  //backward_scale_cpu(@l.x_norm[0], @l.delta[0], l.batch, l.out_c, l.out_w * l.out_h, @l.scale_updates[0]);
  //scale_bias(@l.delta[0], @l.scales[0], l.batch, l.out_c, l.out_h * l.out_w);
  //mean_delta_cpu(@l.delta[0], @l.variance[0], l.batch, l.out_c, l.out_w * l.out_h, @l.mean_delta[0]);
  //variance_delta_cpu(@l.x[0], @l.delta[0], @l.mean[0], @l.variance[0], l.batch, l.out_c, l.out_w * l.out_h, @l.variance_delta[0]);
  //normalize_delta_cpu(@l.x[0], @l.mean[0], @l.variance[0], @l.mean_delta[0], @l.variance_delta[0], l.batch, l.out_c, l.out_w * l.out_h, @l.delta[0]);
  //if l.&type = ltBATCHNORM then
  //    copy_cpu(l.outputs * l.batch, l.delta, 1, state.delta, 1)

{
  // spatial dot (x_norm . delta) then add to scale_updates
  scale_updates.addDots(x_norm, delta);

  // add scales to all delta batches
  delta.add(scales);
  delta.MeansAndVarsDelta(delta, x, mean, variance, mean_delta, variance_delta);
  delta.normalizeDelta(x, mean, variance, mean_delta, variance_delta, delta);
  if layerType = ltBATCHNORM then
      delta.copyTo(state.delta.Data)
}
  batchNormBack(state);

end;

procedure TBatchNormLayer.update(const args: TUpdateArgs);
begin
  //int size = l.nweights;

  biases.axpy(args.learningRate / args.batch, bias_updates);
  bias_updates.Multiply(args.momentum);

  scales.axpy(args.learningRate / args.batch, scale_updates);
  scale_updates.Multiply(args.momentum);

  //axpy_cpu(l.c, args.learning_rate / args.batch, l.bias_updates, 1, l.biases, 1);
  //scal_cpu(l.c, args.momentum, l.bias_updates, 1);
  //axpy_cpu(l.c, args.learning_rate / args.batch, l.scale_updates, 1, l.scales, 1);
  //scal_cpu(l.c, args.momentum, l.scale_updates, 1);

  inherited update(args);
end;

end.

