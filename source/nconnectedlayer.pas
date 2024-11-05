unit nConnectedlayer;
{$ifdef fpc}
{$mode Delphi}{$H+}
{$endif}

interface

uses
  SysUtils, Math
  ,ntensors, NTypes, nBaseLayer
  {$ifdef USE_OPENCL}
  , opencl, clblast, OpenCLHelper
  {$endif}
  ;

type

  { TConnectedLayer }

  TConnectedLayer = class(TBaseLayer)
    constructor Create(const ABatch, AInputs, aOutputs: SizeInt; const AActivationType:TActivationType= acLINEAR; AIsBatchNormalized:boolean=false);
    procedure setBatch(ABatch:SizeInt); override;
    procedure setTrain(ATrain: boolean); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure DeNormalize;
    procedure update(const args: TUpdateArgs); override;
    {$ifdef USE_OPENCL}
    procedure forwardGPU(var state: TNNetState);  override;
    procedure backwardGPU(var state: TNNetState); override;
    procedure updateGPU(const args: TUpdateArgs); override;
    {$endif}
  end;

implementation

{ TConnectedLayer }

constructor TConnectedLayer.Create(const ABatch, AInputs, aOutputs: SizeInt;
  const AActivationType: TActivationType; AIsBatchNormalized: boolean);
var randomRange:Single;
begin
  batch                 := ABatch;
  layerType             := ltCONNECTED;
  ActivationType        := AActivationType;
  inputs                := AInputs;
  inputShape            := [batch, inputs];
  outputs               := aOutputs;
  isBatchNormalized     := AIsBatchNormalized;

  output                := TSingleTensor.Create([batch , outputs], Batch);
  weights               := TSingleTensor.Create([outputs , inputs]);
  randomRange           := sqrt(2/inputs);
  weights.UniformDistribution(-randomRange, randomRange);
  biases                := TSingleTensor.Create([outputs]);

  if isBatchNormalized then begin
    scales              := TSingleTensor.Create([outputs]);
    scale_updates       := TSingleTensor.Create([outputs]);
    scales.fill(1.0);
    mean                := TSingleTensor.Create([outputs]);
    mean_delta          := TSingleTensor.Create([outputs]);
    variance            := TSingleTensor.Create([outputs]);
    variance_delta      := TSingleTensor.Create([outputs]);
    rolling_mean        := TSingleTensor.Create([outputs]);
    rolling_variance    := TSingleTensor.Create([outputs]);
    x                   := TSingleTensor.Create([batch, outputs], Batch);
    x_norm              := TSingleTensor.Create([batch, outputs], Batch)
  end;

end;

procedure TConnectedLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch=Batch then exit();
  Batch := ABatch;
  inputShape[0] := ABatch;

  output.reSize([batch , outputs], Batch) ;

  if FTrain then
      delta.reSize([batch , outputs], Batch) ;

  if isBatchNormalized then begin
    x.reSize([batch, outputs], Batch);
    x_norm.reSize([batch, outputs], Batch);
  end;
end;

procedure TConnectedLayer.setTrain(ATrain: boolean);
begin
  if ATrain=FTrain then exit;
  FTrain := ATrain;

  if FTrain then begin
    delta                 := TSingleTensor.Create([batch , outputs], Batch);
    weight_updates        := TSingleTensor.Create([inputs , outputs]);
    bias_updates          := TSingleTensor.Create([outputs]);
  end else begin
    delta.free;
    weight_updates.free;
    bias_updates.free
  end;
end;

procedure TConnectedLayer.forward(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}

  //output.fill(0);

  {$ifdef USE_TELEMETRY}
  if benchmark then tensorMetrics.start(opGemm);
  {$endif}
  TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, outputs, inputs, 1
    , state.input^, inputs
    , weights, inputs
    , 0, output, outputs);
  {$ifdef USE_TELEMETRY}
  if benchmark then tensorMetrics.finish(opGemm);
  {$endif}
    //state.input.matMul(weights, output, CblasNoTrans, CblasTrans);
  if isBatchNormalized and (batch > 1) then begin
      if state.isTraining then begin
          //mean_cpu(output.data , batch, outputs, 1, mean.data);
          //variance_cpu(output.data , mean, batch, outputs, 1, variance.data);
          output.MeansAndVars(mean, variance);

          //scal_cpu(outputs, 0.95, rolling_mean, 1);
          //axpy_cpu(outputs, 0.05, mean, 1, rolling_mean, 1);
          rolling_mean.Multiply(0.95);
          rolling_mean.axpy(0.05, mean);

          //scal_cpu(outputs, 0.95, rolling_variance, 1);
          //axpy_cpu(outputs, 0.05, variance, 1, rolling_variance, 1);
          rolling_variance.Multiply(0.95);
          rolling_variance.axpy(0.05, variance);

          //copy_cpu(l.outputs * l.batch, l.output, 1, l.x, 1);
          output.CopyTo(x);

          //normalize_cpu(output, mean, variance, batch, outputs, 1);
          output.Normalize(mean, variance);

          //copy_cpu(l.outputs * l.batch, l.output, 1, l.x_norm, 1)
          output.copyTo(x_norm)
      end else
          //normalize each column
          //normalize_cpu(output, rolling_mean, rolling_variance, batch, outputs, 1);
          output.Normalize(rolling_mean, rolling_variance);

      //scale_bias(l.output, l.scales, l.batch, l.outputs, 1);
      output.Multiply(scales);
  end;

  //for i := 0 to batch -1 do
  //    TSingleTensor.axpysvv(outputs, 1, biases.data, 1, output.data+i * outputs, 1);
  output.add(biases);

  //activate_array(l.output, l.outputs * l.batch, l.activation);
  activate();

  //write(#$1b'[2J'#$1b'[H');
  //output.print(psGray);
  //readln;
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TConnectedLayer.backward(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
  //gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);
  Derivative;
  //for i := 0 to batch -1 do
  //    TSingleTensor.axpysvv(outputs, 1, delta.data+i * outputs, 1, bias_updates.data, 1);
  bias_updates.add(delta);

  //if l.batch_normalize then
  if isBatchNormalized and (batch > 1) then begin
      // spatial dot (x_norm . delta) then add to scale_updates
      //backward_scale_cpu(x_norm, delta, batch, outputs, 1, scale_updates);
      scale_updates.addDots(x_norm, delta);

      // add scales to all delta batches
      //scale_bias(delta, scales, batch, outputs, 1);
      delta.add(scales);

      //mean_delta_cpu(delta, variance, batch, outputs, 1, mean_delta);
      //variance_delta_cpu(x, delta, mean, variance, batch, outputs, 1, variance_delta);
      delta.MeansAndVarsDelta(delta, x, mean, variance, mean_delta, variance_delta);

      //normalize_delta_cpu(x, mean, variance, mean_delta, variance_delta, batch, outputs, 1, delta)
      delta.normalizeDelta(x, mean, variance, mean_delta, variance_delta, delta);
  end;

  {$ifdef USE_TELEMETRY}
  if benchmark then tensorMetrics.start(opGemm);
  {$endif}
  //delta.matMul(state.input, weight_updates, CblasTrans, CblasNoTrans);

  //if assigned(state.delta) then
      //sgemm(0, 0, l.batch, l.inputs, l.outputs, 1, l.delta, l.outputs, l.weights, l.inputs, 1, state.delta, l.inputs)

  TSingleTensor.gemm(CblasRowMajor, CblasTrans, CblasNoTrans, outputs, inputs, batch, 1
  , delta, outputs
  , state.input^, inputs
  , 1, weight_updates, inputs);

  if assigned(state.delta) and assigned(state.delta.Data) then
      TSingleTensor.gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch, inputs, outputs, 1
      , delta, outputs
      , weights, inputs
      , 1, state.delta^, inputs);
  {$ifdef USE_TELEMETRY}
   if benchmark then tensorMetrics.finish(opGemm);
  {$endif}
      //delta.matMul(weights, state.delta)

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;

procedure TConnectedLayer.DeNormalize;
var
    i, j: SizeInt;
    _scale: single;
begin
    // tofdo SIMDfy and GPU
    for i := 0 to outputs -1 do
        begin
            _scale := scales.data[i] / max(sqrt(rolling_variance.data[i]), sEPSILON);
            for j := 0 to inputs -1 do
                weights.data[i * inputs+j] := weights.data[i * inputs+j] * _scale;
            biases.data[i] := biases.data[i] - (rolling_mean.data[i] * _scale);
            scales.data[i] := 1;
            rolling_mean.data[i] := 0;
            rolling_variance.data[i] := 1
        end
end;

procedure TConnectedLayer.update(const args: TUpdateArgs);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.start(layerType);
  {$endif}
  //axpy_cpu(l.outputs, args.learning_rate / args.batch, l.bias_updates, 1, l.biases, 1);
  //TSingleTensor.axpysvv(outputs, args.learningRate / args.batch, bias_updates, 1, biases, 1);
  biases.axpy(args.learningRate / args.batch, bias_updates);

  //scal_cpu(l.outputs, args.momentum, l.bias_updates, 1);
  bias_updates.Multiply(args.momentum);
  if isBatchNormalized and (batch > 1) then begin
      //axpy_cpu(l.outputs, args.learning_rate / args.batch, l.scale_updates, 1, l.scales, 1);
      scales.axpy(args.learningRate / args.batch, scale_updates);

      //scal_cpu(l.outputs, args.momentum, l.scale_updates, 1)
      scale_updates.Multiply(args.momentum);
  end;

  //axpy_cpu(l.inputs * l.outputs, -args.decay * args.batch, l.weights, 1, l.weight_updates, 1);
  //TSingleTensor.axpysvv(weight_updates.Size(), -args.decay * args.batch, weights, 1, weight_updates, 1);
  weight_updates.axpy(-args.decay * args.batch, weights);

  //axpy_cpu(l.inputs * l.outputs, args.learning_rate / args.batch, l.weight_updates, 1, l.weights, 1);
  //TSingleTensor.axpysvv(weights.size(), args.learningRate / args.batch, weight_updates, 1, weights, 1);
  weights.axpy(args.learningRate / args.batch, weight_updates);

  //scal_cpu(l.inputs * l.outputs, args.momentum, l.weight_updates, 1)
  weight_updates.Multiply(args.momentum);

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.finish(layerType);
  {$endif}
end;

{$ifdef USE_OPENCL}
//const clearscr=#$1B'[2J'#$1B'[2;1H';
procedure TConnectedLayer.forwardGPU(var state: TNNetState);
var ip,t,w:TSingleTensor;
    ev : array[0..15] of cl_event;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}

  //output.fill(0);

  {$ifdef USE_TELEMETRY}
    if benchmark then tensorMetrics.start(opGemm);
  {$endif}

  //writeln(clearscr);
  if not state.input.wasGPU() then  begin
      state.input.pushToDevice;
  end;
  if not weights.wasGPU() then  begin
      weights.pushToDevice;
  end;
  if not biases.wasGPU() then  begin
      biases.pushToDevice;
  end;
  ev[0]:=nil;
  output.setOCL;
  //ocl.FErr := integer(CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeYes
  //        , batch, outputs, inputs, 1
  //        , state.input.devData, 0, inputs
  //        , weights.devData    , 0, inputs
  //        , 0, output.devData  , 0, outputs
  //        , @ocl.ActiveQueue
  //        , nil//@ev[0]
  //        ));  ocl.checkError();

  ocl.gemm(false, true
          , batch, outputs, inputs, 1
          , state.input.devData, 0, inputs
          , weights.devData    , 0, inputs
          , 0, output.devData  , 0, outputs
          , 0
          , nil
          , nil//@ev[0]
          );


  //ocl.finish();
  {$ifdef USE_TELEMETRY}
  if benchmark then tensorMetrics.finish(opGemm);
  {$endif}

  if isBatchNormalized and (batch > 1) then begin
      if state.isTraining then begin
              output.MeansAndVars(mean, variance);

              rolling_mean.Multiply(0.95);
              rolling_mean.axpy(0.05, mean);

              rolling_variance.Multiply(0.95);
              rolling_variance.axpy(0.05, variance);

              output.CopyTo(x);

              output.Normalize(mean, variance);

              output.copyTo(x_norm)
      end else
          output.Normalize(rolling_mean, rolling_variance);

      output.Multiply(scales);
  end;

  ocl.forwardBias(output.groupSize(), output.devData, 1, biases.devData, 1, output.groups
  , 0  // 1
  , nil//@ev[0]
  , nil//@ev[1]
  );
  //ocl.finish();

  ocl.ActivateArray(output.devData, output.size, longint(ActivationType)
  , 0  //1
  , nil//@ev[1]
  , nil//@ev[2]
  );
  //ocl.finish();

  //ocl.waitForEvents(3, @ev[0]);
  //ocl.freeEvents(3, @ev[0]);
  //ocl.finish();


  //ocl.freeEvents(3, @ev);
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TConnectedLayer.backwardGPU(var state: TNNetState);
var t:TSingleTensor;
    ev : array[0..15] of cl_event;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}

  if not delta.wasGPU() then
      delta.pushToDevice;

  if not state.input.wasGPU() then
      state.input.pushToDevice;

  if not output.wasGPU() then
      output.pushToDevice;

  if not bias_updates.wasGPU() then
      bias_updates.pushToDevice;

  if not weight_updates.wasGPU() then
      weight_updates.pushToDevice;


  ocl.DeriveArray(output.devData, output.size, longint(ActivationType), delta.devData
  , 0
  , nil
  , nil//@ev[0]
  );
  //ocl.finish();

  ocl.backwardBias(bias_updates.size, bias_updates.devData, 1, delta.devData, 1, delta.Groups
  , 0  //1
  , nil//@ev[0]
  , nil//@ev[1]
  );
  //ocl.finish();

  if isBatchNormalized and (batch > 1) then begin
      scale_updates.addDots(x_norm, delta);

      delta.add(scales);

      delta.MeansAndVarsDelta(delta, x, mean, variance, mean_delta, variance_delta);

      delta.normalizeDelta(x, mean, variance, mean_delta, variance_delta, delta);
  end;

  {$ifdef USE_TELEMETRY}
  if benchmark then tensorMetrics.start(opGemm);
  {$endif}
//
  //ocl.FErr := integer(CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeYes, CLBlastTransposeNo
  //    , outputs, inputs, batch, 1
  //    , delta.devData, 0, outputs
  //    , state.input.devData, 0, inputs
  //    , 1, weight_updates.devData, 0, inputs, @ocl.ActiveQueue
  //    , nil//@ev[0]
  //    )); ocl.checkError();

  ocl.gemm(true, false
    , outputs, inputs, batch, 1
    , delta.devData, 0, outputs
    , state.input.devData, 0, inputs
    , 1, weight_updates.devData, 0, inputs
    , 0//1
    , nil//@ev[0]
    , nil//@ev[0]
    );

    //ocl.finish();

  if assigned(state.delta) and assigned(state.delta^.devData) then begin
      if not weights.wasGPU then
          weights.pushToDevice;
      //ocl.FErr := integer(CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo
      //    , batch, inputs, outputs, 1
      //    , delta.devData, 0, outputs
      //    , weights.devData, 0, inputs
      //    , 1, state.delta.devData, 0, inputs, @ocl.ActiveQueue
      //    , nil//@ev[0]
      //    ));  ocl.checkError();
      ocl.gemm( false, false
          , batch, inputs, outputs, 1
          , delta.devData, 0, outputs
          , weights.devData, 0, inputs
          , 1, state.delta.devData, 0, inputs
          , 0//1
          , nil//@ev[0]
          , nil//@ev[0]
          );

  end ;
  //ocl.waitForEvents(2, @ev[0]);
  //ocl.freeEvents(2, @ev[0]);
  //ocl.finish();
  //ocl.freeEvents(1, @ev[0]);


  {$ifdef USE_TELEMETRY}
   if benchmark then tensorMetrics.finish(opGemm);
  {$endif}

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}

end;

procedure TConnectedLayer.updateGPU(const args: TUpdateArgs);
var t:TSingleTensor;
    ev : array[0..15] of cl_event;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.start(layerType);
  {$endif}

  if not biases.wasGPU() then
      biases.pushToDevice;
  if not bias_updates.wasGPU() then
      bias_updates.pushToDevice;
  if not weights.wasGPU() then
      weights.pushToDevice;
  if not weight_updates.wasGPU() then
      weight_updates.pushToDevice;

  ocl.axpy(biases.Size(), args.learningRate / args.batch, bias_updates.devData, 1, biases.devData, 1
  , 0
  , nil
  , nil //@ev[0]
  );
  //ocl.finish();

  ocl.scale(bias_updates.Size(), args.momentum, bias_updates.devData, 1
  , 0  //1
  , nil//@ev[0]
  , nil//@ev[1]
  );
  //ocl.finish();
  if isBatchNormalized and (batch > 1) then begin
      scales.axpy(args.learningRate / args.batch, scale_updates);

      scale_updates.Multiply(args.momentum);
  end;

  ocl.axpy(weight_updates.Size(), -args.decay * args.batch, weights.devData, 1, weight_updates.devData, 1
  , 0
  , nil
  , nil//@ev[2]
  );
  //ocl.finish();

  ocl.axpy(weights.Size(), args.learningRate / args.batch, weight_updates.devData, 1, weights.devData, 1
  , 0  //1
  , nil//@ev[2]
  , nil//@ev[2]
  );
  //ocl.finish();

  ocl.scale(weight_updates.size(), args.momentum, weight_updates.devData, 1
  , 0//1
  , nil//@ev[2]
  , nil//@ev[2]
  );
  //ocl.waitForEvents(3, @ev[0]);
  //ocl.freeEvents(3, @ev[0]);
  //ocl.finish();
  //ocl.freeEvents(5, @ev[0]);

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.update.finish(layerType);
  {$endif}
end;
{$endif}

end.

