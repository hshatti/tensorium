unit nLogisticLayer;
{$ifdef FPC}
{$mode Delphi}{$H+}
{$endif}
interface

uses
  SysUtils, math, nTensors, nTypes, nBaseLayer
  {$ifdef USE_OPENCL}
  , opencl, OpenCLHelper, clblast
  {$endif}
  ;

type

  { TLogisticLayer }

  TLogisticLayer = class (TBaseLayer)
    loss : TSingleTensor;
    constructor Create(const ABatch, AInputs:SizeInt);
    procedure setBatch(ABatch: SizeInt); override;
    procedure setTrain(ATrain: boolean); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    {$ifdef USE_OPENCL}
    procedure forwardGPU(var state: TNNetState); override;
    procedure backwardGPU(var state: TNNetState); override;
    {$endif}
  private
    class procedure logisticCrossEntropy(const pred, truth:TSingleTensor; var delta, error: TSingleTensor); static;
  end;


implementation

{ TLogisticLayer }

constructor TLogisticLayer.Create(const ABatch, AInputs: SizeInt);
begin

  layerType       := ltLOGXENT;
  ActivationType  := acLOGISTIC;
  batch           := aBatch;
  inputs          := AInputs;
  inputShape      := [batch, inputs];
  outputs         := AInputs;
  loss            := TSingleTensor.Create([batch, inputs], batch);
  output          := TSingleTensor.Create([batch, inputs], batch);
  delta           := TSingleTensor.Create([batch, inputs], batch);
  cost            := [0]

end;

procedure TLogisticLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch=Batch then exit();
  Batch := ABatch;
  inputShape[0]     := batch;
  loss.resize([batch, Inputs], batch);
  output.resize([batch, Inputs], batch);
  delta.resize([batch, Inputs], batch);
end;

procedure TLogisticLayer.setTrain(ATrain: boolean);
begin
  if ATrain=FTrain then exit;
  FTrain := ATrain;
end;

procedure TLogisticLayer.forward(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
   if benchmark then metrics.forward.start(layerType);
  {$endif}


  //copy_cpu(l.outputs * l.batch, net.input, 1, l.output, 1);

  state.input.copyTo(output);

  //activate_array(l.output, l.outputs * l.batch, acLOGISTIC);
  Activate;

  if assigned(state.truth.Data) then  begin
    //logistic_x_ent_cpu(l.batch * l.inputs, l.output, net.truth, l.delta, l.loss);
    logisticCrossEntropy(output, state.truth, delta, loss);
    //l.cost := sum_array(l.loss, l.batch * l.inputs)
    cost[0] := loss.sum
  end;
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}

end;

procedure TLogisticLayer.backward(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
   if benchmark then metrics.backward.start(layerType);
  {$endif}

  //axpy_cpu(l.inputs * l.batch, 1, l.delta, 1, net.delta, 1)
  state.delta.Add(delta.Data);

  {$ifdef USE_TELEMETRY}
   if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;

{$ifdef USE_OPENCL}
procedure TLogisticLayer.forwardGPU(var state: TNNetState);
var t:TSingleTensor;
    ev:cl_event;
begin
  {$ifdef USE_TELEMETRY}
   if benchmark then metrics.forward.start(layerType);
  {$endif}
  ev := nil;
  //CLBlastScopy(output.Size, state.input.devData, 0, 1, output.devData, 0, 1, @ocl.ActiveQueue
  //, nil//@ev
  //);
  ocl.copy(output.Size(), state.input.devData, 0, 1, output.devData, 0, 1
  , 0
  , nil
  , nil);
  //ocl.finish;
  output.setOCL;
  delta.setOCL;
  loss.setOCL;

  ocl.ActivateArray(output.devData, output.Size(), longint(ActivationType)
  , 0//1
  , nil//@ev
  , nil//@ev
  );
  //ocl.finish();

  if assigned(state.truth.Data) then  begin
    if not state.truth.wasGPU() then
      state.truth.pushToDevice();

    ocl.crossEntropyLogistic(output.size(), output.devData, state.truth.devData, delta.devData, loss.devData
    , 0//1
    , nil//@ev
    , nil//@ev
    );
    //ocl.finish();

    loss.pullFromDevice();
    cost[0] := loss.sum
  end;
  //ocl.waitForEvents(1, @ev);
  //ocl.finish();

  //t.resize(output.shape);
  //output.pullFromDevice(t);
  //forward(state);
  //writeln('Logistic backward diff :', output.sumSqrDiff(t):1:sDigits);
  //readln;

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TLogisticLayer.backwardGPU(var state: TNNetState);
var t:TSingleTensor;
    ev:cl_event;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
  if not state.delta.wasGPU() then
    state.delta.pushToDevice();
  if not delta.wasGPU() then
    delta.pushToDevice();

  ocl.addvv(delta.size(), state.delta.devData, delta.devData
  , 0
  , nil
  , nil//@ev
  );
  //ocl.waitForEvents(1, @ev);
  //ocl.finish();

  //t.resize(state.delta.shape);
  //state.delta.pullFromDevice(t);
  //backward(state);
  //writeln('Logistic backward diff :', state.delta.sumSqrDiff(t):1:sDigits);
  //readln;

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;
{$endif}

class procedure TLogisticLayer.logisticCrossEntropy(const pred, truth: TSingleTensor; var delta, error: TSingleTensor);
var i:SizeInt;
    t,p:single;
begin
  // todo [logistic_x_ent] simdfy and GPU
  for i := 0 to delta.Size-1 do begin
      t := truth.Data[i];
      p := pred.Data[i];
      error.Data[i] := -t*ln(max(p, sEPSILON)) - (1-t) * ln(max(1 - p, sEPSILON));
      delta.Data[i] := t - p;
  end
end;

end.

