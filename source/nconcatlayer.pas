unit nConcatLayer;
{$ifdef fpc}
{$mode Delphi}
{$endif}
{$pointermath on}

interface

uses
  SysUtils, nTypes, nTensors, nBaseLayer;

type

  { TConcatLayer }

  TConcatLayer=class(TBaseImageLayer)
    inTensors : TArray<TSingleTensor>;
    inputLayers, inputSizes : TArray<SizeInt>;
    groupId : SizeInt;
    constructor Create(const aBatch :SizeInt; const aInputLayers, aInputSizes: TArray<SizeInt>; const aGroups:SizeInt=1;const aGroupId: SizeInt=0);
    procedure setTrain(ATrain: boolean); override;
    procedure setBatch(ABatch: SizeInt); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
  end;

  TRouteLayer = TConcatLayer;


implementation
uses nnet;


{ TConcatLayer }

constructor TConcatLayer.Create(const aBatch: SizeInt; const aInputLayers,
  aInputSizes: TArray<SizeInt>; const aGroups: SizeInt; const aGroupId: SizeInt);
var i:SizeInt;
begin
  layerType       := ltCONCAT;
  batch           := Abatch;
  inputLayers     := aInputLayers;
  inputSizes      := aInputSizes;
  setLength(inTensors, length(aInputLayers));
  outputs         := 0;
  for i:=0 to high(inputSizes) do
    inc(outputs, inputSizes[i]);
  groups          := aGroups;
  groupId         := aGroupId;
  outC            := 0;
  outputs         := outputs div groups;
  inputs          := outputs;
  inputShape      := [batch, inputs];
  //delta           := TSingleTensor.Create([batch, output, batch]);
  output          := TSingleTensor.Create([batch, outputs], batch);

end;

procedure TConcatLayer.setTrain(ATrain: boolean);
begin
  if ATrain=FTrain then exit;
  FTrain := ATrain;
  if FTrain then
    if outC=0 then
      delta := TSingleTensor.Create([batch, outputs], batch)
    else
      delta := TSingleTensor.Create([batch, outC, outH, outW], batch)
  else
    delta.free
end;

procedure TConcatLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch=Batch then exit();
  batch := aBatch;
  c:=0;
  inputShape[0] := batch;
  if outC=0 then begin
    output.resize([batch, outputs], batch);
    if FTrain then
      delta.resize([batch, outputs], batch)
  end else begin
    output.resize([batch, outC, outH, outW], batch);
    if FTrain then
      delta.resize([batch, outC, outH, outW], batch)
  end
end;

procedure TConcatLayer.forward(var state: TNNetState);
var
    i:SizeInt;
    //j, offset, index, input_size, part_input_size: SizeInt;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(layerType);
    {$endif}
    //offset := 0;
    //for i := 0 to high(inputLayers) -1 do
    //    begin
    //        index := inputLayers[i];
    //        input := TNNet(state.net).layers[index].output;
    //        input_size := inputSizes[i];
    //        part_input_size := input_size div groups;
    //        for j := 0 to batch -1 do
    //            copy_cpu(part_input_size, input + j * input_size + part_input_size * group_id, 1, output + offset + j * outputs, 1);
    //        offset := offset + part_input_size
    //    end;
    // todo TConcatLayer : implement forward and backward groups and groupId for multi GPU
    for i:=0 to high(inTensors) do
      inTensors[i] := TNNet(state.net).layers[inputLayers[i]].output;
    output.concat(inTensors);
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(layerType);
    {$endif}
end;

procedure TConcatLayer.backward(var state: TNNetState);
var
    i: SizeInt;
    //j: SizeInt;
    //offset: SizeInt;
    //index: SizeInt;
    //delta: PSingle;
    //input_size: SizeInt;
    //part_input_size: SizeInt;
begin
    //offset := 0;
    //for i := 0 to l.n -1 do
    //    begin
    //        index := l.input_layers[i];
    //        delta := state.net.layers[index].delta;
    //        input_size := l.input_sizes[i];
    //        part_input_size := input_size div l.groups;
    //        for j := 0 to l.batch -1 do
    //            axpy_cpu(part_input_size, 1, l.delta+offset+j * l.outputs, 1, delta+j * input_size+part_input_size * l.group_id, 1);
    //        offset := offset + part_input_size
    //    end
  for i:=0 to high(inTensors) do
    inTensors[i] := TNNet(state.net).layers[inputLayers[i]].delta;
  delta.addConcat(inTensors);
end;

end.

