unit nDropOutLayer;
{$ifdef fpc}
{$mode Delphi}
{$endif}

interface

uses
  SysUtils, nTensors, nTypes, nBaseLayer;

type

  { TDropoutLayer }

  TDropoutLayer = class(TBaseImageLayer)
    probability            : single;
    scale                  : single;
    dropBlock              : boolean;
    dropBlockSizeRel       : single;
    rand                   : TSingleTensor;
    dropBlockSizeAbs       : SizeInt;
    constructor Create(const aBatch, aInputs: SizeInt; const aProbability: single; const aDropblock: boolean; const aDropblock_size_rel: single; const aDropblock_size_abs, aWidth, aHeight, aChannels: SizeInt);
    procedure setBatch(ABatch: SizeInt); override;
    procedure setTrain(ATrain: boolean); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
  end;

implementation

{ TDropoutLayer }

constructor TDropoutLayer.Create(const aBatch, aInputs: SizeInt;
  const aProbability: single; const aDropblock: boolean;
  const aDropblock_size_rel: single; const aDropblock_size_abs, aWidth,
  aHeight, aChannels: SizeInt);
begin
  layerType := ltDROPOUT;
  probability := aProbability;
  dropBlock := aDropblock;
  dropBlockSizeRel := aDropblock_size_rel;
  dropBlockSizeAbs := aDropblock_size_abs;
  if dropblock then
      begin
          w := aWidth;
          outW := w;
          h := aHeight;
          outH := h;
          c := aChannels;
          outC := c;
          if (w <= 0) or (h <= 0) or (c <= 0) then
               raise Exception.Create(format(' Error: DropBlock - there must be positive values for: result.w=%d, result.h=%d, result.c=%d ',[ w, h, c]));
      end;
  inputs := Ainputs;
  outputs := Ainputs;
  batch := Abatch;
  inputShape := [batch, inputs];
  rand := TSingleTensor.Create([batch, inputs], batch);
  scale := 1 / (1.0-probability);
end;

procedure TDropoutLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch=Batch then exit();
  Batch := ABatch;
  inputShape[0] := batch;
  rand.resize([batch, inputs], batch);
end;

procedure TDropoutLayer.setTrain(ATrain: boolean);
begin
  if FTrain = ATrain then exit();
  FTrain := ATrain
end;

procedure TDropoutLayer.forward(var state: TNNetState);
var
    i: SizeInt;
    r: single;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}

    if not state.isTraining then
        exit();
    //rand.UniformDistribution(0,1);
    for i := 0 to batch * inputs -1 do
        begin
            r := random();//rand_uniform(0, 1);
            rand.data[i] := r;
            if r < probability then
            //if rand.Data[i] < probability then
                state.input.Data[i] := 0
            else
                state.input.Data[i] := state.input.Data[i] * scale
        end ;
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TDropoutLayer.backward(var state: TNNetState);
var
    i: SizeInt;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
    if not assigned(state.delta.Data) then
        exit();
    for i := 0 to batch *  inputs -1 do
        begin
            if rand.Data[i] < probability then
                state.delta.Data[i] := 0
            else
                state.delta.Data[i] := state.delta.Data[i] * scale
        end;
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;

end.

