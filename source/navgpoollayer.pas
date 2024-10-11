unit nAvgPoolLayer;
{$ifdef FPC}
{$mode Delphi}
{$endif}

interface

uses
  SysUtils, nTypes, nTensors, nBaseLayer;

type

  { TAvgPoolLayer }

  TAvgPoolLayer = class(TBaseImageLayer)
    constructor Create(const aBatch, aWideth, aHeight, aChannels: SizeInt);
    procedure setBatch(ABatch: SizeInt); override;
    procedure setTrain(ATrain: boolean); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
  end;

implementation

{ TAvgPoolLayer }

constructor TAvgPoolLayer.Create(const aBatch, aWideth, aHeight,
  aChannels: SizeInt);
begin
  layerType := ltAVGPOOL;
  batch := aBatch;
  c := aChannels;
  h := aHeight;
  w := aWideth;
  outW := 1;
  outH := 1;
  outC := c;
  inputShape := [batch, c, h , w];
  outputs := outC;
  inputs := h * w * c;
  output:=TSingleTensor.Create([batch, outputs], batch);
  delta:=TSingleTensor.Create([batch, outputs], batch);
end;

procedure TAvgPoolLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch=Batch then exit();
  batch := ABatch;
  inputShape[0] := batch;
  output.reSize([batch, outputs], batch);
  delta.reSize([batch, outputs], batch);
end;

procedure TAvgPoolLayer.setTrain(ATrain: boolean);
begin
    if ATrain=FTrain then exit();
  FTrain := ATrain
end;

procedure TAvgPoolLayer.forward(var state: TNNetState);
var
    b, i, k, out_index, in_index: SizeInt;
begin
    {$ifdef USE_TELEMeTRY}
    if benchmark then metrics.forward.start(layerType);
    {$endif}
    for b := 0 to batch -1 do
        for k := 0 to c -1 do
            begin
                out_index := k+b * c;
                output.data[out_index] := 0;
                for i := 0 to h * w -1 do
                    begin
                        in_index := i+h * w * (k+b * c);
                        output.data[out_index] := output.data[out_index] + state.input.data[in_index]
                    end;
                output.data[out_index] := output.data[out_index] / (h * w)
            end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(layerType);
    {$endif}
end;

procedure TAvgPoolLayer.backward(var state: TNNetState);
var
    b, i, k, out_index, in_index: SizeInt;
    t:int64;
begin
    for b := 0 to batch -1 do
        for k := 0 to c -1 do
            begin
                out_index := k+b * c;
                for i := 0 to h * w -1 do
                    begin
                        in_index := i+h * w * (k+b * c);
                        state.delta.data[in_index] := state.delta.data[in_index] + (delta.data[out_index] / (h * w))
                    end
            end
end;

end.

