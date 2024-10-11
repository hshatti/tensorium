unit nUpSampleLayer;
{$ifdef FPC}
{$mode Delphi}
{$endif}
interface

uses
  SysUtils, nTensors, nTypes, nBaseLayer;

type

  { TUpSampleLayer }

  TUpSampleLayer = class(TBaseImageLayer)
    reverse : boolean;
    stride : SizeInt;
    scale : single;
    constructor Create(const ABatch, AWidth, AHeight, AChannels: SizeInt;
      AStride: SizeInt; const AScale :Single =1);
    procedure setBatch(ABatch: SizeInt); override;
    procedure setTrain(ATrain: boolean); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
  end;

implementation

{ TUpSampleLayer }

constructor TUpSampleLayer.Create(const ABatch, AWidth, AHeight,
  AChannels: SizeInt; AStride: SizeInt; const AScale: Single);
begin
  layerType := ltUPSAMPLE;
  batch := Abatch;
  w := AWidth;
  h := AHeight;
  c := AChannels;
  outW := AWidth * AStride;
  outH := AHeight * AStride;
  outC := AChannels;
  if AStride < 0 then
      begin
          AStride := -AStride;
          reverse := true;
          outW := AWidth div AStride;
          outH := AHeight div AStride
      end;
  stride := AStride;
  inputShape := [batch, c, h, w];
  outputs := outW * outH * outC;
  inputs := w * h * c;
  scale := AScale;
  //delta := TSingles.Create([batch, c, h, w], batch);
  output := TSingleTensor.Create([batch, outC, outH, outW], batch);
end;

procedure TUpSampleLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch = Batch then exit;
  batch := ABatch;
  output.reSize([batch, c, h, w], batch);
  inputShape[0] := batch;;
  if FTrain then
      Delta.reSize([batch, c,h, w], batch);
end;

procedure TUpSampleLayer.setTrain(ATrain: boolean);
begin
  if FTrain=Atrain then exit;
  FTrain := ATrain;
  if FTrain then
    delta := TSingleTensor.Create([batch, c, h, w], batch)
  else
    delta.free

end;

procedure upsample(const &in: PSingle; const w, h, c, batch,
  stride: SizeInt; const forward: boolean; const scale: single;
  const &out: PSingle);
var i, j, k, b, in_index, out_index:SizeInt;
begin
   for b := 0 to batch-1 do
       for k := 0 to c-1 do
           for j := 0 to h*stride-1 do
               for i := 0 to w*stride-1 do begin
                   in_index := b*w*h*c + k*w*h + (j div stride)*w + i div stride;
                   out_index := b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                   if forward then
                     &out[out_index] := scale*&in[in_index]
                   else
                     &in[in_index] := &in[in_index] + scale*&out[out_index]
               end
end;

procedure TUpSampleLayer.forward(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}
  //fill_cpu(l.outputs * l.batch, 0, l.output, 1);
  output.fill(0);
  if reverse then         // todo [forward_upsample_layer] why not using rverse as a parameter instead of [if else then]
      upsample(output, outW, outH, outC, batch, stride, false, scale, state.input)
  else
      upsample(state.input, w, h, c, batch, stride, true, scale, output);

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TUpSampleLayer.backward(var state: TNNetState);
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(layerType);
  {$endif}
  if reverse then  // todo [backward_upsample] why not passing l.reverse to the function instad of [if then else]
      upsample(delta, outW, outH, outC, batch, stride, true, scale, state.delta)
  else
      upsample(state.delta, w, h, c, batch, stride, false, scale, delta) ;
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(layerType);
  {$endif}
end;

end.

