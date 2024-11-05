unit nNormalizationLayer;
{$ifdef FPC}
{$mode Delphi}
{$endif}

interface

uses
  SysUtils, nTensors, nTypes, nBaseLayer;

type

  { TNormaliationLayer }

  TNormaliationLayer = class(TBaseImageLayer)
    Size: SizeInt;
    Alpha, Beta, Kappa : Single;
    norms, squared : TSingleTensor;
    constructor Create(const aBatch, aWidth, aHeight, aChannels, aSize: SizeInt; const AAlpha, ABeta, AKappa: single);
    procedure setBatch(ABatch: SizeInt); override;
    procedure setTrain(ATrain: boolean); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
  end;

implementation

{ TNormaliationLayer }

constructor TNormaliationLayer.Create(const aBatch, aWidth, aHeight, aChannels,
  aSize: SizeInt; const AAlpha, ABeta, AKappa: single);
begin
  layerType := ltNORMALIZATION;
  batch := Abatch;
  h := aHeight;   outH := h;
  w := aWidth;    outW := w;
  c := aChannels; outC := c;
  kappa    := Akappa;
  size     := Asize;
  alpha    := Aalpha;
  beta     := Abeta;
  output   := TSingleTensor.Create([batch, h, w, c, batch], batch);
  squared  := TSingleTensor.Create([batch, h, w, c, batch], batch);
  norms    := TSingleTensor.Create([batch, h, w, c, batch], batch);
  inputs   := w * h * c;
  outputs  := inputs;
end;

procedure TNormaliationLayer.setBatch(ABatch: SizeInt);
begin
  batch := ABatch;
  output   := TSingleTensor.Create([batch, h, w, c, batch], batch);
  squared  := TSingleTensor.Create([batch, h, w, c, batch], batch);
  norms    := TSingleTensor.Create([batch, h, w, c, batch], batch);
  if Train then
    delta    := TSingleTensor.Create([batch, h, w, c, batch], batch)
  else
    delta.free;
end;

procedure TNormaliationLayer.setTrain(ATrain: boolean);
begin
  if ATrain then
    delta    := TSingleTensor.Create([batch, h, w, c, batch], batch)
  else
    delta.free;
  train := ATrain;
end;

procedure TNormaliationLayer.forward(var state: TNNetState);
var prev, next, b, k : SizeInt;
  nrms, sqrd, inpt : PSingle;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}

  squared.fill(0);
  for b := 0 to batch -1 do
      begin
          sqrd := self.squared.data + w*h*c*b;
          nrms := self.norms.data + w*h*c*b;
          inpt := state.input.data + w*h*c*b;
          TSingleTensor.sqrv(w*h*c, inpt, 1, sqrd, 1);
          FillDWord(nrms[0], w*h, longword(self.kappa));
          for k := 0 to self.size div 2 -1 do
              TSingleTensor.axpysvv(w * h, self.alpha, sqrd + w*h*k, 1, nrms, 1);
          for k := 1 to self.c -1 do
              begin
                  //copy_cpu(w * h, nrms+w * h * (k-1), 1, nrms + w*h*k, 1);
                  move((nrms + w*h*(k-1))^, (nrms + w*h*k)^, w*h*sizeOf(single));
                  prev := k - ((self.size-1) div 2)-1;
                  next := k + (self.size div 2);
                  if prev >= 0 then
                      TSingleTensor.axpysvv(w * h, -self.alpha, sqrd+w * h * prev, 1, nrms + w*h*k, 1);
                  if next < self.c then
                      TSingleTensor.axpysvv(w*h, self.alpha, sqrd + w*h*next, 1, nrms + w*h*k, 1)
              end
      end;
  //pow_cpu(w * h * c * self.batch, -self.beta, self.norms, 1, self.output, 1);
  norms.power(-beta, output.data);
  //mul_cpu(w * h * c * self.batch, state.input, 1, self.output, 1);
  output.Multiply(state.input^);
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}
end;

procedure TNormaliationLayer.backward(var state: TNNetState);
begin
  //pow_cpu(w * h * c * layer.batch, -layer.beta, layer.norms, 1, state.delta, 1);
  norms.power( -beta, state.delta.data);
  //mul_cpu(w * h * c * layer.batch, layer.delta, 1, state.delta, 1)
  state.delta.Multiply(delta);
end;

end.

