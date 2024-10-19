unit nBaseLayer;
{$ifdef FPC}
{$mode Delphi}{$H+}
{$endif}
interface

uses
  SysUtils, ntypes, ntensors, nActivation;

type

  { TBaseLayer }

  TBaseLayer = class(TInterfacedObject)
  protected
    procedure setTrain(ATrain: boolean); virtual; abstract;
  public
    FTrain                   : boolean;
    layerType                : TLayerType;
    inputShape               : TArray<SizeInt>;
    output                   : TSingleTensor;
    Batch, Groups,
      inputs, outputs        : SizeInt;
    weights                  : TSingleTensor;
    biases                   : TSingleTensor;

    weight_updates           : TSingleTensor;
    bias_updates             : TSingleTensor;

    delta                    : TSingleTensor;

    //gradient                 : TSingleTensor;
    //activated                : TSingleTensor;
    ActivationType           : TActivationType;
    id                       : SizeInt;
    isBatchNormalized        : boolean;
    backwardStop, forwardOnly: boolean;
    dontLoad, dontLoadScales  : boolean;
    // for batch normalization
    scales                   : TSingleTensor;
    scale_updates            : TSingleTensor;
    mean                     : TSingleTensor;
    mean_delta               : TSingleTensor;
    variance                 : TSingleTensor;
    variance_delta           : TSingleTensor;
    rolling_mean             : TSingleTensor;
    rolling_variance         : TSingleTensor;
    x                        : TSingleTensor;
    x_norm                   : TSingleTensor;

    // for ADAM optimization
    m                        : TSingleTensor;
    v                        : TSingleTensor;
    bias_m                   : TSingleTensor;
    scale_m                  : TSingleTensor;
    bias_v                   : TSingleTensor;
    scale_v                  : TSingleTensor;

    // Exponential Moving Average
    weights_ema, biases_ema  : TSingleTensor;
    scales_ema               : TSingleTensor;

    // cost array must be nil or size of [1]
    cost                     : TArray<Single>;
    index                    : SizeInt;
    net                      : TObject;
    function getWorkspaceSize():SizeInt; virtual;
    procedure Activate;   inline;
    procedure Derivative; virtual;

    function LayerName:string;
    procedure setBatch(ABatch :SizeInt); virtual; abstract;
    procedure freeBatchNorm;
    destructor Destroy();override;
    procedure forward(var state : TNNetState); virtual; abstract;
    procedure backward(var state : TNNetState); virtual; abstract;
    procedure update(const args : TUpdateArgs); virtual;
    procedure fuseBatchNorm; virtual;
    function getWorkspaceShape:TArray<SizeInt>; virtual; abstract;
    property train:boolean read FTrain write setTrain;
    property workspaceSize:SizeInt read getWorkspaceSize;

  end;

  { TBaseImageLayer }

  TBaseImageLayer = class(TBaseLayer)
    step                     : SizeInt;
    w, h, c                  : SizeInt;
    outW, outH, outC         : SizeInt;
    learningRateScale        : Single;
    procedure batchNorm(var state: TNNetState);
    procedure batchNormBack(var state :TNNetState);
    function getImage():TImageData;
    function getDelta():TImageData;
  end;

{$ifdef USE_TELEMETRY}
  { TMetrics }

  TMetrics = record
    type

      { TAct }

      TAct =record
      private
          m:array[0..999] of int64;
          stack: longint;
          function GetItem(i: TActivationType): int64;
      public
          all:array[low(TActivationType)..high(TActivationType)] of int64;
          procedure start(const a:TActivationType);
          procedure finish(const a:TActivationType);
          function total:int64;
          property Item[i:TActivationType]:int64 read GetItem ;default;
      end;

    { TFw }
       TFw = record
       private
          m:array[0..999] of int64;
          stack: longint;
          function GetItem(i: TLayerType): int64;
       public
          all:array[low(TLayerType)..high(TLayerType)] of int64;
          procedure start(const a:TLayerType);
          procedure finish(const a:TLayerType);
          function total():int64;
          property Item[i:TLayerType]:int64 read GetItem ;default;
       end;

    public

      ops: PTensorMetrics;
      act, grad : TAct;
      forward, backward, update:TFw;
      procedure reset;
      function print:string;
  end;
{$endif}

{$ifdef USE_TELEMETRY}
var
  metrics : TMetrics;
{$endif}


implementation
uses typInfo, nChrono;

{ TBaseLayer }

function TBaseLayer.getWorkspaceSize(): SizeInt;
begin
  result :=0;
end;

procedure TBaseLayer.Activate;
begin
  activate_array(Pointer(output.Data), batch * outputs, ActivationType);
end;

procedure TBaseLayer.Derivative;
begin
  gradient_array(pointer(output.Data), batch * outputs, ActivationType, pointer(Delta.Data));
end;

function TBaseLayer.LayerName: string;
begin
  case layerType of
      ltCONVOLUTIONAL:
          exit('convolutional');
      ltACTIVE:
          exit('activation');
      ltLOCAL:
          exit('local');
      ltDECONVOLUTIONAL:
          exit('deconvolutional');
      ltCONNECTED:
          exit('connected');
      ltRNN:
          exit('rnn');
      ltGRU:
          exit('gru');
      ltLSTM:
          exit('lstm');
      ltCRNN:
          exit('crnn');
      ltMAXPOOL:
          exit('maxpool');
      ltREORG:
          exit('reorg');
      ltAVGPOOL:
          exit('avgpool');
      ltSOFTMAX:
          exit('softmax');
      ltDETECTION:
          exit('detection');
      ltREGION:
          exit('region');
      ltYOLO:
          exit('yolo');
      ltGaussianYOLO:
          exit('Gaussian_yolo');
      ltDROPOUT:
          exit('dropout');
      ltCROP:
          exit('crop');
      ltCOST:
          exit('cost');
      ltROUTE:
          exit('route');
      ltSHORTCUT:
          exit('shortcut');
      ltScaleChannels:
          exit('scale_channels');
      ltSAM:
          exit('sam');
      ltNORMALIZATION:
          exit('normalization');
      ltBATCHNORM:
          exit('batchnorm');
      ltUPSAMPLE:
          exit('upsample');
      //else

  end;
  exit('none')
end;

procedure TBaseLayer.freeBatchNorm;
begin
  scales.free;
  scale_updates.free;
  mean.free;
  variance.free;
  mean_delta.free;
  variance_delta.free;
  rolling_mean.free;
  rolling_variance.free;
  x.free;
  x_norm.free
end;

destructor TBaseLayer.Destroy;
begin
  inherited Destroy;
end;

procedure TBaseLayer.update(const args: TUpdateArgs);
begin
  //
end;

procedure TBaseLayer.fuseBatchNorm;
begin

end;

{ TBaseImageLayer }

procedure TBaseImageLayer.batchNorm(var state: TNNetState);
begin
{$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(ltBATCHNORM);
{$endif}

  if LayerType = ltBATCHNORM then
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
  if benchmark then metrics.forward.finish(ltBATCHNORM);
{$endif}
end;

procedure TBaseImageLayer.batchNormBack(var state: TNNetState);
begin
{$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.start(ltBATCHNORM);
{$endif}

  // spatial dot (x_norm . delta) then add to scale_updates
  scale_updates.addDots(x_norm, delta);

  // add scales to all delta batches
  delta.add(scales);
  delta.MeansAndVarsDelta(delta, x, mean, variance, mean_delta, variance_delta);
  delta.normalizeDelta(x, mean, variance, mean_delta, variance_delta, delta);
  if layerType = ltBATCHNORM then
    delta.copyTo(state.delta.Data);

{$ifdef USE_TELEMETRY}
  if benchmark then metrics.backward.finish(ltBATCHNORM);
{$endif}
end;

function TBaseImageLayer.getImage(): TImageData;
begin
  result.h := outH;
  result.w := outW;
  result.c := outC;
  setLength(result.data, result.c * result.h * result.w );
  Move(output.Data[0], result.Data[0], length(result.Data)*SizeOf(Single))
end;

function TBaseImageLayer.getDelta(): TImageData;
begin
  result.h := outH;
  result.w := outW;
  result.c := outC;
  setLength(result.data, result.c * result.h * result.w );
  Move(delta.Data[0], result.Data[0], length(result.Data)*SizeOf(Single))
end;

{$ifdef USE_TELEMETRY}
{ TMetrics }

procedure TMetrics.reset;
begin
  if assigned(ops) then
    fillchar(PAnsiChar(@ops.all)[0], sizeOf(ops.all), #0);
  fillchar(PAnsiChar(@act.all)[0], sizeOf(act.all), #0);
  fillchar(PAnsiChar(@grad.all)[0], sizeOf(grad.all), #0);
  fillchar(PAnsiChar(@forward.all)[0], sizeOf(forward.all), #0);
  fillchar(PAnsiChar(@backward.all)[0], sizeOf(backward.all), #0);

end;

function TMetrics.print: string;
const uSecPerSec=1000000;
var
  i :TMeasureOps;
  j :TActivationType;
  k :TLayerType;
begin
  result := '';
  if not benchmark then exit;
  result:=sLineBreak;
  for i:= low(ops.all) to high(ops.all) do
    if ops.all[i]<>0 then
      result := result + format('%-15s%10.3f[ms]',[copy(GetEnumName(TypeInfo(TMeasureOps),ord(i)),3), ops.all[i]/uSecPerSec] ) + sLineBreak;
  result := result + '----------------------------' + sLineBreak;
  result := result + format('Total          %10.3f[ms]', [ops.total()/uSecPerSec]) + sLineBreak + sLineBreak;

  for j:= low(act.all) to high(act.all) do
    if act.all[j]<>0 then
      result := result + format('%-15s%10.3f[ms]',[copy(GetEnumName(TypeInfo(TActivationType),ord(j)),3), act.all[j]/uSecPerSec] ) + sLineBreak;
  result := result + '----------------------------' + sLineBreak;
  result := result + format('Total          %10.3f[ms]', [act.total/uSecPerSec]) + sLineBreak + sLineBreak;

  for k:= low(forward.all) to high(forward.all) do
    if forward.all[k]<>0 then
      result := result + format('%-15s%10.3f[ms]',[copy(GetEnumName(TypeInfo(TLayerType),ord(k)),3), forward.all[k]/uSecPerSec] ) + sLineBreak;
  result := result + '----------------------------' + sLineBreak;
  result := result + format('Total          %10.3f[ms]', [forward.total/uSecPerSec]) + sLineBreak + sLineBreak;

end;

{ TMetrics.TAct }

function TMetrics.TAct.GetItem(i: TActivationType): int64;
begin
  result := all[i]
end;

procedure TMetrics.TAct.start(const a: TActivationType);
begin
  m[stack]:=clock;
  inc(stack)
  //all[a] := clock();
end;

procedure TMetrics.TAct.finish(const a: TActivationType);
begin
  dec(stack);
  all[a] := all[a] + clock()- m[stack]
end;

function TMetrics.TAct.total: int64;
var
  i: TActivationType;
begin
  result := 0;
  for i:=low(TActivationType) to high(TActivationType) do
    inc(result, all[i])
end;

{ TMetrics.TFw }

function TMetrics.TFw.GetItem(i: TLayerType): int64;
begin
  result := all[i];
end;

procedure TMetrics.TFw.start(const a: TLayerType);
begin
  m[stack]:=clock;
  inc(stack)
end;

procedure TMetrics.TFw.finish(const a: TLayerType);
begin
  dec(stack);
  all[a] := all[a] + clock()- m[stack]
end;

function TMetrics.TFw.total(): int64;
var
  i: TLayerType;
begin
  result := 0;
  for i:=low(TLayerType) to high(TLayerType) do
    inc(result, all[i])
end;
{$endif USE_TELEMETRY}
initialization

{$ifdef USE_TELEMETRY}
  metrics.ops := @tensorMetrics;
{$endif}

end.

