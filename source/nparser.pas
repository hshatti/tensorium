unit nParser;
{$ifdef fpc}
{$mode Delphi}
{$ModeSwitch advancedrecords}
{$ModeSwitch typehelpers}
{$endif}

interface

uses
  SysUtils, TypInfo,
  ntensors, NTypes, nChrono , nConfig
  , nnet
  , nActivation
  , nBaseLayer
  , nAddLayer
  , nConnectedlayer
  , nConvolutionLayer
  , nDropOutLayer
  , nAvgPoolLayer
  , nCostLayer
  , nSoftmaxLayer
  , nLogisticLayer
  , nConcatLayer
  , nMaxPoolLayer
  , nContrastiveLayer
  , nBatchNormLayer
  , nUpSampleLayer
  , nYoloLayer

  ;

type
  TSizeParams = record
      batch, inputs, h, w, c, index, timeSteps :SizeInt;
      train:boolean;
      net : TNNet;
  end;

  { TDarknetParser }

  TDarknetParser=class

    CFG : TCFGList;
    params : TSizeParams;
    Neural : TNNet;
  protected
    function parseConnected(const opt:TCFGSection):TConnectedLayer;
    function parseConvolutional(const opt:TCFGSection):TConvolutionalLayer;
    function parseMaxPool(const opt:TCFGSection):TMaxPoolLayer;
    function parseLocalAvgPool(const opt:TCFGSection):TLocalAvgPoolLayer;
    function parseAvgPool(const opt:TCFGSection):TAvgPoolLayer;
    function parseDropOut(const opt:TCFGSection):TDropOutLayer;
    function parseConcat(const opt:TCFGSection):TConcatLayer;
    function parseAdd(const opt:TCFGSection):TAddLayer;
    function parseUpSample(const opt:TCFGSection):TUpSampleLayer;
    function parseBatchNorm(const opt:TCFGSection):TBatchNormLayer;
    function parseYolo(const opt:TCFGSection):TYoloLayer;
    function parseLogistic(const opt:TCFGSection):TLogisticLayer;
    function parseSoftmax(const opt:TCFGSection):TSoftmaxLayer;
    function parseCost(const opt:TCFGSection):TCostLayer;
    function parseContrastive(const opt:TCFGSection):TContrastiveLayer;
    procedure parseNet(const options:TCFGSection);

    procedure loadConnectedWeights(var l: TConnectedLayer; var fp: file; const transpose: boolean);
    procedure loadBatchNormWeights(var l: TBatchNormLayer; var fp: file);
    procedure loadConvolutionalWeights(var l: TConvolutionalLayer; var fp: file);
    procedure loadAddWeights(var l: TAddLayer; var fp: file);


  public
    constructor Create(const filename:string; const ABatch:SizeInt=0; const ATimeSteps:SizeInt = 0);
    destructor Destroy; override;
    procedure loadWeights(const filename:string; cutoff:SizeInt=0);

  end;

  { TLearningRatePolicyHelper }

  TLearningRatePolicyHelper = type helper for TLearningRatePolicy
    class function fromString(s:string):TLearningRatePolicy;static;
    function toSring():string;
  end;

  { TLayerTypeHelper }

  TLayerTypeHelper= type helper for TLayerType
    class function fromString(s:string):TLayerType; static;
    function toString():string;
  end;

  { TActivationTypeHelper }

  TActivationTypeHelper=type helper for TActivationType
    class function fromString(s:string):TActivationType; static;
    function toString():string;
  end;

  function fromFile(const filename: string):TStringArray;
  function stringsToInts(const strs:TStringArray):TArray<SizeInt>;

implementation
uses math;

function fromFile(const filename: string):TStringArray;
var f:TextFile;
begin
  result:=nil;
  if not FileExists(filename) then
    raise  EFileNotFoundException.CreateFmt('File [%s] not found',[filename]);
  AssignFile(f,filename);
  reset(f);
  while not EOF(f) do begin
    setLength(result, length(result)+1);
    readln(f, result[high(result)])
  end;
  CloseFile(F);
end;

function stringsToInts(const strs: TStringArray): TArray<SizeInt>;
var i : SizeInt;
begin
  setLength(result, Length(strs));
  for i:= 0 to High(strs) do
    {$ifdef CPU64}
    result[i] := StrToInt64(trim(strs[i]))
    {$else}
    result[i] := StrToInt(trim(strs[i]))
    {$endif};
end;

{ TDarknetParser }

function TDarknetParser.parseConnected(const opt: TCFGSection): TConnectedLayer;
begin
  result := TConnectedLayer.Create(params.batch, params.inputs,
         opt.getInt('output', 1),
         TActivationType.fromString(opt.getStr('activation', 'relu')),
         opt.getBool('batch_normalize', false){, params.net.adam});
end;

function TDarknetParser.parseConvolutional(const opt: TCFGSection): TConvolutionalLayer;
var
  stride, stride_x, n, groups, size, stride_y, dilation, antialiasing,
    padding, assisted_excitation, share_index, h, w, c, batch,
    batch_normalize, sway, rotate, stretch, stretch_sway: SizeInt;
  pad, cbn
    //, binary, xnor, use_bin_output
    , deform: Boolean;
  activation_s: String;
  activation: TActivationType;
  share_layer: TConvolutionalLayer;
begin
  n           := opt.getInt( 'filters', 1);
  groups      := opt.getInt('groups', 1, true);
  size        := opt.getInt( 'size', 1);
  stride      := -1;
  stride_x    := opt.getInt('stride_x', -1, true);
  stride_y    := opt.getInt('stride_y', -1, true);
  if (stride_x < 1) or (stride_y < 1) then
      begin
          stride := opt.getInt( 'stride', 1);
          if stride_x < 1 then
              stride_x := stride;
          if stride_y < 1 then
              stride_y := stride
      end
  else
      stride    := opt.getInt('stride', 1, true);
  dilation      := opt.getInt('dilation', 1, true);
  antialiasing  := opt.getInt('antialiasing', 0, true);
  if size = 1 then
      dilation := 1;
  pad           := opt.getBool( 'pad', false, true);
  padding       := opt.getInt('padding', 0, true);
  if pad then
      padding   := size div 2;
  activation_s  := opt.getStr( 'activation', 'logistic');
  activation    := TActivationType.fromString(activation_s);
  assisted_excitation := opt.getInt('assisted_excitation', 0, true);
  share_index         := opt.getInt('share_index', -1000000000, true);
  share_layer         := nil;
  if share_index >= 0 then
      share_layer   :=  TConvolutionalLayer(params.net.layers[share_index])
  else
      if share_index <> -1000000000 then
          share_layer := TConvolutionalLayer(params.net.layers[params.index+share_index]);
  h := params.h;
  w := params.w;
  c := params.c;
  batch := params.batch;
  if not ((h<>0) and (w<>0) and (c<>0)) then
      raise Exception.Create('Layer before convolutional layer must output image.');
      //error('Layer before convolutional layer must output image.', DARKNET_LOC);
  batch_normalize := opt.getInt('batch_normalize', 0, true);
  cbn := opt.getBool( 'cbn', false, true);
  if cbn then
      batch_normalize := 2;
  //binary         := opt.getBool('binary', false, true);
  //xnor           := opt.getBool('xnor', false, true);
  //use_bin_output := opt.getBool('bin_output', false, true);
  sway           := opt.getInt('sway', 0, true);
  rotate         := opt.getInt('rotate', 0, true);
  stretch        := opt.getInt('stretch', 0, true);
  stretch_sway   := opt.getInt('stretch_sway', 0, true);
  if (sway+rotate+stretch+stretch_sway) > 1 then
      raise Exception.Create('Error: should be used only 1 param: sway=1, rotate=1 or stretch=1 in the [convolutional] layer');
  deform := (sway<>0) or (rotate<>0) or (stretch<>0) or (stretch_sway<>0);
  if deform and (size = 1) then
      raise Exception.Create('Error: params (sway=1, rotate=1 or stretch=1) should be used only with size >=3 in the [convolutional] layer');
  result :=TConvolutionalLayer.Create(batch, h, w, c, n, groups,
          size, stride_x, stride_y, dilation, padding, activation,
          batch_normalize in [1,2], false{params.net.adam}, params.index,
          antialiasing, share_layer, assisted_excitation, deform, params.train);
  //result.flipped        := opt.getBool( 'flipped', false, true);
  //result.dot            := opt.getFloat( 'dot', 0, true);
  //result.sway           := sway;
  //result.rotate         := rotate;
  //result.stretch        := stretch;
  //result.stretch_sway   := stretch_sway;
  //result.angle          := opt.getFloat( 'angle', 15, true);
  //result.grad_centr     := opt.getInt('grad_centr', 0, true);
  //result.reverse        := opt.getBool( 'reverse', false, true);
  //result.coordconv      := opt.getInt('coordconv', 0, true);
  //result.stream         := opt.getInt('stream', -1, true);
  result.waitStreamId   := opt.getInt('wait_stream', -1, true);
  //if params.net.adam then
  //    begin
  //        layer.B1 := params.net.B1;
  //        layer.B2 := params.net.B2;
  //        layer.eps := params.net.eps
  //    end;

end;

function TDarknetParser.parseMaxPool(const opt: TCFGSection): TMaxPoolLayer;
var
  stride, stride_x, stride_y, size, padding, maxpool_depth,
    out_channels, antialiasing, h, w, c, batch: SizeInt;
begin
  stride := opt.getInt( 'stride', 1);
  stride_x := opt.getInt('stride_x', stride, true);
  stride_y := opt.getInt('stride_y', stride, true);
  size := opt.getInt( 'size', stride);
  padding := opt.getInt('padding', size-1, true);
  maxpool_depth := opt.getInt('maxpool_depth', 0, true);
  out_channels := opt.getInt('out_channels', 1, true);
  antialiasing := opt.getInt('antialiasing', 0, true);
  h := params.h;
  w := params.w;
  c := params.c;
  batch := params.batch;
  if (h * w * c=0) then
      raise Exception.Create('Layer before [maxpool] layer must output image.');
  result := TMaxPoolLayer.Create(batch, h, w, c, size, stride_x, stride_y, padding, maxpool_depth, out_channels, antialiasing, false, params.train);
  //Result.maxpool_zero_nonmax := opt.getInt('maxpool_zero_nonmax', 0, true);
end;

function TDarknetParser.parseLocalAvgPool(const opt: TCFGSection): TLocalAvgPoolLayer;
var
  stride, stride_x, stride_y, size, padding, maxpool_depth, out_channels, antialiasing,
    h, w, c, batch: SizeInt;
begin
  stride := opt.getInt( 'stride', 1);
  stride_x := opt.getInt('stride_x', stride, true);
  stride_y := opt.getInt('stride_y', stride, true);
  size := opt.getInt( 'size', stride);
  padding := opt.getInt('padding', size-1, true);
  maxpool_depth := 0;
  out_channels := 1;
  antialiasing := 0;
  h := params.h;
  w := params.w;
  c := params.c;
  batch := params.batch;
  if not ((h<>0) and (w<>0) and (c<>0)) then
      raise Exception.Create('Layer before [local_avgpool] layer must output image.');
  result := TLocalAvgPoolLayer.Create(batch, h, w, c, size, stride_x, stride_y, padding, maxpool_depth, out_channels, antialiasing, true, params.train);

end;

function TDarknetParser.parseAvgPool(const opt: TCFGSection): TAvgPoolLayer;
begin
  if (params.h * params.w * params.c)=0 then
      raise Exception.Create('Layer before avgpool layer must output image.');
  result := TAvgPoolLayer.Create(params.batch, params.w, params.h, params.c);
end;

function TDarknetParser.parseDropOut(const opt: TCFGSection): TDropOutLayer;
var
  probability, dropblock_size_rel: Single;
  dropblock: Boolean;
  dropblock_size_abs: SizeInt;
begin
  probability := opt.getFloat( 'probability', 0.2);
  dropblock := opt.getBool( 'dropblock', false, true);
  dropblock_size_rel := opt.getFloat( 'dropblock_size_rel', 0, true);
  dropblock_size_abs := opt.getInt('dropblock_size_abs', 0, true);
  if (dropblock_size_abs > params.w) or (dropblock_size_abs > params.h) then
      begin
          writeln(' [dropout] - dropblock_size_abs = %d that is bigger than layer size %d x %d ', dropblock_size_abs, params.w, params.h);
          dropblock_size_abs := min(params.w, params.h)
      end;
  if dropblock and not (dropblock_size_rel<>0) and not (dropblock_size_abs<>0) then
      begin
          writeln(' [dropout] - None of the parameters (dropblock_size_rel or dropblock_size_abs) are set, will be used: dropblock_size_abs = 7 ');
          dropblock_size_abs := 7
      end;
  if (dropblock_size_rel<>0) and (dropblock_size_abs<>0) then
      begin
          writeln(' [dropout] - Both parameters are set, only the parameter will be used: dropblock_size_abs = %d ', dropblock_size_abs);
          dropblock_size_rel := 0
      end;
  result := TDropoutLayer.Create(params.batch, params.inputs, probability, dropblock, dropblock_size_rel, dropblock_size_abs, params.w, params.h, params.c);
  result.outW := params.w;
  result.outH := params.h;
  result.outC := params.c;

end;

function TDarknetParser.parseConcat(const opt: TCFGSection): TConcatLayer;
var
  l: String;
  layers : TArray<SizeInt>;
  Sizes : TArray<SizeInt>;
  n , batch, groups, group_id, i, index: SizeInt;
  vals: TStringArray;
  first, next: TBaseImageLayer;
begin
  l := opt.getStr('layers','');
  if l='' then
      raise Exception.Create('Route Layer must specify input layers');
  vals := l.split([',']);
  n:= length(vals);
  setLength(layers, n);
  setLength(sizes, n);
  for i := 0 to n -1 do
      begin
          index := StrToInt(vals[i]);
          if index < 0 then
              index := params.index+index;
          layers[i] := index;
          sizes[i] := params.net.layers[index].outputs
      end;
  batch := params.batch;
  // todo [parser concat] immplement groups and grou_id for multi GPU processing
  groups := opt.getInt('groups', 1, true);
  group_id := opt.getInt('group_id', 0, true);
  result := TConcatLayer.Create(batch, layers, sizes, groups, group_id);
  if not params.net.layers[layers[0]].InheritsFrom(TBaseImageLayer) then exit;
  first := TBaseImageLayer(params.net.layers[layers[0]]);
  result.outw := first.outw;
  result.outh := first.outh;
  result.outc := first.outc;
  for i := 1 to high(layers) do
      begin
          index := layers[i];
          next := TBaseImageLayer(params.net.layers[index]);
          if (next.outw = first.outw) and (next.outh = first.outh) then
              result.outc := result.outc + next.outc
          else
              begin
                  writeln(ErrOutput, ' The width and height of the input layers are different.');
                  result.outH := 0;
                  result.outW := 0;
                  result.outC := 0
              end
      end;
  result.outc := result.outc div result.groups;
  result.w := first.w;
  result.h := first.h;
  result.c := result.outc;
  result.output.reshape([batch, result.outC, result.outH, result.outW], batch);
  //result.stream := opt.getInt('stream', -1, true);
  //result.wait_stream_id := opt.getInt('wait_stream', -1, true);
  //if n > 3 then
  //    write(ErrOutput, ' '#9'    ')
  //else
  //    if n > 1 then
  //        write(ErrOutput, ' '#9'            ')
  //else
  //    write(ErrOutput, ' '#9#9'            ');
  //write(ErrOutput, '           ');

  //if result.groups > 1 then
  //    write(ErrOutput, format('%d/%d', [result.group_id, result.groups]))
  //else
  //    write(ErrOutput, '   ');
  //writeln(ErrOutput, format(' -> %4d x%4d x%4d ', [result.outw, result.outh, result.outc]));

end;

function TDarknetParser.parseAdd(const opt: TCFGSection): TAddLayer;
var
  activation_s, weights_type_str, weights_normalization_str, l: String;
  weights_type: TWeightsType;
  weights_normalization: TWeightsNormalization;
  len, n, index: SizeInt;
  layers, sizes : TArray<SizeInt>;
  sIndex: TStringArray;
  activation: TActivationType;
  layers_output, layers_delta :TArray<TSingleTensor>;
  baseImageLayer : TBaseImageLayer;
  i : SizeInt;
begin
  activation_s := opt.getStr( 'activation', 'linear');
  activation := TActivationType.fromString(activation_s);
  weights_type_str := opt.getStr( 'weights_type', 'none', true);
  weights_type := wtNO_WEIGHTS;
  if (weights_type_str = 'per_feature') or (weights_type_str = 'per_layer') then
      weights_type := wtPER_FEATURE
  else
      if weights_type_str = 'per_channel' then
          weights_type := wtPER_CHANNEL
  else
      if (weights_type_str <> 'none') then
          begin
              writeln(format('Error: Incorrect weights_type = %s '#10' Use one of: none, per_feature, per_channel ', [weights_type_str]));
              raise Exception.Create('Error!')
          end;
  weights_normalization_str := opt.getStr( 'weights_normalization', 'none', true);
  weights_normalization := wnNO_NORMALIZATION;
  if (weights_normalization_str = 'relu') or (weights_normalization_str = 'avg_relu') then
      weights_normalization := wnRELU_NORMALIZATION
  else
      if weights_normalization_str = 'softmax' then
          weights_normalization := wnSOFTMAX_NORMALIZATION
  else
      if weights_type_str <> 'none' then
          begin
              writeln(format('Error: Incorrect weights_normalization = %s '#10' Use one of: none, relu, softmax ',[ weights_normalization_str]));
              raise Exception.Create('Error!')
          end;
  l := trim(opt.getStr('from',''));
  len := length(l);
  if l='' then
      raise Exception.Create('Route Layer must specify input layers: from = ...');
  sIndex:=l.split([',']);
  n := length(sIndex);
  setLength(layers, n);
  setLength(sizes, n);
  //layers_output := AllocMem(n * sizeOf(PSingle));
  //layers_delta := AllocMem(n * sizeOf(PSingle));
  setLength(layers_output ,n);
  setLength(layers_delta ,n);
  //setLength(layers_output_gpu , n);
  //setLength(layers_delta_gpu , n);

  for i := 0 to n -1 do
      begin
          index := StrToInt(sIndex[i]);
          if index < 0 then
              index := params.index+index;
          layers[i] := index;
          sizes[i] := params.net.layers[index].outputs;
          layers_output[i] := params.net.layers[index].output;
          layers_delta[i] := params.net.layers[index].delta
      end;
{$ifdef GPU}
  for i := 0 to n -1 do
      begin
          layers_output_gpu[i] := params.net.layers[layers[i]].output_gpu;
          layers_delta_gpu[i] := params.net.layers[layers[i]].delta_gpu
      end;
{$endif}
  result := TAddLayer.Create(params.batch, layers, sizes, params.w, params.h, params.c, layers_output, layers_delta, weights_type, weights_normalization, activation, params.train);
  //free(layers_output_gpu);
  //free(layers_delta_gpu);
  for i := 0 to high(layers) do
      begin
          index := layers[i];
          if Neural.layers[index].InheritsFrom(TBaseImageLayer) then begin
            baseImageLayer :=  Neural.layers[index] as TBaseImageLayer;
            assert((params.w =baseImageLayer.outW) and (params.h = baseImageLayer.outH), '[parser addLayer] : source layer [width] and [hight] doesn''t match the layer output tensor!');

          end;
          //if (params.w <> net.layers[index].out_w) or (params.h <> net.layers[index].out_h) or (params.c <> net.layers[index].out_c) then
          //    writeln(ErrOutput, format(' (%4d x%4d x%4d) + (%4d x%4d x%4d) ', [params.w, params.h, params.c, net.layers[index].out_w, net.layers[index].out_h, params.net.layers[index].out_c]))
      end;

end;

function TDarknetParser.parseUpSample(const opt: TCFGSection): TUpSampleLayer;
begin
  result := TUpSampleLayer.Create(params.batch, params.w, params.h, params.c, opt.getInt( 'stride', 2), opt.getFloat( 'scale', 1, true));
end;

function TDarknetParser.parseBatchNorm(const opt: TCFGSection): TBatchNormLayer;
begin
  result := TBatchNormLayer.Create(params.batch, params.w, params.h, params.c, params.train);
end;

function TDarknetParser.parseYolo(const opt: TCFGSection): TYoloLayer;
var
  classes, total, num, max_boxes, maxCounter, i, embedding_layer_id: SizeInt;
  mask, counters : TArray<SizeInt>;
  a, cpc, iou_loss, iou_thresh_kind_str, nms_kind, map_file: String;
  le: TBaseConvolutionalLayer;
  vals: TStringArray;
begin
  classes := opt.getInt( 'classes', 20);
  total := opt.getInt( 'num', 1);
  a := opt.getStr( 'mask', '');
  mask := stringsToInts(a.split([',']));
  num := length(mask);
  max_boxes := opt.getInt('max', 200, true);
  result := TYoloLayer.Create(params.batch, params.w, params.h, num, total, mask, classes, max_boxes);
  if result.outputs <> params.inputs then
      raise Exception.CreateFmt('Error: result.outputs == params.inputs, filters= in the [convolutional]-layer doesn''t correspond to classes= or mask= in [yolo]-layer-%d',[params.index]);
  result.showDetails := opt.getBool('show_details', true, true);
  result.maxDelta := opt.getFloat( 'max_delta', MaxSingle, true);

  cpc := trim(opt.getStr( 'counters_per_class', ''));
  if cpc <>'' then begin
    counters := stringsToInts(cpc.split([',']));
    if length(counters) <> classes then
      raise Exception.Create(format(' number of values in counters_per_class = %d doesn''t match with classes = %d ', [length(counters), classes]));
    if length(counters)>0 then begin
      maxCounter := counters[0];
      for i:=1 to high(counters) do
        if counters[i]>maxCounter then
          maxCounter := counters[i];
    end;
  end;

  result.classesMultipliers := TSingleTensor.Create([length(counters)]);
  for i:= 0 to Result.classesMultipliers.size()-1 do
    result.classesMultipliers.Data[i] := maxCounter / counters[i];

  result.labelSmoothEps := opt.getFloat( 'label_smooth_eps', 0.0, true);
  result.scaleXY := opt.getFloat( 'scale_x_y', 1, true);
  result.objectnessSmooth := opt.getBool( 'objectness_smooth', false, true);
  result.newCoords := opt.getBool( 'new_coords', false, true);
  result.iouNormalizer := opt.getFloat( 'iou_normalizer', 0.75, true);
  result.objNormalizer := opt.getFloat( 'obj_normalizer', 1, true);
  result.classNormalizer := opt.getFloat( 'cls_normalizer', 1, true);
  result.deltaNormalizer := opt.getFloat( 'delta_normalizer', 1, true);
  iou_loss := opt.getStr( 'iou_loss', 'mse', true);
  if iou_loss= 'mse' then
      result.iouLoss := ilMSE
  else
      if iou_loss= 'giou' then
          result.iouLoss := ilGIOU
  else
      if iou_loss= 'diou' then
          result.iouLoss := ilDIOU
  else
      if iou_loss= 'ciou' then
          result.iouLoss := ilCIOU
  else
      result.iouLoss := ilIOU;
  //writeln(ErrOutput, format('[yolo] params: iou loss: %s (%d), iou_norm: %2.2f, obj_norm: %2.2f, cls_norm: %2.2f, delta_norm: %2.2f, scale_x_y: %2.2f', [iou_loss, ord(result.iouLoss), result.iouNormalizer, result.objNormalizer, result.classNormalizer, result.deltaNormalizer, result.scaleXY]));
  iou_thresh_kind_str := opt.getStr( 'iou_thresh_kind', 'iou', true);
  if iou_thresh_kind_str= 'iou' then
      result.iouThreshKind :=ilIOU
  else
      if iou_thresh_kind_str= 'giou' then
          result.iouThreshKind := ilGIOU
  else
      if iou_thresh_kind_str= 'diou' then
          result.iouThreshKind := ilDIOU
  else
      if iou_thresh_kind_str= 'ciou' then
          result.iouThreshKind := ilCIOU
  else
      begin
          writeln(ErrOutput, format(' Wrong iou_thresh_kind = %s ', [iou_thresh_kind_str]));
          result.iouThreshKind := ilIOU
      end;
  result.betaNMS := opt.getFloat( 'beta_nms', 0.6, true);
  nms_kind := opt.getStr( 'nms_kind', 'default', true);
  if nms_kind = 'default' then
      result.NMSKind := nmsDEFAULT_NMS
  else
      begin
          if nms_kind = 'greedynms' then
              result.NMSKind := nmsGREEDY_NMS
          else
              if nms_kind = 'diounms' then
                  result.NMSKind := nmsDIOU_NMS
          else
              result.nmsKind := nmsDEFAULT_NMS;
          writeln(format('nms_kind: %s (%d), beta = %f ', [nms_kind, ord(result.NMSKind), result.betaNMS]))
      end;
  result.jitter := opt.getFloat( 'jitter', 0.2);
  result.resize := opt.getFloat( 'resize', 1.0, true);
  result.focalLoss := opt.getBool( 'focal_loss', false, true);
  result.ignoreThresh := opt.getFloat( 'ignore_thresh', 0.5);
  result.truthThresh := opt.getFloat( 'truth_thresh', 1);
  result.iouThresh := opt.getFloat( 'iou_thresh', 1, true);
  result.random := opt.getBool( 'random', false, true);
  result.trackHistorySize := opt.getInt('track_history_size', 5, true);
  result.simThresh := opt.getFloat( 'sim_thresh', 0.8, true);
  result.detsForTrack := opt.getInt('dets_for_track', 1, true);
  result.detsForShow := opt.getInt('dets_for_show', 1, true);
  result.trackCIOUNorm := opt.getFloat( 'track_ciou_norm', 0.01, true);
  embedding_layer_id := opt.getInt('embedding_layer', 999999, true);
  if embedding_layer_id < 0 then
      embedding_layer_id := params.index + embedding_layer_id;
  if (embedding_layer_id <> 999999) and (Neural.layers[embedding_layer_id].InheritsFrom(TBaseConvolutionalLayer)) then
      begin
          write(format(' embedding_layer_id = %d, ', [embedding_layer_id]));
          le := Neural.layers[embedding_layer_id] as TBaseConvolutionalLayer;
          result.embeddingLayerId := embedding_layer_id;
          result.embeddingOutput := TSingleTensor.Create([le.batch , le.outputs], le.Batch);
          result.embeddingSize := le.n div result.n;
          writeln(format(' embedding_size = %d ', [result.embeddingSize]));
          if le.n mod result.n <> 0 then
              writeln(format(' Warning: filters=%d number in embedding_layer=%d isn''t divisable by number of anchors %d ', [le.n, embedding_layer_id, result.n]))
      end;
  map_file := opt.getStr( 'map', '');
  if map_file<>'' then
      result.map := stringsToInts(fromFile(map_file));
  a := opt.getStr('anchors', '');
  if a<>'' then
      begin
         vals := a.Split([',']);
         for i:= 0 to min(length(vals), total * 2)- 1 do
              result.biases.data[i] := StrToFloat(trim(vals[i]));
      end;
end;

function TDarknetParser.parseLogistic(const opt: TCFGSection): TLogisticLayer;
begin
  result := TLogisticLayer.Create(params.batch, params.inputs);
  //result.h := params.h; result.outH := params.h;
  //result.w := params.w; result.outW := params.w;
  //result.c := params.c; result.outC := params.c;
end;

function TDarknetParser.parseSoftmax(const opt: TCFGSection): TSoftmaxLayer;
var
  groups: SizeInt;
  tree_file: String;
begin
  groups := opt.getInt('groups', 1, true);
  result :=TSoftmaxLayer.Create(params.batch, params.inputs, groups);
  result.temperature := opt.getFloat( 'temperature', 1, true);
  tree_file := opt.getStr( 'tree', '');
  if tree_file<>'' then
      result.softmaxTree := [TTree.loadFromFile(tree_file)];
  //result.w := params.w;
  //result.h := params.h;
  //result.c := params.c;
  //result.spatial := trunc(opt.getFloat( 'spatial', 0, true));
  result.noloss := opt.getBool( 'noloss', false, true);
end;

function TDarknetParser.parseCost(const opt: TCFGSection): TCostLayer;
var
    type_s: string;
    costType: TCostType;
    scale: single;
    //layer: cost_layer;
begin
    type_s := opt.getStr( 'type', 'sse');
    costType := TCostType.fromString(type_s);
    scale := opt.getFloat( 'scale', 1, true);
    result := TCostLayer.Create(params.batch, params.inputs, costType, scale);
    //result.ratio := opt.getFloat( 'ratio', 0, true);
    //result.noobject_scale := options.getFloat( 'noobj', 1, true);
    //result.thresh := options.getFloat( 'thresh', 0, true);
    //exit(layer)
end;

function TDarknetParser.parseContrastive(const opt: TCFGSection): TContrastiveLayer;
var
  classes, yolo_layer_id: SizeInt;
  yolo_layer: TYoloLayer;
begin
  classes := opt.getInt( 'classes', 1000);
  yolo_layer := nil;
  yolo_layer_id := opt.getInt('yolo_layer', 0, true);
  if yolo_layer_id < 0 then
      yolo_layer_id := params.index+yolo_layer_id;
  if yolo_layer_id <> 0 then
      yolo_layer := TYoloLayer(params.net.layers[yolo_layer_id]);
  if yolo_layer.layerType <> ltYOLO then
      begin
          writeln(format(' Error: [contrastive] layer should point to the [yolo] layer instead of %d layer! ', [yolo_layer_id]));
          Exception.Create('Error!')
      end;
  result := TContrastiveLayer.Create(params.batch, params.w, params.h, params.c, classes, params.inputs, Pointer(yolo_layer));
  result.temperature := opt.getFloat( 'temperature', 1, true);
  result.steps := params.timeSteps;
  result.classNormalizer := opt.getFloat( 'cls_normalizer', 1, true);
  result.maxDelta := opt.getFloat( 'max_delta', MaxSingle, true);
  result.contrastive_neg_max := opt.getInt('contrastive_neg_max', 3, true);
end;

constructor TDarknetParser.Create(const filename: string;
  const ABatch: SizeInt; const ATimeSteps: SizeInt);
var
  lt : TLayerType;
  layers : TArray<TBaseLayer>;
  baseImageLayer : TBaseImageLayer;
  baseConvLayer : TBaseConvolutionalLayer;
  i, count : SizeInt;

begin
  CFG.loadFromFile(filename);
  Neural := TNNet.Create([]);
  if CFG.Count()>0 then
    if (CFG.Sections[0].typeName='net') or (CFG.Sections[0].typeName='network') then
      parseNet(CFG.Sections[0]) // parses net, also sets params.net to Neural;
    else
      raise Exception.Create('1st section in config file must be a [net] parameters.');
  if ABatch>0 then
    neural.batch:= ABatch;
  if ATimeSteps>0 then
    neural.batch:= ATimeSteps;
  if neural.batch < neural.timeSteps then
    neural.batch := neural.timeSteps;

  params.batch:=neural.batch;
  params.timeSteps:=neural.timeSteps;

  if params.w*params.h*params.c<>0 then
    Neural.input.reSize([Neural.batch, params.c, params.h, params.w], Neural.batch)
  else
    Neural.input.resize([Neural.batch, CFG.Sections[0].getInt('inputs',0, true)], Neural.batch);

  params.inputs:=Neural.input.groupSize();

  setLength(layers, CFG.count -1);
  Neural.layers := layers;
  count := 0;
  for i:=1 to CFG.Count()-1 do begin
    params.index := count;
    lt := TLayerType.fromString(CFG.Sections[i].typeName);
    case lt of
      ltCONVOLUTIONAL  :
        begin
          layers[count] := parseConvolutional(CFG.Sections[i]);
        end;
      ltCONNECTED  :
        begin
          layers[count] := parseConnected(CFG.Sections[i]);
        end;
      ltMAXPOOL  :
        begin
          layers[count] := parseMaxPool(CFG.Sections[i]);
        end;
      ltLOCAL_AVGPOOL  :
        begin
          layers[count] := parseLocalAvgPool(CFG.Sections[i]);
        end;
      ltSOFTMAX  :
        begin
          layers[count] := parseSoftmax(CFG.Sections[i]);
        end;
      ltLOGXENT  :
        begin
          layers[count] := parseLogistic(CFG.Sections[i]);
        end;
      ltDROPOUT  :
        begin
          layers[count] := parseDropOut(CFG.Sections[i]);
        end;
      ltROUTE  :
        begin
          layers[count] := parseConcat(CFG.Sections[i]);
        end;
      ltCOST  :
        begin
          layers[count] := parseCost(CFG.Sections[i]);
        end;
      ltAVGPOOL  :
        begin
          layers[count] := parseAvgPool(CFG.Sections[i]);
        end;
      ltSHORTCUT  :
        begin
          layers[count] := parseAdd(CFG.Sections[i]);
        end;
      ltBATCHNORM  :
        begin
          layers[count] := parseBatchNorm(CFG.Sections[i]);
        end;
      ltYOLO  :
        begin
          layers[count] := parseYolo(CFG.Sections[i]);
        end;
      ltUPSAMPLE  :
        begin
          layers[count] := parseUpSample(CFG.Sections[i]);
        end;
      ltCONTRASTIVE :
        begin
          layers[count] := parseContrastive(CFG.Sections[i]);
        end;
      else
        raise Exception.CreateFmt('[Parser][%s] layer is not yet implemented!', [lt.toString()]);
    end;
    //layers[count].clip := CFG.Sections[i].getFloat( 'clip', 0, true);
    //layers[count].dynamic_minibatch := net.dynamic_minibatch;
    layers[count].forwardOnly := CFG.Sections[i].getBool( 'onlyforward', false, true);
    //layers[count].dont_update := CFG.Sections[i].getBool( 'dont_update', false, true);
    //layers[count].burnin_update := CFG.Sections[i].getBool( 'burnin_update', false, true);
    layers[count].backwardStop := CFG.Sections[i].getBool( 'stopbackward', false, true);
    //layers[count].train_only_bn := CFG.Sections[i].getBool( 'train_only_bn', false, true);
    layers[count].dontLoad := CFG.Sections[i].getBool( 'dontload', false, true);
    layers[count].dontLoadScales := CFG.Sections[i].getBool( 'dontloadscales', false, true);
    //layers[count].learning_rate_scale := CFG.Sections[i].getFloat( 'learning_rate', 1, true);

    params.inputs := layers[count].outputs;
    if layers[count].InheritsFrom(TBaseImageLayer) then begin
      baseImageLayer  := layers[count] as TBaseImageLayer;
      params.h := baseImageLayer.outH;
      params.w := baseImageLayer.outW;
      params.c := baseImageLayer.outC;
    end;
    if layers[count].InheritsFrom(TBaseConvolutionalLayer) then begin
      baseConvLayer := layers[count] as TBaseConvolutionalLayer;
      if baseConvLayer.antialiasing<>0 then
        begin
            params.h := baseConvLayer.inputLayer.outH;
            params.w := baseConvLayer.inputLayer.outW;
            params.c := baseConvLayer.inputLayer.outC;
            params.inputs := baseConvLayer.inputLayer.outputs
        end
    end;

    inc(count);
  end;
  Neural.setLayers(layers);
end;

destructor TDarknetParser.Destroy;
begin
  freeAndNil(Neural);
  inherited Destroy;
end;

procedure TDarknetParser.parseNet(const options: TCFGSection);
var
  i, subdivs, mini_batch, _c, _h, _w, n, step : SizeInt;
  lvals, pvals, svals : TArray<string>;
  scales, seq_scales:TArray<single>;
  steps : TArray<SizeInt>;
  sequence_scale, scale : single;
  l, p, s : string;
begin

  Neural.maxBatches := options.getInt( 'max_batches', 0);
  Neural.batch := options.getInt( 'batch', 1);
  Neural.learningRate := options.getFloat( 'learning_rate', 0.001);
  Neural.learningRateMin := options.getFloat( 'learning_rate_min', 0.00001, true);
  Neural.batchesPerCycle := options.getInt('sgdr_cycle', Neural.maxBatches, true);
  Neural.batchesCycleMult := options.getInt('sgdr_mult', 2, true);
  Neural.momentum := options.getFloat( 'momentum', 0.9);
  Neural.decay := options.getFloat( 'decay', 0.0001);
  subdivs := options.getInt( 'subdivisions', 1);
  Neural.timeSteps := options.getInt('time_steps', 1, true);
  //Neural.track := options.getInt('track', 0, true);
  //Neural.augment_speed := options.getInt('augment_speed', 2, true);
  //Neural.sequential_subdivisions := options.getInt('sequential_subdivisions', subdivs, true);
  //Neural.init_sequential_subdivisions := Neural.sequential_subdivisions;
  //if Neural.sequential_subdivisions > subdivs then begin
      //Neural.init_sequential_subdivisions := subdivs; Neural.sequential_subdivisions := subdivs;
  //end;
  //Neural.try_fix_nan := options.getInt('try_fix_nan', 0, true);
  Neural.batch := Neural.batch div subdivs;
  mini_batch := Neural.batch;
  Neural.batch := Neural.batch * Neural.timeSteps;
  Neural.subdivisions := subdivs;
  //Neural.weights_reject_freq := options.getInt('weights_reject_freq', 0, true);
  Neural.equiDistantPoint := options.getInt('equidistant_point', 0, true);
  Neural.badLabelsRejectionPercentage := options.getFloat( 'badlabels_rejection_percentage', 0, true);
  Neural.numSigmasRejectBadlabels := options.getInt( 'num_sigmas_reject_badlabels', 0, true);
  Neural.EMA_Alpha := options.getFloat( 'ema_alpha', 0, true);
  //Neural.badlabels_reject_threshold [0]:= 0;
  Neural.deltaRollingMax := 0;
  Neural.deltaRollingAvg := 0;
  Neural.deltaRollingStdDev := 0;
  Neural.seen := 0;
  Neural.currentIteration := 0;
  Neural.lossScale := options.getFloat( 'loss_scale', 1, true);
  Neural.dynamicMiniBatch := options.getInt('dynamic_minibatch', 0, true);
  //Neural.optimized_memory := options.getInt('optimized_memory', 0, true);
  //Neural.workspace_size_limit := trunc(1024 * 1024 * options.getFloat( 'workspace_size_limit_MB', 1024, true));
  //Neural.adam := options.getBool( 'adam', false, true);
  //if Neural.adam then
  //    begin
  //        Neural.B1 := options.getFloat( 'B1', 0.9);
  //        Neural.B2 := options.getFloat( 'B2', 0.999);
  //        Neural.eps := options.getFloat( 'eps', 0.000001)
  //    end;
  _h := options.getInt('height', 0, true);
  _w := options.getInt('width', 0, true);
  _c := options.getInt('channels', 0, true);

  params.c         := _c;
  params.h         := _h;
  params.w         := _w;
  params.train     := false;
  params.net       := Neural;

  Neural.maxCrop := options.getInt('max_crop', _w * 2, true);
  Neural.minCrop := options.getInt('min_crop', _w, true);
  //Neural.flip := options.getInt('flip', 1, true);
  //Neural.blur := options.getInt('blur', 0, true);
  //Neural.gaussian_noise := options.getInt('gaussian_noise', 0, true);
  //Neural.mixup := options.getInt('mixup', 0, true);
  //cutmix := options.getBool( 'cutmix', false, true);
  //mosaic := options.getBool( 'mosaic', false, true);
  //if mosaic and cutmix then
      //Neural.mixup := 4
  //else
      //if cutmix then
          //Neural.mixup := 2
  //else
      //if mosaic then
          //Neural.mixup := 3;
  //Neural.letter_box := options.getInt('letter_box', 0, true);
  //Neural.mosaic_bound := options.getInt('mosaic_bound', 0, true);
  //Neural.contrastive := options.getBool( 'contrastive', false, true);
  //Neural.contrastive_jit_flip := options.getBool( 'contrastive_jit_flip', false, true);
  //Neural.contrastive_color := options.getBool( 'contrastive_color', false, true);
  //Neural.unsupervised := options.getBool( 'unsupervised', false, true);
  //if (Neural.contrastive) and (mini_batch < 2) then
      //raise Exception.Create('Error: mini_batch size (batch/subdivisions) should be higher than 1 for Contrastive loss!');
  //Neural.label_smooth_eps := options.getFloat( 'label_smooth_eps', 0.0, true);
  //Neural.resize_step := options.getInt('resize_step', 32, true);
  //Neural.attention := options.getInt('attention', 0, true);
  //Neural.adversarial_lr := options.getFloat( 'adversarial_lr', 0, true);
  //Neural.max_chart_loss := options.getFloat( 'max_chart_loss', 20.0, true);
  //Neural.angle := options.getFloat( 'angle', 0, true);
  //Neural.aspect := options.getFloat( 'aspect', 1, true);
  //Neural.saturation := options.getFloat( 'saturation', 1, true);
  //Neural.exposure := options.getFloat( 'exposure', 1, true);
  //Neural.hue := options.getFloat( 'hue', 0, true);
  //Neural.power := options.getFloat( 'power', 4, true);
  //if (Neural.inputs=0) and (Neural.h * Neural.w * Neural.c=0) then
      //raise Exception.create('No input parameters supplied');
  //policy_s := options.getStr( 'policy', 'constant');
  Neural.policy := TLearningRatePolicy.fromString(options.getStr( 'policy', 'constant'));
  Neural.burnIn := options.getInt('burn_in', 0, true);

  case Neural.policy of
    lrpSTEP:
      begin
          Neural.step := options.getInt( 'step', 1);
          Neural.scale := options.getFloat( 'scale', 1)
      end;
    lrpSTEPS, lrpSGDR:
      begin
          l := options.getStr('steps','');
          p := options.getStr('scales','');
          s := options.getStr('seq_scales','');
          if (Neural.policy = lrpSTEPS) and ((l='') or (p='')) then
              raise Exception.Create('STEPS policy must have steps and scales in cfg file');
          if l<>'' then
              begin
                  lvals := l.split([',']);
                  svals := s.split([',']);
                  pvals := p.split([',']);
                  n:= length(lvals);
                  setLength(steps, n);// := TIntegers.Create(n);
                  setLength(scales, n);// := TSingles.Create(n);
                  setLength(seq_scales, n);// := TSingles.Create(n);
                  for i := 0 to n -1 do
                      begin
                          scale := 1.0;
                          if i<length(pvals) then
                                  trystrToFloat(pvals[i], scale);
                          sequence_scale := 1.0;
                          if i<length(svals) then
                                  tryStrToFloat(svals[i], sequence_scale);
                          //TryStrToInt64(lvals[i], step);
                          steps[i] := StrToInt(lvals[i]);
                          scales[i] := scale;
                          seq_scales[i] := sequence_scale
                      end;
                  Neural.scales := scales;
                  Neural.steps := steps;
                  Neural.seq_scales := seq_scales;
                  Neural.num_steps := n
              end
      end;
    lrpEXP:
      Neural.gamma := options.getFloat( 'gamma', 1);
    lrpSIG:
      begin
          Neural.gamma := options.getFloat( 'gamma', 1);
          Neural.step := options.getInt( 'step', 1)
      end;
    lrpPOLY, lrpRANDOM:

  end


end;

procedure TDarknetParser.loadConnectedWeights(var l: TConnectedLayer;
  var fp: file; const transpose: boolean);
var
  buf: TSingleTensor;
begin
    l.biases.loadFromFile(fp);
    l.weights.loadFromFile(fp);
    if transpose then begin
        buf.resize([l.weights.w, l.weights.h()]);
        l.weights.matTranspose(buf);
        l.weights := buf
    end;
    if l.isBatchNormalized and (not l.dontLoadScales) then
        begin
            l.scales.loadFromFile(fp);
            l.rolling_mean.loadFromFile(fp);
            l.rolling_variance.loadFromFile(fp)
        end;
end;

procedure TDarknetParser.loadBatchNormWeights(var l: TBatchNormLayer;
  var fp: file);
begin

    l.biases.loadFromFile(fp);
    l.scales.loadFromFile(fp);
    l.rolling_mean.loadFromFile(fp);
    l.rolling_variance.loadFromFile(fp);
end;

procedure TDarknetParser.loadConvolutionalWeights(var l: TConvolutionalLayer;
  var fp: file);
var
  //i ,
  num, read_bytes: SizeInt;
begin
  num := l.weights.Size();
  read_bytes := l.biases.loadFromFile(fp);
  //BlockRead(fp, l.biases.data[0], sizeof(single)* l.biases.size(), read_bytes);
  if (read_bytes > 0) and (read_bytes < l.n) then    // todo [load_convolutional_weights] Read_Bytes should be divided by size of(single)?
      writeln(format(#10' Warning: Unexpected end of wights-file! l.biases - l.index = %d', [l.index]));
  if l.isBatchNormalized and (not l.dontloadscales) then
      begin
          read_bytes := l.scales.loadFromFile(fp);
          //BlockRead(fp, l.scales.data[0], sizeof(single) * l.scales.Size(), read_bytes);
          if (read_bytes > 0) and (read_bytes < l.n) then
              writeln(format(#10' Warning: Unexpected end of wights-file! l.scales - l.index = %d', [l.index]));

          read_bytes := l.rolling_mean.loadFromFile(fp);
          //BlockRead(fp, l.rolling_mean.data[0], sizeof(single) * l.rolling_mean.size(), read_bytes);
          if (read_bytes > 0) and (read_bytes < l.n) then
              writeln(format(#10' Warning: Unexpected end of wights-file! l.rolling_mean - l.index = %d', [l.index]));

          read_bytes := l.rolling_variance.loadFromFile(fp);
          //BlockRead(fp, l.rolling_variance.data[0], sizeof(single) * l.rolling_variance.size(), read_bytes);
          if (read_bytes > 0) and (read_bytes < l.n) then
              writeln(format(#10' Warning: Unexpected end of wights-file! l.rolling_variance - l.index = %d', [l.index]));
          //if false then
          //    begin
          //        for i := 0 to l.n -1 do
          //            write(format('%g, ', [l.rolling_mean.data[i]]));
          //        writeln('');
          //        for i := 0 to l.n -1 do
          //            write(format('%g, ', [l.rolling_variance.data[i]]));
          //        writeln('')
          //    end;
          //if false then
          //    begin
          //        fill_cpu(l.n, 0, l.rolling_mean, 1);
          //        fill_cpu(l.n, 0, l.rolling_variance, 1)
          //    end
      end;
  read_bytes := l.weights.loadFromFile(fp);
  //BlockRead(fp, l.weights.data[0], sizeof(single) * l.weights.Size(), read_bytes);
  if (read_bytes > 0) and (read_bytes < l.n) then
      writeln(format(#10' Warning: Unexpected end of wights-file! l.weights - l.index = %d', [l.index]));
  //if l.flipped then
  //    transpose_matrix(l.weights, (l.c div l.groups) * l.size * l.size, l.n);

  //writeln(#$1B'[2J', #$1B'[1H', sLineBreak, 'Layer : ', l.index);
  //writeln(sLineBreak, 'biases :');
  //l.biases.printDesc();
  //
  //writeln(sLineBreak, 'BatchNorm : ', boolToStr(l.isBatchNormalized, true));
  //if l.isBatchNormalized then
  //begin
  //    writeln(sLineBreak, 'scales :');
  //    l.scales.printDesc();
  //
  //    writeln(sLineBreak, 'rollingMean :');
  //    l.rolling_mean.printDesc();
  //
  //    writeln(sLineBreak, 'rollingVariance :');
  //    l.rolling_variance.printDesc();
  //end;
  //
  //writeln(sLineBreak, 'weights :');
  //l.weights.printDesc();
  //writeln(slineBreak, 'Pos : ', FilePos(fp));
  //readln()
end;

procedure TDarknetParser.loadAddWeights(var l: TAddLayer; var fp: file);
var read_bytes, num:SizeInt;
begin
  num := l.weights.size();
  read_bytes := l.weights.loadFromFile(fp);
  //BlockRead(fp, l.weights.data[0], sizeof(single) * l.weights.size(), read_bytes);
  if (read_bytes > 0) and (read_bytes < num) then
      writeln(format(#10' Warning: Unexpected end of wights-file! l.weights - l.index = %d ', [l.index]));
{$ifdef GPU}
  if gpu_index >= 0 then
      push_shortcut_layer(l)
{$endif}

end;

procedure TDarknetParser.loadWeights(const filename: string; cutoff: SizeInt);
var
  fp : file;
  o, iseen, i : SizeInt;
  major, minor, revision : uint32;
  transpose: Boolean;
  l : TBaseLayer;
begin
  assert(fileExists(FileName),'File not fount :'+ filename);
  if cutoff=0 then
    cutoff := Neural.layerCount() ;
  assignfile(fp, filename);
  reset(fp,1);
  BlockRead(fp, major, sizeof(int32) * 1, o);
  BlockRead(fp, minor, sizeof(int32) * 1, o);
  BlockRead(fp, revision, sizeof(int32) * 1, o);
  if (major * 10+minor) >= 2 then
      begin
          //writeln(' seen size = 64');
          iseen := 0;
          BlockRead(fp, iseen, sizeof(uint64)* 1, o);
          Neural.seen := iseen
      end
  else
      begin
          //writeln(' seen size = 32');
          iseen := 0;
          BlockRead(fp, iseen, sizeof(uint32)* 1, o);
          Neural.seen := iseen
      end;
  Neural.currentIteration := Neural.getTrainedBatchs();
  //writeln(format(', trained: %.0f K-images (%.0f Kilo-batches_64) ', [Neural.seen[0] / 1000, Neural.seen[0] / 64000]));
  transpose := (major > 1000) or (minor > 1000);
  i := 0;
  while (i < Neural.layerCount()) and (i < cutoff) do begin
      l := Neural.layers[i];
      if l.dontload then
          continue;
      case l.layerType of
        ltCONVOLUTIONAL:
          if (l is TBaseConvolutionalLayer) and ((l as TConvolutionalLayer).shareLayer = nil) then
            loadConvolutionalWeights(TConvolutionalLayer(l), fp);
        ltSHORTCUT :
          if (l.weights.Size() > 0) then
            loadAddWeights(TAddLayer(l), fp);
        //ltIMPLICIT :
        //  load_implicit_weights(l, fp);
        ltCONNECTED :
          loadConnectedWeights(TConnectedLayer(l), fp, transpose);
        ltBATCHNORM :
          loadBatchNormWeights(TBatchNormLayer(l), fp);
        //ltCRNN :
        //  begin
        //      load_convolutional_weights(l.input_layer[0], fp);
        //      load_convolutional_weights(l.self_layer[0], fp);
        //      load_convolutional_weights(l.output_layer[0], fp)
        //  end;
        //ltRNN :
        //  begin
        //      load_connected_weights(l.input_layer[0], fp, transpose);
        //      load_connected_weights(l.self_layer[0], fp, transpose);
        //      load_connected_weights(l.output_layer[0], fp, transpose)
        //  end;
        //ltGRU :
        //  begin
        //      load_connected_weights(l.wz[0], fp, transpose);
        //      load_connected_weights(l.wr[0], fp, transpose);
        //      load_connected_weights(l.wh[0], fp, transpose);
        //      load_connected_weights(l.uz[0], fp, transpose);
        //      load_connected_weights(l.ur[0], fp, transpose);
        //      load_connected_weights(l.uh[0], fp, transpose)
        //  end;
        //ltLSTM :
        //  begin
        //      load_connected_weights(l.wf[0], fp, transpose);
        //      load_connected_weights(l.wi[0], fp, transpose);
        //      load_connected_weights(l.wg[0], fp, transpose);
        //      load_connected_weights(l.wo[0], fp, transpose);
        //      load_connected_weights(l.uf[0], fp, transpose);
        //      load_connected_weights(l.ui[0], fp, transpose);
        //      load_connected_weights(l.ug[0], fp, transpose);
        //      load_connected_weights(l.uo[0], fp, transpose)
        //  end;
        //ltConvLSTM :
        //  begin
        //      if l.peephole then
        //          begin
        //              load_convolutional_weights(l.vf[0], fp);
        //              load_convolutional_weights(l.vi[0], fp);
        //              load_convolutional_weights(l.vo[0], fp)
        //          end;
        //      load_convolutional_weights(l.wf[0], fp);
        //      if not l.bottleneck then
        //          begin
        //              load_convolutional_weights(l.wi[0], fp);
        //              load_convolutional_weights(l.wg[0], fp);
        //              load_convolutional_weights(l.wo[0], fp)
        //          end;
        //      load_convolutional_weights(l.uf[0], fp);
        //      load_convolutional_weights(l.ui[0], fp);
        //      load_convolutional_weights(l.ug[0], fp);
        //      load_convolutional_weights(l.uo[0], fp)
        //  end;
        //ltLOCAL :
        //  begin
        //      locations := l.out_w * l.out_h;
        //      size := l.size * l.size * l.c * l.n * locations;
        //      BlockRead(fp, l.biases, sizeof(Single)* l.outputs, o);
        //      BlockRead(fp, l.weights, sizeof(Single)* size, o);
        //  end;

      end;
      if eof(fp) then
          break;
      inc(i)
  end;
  closeFile(fp)
end;

{ TLearningRatePolicyHelper }

class function TLearningRatePolicyHelper.fromString(s: string): TLearningRatePolicy;
begin
  s := LowerCase(s);
  if s = 'random' then
      exit(lrpCONSTANT);
  if s = 'poly' then
      exit(lrpPOLY);
  if s = 'constant' then
      exit(lrpCONSTANT);
  if s = 'step' then
      exit(lrpSTEP);
  if s = 'exp' then
      exit(lrpEXP);
  if s = 'sigmoid' then
      exit(lrpSIG);
  if s = 'steps' then
      exit(lrpSTEPS);
  if s = 'sgdr' then
      exit(lrpSGDR);
  if s = 'cost' then
      exit(lrpCOST);
  //writeln(ErrOutput, format('Couldn''t find policy %s, going with constant', [s]));
  exit(lrpCONSTANT)

end;

function TLearningRatePolicyHelper.toSring(): string;
begin
  result := copy(GetEnumName(TypeInfo(TLearningRatePolicy), ord(Self)), 4, 20)
end;

{ TLayerTypeHelper }

class function TLayerTypeHelper.fromString(s: string): TLayerType;
var i: SizeInt;
begin
  s := LowerCase(s);
  if s = 'shortcut'     then exit(ltSHORTCUT);
  if s = 'scale_channels' then exit(ltScaleChannels);
  if s = 'sam'          then exit(ltSAM);
  if s = 'crop'         then exit(ltCROP);
  if s = 'cost'         then exit(ltCOST);
  if s = 'detection'    then exit(ltDETECTION);
  if s = 'region'       then exit(ltREGION);
  if s = 'yolo'         then exit(ltYOLO);
  if s = 'Gaussian_yolo' then exit(ltGaussianYOLO);
  if s = 'iseg'         then exit(ltISEG);
  if s = 'local'        then exit(ltLOCAL);
  if (s = 'conv') or
            (s = 'convolutional') then exit(ltCONVOLUTIONAL);
  if s = 'activation'   then exit(ltACTIVE);
  if (s = 'net') or
            (s = 'network') then exit(ltNETWORK);
  if s = 'crnn'         then exit(ltCRNN);
  if s = 'gru'          then exit(ltGRU);
  if s = 'lstm'         then exit(ltLSTM);
  if s = 'conv_lstm'    then exit(ltConvLSTM);
  if s = 'history'      then exit(ltHISTORY);
  if s = 'rnn'          then exit(ltRNN);
  if (s = 'conn')       or (s = 'connected') then exit(ltCONNECTED);
  if (s = 'max')        or (s = 'maxpool') then exit(ltMAXPOOL);
  if (s = 'local_avg')  or (s = 'local_avgpool') then exit(ltLOCAL_AVGPOOL);
  if s = 'reorg3d'      then exit(ltREORG);
  if s = 'reorg'        then exit(ltREORG_OLD);
  if (s = 'avg')        or (s = 'avgpool') then exit(ltAVGPOOL);
  if s = 'dropout'      then exit(ltDROPOUT);
  if s = 'logistic'     then exit(ltLOGXENT);
  if (s = 'lrn') or (s = 'normalization') then exit(ltNORMALIZATION);
  if s = 'batchnorm'    then exit(ltBATCHNORM);
  if (s = 'soft') or (s = 'softmax') then exit(ltSOFTMAX);
  if s = 'constrative'  then exit(ltCONTRASTIVE);
  if s = 'route'        then exit(ltROUTE);
  if s = 'upsample'     then exit(ltUPSAMPLE);
  if (s = 'empty') or (s = 'silence') then exit(ltEMPTY);
  if s = 'implicit'       then exit(ltIMPLICIT);
  if s = 'l2norm'       then exit(ltL2NORM);
  if (s = 'deconv') or
            (s = 'deconvolutional') then exit(ltDECONVOLUTIONAL);
  exit(ltBLANK)

  //result := TLayerType(GetEnumValue(TypeInfo(TLayerType), 'lt'+s));
end;

function TLayerTypeHelper.toString: string;
begin
  result := copy(GetEnumName(TypeInfo(TLayerType), ord(Self)), 3, 20)
end;

{ TActivationTypeHelper }

class function TActivationTypeHelper.fromString(s: string): TActivationType;
begin
    s := LowerCase(s);
    if s = 'logistic' then  exit(acLOGISTIC);
    if s = 'swish' then exit(acSWISH);
    if s = 'mish' then exit(acMISH);
    if s = 'hard_mish' then exit(acHARD_MISH);
    if s = 'normalize_channels' then exit(acNORM_CHAN);
    if s = 'normalize_channels_softmax' then exit(acNORM_CHAN_SOFTMAX);
    if s = 'normalize_channels_softmax_maxval' then exit(acNORM_CHAN_SOFTMAX_MAXVAL);
    if s = 'loggy' then  exit(acLOGGY);
    if s = 'relu' then  exit(acRELU);
    if s = 'relu6' then  exit(acRELU6);
    if s = 'elu' then  exit(acELU);
    if s = 'selu' then  exit(acSELU);
    if s = 'gelu' then  exit(acGELU);
    if s = 'relie' then  exit(acRELIE);
    if s = 'plse' then  exit(acPLSE);
    if s = 'hardtan' then  exit(acHARDTAN);
    if s = 'lhtan' then  exit(acLHTAN);
    if s = 'linear' then  exit(acLINEAR);
    if s = 'ramp' then  exit(acRAMP);
    if s = 'revleaky' then  exit(acREVLEAKY);
    if s = 'leaky' then  exit(acLEAKY);
    if s = 'tanh' then  exit(acTANH);
    if s = 'stair' then  exit(acSTAIR);
    //writeln('Couldn''t find activation function ',s,' going with ReLU');
    exit(acRELU)
end;
function TActivationTypeHelper.toString(): string;
begin
  result := copy(GetEnumName(TypeInfo(TActivationType), ord(Self)), 3, 20)
end;

end.

