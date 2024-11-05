unit nnet;
{$ifdef FPC}
{$mode Delphi}{$H+}
{$endif}

interface

uses
  SysUtils, Math, typInfo
  {$ifdef MSWINDOWS}, Windows{$endif}
  , nTensors, nBaseLayer, NTypes, nYoloLayer
  {$ifdef USE_OPENCL}
  , opencl, OpenCLHelper
  {$endif}
  ;

type

  { TNNet }

  TNNet = class(TInterfacedObject)
    OnForward, OnBackward: procedure(var state: TNNetState);
    FBatch: SizeInt;
    subDivisions: SizeInt;
    timeSteps : SizeInt;
    maxBatches: SizeInt;
    EMA_Alpha: single;
    input: TSingleTensor;
    layers: TArray<TBaseLayer>;
    seen: SizeInt;
    numBoxes : SizeInt;
    maxCrop, minCrop : SizeInt;
    dynamicMiniBatch : SizeInt;
    currentIteration: SizeInt;
    currentSubDivision: SizeInt;
    epoch: SizeInt;
    adversarial : boolean;
    lossScale   : single;
    rejectThreshold
      ,badLabelsRejectionPercentage :Single;

    numSigmasRejectBadlabels
      ,equiDistantPoint: SizeInt;

    policy: TLearningRatePolicy;
    burnIn: SizeInt;
    step: SizeInt;
    steps: TArray<SizeInt>;
    Scales: TArray<single>;
    seq_scales : TArray<single>;
    num_steps : SizeInt;
    scale: single;
    gamma: single;
    power: single;
    learningRate: single;
    learningRateMin: single;
    momentum: single;
    decay: single;
    batchesPerCycle: SizeInt;
    batchesCycleMult: SizeInt;

    deltaRollingMax: single;
    deltaRollingAvg: single;
    deltaRollingStdDev: single;
    totalBBox: SizeInt;
    rewrittenBBox: SizeInt;

    truth: TArray<single>;

    //outTensor             : TSingleTensor;

    //TrainablesX, TrainablesY : TArray<single>;
    workspace: TSingleTensor;

    constructor Create(const aLayers: TArray<TBaseLayer>);
    procedure setLayers(ALayers:TArray<TBaseLayer>);
    procedure setTraining(const training: boolean);
    procedure setBatch(ABatch: SizeInt);
    function layerCount(): SizeInt;
    function avgCost(): single;
    function getTrainedBatchs: SizeInt;
    function computeCurrentLearningRate: single;
    procedure fuseBatchNorm;
    procedure forward(var state: TNNetState); overload;
    procedure backward(var state: TNNetState); overload;
    procedure update(); overload;
    function predict(const tensor: TSingleTensor): PSingleTensor;
    function Propagate( AInput, ATruth: PSingle): single;
    function trainEpoch(const Data: TData;  const randomSample: boolean = False ; batchCount: SizeInt = 0): single;
    function output(): PSingleTensor;
    function cost(): single;
    function classCount(): SizeInt;
    function Detections(const aWidth, aHeight: SizeInt; const aThresh: single = 0.5; const aBatch: SizeInt=0): TDetections;
    procedure freeLayers();
    destructor Destroy; override;

    property batch: SizeInt read FBatch write setBatch;
  end;


implementation

const
  DEFAULT_LEARNING_RATE = 0.001;
  DEFAULT_MOMENTUM = 0.9;
  DEFAULT_DECAY = 0.0001;

  { TNNet }

constructor TNNet.Create(const aLayers: TArray<TBaseLayer>);
begin
  OnForward := nil;
  OnBackward := nil;
  seen := 0;
  //inputShape := layers[0].Shape;
  subDivisions := 1;
  learningRate := DEFAULT_LEARNING_RATE;
  learningRateMin := 0.00001;
  momentum := DEFAULT_MOMENTUM;
  decay := DEFAULT_DECAY;
  //outTensor.reSize(output.Shape, output.Groups);
  setLayers(aLayers);
  if assigned(Layers) then
    FBatch := Layers[0].Batch;
end;

procedure TNNet.setLayers(ALayers: TArray<TBaseLayer>);
var
  i, wsSize, lSize:SizeInt;
begin
  wsSize := 0;
  for i := 0 to High(ALayers) do begin
    ALayers[i].id := i;
    lSize := ALayers[i].workspaceSize;
    ALayers[i].net := Self;
    if wsSize < lSize then
      wsSize := lSize;
  end;
  Layers := ALayers;
  workSpace.resize([wsSize]);
end;

procedure TNNet.setTraining(const training: boolean);
var
  i: SizeInt;
begin
  try
    for i := 0 to high(Layers) do
      Layers[i].train := training;

  except on E : Exception do
    raise Exception.Create('['+Layers[i].ClassName+'] : ' + E.Message)
  end;
end;

procedure TNNet.setBatch(ABatch: SizeInt);
var
  i, wsSize: SizeInt;
begin
  wsSize :=0;
  try
    for i := 0 to High(Layers) do begin
      Layers[i].setBatch(ABatch);
      wsSize := max(wsSize, layers[i].getWorkspaceSize());
    end;
  except on E : Exception do
    raise Exception.Create('['+Layers[i].ClassName+'] : ' + E.Message)
  end;
  workspace.reSize([wsSize]);
  FBatch := ABatch;
end;

function TNNet.layerCount(): SizeInt;
begin
  Result := length(Layers);
end;

function TNNet.avgCost(): single;
var
  i, c: SizeInt;
begin
  c := 0;
  Result := 0;
  for i := 0 to High(Layers) do
    if assigned(Layers[i].Cost) then
    begin
      Result := Result + Layers[i].Cost[0];
      Inc(c);
    end;
  Result := Result / c;
end;

function TNNet.getTrainedBatchs: SizeInt;
begin
  Result := seen div (batch * subDivisions);
end;

function TNNet.computeCurrentLearningRate: single;
var
  batch_num, i, last_iteration_start, cycle_size: SizeInt;
  rate, _cost: single;
begin
  batch_num := getTrainedBatchs;
  if batch_num < burnIn then
    exit(learningRate * Math.power(batch_num / burnIn, power));
  case policy of
    lrpCONSTANT:
      exit(learningRate);
    lrpSTEP:
      exit(learningRate * Math.power(scale, batch_num div step));
    lrpSTEPS:
    begin
      rate := learningRate;
      for i := 0 to high(steps) - 1 do  begin
        if steps[i] > batch_num then
          exit(rate);
        rate := rate * Scales[i];
      end;
      exit(rate);
    end;
    lrpEXP:
      exit(learningRate * Math.power(gamma, batch_num));
    lrpPOLY:
      exit(learningRate * Math.power(1 - batch_num / maxBatches, power));
    lrpRANDOM:
      // todo make random thread safe
      exit(learningRate * Math.power(random, power));
    lrpSIG:
      exit(learningRate * (1.0 / (1.0 + exp(gamma * (
      +- step)))));
    lrpSGDR:
      begin
        last_iteration_start := 0;
        cycle_size := batchesPerCycle;
        while ((last_iteration_start + cycle_size) < batch_num) do
        begin
          last_iteration_start := last_iteration_start + cycle_size;
          cycle_size := cycle_size * batchesCycleMult;
        end;
        rate := learningRateMin + 0.5 * (learningRate - learningRateMin) *
          (1.0 + cos((batch_num - last_iteration_start) * {3.14159265} PI / cycle_size));
        exit(rate);
      end;
    lrpCOST :
      begin
        _cost := cost() / batch;
        exit(learningRate * sqr(_cost));
      end
    else
    begin
      writeln(ErrOutput, 'Unknowen Policy!');
      exit(learningRate);
    end
  end;
end;

procedure TNNet.fuseBatchNorm;
var i:sizeInt;
begin
  for i := 0 to layerCount() -1 do
    layers[i].fuseBatchNorm;
end;

procedure TNNet.forward(var state: TNNetState);
var
  i: SizeInt;
  currentLayer: TBaseLayer;
  {$ifdef USE_OPENCL}
  ev : cl_event;
  {$endif}
begin
  state.workspace := workspace;
  for i := 0 to High(Layers) do
  begin
    state.index := i;
    currentLayer := layers[i];
    if state.isTraining and assigned(currentLayer.delta.Data) and currentLayer.train then begin
      //currentLayer.delta.Multiply(0);
    {$ifdef USE_OPENCL}
      ocl.fill(currentLayer.delta.size(), currentLayer.delta.devData, 0, 1, 0, nil, @ev);
      //ocl.waitForEvents(1, @ev);
    {$else}
      currentLayer.delta.fill(0);
    {$endif}
    end;
    if assigned(OnForward) then OnForward(state);

    {$ifdef USE_OPENCL}
    currentLayer.forwardGPU(state);
    {$else}
    currentLayer.forward(state);
    {$endif}
    // todo temporary OpenCL output workaroundB
    state.input := @currentLayer.output;
    {$ifdef USE_OPENCL}
    //if (i+1<Length(Layers)) and (currentLayer.layerType=ltCONVOLUTIONAL) and (layers[i+1].layerType<>ltCONVOLUTIONAL) then
    //  currentLayer.output.pullFromDevice;
    //if currentLayer.layerType=ltCONVOLUTIONAL then
    //  currentLayer.output.pullFromDevice;
    //if currentLayer.layerType=ltCONVOLUTIONAL then
    //  state.input.pullFromDevice;
    {$endif}
  end;
end;

procedure TNNet.backward(var state: TNNetState);
var
  i: SizeInt;
  original_input: PSingleTensor;
  original_delta : PSingleTensor;
  prev: TBaseLayer;
  current: TBaseLayer;
begin
  original_input := @input;
  original_delta := state.delta;
  state.workspace := workspace;
  for i := High(Layers) downto 0 do
  begin
    state.index := i;
    if i = 0 then
    begin
      state.input := original_input;
      state.delta := original_delta;
    end
    else
    begin
      prev := Layers[i - 1];
      state.input := @prev.output;
      state.delta := @prev.delta;
    end;
    current := Layers[i];
    if current.backwardStop then
      break;
    if current.forwardOnly then
      continue;
    {$ifdef USE_OPENCL}
    current.backwardGPU(state);
    {$else}
    current.backward(state);
    {$endif}
    if assigned(OnBackward) then
      OnBackward(state);
  end;

end;

procedure TNNet.update();
var
  i: SizeInt;
  update_batch: SizeInt;
  rate: single;
  current: TBaseLayer;
  arg: TUpdateArgs;
begin
  update_batch := batch * subdivisions;
  rate := computeCurrentLearningRate();
  for i := 0 to high(Layers) - 1 do
  begin
    current := Layers[i];
    if not current.train then
      continue;
//    if assigned(pointer(current.update)) then
    begin
      arg.batch := update_batch;
      arg.learningRate := rate;
      arg.momentum := momentum;
      arg.decay := decay;
      {$ifdef USE_OPENCL}
      current.updateGPU(arg);
      {$else}
      current.update(arg);
      {$endif}
    end;
  end;
end;

function TNNet.Propagate(AInput, ATruth: PSingle): single;
var
  state: TNNetState;
begin
  {$ifdef GPU}
    if gpu_index >= 0 then
        exit(train_network_datum_gpu(net, x, y));
  {$endif}
  //state := default(TNetworkState);
  //net.seen[0] :=  net.seen[0] + net.batch;
  //state.index := 0;
  //state.net := @net;
  //state.input := x;
  //state.delta := nil;
  //state.truth := y;
  //state.train := true;
  //forward_network(net, state);
  //backward_network(net, state);
  //error := get_network_cost(net);
  //if  state.net.total_bbox[0] > 0 then
  //    writeln(ErrOutput, format(' total_bbox = %d, rewritten_bbox = %f %% '#10'',  [state.net.total_bbox[0], 100 * state.net.rewritten_bbox[0] /  state.net.total_bbox[0]]));
  //exit(error)
  state := Default(TNNetState);
  state.net := Self;
  state.isTraining := True;
  state.input := @input;
  state.workspace := workspace;
  if assigned(state.delta) then
    state.delta.free;
  state.index := 0;
  state.truth.reShape(output().Shape, batch);
  {$ifdef USE_OPENCL}
  if not assigned(state.truth.devData) then
    state.truth.devData := ocl.createDeviceBuffer(state.truth.byteSize());
  state.truth.setCPU;
  {$endif}
  state.truth.Data := Pointer(ATruth);
  Inc(seen, batch);
  forward(state);
  backward(state);
  Result := cost();
end;

function TNNet.predict(const tensor: TSingleTensor): PSingleTensor;
var
  state: TNNetState;
begin
  state := Default(TNNetState);
  state.net := self;
  state.isTraining := False;
  state.input := @tensor;
  //state.delta.free;
  state.index := 0;
  forward(state);
  Result := output();
end;

function TNNet.trainEpoch(const Data: TData; const randomSample: boolean; batchCount: SizeInt): single;
var

  err: single;
  i: SizeInt;
begin
  assert(Data.X.h() mod batch =
    0, 'Please ensure that Dataset rows are multiple of batches!');

  if batchCount = 0 then
  begin
    batchCount := Data.X.Size() div (layers[0].inputs * batch);
  end;

  // todo [trainEpoch], still primitive implementation, revisit trainEpoch later for batch training

  input.reSize(layers[0].inputShape, Batch);
  if not assigned(self.truth) then
    setLength(self.Truth, Data.Y.Size());

  Result := 0;
  for i := 0 to batchCount - 1 do
  begin
    if randomSample then
      Data.getRandomBatch(batch, pointer(input.Data), pointer(Self.truth))
    else
      Data.getBatch(batch, i * batch, pointer(input.Data), pointer(Self.truth));
    {$ifdef USE_OPENCL}
    input.setCPU;
    {$endif}
    currentSubDivision := i;
    Propagate(pointer(input.Data), pointer(Self.truth));
    err := cost();
    Result := Result + err;
    //if wait_key then begin
    //    sleep(5);
    {$ifdef _MSWINDOWS}
              if GetKeyState(VK_ESCAPE)<0 then begin
                  writeln(sLineBreak, '[ESC] Pressed!, training terminated!') ;
                  break
              end;
    {$endif}
    //end;
    //wait_key_cv(5)
    update();
  end;
  Inc(currentIteration);
  Result := Result / (batchCount * batch);
end;

function TNNet.output: PSingleTensor;
var
  i: SizeInt;
begin
  // TODO [TNNet.output] GPU
  //if length(Layers) = 0 then exit(default(TTensor<Single>));
  for i := High(Layers) downto 0 do
    if layers[i].layerType <> ltCOST then
      exit(@layers[i].output);
end;

function TNNet.cost(): single;
var
  i, Count: SizeInt;
begin
  Result := 0;
  Count := 0;
  for i := 0 to High(Layers) do
    if Assigned(layers[i].cost) then
    begin
      Result := Result + layers[i].cost[0];
      Inc(Count);
    end;
  Result := Result / Count;
end;

function TNNet.classCount(): SizeInt;
var i: SizeInt;
begin
  result :=0;
  for i := high(Layers) downto 0 do
    if Layers[i].layerType in [ltYolo{, ltGaussianYOLO, ltREGION, ltDETECTION}] then
      exit(TYoloLayer(layers[I]).classes);

end;

function TNNet.Detections(const aWidth, aHeight: SizeInt;
  const aThresh: single; const aBatch: SizeInt): TDetections;
var detCount, i:SizeInt; l : TBaseLayer;
begin
  result := nil;
  l := nil;
  detCount := 0;
  for i:=0 to layerCount()-1 do
    case layers[i].layerType of
      ltYOLO:
        begin
          if not assigned(l) then l := layers[i];
          detCount :=  detCount + TYoloLayer(layers[i]).getDetectionCount(aThresh, aBatch);
        end;
      ltGaussianYOLO:
        begin
          if not assigned(l) then l := layers[i];
          //detCount := detCount + TGaussianYoloLayer(layers[i]).getDetectionCount(aThresh, aBatch);

        end;
      ltDETECTION:
        begin
          if not assigned(l) then l := layers[i];
          //detCount :=  detCount + TDetectionLayer(layers[i]).getDetectionCount(aThresh, aBatch);

        end;
      ltREGION:
        begin
          if not assigned(l) then l := layers[i];
          //detCount :=  detCount + TRegionLayer(layers[i]).getDetectionCount(aThresh, aBatch);

        end;
    end;
  setLength(result, detCount);
  if not assigned(result) then exit;

  for i := 0 to detCount -1 do
      begin
          setLength(result[i].prob, TYoloLayer(l).classes);
          //if l.&type = ltGaussianYOLO then
          //    setLength(result[i].uc, 4)
          //else
          //    result[i].uc := nil;
          //if (l.layerType=ltDETECTION) and (TDetectionLayer(l).coords > 4) then
          //    setLength(result[i].mask, l.coords-4)
          //else
          //    result[i].mask := nil;
          if assigned(TYoloLayer(l).embeddingOutput.data) then
              setLength(result[i].embeddings, TYoloLayer(l).embeddingSize)
          else
              result[i].embeddings := nil;
          result[i].embedding_size := TYoloLayer(l).embeddingSize;
      end;
  detCount := 0;
  for i:=0 to layerCount()-1 do
    case layers[i].layerType of
      ltYOLO:
        begin
          if detCount<length(result) then
            detCount := detCount + TYoloLayer(layers[i]).getDetections(aWidth, aHeight, self.input.w(), self.input.h(), aThresh, @result[detCount], true, true, ABatch);
        end;
      ltGaussianYOLO:
        begin
          if detCount<length(result) then
            //TGaussianYoloLayer(layers[i]).getDetections(aWidth, aHeight, self.input.w(), self.input.h(), thersh, nil, @result[detCpunt], true, true);
          //detCount := detCount + TGaussianYoloLayer(layers[i]).getDetectionCount(aThresh, aBatch);

        end;
      ltDETECTION:
        begin
          if detCount<length(result) then
            //TDetectionLayer(layers[i]).getDetections(aWidth, aHeight, self.input.w(), self.input.h(), thersh, nil, @result[detCpunt], true, true);
          //detCount :=  detCount + TDetectionLayer(layers[i]).getDetectionCount(aThresh, aBatch);

        end;
      ltREGION:
        begin
          if detCount<length(result) then
            //TRegionLayer(layers[i]).getDetections(aWidth, aHeight, self.input.w(), self.input.h(), thersh, nil, @result[detCpunt], true, true);
          //detCount :=  detCount + TRegionLayer(layers[i]).getDetectionCount(aThresh, aBatch);

        end;
    end;


end;

procedure TNNet.freeLayers();
var
  i: SizeInt;
begin
  for i := 0 to High(Layers) do
    FreeAndNil(Layers[i]);
end;

destructor TNNet.Destroy;
begin
  freeLayers;
  inherited Destroy;
end;

end.
