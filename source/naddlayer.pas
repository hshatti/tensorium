unit nAddLayer;
{$ifdef fpc}
{$mode Delphi}
{$endif}
{$pointermath on}

interface

uses
  SysUtils, nTypes, nTensors, nBaseLayer, nActivation, nChrono;

type

  { TAddLayer }

  TAddLayer=class(TBaseImageLayer)
    inputLayers, inputSizes     : TArray<SizeInt>;
    LayersOutput, layersDelta   : TArray<TSingleTensor>;
    weightSums                  : TArray<single>;
    weightMaxes                 : TArray<single>;
    weightsType                 : TWeightsType;
    weightsNormalization        : TWeightsNormalization;
    activationInput             : TSingleTensor;
    constructor Create(const aBatch: SizeInt; const aInputLayers,
      aInputSizes: TArray<SizeInt>; const aWidth, aHeight, aChannels: SizeInt;
  const aLayersOutput, aLayersDelta: TArray<TSingleTensor>;
  const aWeightsType: TWeightsType;
  const aWeightsNormalization: TWeightsNormalization;
  const aActivation: TActivationType; const ATrain: boolean);

    procedure setTrain(ATrain: boolean); override;
    procedure setBatch(ABatch: SizeInt); override;
    procedure fuseBatchNorm;  override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
    procedure update(const args: TUpdateArgs); override;
  end;

{.$define USE_MULTITHREADING}
implementation
uses math, nnet
{$ifdef USE_MULTITHREADING}  , steroids  {$endif}
  ;

{ TAddLayer }

constructor TAddLayer.Create(const aBatch: SizeInt;
  const aInputLayers, aInputSizes: TArray<SizeInt>; const aWidth, aHeight,
  aChannels: SizeInt; const aLayersOutput, aLayersDelta: TArray<TSingleTensor>;
  const aWeightsType: TWeightsType;
  const aWeightsNormalization: TWeightsNormalization;
  const aActivation: TActivationType; const ATrain: boolean);
begin
  LayerType := ltADD;
  batch := ABatch;
  ActivationType := aActivation;
  inputLayers := ainputLayers;
  inputSizes := aInputSizes;
  layersOutput := aLayersOutput;
  layersDelta := alayersDelta;
  weightsType := aWeightsType;
  weightsNormalization := aWeightsNormalization;
  learningRateScale := 1;
  w := aWidth   ; outW := w;
  h := aHeight  ; outH := h;
  c := aChannels; outC := c;
  outputs := w * h * c;
  inputs := outputs;
  FTrain := ATrain;
  //index := inputLayers[0];
  inputShape := [batch, c, h, w];
  output := TSingleTensor.Create([batch, c, h, w], batch);
  //nweights := 0;

  case weightsType of
    wtPER_FEATURE :
      begin
          setLength(weightSums, 1);
          setLength(weightMaxes, 1);
          weights := TSingleTensor.Create([length(inputSizes)+1]);
          //scale := sqrt(2 / result.nweights);
          weights.fill(1);
          if FTrain then
              weight_updates := TSingleTensor.Create([length(inputSizes)+1]);
      end;
    wtPER_CHANNEL :
      begin
          setLength(weightSums, c);
          setLength(weightMaxes, c);
          weights := TSingleTensor.Create([length(inputSizes)+1, c]);
          //scale := sqrt(2 / result.nweights);
          weights.fill(1);
          if FTrain then
              weight_updates := TSingleTensor.Create([(length(inputSizes)+1), c]);
      end;
  end;
{$ifndef GPU}
  if (weightsType <> wtNO_WEIGHTS) and (ActivationType in [acSWISH, acMISH]) then
      activationInput := TSingleTensor.Create([batch, c, h, w], batch);
{$endif}

end;

procedure TAddLayer.setTrain(ATrain: boolean);
begin
  if FTrain = ATrain then exit;
  FTrain := ATrain;
  if FTrain then begin
    case weightsType of
      wtPER_FEATURE:
        weight_updates := TSingleTensor.Create([(length(inputSizes)+1)]);
      wtPER_CHANNEL:
        weight_updates := TSingleTensor.Create([(length(inputSizes)+1), c]);
    end;
  end else
    weight_updates.Free()

end;

procedure TAddLayer.setBatch(ABatch: SizeInt);
begin
  if batch = ABatch then exit;
  batch := ABatch;
  inputShape[0] := batch;
  output.resize([batch, c, h, w], batch);
  {$ifndef GPU}
    if (weightsType <> wtNO_WEIGHTS) and (ActivationType in [acSWISH, acMISH]) then
        activationInput.resize([batch, c, h, w], batch);
  {$endif}

end;

function lrelu(const src: single):single;inline;
begin
    if src > sEPSILON then
        exit(src);
    exit(sEPSILON)
end;

function relu(const a:single; const index:SizeInt; const data:PSingle):single; overload; inline;
begin
  result := ord(a>0)*a
end;

function softmax(const a:single; const index:SizeInt; const data:PSingle):single; overload; inline;
begin
  result := exp(a-data[index])
end;

function relu(const a:single):single; overload; inline;
begin
  result := ord(a>0)*a
end;

procedure TAddLayer.fuseBatchNorm;
var
    layer_step, n, chan, i, w_index: SizeInt;
    sum, max_val, w : single;
begin
  if not assigned(weights.data) or (weightsNormalization = wnNO_NORMALIZATION) then exit;
  n := length(inputLayers);
  layer_step := weights.size() div (n+1);
  for chan := 0 to layer_step -1 do
    begin
      sum := 1; max_val := -MaxSingle;
      if weightsNormalization = wnSOFTMAX_NORMALIZATION then
        for i := 0 to (n+1) -1 do
          begin
            w_index := chan+i * layer_step;
            w := weights.data[w_index];
            if max_val < w then
                max_val := w
          end;
      sum := sEPSILON;
      for i := 0 to (n+1) -1 do
        begin
          w_index := chan + i * layer_step;
          w := weights.data[w_index];
          if weightsNormalization = wnRELU_NORMALIZATION then
            sum := sum + lrelu(w)
          else
            if weightsNormalization = wnSOFTMAX_NORMALIZATION then
              sum := sum + exp(w-max_val)
        end;
      for i := 0 to (n+1) -1 do
        begin
          w_index := chan + i * layer_step;
          w := weights.data[w_index];
          if weightsNormalization = wnRELU_NORMALIZATION then
            w := lrelu(w) / sum
          else
            if weightsNormalization = wnSOFTMAX_NORMALIZATION then
              w := exp(w-max_val) / sum;
          weights.data[w_index] := w
        end
    end;
  weightsNormalization := wnNO_NORMALIZATION;
end;

procedure add_multilayer(layers_output: TArray<TSingleTensor>; input, output: TSingleTensor; weights: TSingleTensor;
  weights_normalization: TWeightsNormalization; weightMaxes, weightSums: TArray<Single>);
var
    batch, src_i, i, weights_index, add_outputs, add_index,
    _c, cId, layers, step, nweights, outputs: SizeInt;
    w: single;
    II, OO, add:PSingle;
begin
  layers := length(layers_output);
  nweights := weights.size();
  outputs := output.Size() div output.Groups;
  if (nweights > 0) then begin
      _c := nweights div (layers+1);  {_c = 1 in case of wtPER_FEATURE, or outC in case of wtPER_CHANNEL}
      step := outputs div _c;  // step now is outW * outW
  end;

  case weights_normalization of
    wnRELU_NORMALIZATION:
      weights.Sums(pointer(weightSums), layers+1, relu);
    wnSOFTMAX_NORMALIZATION:
      begin
        weights.maxs(pointer(weightMaxes), layers+1);
        weights.sums(pointer(weightSums), layers+1, softmax, pointer(weightMaxes));
      end;
  end;
  if weights_normalization in [wnRELU_NORMALIZATION, wnSOFTMAX_NORMALIZATION] then
      for i:=0 to high(weightSums) do
        weightSums[i] := max(weightSums[i], sEPSILON);  // avoid division by zero


  case weights_normalization of
    wnRELU_NORMALIZATION :
      for batch :=0 to output.groups -1 do
          for cId :=0 to _c -1 do begin
              w := relu(weights.data[cId]) / weightSums[cId];
              OO := output.data + batch*outputs + cId*step;
              II := input.data + batch*outputs + cId*step;
              for i :=0 to step-1 do
                  OO[i] := II[i] * w
          end;

    wnSOFTMAX_NORMALIZATION :
        for batch :=0 to output.groups -1 do
            for cId :=0 to _c -1 do begin
                w := exp(weights.data[cId]- weightMaxes[cId]) / weightSums[cId];
                OO := output.data + batch*outputs + cId*step;
                II := input.data + batch*outputs + cId*step;
                for i :=0 to step-1 do
                    OO[i] := II[i] * w
            end;
    else
        input.copyTo(output)
  end;

  case weights_normalization of
    wnRELU_NORMALIZATION :
      for batch :=0 to output.groups -1 do begin
        for i := 0 to layers -1 do begin
          add_outputs := layers_output[i].groupSize();
          add_index :=0;
          cId := 0;
          weights_index:= (i+1) * _c;
          while add_index< add_outputs do begin
            add := layers_output[i].Data;
            w := weights.data[weights_index];
            w := relu(w) / weightSums[cId];
            inc(weights_index);
            OO := output.data + batch*outputs + add_index;
            II := add + batch*add_outputs + add_index;
            //for src_i :=0 to step-1 do
            //    OO[src_i] := w * II[src_i] + OO[src_i];
            TSingleTensor.axpysvv(step, w, II, 1, OO, 1);
            inc(add_index, step);
            inc(cId);
          end;
        end;
      end;
    wnSOFTMAX_NORMALIZATION :
      for batch :=0 to output.groups -1 do begin
        for i := 0 to layers -1 do begin
          add_outputs := layers_output[i].groupSize();
          add_index :=0;
          cId := 0;
          weights_index:= (i+1) * _c;
          while add_index< add_outputs do begin
            add := layers_output[i].Data;
            w := weights.data[weights_index];
            w := exp(w-weightMaxes[cId]) / weightSums[cId];
            inc(weights_index);
            OO := output.data + batch*outputs + add_index;
            II := add + batch*add_outputs + add_index;
            //for src_i :=0 to step-1 do
            //    OO[src_i] := w * II[src_i] + OO[src_i];
            TSingleTensor.axpysvv(step, w, II, 1, OO, 1);
            inc(add_index, step);
            inc(cId);
          end;
        end;
      end;
    else
      for i:= 0 to layers -1 do begin
        output.Add(layers_output[i]);
        //add_outputs := layers_output[i].outputSize();
        //add := layers_output[i].Data;
        //for batch :=0 to output.groups -1 do begin
        //  II := add + batch*add_outputs;
        //  OO := output.data + batch*outputs;
        //  for src_i :=0 to min(outputs, add_outputs)-1 do
        //      OO[src_i] := OO[src_i] + II[src_i]
        //end;
      end;
  end

    // todo SIMDfy
end;

procedure backward_add_multilayer(layers_delta: TArray<TSingleTensor>; delta_out, delta_in, weights, weight_updates, input: TSingleTensor;
  layers_output: TArray<TSingleTensor>; weights_normalization: TWeightsNormalization; weightMaxes, weightSums: TArray<single>);
var
    layers, nweights, _c, step, cId, src_i, batch, i, weights_index, add_outputs, add_index, out_index, outputs: SizeInt;
    grad, w: single;
    add, layer_delta, DII, DOO, II : PSingle;
begin
    layers := length(layers_output);
    nweights := weights.size();
    outputs := delta_out.groupSize();
    step := 0;
    if (nweights > 0) then begin
        _c := nweights div (layers+1);  {_c = 1 in case of wtPER_FEATURE, or outC in case of wtPER_CHANNEL}
        step := outputs div _c;  // step now is outW * outW
    end;
    // todo SIMDfy
    case weights_normalization of
      wnRELU_NORMALIZATION:
        weights.Sums(pointer(weightSums), layers+1, relu);
      wnSOFTMAX_NORMALIZATION:
        begin
          weights.maxs(pointer(weightMaxes), layers+1);
          weights.Sums(pointer(weightSums), layers+1, softmax, pointer(weightMaxes));
        end;
    end;

    for i :=0 to high(weightSums) do
        weightSums[i] := max(weightSums[i], sEPSILON);  // avoid division by zeros

    grad:=1;
    case weights_normalization of
      wnRELU_NORMALIZATION:
        for batch:=0 to delta_out.Groups-1 do
            for cId :=0 to _c-1 do begin
                w   := relu(weights.data[cId])/ weightSums[cId];
                //grad := w;// exists only GPU kernel but not in CPU!, why?
                DOO := delta_out.Data + batch*outputs + cId*step;
                DII := delta_in.Data + batch*outputs + cId*step;
                II  := input.Data + batch*outputs + cId*step;
                //for i:=0 to step -1 do
                //    DOO[i] := DOO[i] + DII[i] *w;
                TSingleTensor.axpysvv(step, w, DII, 1, DOO, 1);
                for i:=0 to step -1 do
                    weight_updates.Data[cId] :=  grad * DII[i]*II[i] + weight_updates.Data[cId]; // possible axypz>
            end;

      wnSOFTMAX_NORMALIZATION:
        for batch:=0 to delta_out.Groups-1 do
            for cId :=0 to _c-1 do begin
                w   := exp(weights.data[cId] - weightMaxes[cId])/ weightSums[cId];
                //grad := w*(1-w);// exists only GPU kernel but not in CPU!, why?
                DOO := delta_out.Data + batch*outputs + cId*step;
                DII := delta_in.Data + batch*outputs + cId*step;
                II  := input.Data + batch*outputs + cId*step;
                //for i:=0 to step -1 do
                    //DOO[i] := w * DII[i] + DOO[i];
                TSingleTensor.axpysvv(step, w, DII, 1, DOO, 1);
                for i:=0 to step -1 do
                    weight_updates.Data[cId] := grad * DII[i]*II[i] +  weight_updates.Data[cId];  //possible axypz?
            end;
      else
        delta_out.Add(delta_in);
    end;

    case weights_normalization of
      wnRELU_NORMALIZATION:
        for batch :=0 to delta_out.groups -1 do begin
          for i := 0 to layers -1 do begin
            add_outputs   := layers_output[i].groupSize();

            add_index     := 0;
            cId           := 0;
            weights_index := (i+1) * _c;
            while add_index< add_outputs do begin
              add                           := layers_output[i].Data + batch*add_outputs + add_index;
              II                            := layers_delta[i].Data  + batch*add_outputs + add_index;
              DII                           := delta_in.Data         + batch*outputs     + add_index ;

              w                             := weights.data[weights_index];
              w                             := relu(w)/weightSums[cId];
              //grad                          := w;// exists only GPU kernel but not in CPU!, why?
              //for src_i:=0 to step-1 do
              //  II[src_i] := w * DII[src_i] + II[src_i]; // axpy
              TSingleTensor.axpysvv(step, w, DII, 1, II, 1);

              for src_i:=0 to step-1 do
                weight_updates.Data[weights_index] := grad * DII[src_i] * add[src_i] + weight_updates.Data[weights_index];  // possible axypz

              inc(weights_index);
              inc(cId);

              inc(add_index, step);
              inc(add, step);
              inc(DII, step);
              inc(II,  step);
            end;
          end;
        end;
      wnSOFTMAX_NORMALIZATION:
        for batch :=0 to delta_out.groups -1 do begin
          for i := 0 to layers -1 do begin
            add_outputs   := layers_output[i].groupSize();

            add_index     := 0;
            cId           := 0;
            weights_index := (i+1) * _c;
            while add_index< add_outputs do begin
              add                           := layers_output[i].Data + batch*add_outputs + add_index;
              II                            := layers_delta[i].Data  + batch*add_outputs + add_index;
              DII                           := delta_in.Data         + batch*outputs     + add_index ;

              w                             := weights.data[weights_index];
              w                             := exp(w-weightMaxes[cId])/weightSums[cId];
              //grad                          := w*(1-w);// exists only GPU kernel but not in CPU!, why?
              //for src_i:=0 to step-1 do
                //II[src_i] := w * DII[src_i] + II[src_i];   // axpy
              TSingleTensor.axpysvv(step, w, DII, 1, II, 1);

              for src_i:=0 to step-1 do
                weight_updates.data[weights_index] := grad * DII[src_i] * add[src_i] + weight_updates.data[weights_index];  // posible axypz

              inc(weights_index);
              inc(cId);

              inc(add_index, step);
              inc(add, step);
              inc(DII, step);
              inc(II,  step);
            end;
          end;
        end;
      else
        for i:= 0 to layers -1 do begin
          layers_delta[i].Add(delta_in);
          //add_outputs := layers_delta[i].outputSize();
          //layer_delta := layers_delta[i].Data;
          //for batch :=0 to delta_out.groups -1 do begin
          //  DII := delta_in.data + batch*outputs;
          //  DOO := layer_delta + batch*add_outputs;
          //  for src_i :=0 to min(outputs, add_outputs)-1 do
          //      DOO[src_i] := DOO[src_i] + DII[src_i]
          //end;
        end;
    end;
end;

(*
type
  PSTParams = ^TSTParams;
  TSTParams = record
     src_outputs:SizeInt;
     weights_normalization:TWeightsNormalization;
     weights, &in, &out:PSingle;
     N, layer_step, step : SizeInt;
     outputs_of_layers: TArray<SizeInt>;
     layers_output:TArray<PSingle>;

  end;


procedure shortcut_multilayer(size: SizeInt; src_outputs: SizeInt;
  batch: SizeInt; n: SizeInt; outputs_of_layers: TArray<SizeInt>;
  layers_output: TArray<PSingle>; &out: Psingle; &in: Psingle; weights: Psingle;
  nweights: SizeInt; weights_normalization: TWeightsNormalization);
var
    id, src_id, src_i, src_b, i, layer_step, step, weights_index, add_outputs, add_index, out_index: SizeInt;
    sum, max_val, w: single; add:PSingle;
begin

    layer_step := nweights div (n+1);
    if (nweights > 0) then
        step := src_outputs div layer_step;
    for id := 0 to size-1 do
        begin
            src_id := id;
            src_i := src_id mod src_outputs;
            src_id := src_id div src_outputs;
            src_b := src_id;
            sum := 1; max_val := -MaxSingle;
            if assigned(weights) and boolean(weights_normalization) then
                begin
                    if weights_normalization = wnSOFTMAX_NORMALIZATION then
                        for i := 0 to (n+1) -1 do
                            begin
                                weights_index := src_i div step+i * layer_step;
                                w := weights[weights_index];
                                if max_val < w then
                                    max_val := w
                            end;
                    sum := sEPSILON;
                    for i := 0 to (n+1) -1 do
                        begin
                            weights_index := src_i div step+i * layer_step;
                            w := weights[weights_index];
                            if weights_normalization = wnRELU_NORMALIZATION then
                                sum := sum + relu(w)
                            else
                                if weights_normalization = wnSOFTMAX_NORMALIZATION then
                                    sum := sum + exp(w-max_val)
                        end;
                end;
            if assigned(weights) then
                begin
                    w := weights[src_i div step];
                    if weights_normalization = wnRELU_NORMALIZATION then
                        w := relu(w) / sum
                    else
                        if weights_normalization = wnSOFTMAX_NORMALIZATION then
                            w := exp(w-max_val) / sum;
                    &out[id] := &in[id] * w
                end
            else
                &out[id] := &in[id];

            for i := 0 to n -1 do
                begin
                    add_outputs := outputs_of_layers[i];
                    if src_i < add_outputs then
                        begin
                            add_index := add_outputs * src_b+src_i;
                            out_index := id;
                            add := layers_output[i];
                            if assigned(weights) then
                                begin
                                    weights_index := src_i div step+(i+1) * layer_step;
                                    w := weights[weights_index];
                                    if weights_normalization = wnRELU_NORMALIZATION then
                                        w := relu(w) / sum
                                    else
                                        if weights_normalization = wnSOFTMAX_NORMALIZATION then
                                            w := exp(w-max_val) / sum;
                                    &out[out_index] := &out[out_index] + (add[add_index] * w)
                                end
                            else
                                &out[out_index] := &out[out_index] + add[add_index]
                        end
                end
        end;
    // todo SIMDfy
end;


procedure backward_shortcut_multilayer(size: SizeInt; src_outputs: SizeInt;
  batch: SizeInt; n: SizeInt; outputs_of_layers: TArray<SizeInt>;
  layers_delta: TArray<PSingle>; delta_out: Psingle; delta_in: Psingle;
  weights: Psingle; weight_updates: Psingle; nweights: SizeInt; &in: Psingle;
  layers_output: TArray<PSingle>; weights_normalization: TWeightsNormalization);
var
    layer_step: SizeInt;
    step: SizeInt;
    id: SizeInt;
    src_id: SizeInt;
    src_i: SizeInt;
    src_b: SizeInt;
    grad, sum, max_val, w, eps: single;
    i: SizeInt;
    weights_index: SizeInt;
    add, layer_delta : PSingle;
    add_outputs: SizeInt;
    add_index: SizeInt;
    out_index: SizeInt;
begin
    layer_step := nweights div (n+1);
    step := 0;
    if (nweights > 0) then
        step := src_outputs div layer_step;
    // todo SIMDfy
    for id := 0 to size -1 do
        begin
            src_id := id;
            src_i := src_id mod src_outputs;
            src_id := src_id div src_outputs;
            src_b := src_id;
            grad := 1; sum := 1; max_val := -MaxSingle;
            if assigned(weights) and boolean(weights_normalization) then
                begin
                    if weights_normalization = wnSOFTMAX_NORMALIZATION then
                        for i := 0 to (n+1) -1 do
                            begin
                                weights_index := src_i div step+i * layer_step;
                                w := weights[weights_index];
                                if max_val < w then
                                    max_val := w
                            end;
                    sum := sEPSILON;
                    for i := 0 to (n+1) -1 do
                        begin
                            weights_index := src_i div step+i * layer_step;
                            w := weights[weights_index];
                            if weights_normalization = wnRELU_NORMALIZATION then
                                sum := sum + relu(w)
                            else
                                if weights_normalization = wnSOFTMAX_NORMALIZATION then
                                    sum := sum + exp(w-max_val)
                        end
                end;
            if assigned(weights) then
                begin
                    w := weights[src_i div step];
                    if weights_normalization = wnRELU_NORMALIZATION then
                        w := relu(w) / sum
                    else
                        if weights_normalization = wnSOFTMAX_NORMALIZATION then
                            w := exp(w-max_val) / sum;
                    delta_out[id] := delta_out[id] + (delta_in[id] * w);
                    weight_updates[src_i div step] := weight_updates[src_i div step] + (delta_in[id] * &in[id] * grad)
                end
            else
                delta_out[id] := delta_out[id] + delta_in[id];
            for i := 0 to n -1 do
                begin
                    add_outputs := outputs_of_layers[i];
                    if src_i < add_outputs then
                        begin
                            add_index := add_outputs * src_b+src_i;
                            out_index := id;
                            layer_delta := layers_delta[i];
                            if assigned(weights) then
                                begin
                                    add := layers_output[i];
                                    weights_index := src_i div step+(i+1) * layer_step;
                                    w := weights[weights_index];
                                    if weights_normalization = wnRELU_NORMALIZATION then
                                        w := relu(w) / sum
                                    else
                                        if weights_normalization = wnSOFTMAX_NORMALIZATION then
                                            w := exp(w-max_val) / sum;
                                    layer_delta[add_index] := layer_delta[add_index] + (delta_in[id] * w);
                                    weight_updates[weights_index] := weight_updates[weights_index] + (delta_in[id] * add[add_index] * grad)
                                end
                            else
                                layer_delta[add_index] := layer_delta[add_index] + delta_in[id]
                        end
                end
        end
end;
*)

procedure TAddLayer.forward(var state: TNNetState);
var
    //from_w: longint;
    //from_h: longint;
    //from_c: longint;
    lOutput: TBaseLayer;
    a,b:PSingle;
    i: longint;
    net : TNNet;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(layerType);
    {$endif}

    //from_w := state.net.layers[l.index].w;
    //from_h := state.net.layers[l.index].h;
    //from_c := state.net.layers[l.index].c;
    //if (l.nweights = 0) and (l.n = 1) and (from_w = l.w) and (from_h = l.h) and (from_c = l.c) then
    //    begin
    //        a:=state.input;
    //        b:=state.net.layers[l.index].output;
    //        for i := 0 to l.batch * l.w * l.h * l.c -1 do
    //            l.output[i] := a[i] + b[i]
    //            //l.output[i] := state.input[i]+state.net.layers[l.index].output[i]
    //    end
    //else
    //    shortcut_multilayer_cpu(outputs * batch, outputs, batch, features, inputSizes, layersOutput, output, state.input, weights, nweights, weightsNormalization);
    net := TNNet(state.net);
    lOutput := net.layers[inputLayers[0]];
    if (length(inputLayers)=1) and (lOutput.output.Size()=Output.Size()) then
        begin
          state.input.copyTo(output);
          output.Add(lOutput.output);
        end
    else
        add_multilayer(layersOutput, state.input, output, weights, weightsNormalization, weightSums, weightMaxes);

    case ActivationType of
      acSWISH :
        activate_array_swish(output, output.Size(), activationInput, output);
      acMISH :
        activate_array_mish(output, output.Size(), activationInput, output)
    else
        //activate_array_cpu_custom(l.output, l.outputs * l.batch, l.activation)
        //activate_array(l.output, l.outputs * l.batch, l.activation);
      activate()
    end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(layerType);
    {$endif}
end;

procedure TAddLayer.backward(var state: TNNetState);
begin
  case ActivationType of
    acSWISH :
      gradient_array_swish(output, outputs * batch, activationInput, delta);
    acMISH :
      gradient_array_mish(outputs * batch, activationInput, delta)
  else
      gradient_array(output, outputs * batch, ActivationType, delta)
  end;
  backward_add_multilayer(layersDelta, state.delta, delta, weights, weight_updates, state.input, layersOutput, weightsNormalization, weightMaxes, weightSums)
end;

procedure TAddLayer.update(const args: TUpdateArgs);
var
    learning_rate: single;
begin
  if assigned(weights.data) then
      begin
          //learning_rate := arg.learning_rate * learning_Rate_Scale;
          //axpy_cpu(l.nweights, -arg.decay * arg.batch, l.weights, 1, l.weight_updates, 1);
          //axpy_cpu(l.nweights, arg.learning_rate / arg.batch, l.weight_updates, 1, l.weights, 1);
          //scal_cpu(l.nweights, arg.momentum, l.weight_updates, 1)
          learning_rate := args.learningRate * learningRateScale;
          weight_updates.axpy( -args.decay * args.batch, weights);
          weights.axpy(args.learningRate / args.batch, weight_updates);
          weight_updates.multiply(args.momentum)
      end;
  inherited update(args);
end;

//const N = 4;
//  batch =6;
//var ipt, opt1, opt2, weight, weightUpdates1, weightUpdates2, deltaIpt, deltaOut : TSingleTensor;
//    ipts, opts1, opts2 : TArray<TSingleTensor>;
//    sizes : TArray<SizeInt>;
//    s, totalSize : SizeInt ;
//    i:SizeInt;
//    pp,oo1, oo2 : TArray<PSingle>;
//    ws, wm : TArray<single> ;
//    t : clock_t;
//    str:shortstring;
//initialization
//  sDigits:=4;
//  setLength(ipts, N);
//  setLength(opts1, N);
//  setLength(opts2, N);
//  setLength(Sizes, N);
//  setLength(pp, N);
//  setLength(oo1, N);
//  setLength(oo2, N);
//  setLength(ws, 10);
//  setLength(wm, 10);
//  repeat
//    write(#27'[1J');
//    write(#27'[1H');
//    FillDWord(ws[0], 10, 0);
//    FillDWord(wm[0], 10, 0);
//    totalSize :=0;
//    for i:=0 to N-1 do begin
//      ipts[i].reSize([batch, 10*(i+2)], batch);
//      ipts[i].UniformDistribution(-10, 10);
//      opts1[i].reSize([batch, 10*(i+2)], batch);
//      opts1[i].UniformDistribution(-10, 10);
//      opts2[i].reSize([batch, 10*(i+2)], batch);
//      opts1[i].copyTo(opts2[i]);
//      pp[i] := ipts[i].data;
//      oo1[i] := opts1[i].data;
//      oo2[i] := opts2[i].data;
//      //ipts[i].print(psGray);
//      s := ipts[i].size();
//      sizes[i] := s div batch;
//      totalSize := max(totalSize , ipts[i].outputSize());
//    end;
//    //weight.resize([N+1, 10]);
//    weight.UniformDistribution(-10,10);
//
//    //weightUpdates1.resize([N+1, 10]);
//    weightUpdates1.UniformDistribution(-10,10);
//
//    //weightUpdates2.resize([N+1, 10]);
//    weightUpdates1.copyTo(weightUpdates2);
//
//    ipt.resize([batch, totalSize], batch);
//    ipt.UniformDistribution(-1,1);
//
//    deltaIpt.reSize([batch, totalSize], batch);
//    deltaIpt.UniformDistribution(-10, 10);
//
//    opt1.resize([batch, totalSize], batch);
//    opt1.UniformDistribution(-10, 10);
//    opt2.resize([batch, totalSize], batch);
//    opt1.CopyTo(opt2);
//
//
//    //t := clock();
//    //add_multilayer(ipts, ipt, opt1, weight ,wnSOFTMAX_NORMALIZATION, wm, ws);
//    //writeln((clock()-t)/CLOCKS_PER_uS:1:3, ' uS.');
//    //opt1.print(psGray);
//    //
//    //t := clock();
//    //shortcut_multilayer(opt2.size(), opt2.outputSize(), batch, N, sizes, pp, opt2, ipt, weight , weight.size(), wnSOFTMAX_NORMALIZATION);
//    //writeln((clock()-t)/CLOCKS_PER_uS:1:3, ' uS.');
//    //opt2.Subtract(opt1);
//    //opt2.print(psGray);
//
////
//    t := clock();
//    backward_add_multilayer(opts1, opt1, deltaIpt, weight, weightUpdates1,
//              ipt, ipts, wnNO_NORMALIZATION, wm, ws);
//    writeln((clock()-t)/CLOCKS_PER_uS:1:3, ' uS.');
//
//
//    t := clock();
//    backward_shortcut_multilayer(ipt.size(), totalSize, batch, N, sizes, oo2, opt2.data, deltaIpt.data, weight.data, weightUpdates2.Data,
//              weight.size(), ipt.data, pp, wnNO_NORMALIZATION);
//    writeln((clock()-t)/CLOCKS_PER_uS:1:3, ' uS.');
//
//    weightUpdates1.print(psGray);
//    weightUpdates2.Subtract(weightUpdates1);
//    weightUpdates2.print(psGray);
//    opt1.print(psGray);
//    opt2.Subtract(opt1);
//    opt2.print(psGray);
//    for i:=0 to high(opts2) do begin
//      opts1[i].print(psGray);
//      opts2[i].Subtract(opts1[i]);
//      opts2[i].print(psGray);
//    end;
//
//    readln(str);
//  until lowerCase(s)='q'
//
end.

