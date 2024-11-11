unit nYoloLayer;

{$ifdef fpc}
{$mode Delphi}
{$endif}
{$pointermath on}

interface


uses
  classes, SysUtils,ntensors, nTypes, nBaseLayer, nActivation;

type

  PTrainYoloArgs = ^TTrainYoloArgs;
  TTrainYoloArgs = record
    //l                   : TBaseImageLayer;
    state               : PNNetState ;
    b                   : SizeInt;
    tot_iou             : single;
    tot_giou_loss       : single;
    tot_iou_loss        : single;
    count, class_count   : SizeInt;
  end;

  { TYoloLayer }

  TYoloLayer=class(TBaseImageLayer)
  private
    function entryIndex(const Abatch, location, entry: SizeInt): SizeInt; inline;
    function compareClass(const class_index, stride: SizeInt{; const objectness: single; const class_id: SizeInt}; const conf_thresh: single):boolean;
    function getBox(const n, index, col, row, netWidth, netHeight, stride:SizeInt):TBox;
    function deltaBox(const truth: TBox; const n, index, i, j, netWidth, netHeight, stride: SizeInt; const iou_normalizer: single; const iou_loss: TIOULoss; const accumulate: boolean; var reWrittenBBox:SizeInt):TIOUs;
    procedure deltaClass(const index, class_id, stride: SizeInt; const avg_cat: PSingle);
    procedure averageDeltas(const class_index, box_index, stride: SizeInt);
    procedure processBatch(arg:pointer);
  public
    n, classes, maxBoxes, total : SizeInt;
    tuneSize, TruthSize, truths, embeddingSize, embeddingLayerId, trackHistorySize, detsForTrack, detsForShow : SizeInt;
    Mask, Labels, classIds, map : TSizeIntTensor;
    classesMultipliers : TSingleTensor;
    Threads    : TArray<TThread>;
    ThreadArgs : TArray<TTrainYoloArgs>;
    newCoords, showDetails, objectnessSmooth, focalLoss, random : boolean;
    scaleXY, objNormalizer, iouNormalizer, ignoreThresh, truthThresh, maxDelta
      , labelSmoothEps, classNormalizer, deltaNormalizer, iouThresh, simThresh, betaNMS, jitter, resize, trackCIOUNorm : single;
    iouThreshKind : TIOULoss;
    iouLoss :TIOULoss;
    NMSKind : TNMSKind;
    embeddingOutput : TSingleTensor;
    constructor Create(const ABatch, aWidth, aHeight, aN, ATotal: SizeInt; const AMask: TArray<SizeInt>; const aClasses, aMaxBoxes: SizeInt; const ATrain: boolean=false);
    procedure setBatch(ABatch: SizeInt); override;
    procedure setTrain(ATrain: boolean); override;

    function getDetectionCount(const aThreshold:Single; const ABatch:SizeInt=0):SizeInt;

    function getDetections(const _w, _h, netw, neth: SizeInt; const thresh: single; const dets: PDetection;  const relative, letter: boolean; const abatch: SizeInt=0): SizeInt;

    procedure correctBoxes(const dets: PDetection; const n, w, h, netw, neth: SizeInt; const relative, letter: boolean);
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
  end;

implementation
uses math, nnet, steroids;

function ifthen(const val:boolean;const iftrue:single ; const iffalse:single =0.0):single;overload; inline;
begin
  if val then result:=iftrue else result:=iffalse;
end;

function fix_nan_inf(const val: single):single;
begin
    if isnan(val) or IsInfinite(val) then
        exit(0);
    exit(val)
end;

function clip_value(const val: single; const max_val: single):single;
begin
    if val > max_val then
        exit( max_val);
    if val < -max_val then
        exit( -max_val);
    exit(val)
end;

{ TYoloLayer }

function TYoloLayer.entryIndex(const Abatch, location, entry: SizeInt): SizeInt;
var _n, loc:SizeInt;
begin
  _n := location div (w * h);
  loc := location mod (w * h);
  result := aBatch*outputs + _n*(classes + 4+1)*w*h + entry*w*h + loc
end;

function TYoloLayer.getBox(const n, index, col, row, netWidth, netHeight,
  stride: SizeInt): TBox;
var n2 :SizeInt;
begin
  n2 := 2*n;
  result.x := (col + output.data[index+0 * stride]) / w;
  result.y := (row + output.data[index+1 * stride]) / h;
  if newCoords then
      begin
          result.w := sqr(output.data[index+2 * stride]) * 4 * biases.data[n2] / netWidth;
          result.h := sqr(output.data[index+3 * stride]) * 4 * biases.data[n2 + 1] / netHeight
      end
  else
      begin
          result.w := exp(output.data[index+2 * stride]) * biases.data[n2] / netWidth;
          result.h := exp(output.data[index+3 * stride]) * biases.data[n2 + 1] / netHeight
      end;
end;

function TYoloLayer.deltaBox(const truth: TBox; const n, index, i, j, netWidth,
  netHeight, stride: SizeInt; const iou_normalizer: single;
  const iou_loss: TIOULoss; const accumulate: boolean;
  var reWrittenBBox: SizeInt): TIOUs;
var
    pred: TBox;
    scale, tx, ty, tw, th, dx, dy, dw, dh: single;
begin
    scale := 2-truth.w * truth.h;
    if (delta.data[index+0 * stride]<>0) or (delta.data[index+1 * stride]<>0) or (delta.data[index+2 * stride]<>0) or (delta.data[index+3 * stride]<>0) then
        inc(reWrittenBBox);
    result := default(TIOUs);
    pred := getBox(n, index, i, j, netWidth, netHeight, stride);
    result.iou  := pred.iou(truth);
    result.giou := pred.giou(truth);
    result.diou := pred.diou(truth);
    result.ciou := pred.ciou(truth);
    if pred.w = 0 then
        pred.w := 1.0;
    if pred.h = 0 then
        pred.h := 1.0;
    if iou_loss = ilMSE then
        begin
            tx := (truth.x * w-i);
            ty := (truth.y * h-j);
            tw := ln(truth.w * w / biases.data[2 * n]);
            th := ln(truth.h * h / biases.data[2 * n+1]);
            if newCoords then
                begin
                    tw := sqrt(truth.w * w / (4 * biases.data[2 * n]));
                    th := sqrt(truth.h * h / (4 * biases.data[2 * n+1]))
                end;
            delta.data[index+0 * stride] := delta.data[index+0 * stride] + (scale * (tx - output.data[index+0 * stride]) * iou_normalizer);
            delta.data[index+1 * stride] := delta.data[index+1 * stride] + (scale * (ty - output.data[index+1 * stride]) * iou_normalizer);
            delta.data[index+2 * stride] := delta.data[index+2 * stride] + (scale * (tw - output.data[index+2 * stride]) * iou_normalizer);
            delta.data[index+3 * stride] := delta.data[index+3 * stride] + (scale * (th - output.data[index+3 * stride]) * iou_normalizer)
        end
    else
        begin
            result.dx_iou := pred.dx_iou( truth, iou_loss);
            dx := result.dx_iou.dt;
            dy := result.dx_iou.db;
            dw := result.dx_iou.dl;
            dh := result.dx_iou.dr;
            if newCoords then

            else
                begin
                    dw := dw * exp(output.data[index+2 * stride]);
                    dh := dh * exp(output.data[index+3 * stride])
                end;
            dx := dx * iou_normalizer;
            dy := dy * iou_normalizer;
            dw := dw * iou_normalizer;
            dh := dh * iou_normalizer;
            dx := fix_nan_inf(dx);
            dy := fix_nan_inf(dy);
            dw := fix_nan_inf(dw);
            dh := fix_nan_inf(dh);
            if maxDelta <> MaxSingle then
                begin
                    dx := clip_value(dx, maxDelta);
                    dy := clip_value(dy, maxDelta);
                    dw := clip_value(dw, maxDelta);
                    dh := clip_value(dh, maxDelta)
                end;
            if not accumulate then
                begin
                    delta.data[index+0 * stride] := 0;
                    delta.data[index+1 * stride] := 0;
                    delta.data[index+2 * stride] := 0;
                    delta.data[index+3 * stride] := 0
                end;
            delta.data[index+0 * stride] := delta.data[index+0 * stride] + dx;
            delta.data[index+1 * stride] := delta.data[index+1 * stride] + dy;
            delta.data[index+2 * stride] := delta.data[index+2 * stride] + dw;
            delta.data[index+3 * stride] := delta.data[index+3 * stride] + dh
        end;
end;

procedure TYoloLayer.deltaClass(const index, class_id, stride: SizeInt; const avg_cat: PSingle);
var
    _n, ti: SizeInt;
    y_true, result_delta, alpha, pt, grad: single;
begin
    if delta.data[index+stride * class_id]<>0 then
        begin
            y_true := 1;
            if labelSmoothEps<>0 then
                y_true := y_true * (1-labelSmoothEps)+0.5 * labelSmoothEps;
            result_delta := y_true-output.data[index+stride * class_id];
            if not isnan(result_delta) and not IsInfinite(result_delta) then
                delta.data[index+stride * class_id] := result_delta;
            if assigned(classesMultipliers.data) then
                delta.data[index+stride * class_id] := delta.data[index+stride * class_id] * classesMultipliers.data[class_id];
            if assigned(avg_cat) then
                 avg_cat[0] := avg_cat[0] + output.data[index+stride * class_id];
            exit()
        end;
    if focalLoss then
        begin
            alpha := 0.5;
            ti := index+stride * class_id;
            pt := output.data[ti] + sEPSILON;
            grad := -(1-pt) * (2 * pt * ln(pt)+pt-1);
            for _n := 0 to classes -1 do
                begin
                    delta.data[index+stride * _n] := ((ifthen((_n = class_id), 1, 0))-output.data[index+stride * _n]);
                    delta.data[index+stride * _n] := delta.data[index+stride * _n] * (alpha * grad);
                    if (_n = class_id) and assigned(avg_cat) then
                        avg_cat[0] := avg_cat[0] + output.data[index+stride * _n]
                end
        end
    else
        for _n := 0 to classes -1 do
            begin
                y_true := (ifthen((_n = class_id), 1, 0));
                if labelSmoothEps<>0 then
                    y_true := y_true * (1-labelSmoothEps)+0.5 * labelSmoothEps;
                result_delta := y_true-output.data[index+stride * _n];
                if not isnan(result_delta) and not IsInfinite(result_delta) then
                    delta.data[index+stride * _n] := result_delta;
                if assigned(classesMultipliers.data) and (_n = class_id) then
                    delta.data[index+stride * class_id] := delta.data[index+stride * class_id] * (classesMultipliers.data[class_id] * classNormalizer);
                if (_n = class_id) and assigned(avg_cat) then
                    avg_cat[0] := avg_cat[0] + output.data[index+stride * _n]
            end
end;

procedure TYoloLayer.averageDeltas(const class_index, box_index, stride: SizeInt);
var
    classes_in_one_box: SizeInt;
    c: SizeInt;
begin
    classes_in_one_box := 0;
    for c := 0 to classes -1 do
        if delta.data[class_index+stride * c] > 0 then
            inc(classes_in_one_box);
    if classes_in_one_box > 0 then
        begin
            delta.data[box_index+0 * stride] := delta.data[box_index+0 * stride] / classes_in_one_box;
            delta.data[box_index+1 * stride] := delta.data[box_index+1 * stride] / classes_in_one_box;
            delta.data[box_index+2 * stride] := delta.data[box_index+2 * stride] / classes_in_one_box;
            delta.data[box_index+3 * stride] := delta.data[box_index+3 * stride] / classes_in_one_box
        end
end;


function TYoloLayer.compareClass(const class_index, stride: SizeInt;
  const conf_thresh: single): boolean;
var
    j: SizeInt;
begin
    for j := 0 to classes -1 do
      if {objectness*}output.Data[class_index+stride * j] > conf_thresh then
        exit(true);
    exit(false)
end;

constructor TYoloLayer.Create(const ABatch, aWidth, aHeight, aN,
  ATotal: SizeInt; const AMask: TArray<SizeInt>; const aClasses,
  aMaxBoxes: SizeInt; const ATrain: boolean);
var
    i: SizeInt;
begin
    layerType := ltYOLO;
    n := An;
    total := Atotal;
    batch := Abatch;
    w := aWidth;
    h := aHeight;
    c := n * (aClasses+4+1);
    classes := aClasses;
    outW := w;
    outH := h;
    outC := c;
    if assigned(AMask) then
        mask := AMask
    else
        begin
            mask := TSizeIntTensor.Create([n]);
            mask.fill(0, 1);
        end;
    outputs            :=  n * (classes+4+1) * h * w;
    inputs             := outputs;
    inputShape         := [batch, n, (classes+4+1), h, w];
    maxBoxes           := aMaxBoxes;
    truthSize          := 4+2;
    truths             := maxBoxes * truthSize;

    //setLength(labels, batch * result.w * result.h * result.n);// := longint(xcalloc(batch * result.w * result.h * result.n, sizeof(int)));
    labels             := TSizeIntTensor.Create([batch, n, h, w], batch);
    //for i := 0 to batch * result.w * result.h * result.n -1 do
    //    labels[i] := -1;
    labels.Fill(-1);
    //setLength(result.class_ids, batch * result.w * result.h * result.n) ;//:= longint(xcalloc(batch * result.w * result.h * result.n, sizeof(int)));
    classIds := TSizeIntTensor.create([batch, n, h, w], batch);
    //for i := 0 to batch * result.w * result.h * result.n -1 do
    //    result.class_ids[i] := -1;
    classIds.Fill(-1);

    biases := TSingleTensor.Create([total * 2]);
    biases.fill(0.5);
    output := TSingleTensor.Create([batch, n, (classes+4+1), h, w], batch);
    FTrain := aTrain;
    if FTrain then  begin
        delta := TSingleTensor.Create([batch, n, (classes+4+1), h, w], batch);
        //bias_updates       := TSingleTensor.Create([n * 2]);
        setLength(threads, batch);
        setLength(ThreadArgs, batch)
    end;
    cost := [0];//TSingles.Create(1);
    randomize;
end;

procedure TYoloLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch = batch then exit;
  batch              := ABatch;
  inputShape[0]      := Batch;

  labels.resize([batch, n, h, w], batch);
  classIds.resize([batch, n, h, w], batch);

  output.resize([batch, n, (classes+4+1), h, w], batch);

  labels.Fill(-1);

  if FTrain then begin
      delta.resize([batch, n, (classes+4+1), h, w], batch);
      setLength(threads, batch);
      setLength(ThreadArgs, batch)
  end;

end;

procedure TYoloLayer.setTrain(ATrain: boolean);
begin
  if ATrain = FTrain then exit;
  FTrain := ATrain;
  if FTrain then begin
      delta              := TSingleTensor.Create([batch, (classes+4+1), n, h, w], batch);
      //bias_updates       := TSingleTensor.Create([n * 2]);
      setLength(threads, batch);
      setLength(ThreadArgs, batch)
  end else begin
      delta.free;
      setLength(threads, 0);
      setLength(ThreadArgs, 0)
      //bias_updates.free
  end;
end;

function TYoloLayer.getDetectionCount(const aThreshold: Single;
  const ABatch: SizeInt): SizeInt;
var
    i, _n, obj_index: SizeInt;
begin
    result := 0;
    for i := 0 to w * h -1 do
        for _n := 0 to n -1 do
            begin
                //obj_index := aBatch*outputs + _n*(classes + 4+1)*w*h + 4*w*h + i;
                obj_index := entryIndex(ABatch, _n * w * h + i, 4);
                if //not isnan(l.output[obj_index]) and
                   (output.Data[obj_index] > aThreshold) then
                    inc(result)
            end;
end;

function TYoloLayer.getDetections(const _w, _h, netw, neth: SizeInt;
  const thresh: single; const dets: PDetection;
  const relative, letter: boolean; const abatch: SizeInt): SizeInt;
var
    i, j, _n, row, col, obj_index, box_index, class_index: SizeInt;
    predictions: PSingle;
    objectness, prob: single;
begin
    predictions := pointer(output.Data);
    result := 0;
    for i := 0 to w * h -1 do
        begin
            row := i div w;
            col := i mod w;
            for _n := 0 to n -1 do
                begin
                    obj_index := entryIndex(abatch, _n * w * h+i, 4);
                    objectness := predictions[obj_index];
                    if //not isnan(objectness) and
                       (objectness > thresh) then
                        begin
                            box_index := entryIndex(abatch, _n * w * h+i, 0);
                            dets[result].bbox := getBox(mask.Data[_n], box_index, col, row, netw, neth, w * h);
                            dets[result].objectness := objectness;
                            dets[result].classes := classes;
                            if assigned(embeddingOutput.data) then
                                TensorUtils.get_embedding(pointer(embeddingOutput.data), w, h, n * embeddingSize, embeddingSize, col, row, _n, aBatch, pointer(dets[result].embeddings));
                            for j := 0 to classes -1 do
                                begin
                                    class_index := entryIndex(abatch, _n * w * h+i, 4+1+j);
                                    prob := objectness * predictions[class_index];
                                    if (prob > thresh) then
                                        dets[result].prob[j] := prob
                                    else
                                        dets[result].prob[j] := 0
                                end;
                            inc(result)
                        end
                end
        end;
    correctBoxes(dets, result, _w, _h, netw, neth, relative, letter);
    exit(result)
end;

procedure TYoloLayer.correctBoxes(const dets: PDetection; const n, w, h, netw,
  neth: SizeInt; const relative, letter: boolean);
var
    i, new_w, new_h: SizeInt;
    deltaw, deltah, ratiow, ratioh: single;
    b: TBox;
begin
    new_w := 0;
    new_h := 0;
    if letter then
        begin
            if (netw / w) < (neth / h) then
                begin
                    new_w := netw;
                    new_h := (h * netw) div w
                end
            else
                begin
                    new_h := neth;
                    new_w := (w * neth) div h
                end
        end
    else
        begin
            new_w := netw;
            new_h := neth
        end;
    deltaw := netw -new_w;
    deltah := neth -new_h;
    ratiow := new_w / netw;
    ratioh := new_h / neth;
    for i := 0 to n -1 do
        begin
            b := dets[i].bbox;
            b.x := (b.x-deltaw / 2.0 / netw) / ratiow;
            b.y := (b.y-deltah / 2.0 / neth) / ratioh;
            b.w := b.w * (1 / ratiow);
            b.h := b.h * (1 / ratioh);
            if not relative then
                begin
                    b.x := b.x * w;
                    b.w := b.w * w;
                    b.y := b.y * h;
                    b.h := b.h * h
                end;
            dets[i].bbox := b
        end
end;

procedure TYoloLayer.processBatch(arg: pointer);
var
    Args:PTrainYoloArgs absolute arg;
    b, stride, obj_index, box_index, i, j, _n, t
      , best_match_t, best_t, class_id, class_index, cl_id, mask_n
      , best_n, netWidth, netHeight, truth_in_index, track_id, truth_out_index: SizeInt;
    state: PNNetState;
    net: TNNet;
    pred, truth, truthShift : TBox;
    iou, tot_giou, tot_diou, tot_ciou, tot_diou_loss
      , tot_ciou_loss, recall, recall75
      , best_iou, best_match_iou
      , avg_cat, avg_obj, avg_anyobj, objectness , delta_obj
      , iou_multiplier, scale, class_multiplier: Single;
    class_id_match, found_object: Boolean;
    all_ious: TIOUs;

begin
  //l := args.l;
  state := args.state;
  net := TNNet(state.net);
  netWidth  := net.input.w();
  netHeight := net.input.h();;
  b := args.b;
  tot_giou := 0;
  tot_diou := 0;
  tot_ciou := 0;
  tot_diou_loss := 0;
  tot_ciou_loss := 0;
  recall := 0;
  recall75 := 0;
  avg_cat := 0;
  avg_obj := 0;
  avg_anyobj := 0;
  stride := self.w * self.h;
  for j := 0 to self.h -1 do
      for i := 0 to self.w -1 do
          for _n := 0 to self.n -1 do
              begin
                  class_index := entryIndex(b, _n*self.w*self.h + j*self.w + i, 4+1);
                  obj_index := entryIndex(b, _n*self.w*self.h + j*self.w + i, 4);
                  box_index := entryIndex(b, _n*self.w*self.h + j*self.w + i, 0);
                  pred := getBox(self.mask.data[_n], box_index, i, j, netWidth, netHeight, stride);
                  best_match_iou := 0;
                  best_match_t := 0;
                  best_iou := 0;
                  best_t := 0;
                  for t := 0 to self.maxBoxes -1 do
                      begin
                          truth := TBox.fromFloat(Pointer(state.truth.data + t*self.TruthSize + b*self.truths), 1);
                          if truth.x=0 then
                              break;
                          class_id := trunc(state.truth.data[t*self.TruthSize + b*self.truths + 4]);
                          if (class_id >= self.classes) or (class_id < 0) then
                              begin
                                  writeln(format(#10' Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] ', [class_id, self.classes, self.classes-1]));
                                  writeln(format(#10' truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f, class_id = %d ', [truth.x, truth.y, truth.w, truth.h, class_id]));
                                  continue
                              end;
                          objectness := self.output.Data[obj_index];
                          if isnan(objectness) or IsInfinite(objectness) then
                              self.output.Data[obj_index] := 0;
                          class_id_match := compareClass(class_index, stride{, objectness, class_id}, 0.25);
                          iou := pred.iou(truth);
                          if (iou > best_match_iou) and class_id_match then
                              begin
                                  best_match_iou := iou;
                                  best_match_t := t
                              end;
                          if iou > best_iou then
                              begin
                                  best_iou := iou;
                                  best_t := t
                              end
                      end;
                  avg_anyobj := avg_anyobj + self.output.data[obj_index];
                  self.delta.Data[obj_index] := self.objNormalizer * (-self.output.data[obj_index]);
                  if best_match_iou > self.ignoreThresh then
                      begin
                          if self.objectnessSmooth then
                              begin
                                  delta_obj := self.objNormalizer * (best_match_iou-self.output.data[obj_index]);
                                  if delta_obj > self.delta.data[obj_index] then
                                      self.delta.data[obj_index] := delta_obj
                              end
                          else
                              self.delta.data[obj_index] := 0
                      end
                  else
                      if net.adversarial then
                          begin
                              //stride := self.w * self.h;
                              scale := pred.w * pred.h;
                              if scale > 0 then
                                  scale := sqrt(scale);
                              self.delta.data[obj_index] := scale * self.objNormalizer * (-self.output.data[obj_index]);
                              found_object := false;
                              for cl_id := 0 to self.classes -1 do
                                  if self.output.data[class_index+stride * cl_id] * self.output.data[obj_index] > 0.25 then
                                      begin
                                          self.delta.data[class_index+stride * cl_id] := scale * (-self.output.data[class_index+stride * cl_id]);
                                          found_object := true
                                      end;
                              if found_object then
                                  begin
                                      for cl_id := 0 to self.classes -1 do
                                          if self.output.data[class_index+stride * cl_id] * self.output.data[obj_index] < 0.25 then
                                              self.delta.data[class_index+stride * cl_id] := scale * (1-self.output.data[class_index+stride * cl_id]);
                                      self.delta.data[box_index+0 * stride] := self.delta.data[box_index+0 * stride] + (scale * (-self.output.data[box_index+0 * stride]));
                                      self.delta.data[box_index+1 * stride] := self.delta.data[box_index+1 * stride] + (scale * (-self.output.data[box_index+1 * stride]));
                                      self.delta.data[box_index+2 * stride] := self.delta.data[box_index+2 * stride] + (scale * (-self.output.data[box_index+2 * stride]));
                                      self.delta.data[box_index+3 * stride] := self.delta.data[box_index+3 * stride] + (scale * (-self.output.data[box_index+3 * stride]))
                                  end
                          end;
                  if best_iou > self.truthThresh then
                      begin
                          iou_multiplier := best_iou * best_iou;
                          if self.objectnessSmooth then
                              self.delta.data[obj_index] := self.objNormalizer * (iou_multiplier-self.output.data[obj_index])
                          else
                              self.delta.data[obj_index] := self.objNormalizer * (1-self.output.data[obj_index]);
                          class_id := trunc(state.truth.data[best_t*self.TruthSize + b*self.truths+4]);
                          if assigned(self.map.data) then
                              class_id := self.map.data[class_id];
                          deltaClass(class_index, class_id, stride, nil);
                          class_multiplier := ifthen(assigned(self.classesMultipliers.Data), self.classesMultipliers.Data[class_id], 1.0);
                          if self.objectnessSmooth then
                              self.delta.data[class_index+stride * class_id] := class_multiplier * (iou_multiplier-self.output.data[class_index+stride * class_id]);
                          truth := TBox.fromFloat(pointer(state.truth.data + best_t * self.TruthSize+b * self.truths), 1);
                          deltaBox(truth, self.mask.data[_n], box_index, i, j, netWidth, netHeight, stride, self.iouNormalizer * class_multiplier, self.iouLoss, true, net.rewrittenBBox);
                          inc(net.totalBBox)
                      end
              end;
  for t := 0 to self.maxBoxes -1 do
      begin
          truth := TBox.fromFloat(pointer(state.truth.Data + t*self.TruthSize + b*self.truths), 1);
          if truth.x=0 then
              break;
          if (truth.x < 0) or (truth.y < 0) or (truth.x > 1) or (truth.y > 1) or (truth.w < 0) or (truth.h < 0) then
              begin
                  writeln(format(' Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f ',[ truth.x, truth.y, truth.w, truth.h]));
                  //sprintf(buff, 'echo "Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f" >> bad_labeself.list', truth.x, truth.y, truth.w, truth.h);
                  //system(buff)
              end;
          class_id := trunc(state.truth.data[t*self.TruthSize + b*self.truths + 4]);
          if (class_id >= self.classes) or (class_id < 0) then
              continue;
          best_iou := 0;
          best_n := 0;
          i := trunc(truth.x * self.w);
          j := trunc(truth.y * self.h);
          truthShift := truth;
          truthShift.x := 0;truthShift.y := 0;
          for _n := 0 to self.total -1 do
              begin
                  pred := default(TBox);
                  pred.w := self.biases.data[2 * _n] / netWidth;
                  pred.h := self.biases.data[2 * _n+1] / netHeight;
                  iou := pred.iou(truthShift);
                  if iou > best_iou then
                      begin
                          best_iou := iou;
                          best_n := _n
                      end
              end;
          mask.indexOf(best_n);
          mask_n := self.mask.indexOf(best_n);
          if mask_n >= 0 then
              begin
                  class_id := trunc(state.truth.data[t * self.TruthSize+b * self.truths+4]);
                  if assigned(self.map.data) then
                      class_id := self.map.data[class_id];
                  box_index := entryIndex(b, mask_n * self.w * self.h+j * self.w+i, 0);
                  if assigned(self.classesMultipliers.Data) then
                      class_multiplier := self.classesMultipliers.Data[class_id]
                  else
                      class_multiplier := 1.0;
                  all_ious := deltaBox(truth, best_n, box_index, i, j, netWidth, netHeight, stride, self.iouNormalizer * class_multiplier, self.iouLoss, true, net.rewrittenBBox);
                  inc(net.totalBBox);
                  truth_in_index := t * self.TruthSize+b * self.truths+5;
                  track_id := trunc(state.truth.data[truth_in_index]);
                  truth_out_index := b * self.n * self.w * self.h+mask_n * self.w * self.h+j * self.w+i;
                  self.labels.Data[truth_out_index] := track_id;
                  self.classIds.Data[truth_out_index] := class_id;
                  args.tot_iou := args.tot_iou + all_ious.iou;
                  args.tot_iou_loss := args.tot_iou_loss + (1-all_ious.iou);
                  tot_giou := tot_giou + all_ious.giou;
                  args.tot_giou_loss := args.tot_giou_loss + (1-all_ious.giou);
                  tot_diou := tot_diou + all_ious.diou;
                  tot_diou_loss := tot_diou_loss + (1-all_ious.diou);
                  tot_ciou := tot_ciou + all_ious.ciou;
                  tot_ciou_loss := tot_ciou_loss + (1-all_ious.ciou);
                  obj_index := entryIndex(b, mask_n * self.w * self.h+j * self.w+i, 4);
                  avg_obj := avg_obj + self.output.Data[obj_index];
                  if self.objectnessSmooth then
                      begin
                          delta_obj := class_multiplier * self.objNormalizer * (1-self.output.Data[obj_index]);
                          if self.delta.Data[obj_index] = 0 then
                              self.delta.Data[obj_index] := delta_obj
                      end
                  else
                      self.delta.Data[obj_index] := class_multiplier * self.objNormalizer * (1-self.output.Data[obj_index]);
                  class_index := entryIndex(b, mask_n * self.w * self.h+j * self.w+i, 4+1);
                  deltaClass(class_index, class_id, stride,  @avg_cat);
                  inc(args.count);
                  inc(args.class_count);
                  if all_ious.iou > 0.5 then
                      recall := recall + 1;
                  if all_ious.iou > 0.75 then
                      recall75 := recall75 + 1
              end;
          for _n := 0 to self.total -1 do
              begin
                  mask_n := self.mask.indexOf(_n);
                  if (mask_n >= 0) and (_n <> best_n) and (self.iouThresh < 1.0) then
                      begin
                          pred := Default(TBox);
                          pred.w := self.biases.data[2 * _n] / netWidth;
                          pred.h := self.biases.data[2 * _n+1] / netHeight;
                          iou := pred.iouKind(truthShift, self.iouThreshKind);
                          if iou > self.iouThresh then
                              begin
                                  class_id := trunc(state.truth.Data[t * self.truthSize + b*self.truths + 4]);
                                  if assigned(self.map.Data) then
                                      class_id := self.map.Data[class_id];
                                  box_index := entryIndex(b, mask_n * stride + j * self.w+i, 0);
                                  if assigned(self.classesMultipliers.Data) then
                                      class_multiplier := self.classesMultipliers.Data[class_id]
                                  else
                                      class_multiplier := 1.0;
                                  all_ious := deltaBox(truth, _n, box_index, i, j, netWidth, netHeight, stride, self.iouNormalizer * class_multiplier, self.iouLoss, true, net.rewrittenBBox);
                                  inc(net.totalBBox);
                                  args.tot_iou := args.tot_iou + all_ious.iou;
                                  args.tot_iou_loss := args.tot_iou_loss + (1-all_ious.iou);
                                  tot_giou := tot_giou + all_ious.giou;
                                  args.tot_giou_loss := args.tot_giou_loss + (1-all_ious.giou);
                                  tot_diou := tot_diou + all_ious.diou;
                                  tot_diou_loss := tot_diou_loss + (1-all_ious.diou);
                                  tot_ciou := tot_ciou + all_ious.ciou;
                                  tot_ciou_loss := tot_ciou_loss + (1-all_ious.ciou);
                                  obj_index := entryIndex(b, mask_n * stride + j*self.w+i, 4);
                                  avg_obj := avg_obj + self.output.Data[obj_index];
                                  if self.objectnessSmooth then
                                      begin
                                          delta_obj := class_multiplier * self.objNormalizer * (1-self.output.Data[obj_index]);
                                          if self.delta.data[obj_index] = 0 then
                                              self.delta.data[obj_index] := delta_obj
                                      end
                                  else
                                      self.delta.data[obj_index] := class_multiplier * self.objNormalizer * (1-self.output.data[obj_index]);
                                  class_index := entryIndex(b, mask_n * stride+j * self.w+i, 4+1);
                                  deltaClass(class_index, class_id, stride,  @avg_cat);
                                  inc(args.count);
                                  inc(args.class_count);
                                  if all_ious.iou > 0.5 then
                                      recall := recall + 1;
                                  if all_ious.iou > 0.75 then
                                      recall75 := recall75 + 1
                              end
                      end
              end
      end;
  if self.iouThresh < 1.0 then
      for j := 0 to self.h -1 do
          for i := 0 to self.w -1 do
              for _n := 0 to self.n -1 do
                  begin
                      obj_index := entryIndex(b, _n*stride + j * self.w+i, 4);
                      box_index := entryIndex(b, _n*stride + j * self.w+i, 0);
                      class_index := entryIndex(b, _n*stride + j * self.w+i, 4+1);
                      //stride := self.w * self.h;
                      if self.delta.data[obj_index] <> 0 then
                          averageDeltas(class_index, box_index, stride)
                  end;
end;

procedure TYoloLayer.forward(var state: TNNetState);
var bbox_index , obj_index, iteration_num, b, _n, start_point: SizeInt;
    counter_reject, counter_all, i, progress_it, counter, num_deltas_per_anchor: SizeInt;
    net : TNNet;
    progress, ep_loss_threshold, cur_max, cur_avg , cur_std, rolling_std
      , rolling_max, rolling_avg, final_badlebels_threshold
      , cur_percent, progress_badlabels, badlabels_threshold, loss: Single;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(layerType);
  {$endif}

  state.input.copyTo(output);
  for b := 0 to batch -1 do
      for _n := 0 to n -1 do
          begin
              bbox_index := entryIndex(b, _n * w * h, 0);
              if not newCoords then
                  begin
                      activate_array(pointer(output.data + bbox_index), 2 * w * h, acLOGISTIC);
                      obj_index := entryIndex(b, _n * w * h, 4);
                      activate_array(pointer(output.Data + obj_index), (1+classes) * w * h, acLOGISTIC)
                  end;
              //scal_add_cpu(2 * l.w * l.h, l.scale_x_y, -0.5 * (l.scale_x_y-1), l.output+bbox_index, 1)
              if scaleXY<>1 then
                  output.FusedMultiplyAdd(scaleXY, 0.5*(1-scaleXY), bbox_index, 2*w*h);
          end;

  if state.isTraining then begin

    delta.fill(0);

    labels.fill(-1);
    classIds.fill(-1);

    //tot_iou := 0;
    //tot_giou := 0;
    //tot_diou := 0;
    //tot_ciou := 0;
    //tot_iou_loss := 0;
    //tot_giou_loss := 0;
    //tot_diou_loss := 0;
    //tot_ciou_loss := 0;
    //recall := 0;
    //recall75 := 0;
    //avg_cat := 0;
    //avg_obj := 0;
    //avg_anyobj := 0;
    //count := 0;
    //class_count := 0;
    cost[0] := 0;
    for b := 0 to batch -1 do
        begin
            //ThreadArgs[b].l := self;
            ThreadArgs[b].state := @state;
            ThreadArgs[b].b := b;
            ThreadArgs[b].tot_iou := 0;
            ThreadArgs[b].tot_iou_loss := 0;
            ThreadArgs[b].tot_giou_loss := 0;
            ThreadArgs[b].count := 0;
            ThreadArgs[b].class_count := 0;
            threads[b] := ExecuteInThread(processBatch, @threadArgs[b]);
            //if pthread_create( and threads[b], 0, process_batch,  and (yolo_args[b])) then
                //error('Thread creation failed', DARKNET_LOC)
        end;
    for b := 0 to batch -1 do
        begin
            //pthread_join(threads[b], 0);
            threads[b].WaitFor;
            //tot_iou       := tot_iou       + threadArgs[b].tot_iou;
            //tot_iou_loss  := tot_iou_loss  + threadArgs[b].tot_iou_loss;
            //tot_giou_loss := tot_giou_loss + threadArgs[b].tot_giou_loss;
            //count         := count         + threadArgs[b].count;
            //class_count   := class_count   + threadArgs[b].class_count
        end;
    //free(yolo_args);
    //free(threads);
    net := TNNet(state.net);
    iteration_num := net.currentIteration;//get_current_iteration(state.net);
    start_point   := net.maxBatches * 3 div 4;
    if ((net.badlabelsrejectionpercentage<>0) and (start_point < iteration_num)) or ((net.numSigmasRejectBadlabels<>0) and (start_point < iteration_num)) or ((net.equiDistantPoint<>0) and (net.equiDistantPoint < iteration_num)) then
        begin
            progress_it := iteration_num - net.equiDistantPoint;
            progress := progress_it / (net.maxBatches - net.equiDistantPoint);
            ep_loss_threshold := (net.deltaRollingAvg) * progress * 1.4;
            cur_max := 0;
            cur_avg := 0;
            counter := 0;
            //for i := 0 to batch * outputs -1 do
            //    if delta.data[i] <> 0 then
            //        begin
            //            inc(counter);
            //            cur_avg := cur_avg + abs(delta.data[i]);
            //            if cur_max < abs(delta.data[i]) then
            //                cur_max := abs(delta.data[i])
            //        end;
            //cur_avg := cur_avg / counter;
            counter := delta.countNotValue(0);
            cur_max := delta.maxAbs();
            cur_avg := delta.sumAbs() / counter;
            if net.deltaRollingMax = 0 then
                net.deltaRollingMax := cur_max;
            net.deltaRollingMax := net.deltaRollingMax * 0.99 + cur_max * 0.01;
            net.deltaRollingAvg := net.deltaRollingAvg * 0.99 + cur_avg * 0.01;
            if (net.numSigmasRejectBadlabels<>0) and (start_point < iteration_num) then
                begin
                    rolling_std := net.deltaRollingStdDev;
                    rolling_max := net.deltaRollingMax;
                    rolling_avg := net.deltaRollingAvg;
                    progress_badlabels := (iteration_num-start_point) / (start_point);
                    cur_std := 0;
                    counter := 0;
                    //for i := 0 to batch * outputs -1 do
                    //    if delta[i] <> 0 then
                    //        begin
                    //            inc(counter);
                    //            cur_std := cur_std + sqr(delta[i]-rolling_avg{, 2})
                    //        end;
                    cur_std := delta.sumSqrDiff(rolling_avg);
                    cur_std := sqrt(cur_std / counter);
                    net.deltaRollingStdDev := net.deltaRollingStdDev * 0.99+cur_std * 0.01;
                    final_badlebels_threshold := rolling_avg + rolling_std*net.numSigmasRejectBadlabels;
                    badlabels_threshold := rolling_max - progress_badlabels * abs(rolling_max - final_badlebels_threshold);
                    badlabels_threshold := max(final_badlebels_threshold, badlabels_threshold);
                    //for i := 0 to delta.size() do
                    //    if abs(delta.data[i]) > badlabels_threshold then
                    //        delta.data[i] := 0;
                    delta.absThreshold(badlabels_threshold, @delta.Zero);
                    //writeln(format(' rolling_std = %f, rolling_max = %f, rolling_avg = %f ', [rolling_std, rolling_max, rolling_avg]));
                    //writeln(format(' badlabels loss_threshold = %f, start_it = %d, progress = %f ', [badlabels_threshold, start_point, progress_badlabels * 100]));
                    ep_loss_threshold := min(final_badlebels_threshold, rolling_avg) * progress
                end;
            if (net.badLabelsRejectionPercentage<>0) and (start_point < iteration_num) then
                begin
                    if net.rejectThreshold = 0 then
                        net.rejectThreshold := net.deltaRollingAvg;
                    //writeln(' badlabels_reject_threshold = %f ', net.badlabels_reject_threshold[0]);
                    num_deltas_per_anchor := (classes+4+1);
                    //counter_reject := 0;
                    //counter_all := 0;
                    //for i := 0 to delta.size() -1 do
                    //    if delta.data[i] <> 0 then
                    //        begin
                    //            inc(counter_all);
                    //            if abs(delta.data[i]) > net.rejectThreshold then
                    //                begin
                    //                    inc(counter_reject);
                    //                    delta.data[i] := 0
                    //                end
                    //        end;
                    counter_all := delta.countNotValue(0);
                    counter_reject:= delta.absThreshold(net.rejectThreshold, @delta.zero);
                    cur_percent := 100 * (counter_reject * num_deltas_per_anchor / counter_all);
                    if cur_percent > net.badLabelsRejectionPercentage then
                        begin
                            net.rejectThreshold := net.rejectThreshold + 0.01;
                            //writeln(' increase!!! ')
                        end
                    else
                        if  net.rejectThreshold > 0.01 then
                            begin
                                net.rejectThreshold := net.rejectThreshold - 0.01;
                                //writeln(' decrease!!! ')
                            end;
                    //writeln(format(' badlabels_reject_threshold = %f, cur_percent = %f, badlabels_rejection_percentage = %f, delta_rolling_max = %f ', [state.net.badlabels_reject_threshold[0], cur_percent, state.net.badlabels_rejection_percentage, state.net.delta_rolling_max[0]]))
                end;
            if (net.equiDistantPoint<>0) and (net.equiDistantPoint < iteration_num) then
                begin
                    //writeln(format(' equidistant_point loss_threshold = %f, start_it = %d, progress = %3.1f %% ', [ep_loss_threshold, state.net.equidistant_point, progress * 100]));
                    //for i := 0 to delta.size()-1 do
                    //    if abs(delta.data[i]) < ep_loss_threshold then
                    //        delta.data[i] := 0
                   delta.absThreshold(ep_loss_threshold,nil, @delta.zero);
                end
        end;
    //if count = 0 then
    //    count := 1;
    //if class_count = 0 then
    //    class_count := 1;
    if showDetails then
        begin
            //loss := sqr(mag_array(l.delta, l.outputs * l.batch){, 2});
            loss :=  delta.sumSquares();
            cost[0] := loss;
            //loss := loss / batch;
            //writeln(ErrOutput, format('v3 (%s loss, Normalizer: (iou: %.2f, obj: %.2f, cls: %.2f) Region %d Avg (IOU: %f), count: %d, total_loss = %f ', [(ifthen(iou_loss = ilMSE, 'mse', (ifthen(iou_loss = ilGIOU, 'giou', 'iou')))), iou_normalizer, obj_normalizer, cls_normalizer, state.index, tot_iou / count, count, loss]))
        end
  end;

  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.finish(layerType);
  {$endif}

end;

procedure TYoloLayer.backward(var state: TNNetState);
begin
  state.delta.add(delta)
end;

end.

