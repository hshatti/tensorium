unit nContrastiveLayer;
{$ifdef FPC}
{$mode Delphi}
{$endif}

interface

uses
  SysUtils, nTensors, nTypes, nBaseLayer, nConvolutionLayer, nYoloLayer;

type

  {$if not declared(TYoloLayer)}
     TContrastiveLayer=class;
     TYoloLayer = TContrastiveLayer;
  {$endif}

  { TContrastiveLayer }

  TContrastiveLayer=class(TBaseImageLayer)
    n, truths, contrastive_neg_max, max_boxes, classes, embeddingSize, steps : SizeInt;
    Labels, classIds : TTensor<SizeInt>;
    temperature, maxDelta, classNormalizer : single;
    cos_sim, exp_cos_sim, p_contrastive :TSingleTensor;
    Detection : boolean;
    yoloLayer: TYoloLayer;

    loss : TArray<single>;
    constructor Create(const ABatch, aWidth, aHeight, aChannels:SizeInt; aClasses:SizeInt; const aInputs: SizeInt; const aYoloLayer: TYoloLayer);
    procedure setBatch(ABatch: SizeInt); override;
    procedure setTrain(ATrain: boolean); override;
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
  end;

implementation
uses math;
{ TContrastiveLayer }

constructor TContrastiveLayer.Create(const ABatch, aWidth, aHeight,
  aChannels: SizeInt; aClasses: SizeInt; const aInputs: SizeInt;
  const aYoloLayer: TYoloLayer);
var
    step, max_contr_size: SizeInt;
begin
    layerType := ltCONTRASTIVE;
    batch := Abatch;
    inputs := AInputs;

    w := AWidth;
    h := AHeight;
    c := AChannels;
    temperature := 1;
    max_boxes := 0;
    if assigned(yoloLayer) then
        begin
            detection := true;
            max_boxes := yoloLayer.maxBoxes;
            labels := yoloLayer.labels;
            classIds := yoloLayer.classIds;
            n := yoloLayer.n;
            classes := yoloLayer.classes;
            embeddingSize := inputs div (n * h * w);
            truths := yoloLayer.truths;
            if embeddingSize <> yoloLayer.embeddingSize then
                raise Exception.create(format(' Error: [contrastive] embedding_size=%d isn''t equal to [yolo] embedding_size=%d. They should use the same [convolutional] layer ', [embeddingSize, yoloLayer.embeddingSize]));
            if inputs mod (n * h * w) <> 0 then
                writeln (' Warning: filters= number in the previous (embedding) layer isn''t divisable by number of anchors ', n)
        end
    else
        begin
            detection := false;
            labels.ReSize([batch]);
            n := 1;
            classes := AClasses;
            embeddingSize := c
        end;
    inputShape := [batch, inputs];;
    outputs := inputs;
    loss := [0];
    output := TSingleTensor.Create([batch, inputs], batch);
    delta := TSingleTensor.Create([batch, inputs], batch);
//    cost := TSingles.Create(1);
    step := batch * n * h * w;
    //cos_sim := nil;
    //exp_cos_sim := nil;
    //p_constrastive := nil;
    if not detection then
        begin
            cos_sim := TSingleTensor.Create([step * step]);
            exp_cos_sim := TSingleTensor.Create([step * step]);
            p_contrastive := TSingleTensor.Create([step * step])
        end;
end;

procedure TContrastiveLayer.setBatch(ABatch: SizeInt);
begin
    if ABatch = batch then exit();
    batch := ABatch;
    inputShape[0] := batch;
    output.resize([batch, inputs], batch);
    delta.resize([batch, inputs], batch);
end;

procedure TContrastiveLayer.setTrain(ATrain: boolean);
begin
  if FTrain = ATrain then exit();
  FTrain := ATrain
end;

type
  TContrastiveParams = record
    sim, exp_sim, P:single;
    i, j:SizeInt;
    time_step_i, time_step_j: SizeInt;
  end;

function cosine_similarity(const A, B: PSingle; const feature_size: SizeInt):single;
var
    mul, d_a, d_b, divider: single;
    i: SizeInt;
begin
    mul := 0.0; d_a := 0.0; d_b := 0.0;
    for i := 0 to feature_size -1 do
        begin
            mul := mul + (A[i] * B[i]);
            d_a := d_a + (A[i] * A[i]);
            d_b := d_b + (B[i] * B[i])
        end;
    divider := sqrt(d_a) * sqrt(d_b);
    if divider > 0 then
        exit(mul / divider);
    result := 0;
end;

function find_sim(const i, j: SizeInt; const contrast_p: TArray<TContrastiveParams>; const contrast_p_size:SizeInt): single;
var
    z: SizeInt;
begin
    for z := 0 to contrast_p_size-1 do
        if (contrast_p[z].i = i) and (contrast_p[z].j = j) then
            exit(contrast_p[z].sim);
    raise exception.Create(format(' Error: find_sim(): sim isn''t found: i = %zu, j = %zu, z = %zu ',[ i, j, z]));
end;

function P_contrastive_f(const i, l: SizeInt; const labels: PSizeInt; const z: TArray<TArray<single>>;
  const feature_size: SizeInt; const temperature: single; const contrast_p: TArray<TContrastiveParams>; const contrast_p_size:SizeInt): single;
var
    sim, numerator, denominator: single;
    k: SizeInt;
    cp: TContrastiveParams;
begin
    if (i = l) then
            raise exception.Create(format(' Error: in P_constrastive must be i != l, while i = %d, l = %d ', [i, l]));
    sim := find_sim(i, l, contrast_p, contrast_p_size);
    numerator := exp(sim / temperature);
    denominator := 0;
    for k := 0 to contrast_p_size-1 do
        begin
            cp := contrast_p[k];
            if (cp.i <> i) and (cp.j = l) then
                denominator := denominator + cp.exp_sim
        end;
    result := 0.9999;
    if denominator <> 0 then
        result := numerator / denominator;
    if result > 1 then
        result := 0.9999;
end;

function p_contrastive_f_det(const il: SizeInt; const labels: PSizeInt; const z: TArray<TArray<Single>>;
  const feature_size: SizeInt; const temperature: Single; const contrast_p: TArray<TContrastiveParams>; const contrast_p_size:SizeInt): Single;
var
    sim, numerator, denominator: single;
    i, j, k: SizeInt;
    cp: TContrastiveParams;
begin
    sim := contrast_p[il].sim;
    i := contrast_p[il].i;
    j := contrast_p[il].j;
    numerator := exp(sim / temperature);
    denominator := 0;
    for k := 0 to contrast_p_size-1 do
        begin
            cp := contrast_p[k];
            if (cp.i <> i) and (cp.j = j) then
                denominator := denominator + cp.exp_sim
        end;
    result := 0.9999;
    if denominator <> 0 then
        result := numerator / denominator;
    if result > 1 then
        result := 0.9999;
end;

function calc_P_contrastive(const i, l: SizeInt; const labels: PSizeInt; const num_of_samples: SizeInt; const z: TArray<TArray<single>>; const feature_size: SizeInt; const temperature: single; const cos_sim, exp_cos_sim: PSingle):single;
var
    numerator, denominator: single;
    k: SizeInt;
begin
    if i = l then
            raise exception.Create(format(' Error: in P_constrastive must be i != l, while i = %zu, l = %zu ', [i, l]));
    numerator := exp_cos_sim[i * num_of_samples+l];
    denominator := 0;
    for k := 0 to num_of_samples -1 do
        if k <> i then
            denominator := denominator + exp_cos_sim[k * num_of_samples+l];
    result := numerator / denominator;
end;

procedure grad_contrastive_loss_positive(const i: SizeInt; const labels: PSizeInt; const num_of_samples: SizeInt; const z: TArray<TArray<single>>; const feature_size: SizeInt; const temperature: single; const cos_sim: PSingle; const p_constrastive: PSingle; delta: PSingle; const wh: SizeInt);
var
    j, N, m, out_i: SizeInt;
    vec_len, mult, sim, P, d: single;
begin
    vec_len := sqrt(SumOfSquares(PSingle(z[i]), feature_size));
    N := 0;
    for j := 0 to num_of_samples -1 do
        if labels[i] = labels[j] then
            inc(N);
    if (N = 0) or (temperature = 0) or (vec_len = 0) then
        raise exception.Create(format(' Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f ', [N, temperature, vec_len]));
    mult := 1 / ((N-1) * temperature * vec_len);
    for j := 0 to num_of_samples -1 do
        if (i <> j) and (labels[i] = labels[j]) then
            begin
                sim := cos_sim[i * num_of_samples+j];
                P := p_constrastive[i * num_of_samples+j];
                for m := 0 to feature_size -1 do
                    begin
                        d := mult * (sim * z[i][m]-z[j][m]) * (1-P);
                        out_i := m * wh;
                        delta[out_i] := delta[out_i] - d
                    end
            end
end;

procedure grad_contrastive_loss_negative(const i: SizeInt; const labels: PSizeInt; const num_of_samples: SizeInt; const z: TArray<TArray<single>>; const feature_size: SizeInt; const temperature: single; const cos_sim: PSingle; const p_constrastive: PSingle; delta: PSingle; const wh: SizeInt);
var
    j, N, k, m, out_i: SizeInt;
    vec_len, mult, sim, P, d: single;
begin
    vec_len := sqrt(SumOfSquares(PSingle(z[i]), feature_size));
    N := 0;
    for j := 0 to num_of_samples -1 do
        if labels[i] = labels[j] then
            inc(N);
    if (N = 0) or (temperature = 0) or (vec_len = 0) then
            raise exception.Create(format(' Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f ', [N, temperature, vec_len]));
    mult := 1 / ((N-1) * temperature * vec_len);
    for j := 0 to num_of_samples -1 do
        if (i <> j) and (labels[i] = labels[j]) then
            begin
                for k := 0 to num_of_samples -1 do
                    if (k <> i) and (k <> j) and (labels[k] >= 0) then
                        begin
                            sim := cos_sim[i * num_of_samples+k];
                            P := p_constrastive[i * num_of_samples+k];
                            for m := 0 to feature_size -1 do
                                begin
                                    d := mult * (z[k][m]-sim * z[i][m]) * P;
                                    out_i := m * wh;
                                    delta[out_i] := delta[out_i] - d
                                end
                        end
            end
end;

function get_sim_P_index(const i, j: SizeInt; contrast_p: TArray<TContrastiveParams>;
  const contrast_p_size: SizeInt): SizeInt;
var
    z: SizeInt;
begin
    for z := 0 to contrast_p_size -1 do
        if (contrast_p[z].i = i) and (contrast_p[z].j = j) then
            break;
    if z = contrast_p_size then
        exit(-1);
    result := z
end;

procedure grad_contrastive_loss_positive_f(const i: SizeInt; const class_ids, labels: PSizeInt; const num_of_samples: SizeInt;
  const z: TArray<TArray<Single>>;
  const feature_size: SizeInt; const temperature: single; delta: PSingle; const wh: SizeInt;
  const contrast_p: TArray<TContrastiveParams>; const contrast_p_size: SizeInt);
var
    j, N, sim_P_i, m, out_i: SizeInt;
    vec_len, mult, sim, P, d: single;
begin
    vec_len := sqrt(sumOfSquares(PSingle(z[i]), feature_size));
    N := 0;
    for j := 0 to num_of_samples -1 do
        if (labels[i] = labels[j]) and (labels[i] >= 0) then
            inc(N);
    if (N = 0) or (temperature = 0) or (vec_len = 0) then
            raise exception.Create(format(' Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f, labels[i] = %d ', [N, temperature, vec_len, labels[i]]));
    mult := 1 / ((N-1) * temperature * vec_len);
    for j := 0 to num_of_samples -1 do
        if (i <> j) and (labels[i] = labels[j]) and (labels[i] >= 0) then
            begin
                sim_P_i := get_sim_P_index(i, j, contrast_p, contrast_p_size);
                if sim_P_i < 0 then
                    continue;
                sim := contrast_p[sim_P_i].sim;
                P := contrast_p[sim_P_i].P;
                for m := 0 to feature_size -1 do
                    begin
                        d := mult * (sim * z[i][m]-z[j][m]) * (1-P);
                        out_i := m * wh;
                        delta[out_i] := delta[out_i] - d
                    end
            end
end;

procedure grad_contrastive_loss_negative_f(const i: SizeInt; const class_ids, labels: PSizeInt; const num_of_samples: SizeInt;
  z: TArray<TArray<single>>;const feature_size: SizeInt; const temperature: single; delta: PSingle; const wh: SizeInt;
  const contrast_p: TArray<TContrastiveParams>; const contrast_p_size: SizeInt; const neg_max: SizeInt);
var
    j, N, neg_counter, k, sim_P_i, m, out_i: SizeInt;
    vec_len, mult, sim, P, d: single;
begin
    vec_len := sqrt(sumOfSquares(PSingle(z[i]), feature_size));
    N := 0;
    for j := 0 to num_of_samples -1 do
        if (labels[i] = labels[j]) and (labels[i] >= 0) then
            inc(N);
    if (N = 0) or (temperature = 0) or (vec_len = 0) then
            raise exception.Create(format(' Error: N == 0 || temperature == 0 || vec_len == 0. N=%f, temperature=%f, vec_len=%f, labels[i] = %d ', [N, temperature, vec_len, labels[i]]));
    mult := 1 / ((N-1) * temperature * vec_len);
    neg_counter := 0;
    for j := 0 to num_of_samples -1 do
        if (labels[i] >= 0) and (labels[i] = labels[j]) and (i <> j) then
            begin
                for k := 0 to num_of_samples -1 do
                    if (k <> i) and (k <> j) and (labels[k] <> labels[i]) and (class_ids[j] = class_ids[k]) then
                        begin
                            inc(neg_counter);
                            sim_P_i := get_sim_P_index(i, k, contrast_p, contrast_p_size);
                            if sim_P_i < 0 then
                                continue;
                            sim := contrast_p[sim_P_i].sim;
                            P := contrast_p[sim_P_i].P;
                            for m := 0 to feature_size -1 do
                                begin
                                    d := mult * (z[k][m]-sim * z[i][m]) * P;
                                    out_i := m * wh;
                                    delta[out_i] := delta[out_i] - d
                                end;
                            if neg_counter >= neg_max then
                                exit()
                        end
            end
end;

procedure TContrastiveLayer.forward(var state: TNNetState);
var
    truth_thresh, max_truth, truth_prob, exp_sim, sim, P: single;
    z: TArray<TArray<Single>>;
    max_sim_same, max_sim_diff : TSingleTensor;//TArray<Single>;
    contrast_p : TArray<TContrastiveParams>;
    mini_batch, b, _n, _w, _h, z_index, b2, n2, h2, w2,
      contrast_p_index, step, contrast_p_size,
      z_index2, time_step_i, time_step_j, i,
      good_sims, all_sims, same_sim, diff_sim, contr_size, max_contr_size,
      k,  q, bd, nd, hd, wd, delta_index, wh: SizeInt;
begin

    if not state.isTraining then
        exit();
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(layerType);
    {$endif}

    truth_thresh := state.label_smooth_eps;
    mini_batch := batch div steps;
    //fill_cpu(l.batch * l.inputs, 0, l.delta, 1);
    delta.Fill(0);
    if not detection then
        begin
            for b := 0 to batch -1 do
                begin
                    if state.adversarial then
                        labels.Data[b] := b mod 2
                    else
                        labels.Data[b] := b div 2
                end;
            for b := 0 to batch -1 do
                for _h := 0 to h -1 do
                    for _w := 0 to w -1 do
                        begin
                            max_truth := 0;
                            for _n := 0 to classes -1 do
                                begin
                                    truth_prob := state.truth.data[b * classes + _n];
                                    if truth_prob > truth_thresh then
                                        begin
                                            max_truth := truth_prob;
                                            labels.Data[b] := _n
                                        end
                                end
                        end
        end;
    setLength(z, batch * n * h * w);
    for b := 0 to batch -1 do
        for _n := 0 to n -1 do
            for _h := 0 to h -1 do
                for _w := 0 to w -1 do
                    begin
                        z_index := ((b*n + _n)*h +_h)*w + _w;
                        if labels.data[z_index] < 0 then
                            continue;
                        setLength(z[z_index], embeddingSize);
                        TensorUtils.get_embedding(state.input, w, h, c, embeddingSize, _w, _h, _n, b, pointer(z[z_index]))
                    end;
    contrast_p_index := 0;
    step := batch * n * h * w;
    contrast_p_size := step;
    if not detection then
        contrast_p_size := batch * batch;
    setLength(contrast_p, contrast_p_size);
    //setLength(max_sim_same, batch * inputs);
    //setLength(max_sim_diff, batch * inputs);
    //fill_cpu(batch * inputs, -10, @max_sim_same[0], 1);
    //fill_cpu(batch * inputs, -10, @max_sim_diff[0], 1);
    max_sim_same.resize([batch, inputs]);
    max_sim_diff.resize([batch, inputs]);
    max_sim_same.fill(-10);
    max_sim_diff.fill(-10);
    for b := 0 to batch -1 do
        for _n := 0 to n -1 do
            for _h := 0 to h -1 do
                for _w := 0 to w -1 do
                    begin
                        z_index := ((b*n + _n)*h + _h)*w + _w;
                        if labels.data[z_index] < 0 then
                            continue;
                        for b2 := 0 to batch -1 do
                            for n2 := 0 to n -1 do
                                for h2 := 0 to h -1 do
                                    for w2 := 0 to w -1 do
                                        begin
                                            z_index2 := ((b2 * n + n2) * h + h2) * w + w2;
                                            if labels.data[z_index2] < 0 then
                                                continue;
                                            if z_index = z_index2 then
                                                continue;
                                            if detection then
                                                if classIds.data[z_index] <> classIds.data[z_index2] then
                                                    continue;
                                            time_step_i := b div mini_batch;
                                            time_step_j := b2 div mini_batch;
                                            if time_step_i <> time_step_j then
                                                continue;
                                            step := batch * n * h * w;
                                            sim := cosine_similarity(@z[z_index][0], @z[z_index2][0], embeddingSize);
                                            exp_sim := exp(sim / temperature);
                                            if not detection then
                                                begin
                                                    cos_sim.data[z_index * step+z_index2] := sim;
                                                    exp_cos_sim.data[z_index * step+z_index2] := exp_sim
                                                end;
                                            if (labels.data[z_index] = labels.data[z_index2]) and (max_sim_same.data[z_index] < sim) then
                                                max_sim_same.data[z_index] := sim;
                                            if (labels.data[z_index] <> labels.data[z_index2]) and (max_sim_diff.data[z_index] < sim) then
                                                max_sim_diff.data[z_index] := sim;
                                            contrast_p[contrast_p_index].sim := sim;
                                            contrast_p[contrast_p_index].exp_sim := exp_sim;
                                            contrast_p[contrast_p_index].i := z_index;
                                            contrast_p[contrast_p_index].j := z_index2;
                                            contrast_p[contrast_p_index].time_step_i := time_step_i;
                                            contrast_p[contrast_p_index].time_step_j := time_step_j;
                                            inc(contrast_p_index);
                                            if (contrast_p_index+1) >= contrast_p_size then
                                                begin
                                                    contrast_p_size := contrast_p_index+1;
                                                    setLength(contrast_p , contrast_p_size)
                                                end;
                                            if (sim > 1.001) or (sim < -1.001) then
                                                writeln(format(' sim = %f, ', [sim]))
                                        end
                    end;
    good_sims := 0; all_sims := 0; same_sim := 0; diff_sim := 0;
    for i := 0 to batch * inputs -1 do
        if (max_sim_same.data[i] >= -1) and (max_sim_diff.data[i] >= -1) then
            begin
                if max_sim_same.data[i] >= -1 then
                    inc(same_sim);
                if max_sim_diff.data[i] >= -1 then
                    inc(diff_sim);
                inc(all_sims);
                if max_sim_diff.data[i] < max_sim_same.data[i] then
                    inc(good_sims)
            end;
    if all_sims > 0 then
        loss[0] := 100 * good_sims div all_sims
    else
         loss[0] := -1;
    writeln(format(' Contrast accuracy = %f %%, all = %d, good = %d, same = %d, diff = %d ', [loss[0], all_sims, good_sims, same_sim, diff_sim]));
    //free(max_sim_same);
    //free(max_sim_diff);
    contr_size := contrast_p_index;
    if detection then
        begin
{$ifdef GPU}
            max_contr_size := (l.max_boxes * l.batch) * (l.max_boxes * l.batch);
            if max_contr_size < contr_size then
                begin
                    writeln(format(' Error: too large number of bboxes: contr_size = %d > max_contr_size  = %d ', [contr_size, max_contr_size]));
                    Exception.Create('Error!')
                end;
            labels := nil;
            if contr_size > 2 then
                begin
                    cuda_push_array(single(l.contrast_p_gpu), single(contrast_p), contr_size * sizeof(contrastive_params) div 4);
                    P_constrastive_f_det_gpu(labels, l.embedding_size, l.temperature, l.contrast_p_gpu, contr_size);
                    cuda_pull_array(single(l.contrast_p_gpu), single(contrast_p), contr_size * sizeof(contrastive_params) div 4)
                end;
{$else}
            for k := 0 to contr_size -1 do
                contrast_p[k].P := P_contrastive_f_det(k, labels.Data, z, embeddingSize, temperature, contrast_p, contr_size)
{$endif}
        end
    else
        for b := 0 to batch -1 do
            for _n := 0 to n -1 do
                for _h := 0 to h -1 do
                    for _w := 0 to w -1 do
                        begin
                            z_index := ((b*n + _n)*h + _h)*w +_w;
                            if labels.data[z_index] < 0 then
                                continue;
                            for b2 := 0 to batch -1 do
                                for n2 := 0 to n -1 do
                                    for h2 := 0 to h -1 do
                                        for w2 := 0 to w -1 do
                                            begin
                                                z_index2 := ((b2*n + n2)*h + h2)*w + w2;
                                                if labels.data[z_index2] < 0 then
                                                    continue;
                                                if z_index = z_index2 then
                                                    continue;
                                                if detection then
                                                    if classIds.data[z_index] <> classIds.data[z_index2] then
                                                        continue;
                                                time_step_i := b div mini_batch;
                                                time_step_j := b2 div mini_batch;
                                                if time_step_i <> time_step_j then
                                                    continue;
                                                step := batch * n * h * w;
                                                P := -10;
                                                if detection then
                                                    P := P_contrastive_f(z_index, z_index2, labels.data, z, embeddingSize, temperature, pointer(contrast_p), contr_size)
                                                else
                                                    begin
                                                        P := calc_p_contrastive(z_index, z_index2, labels.data, step, z, embeddingSize, temperature, cos_sim, exp_cos_sim);
                                                        p_contrastive.Data[z_index * step+z_index2] := P
                                                    end;
                                                for q := 0 to contr_size -1 do
                                                    if (contrast_p[q].i = z_index) and (contrast_p[q].j = z_index2) then
                                                        begin
                                                            contrast_p[q].P := P;
                                                            break
                                                        end
                                            end
                        end;
    bd := 0;
    for bd := 0 to batch -1 do
        for nd := 0 to n -1 do
            for hd := 0 to h -1 do
                for wd := 0 to w -1 do
                    begin
                        z_index := ((bd * n + nd) * h + hd) * w + wd;
                        step := batch * n * h * w;
                        if labels.Data[z_index] < 0 then
                            continue;
                        delta_index := ((bd*n + nd)*h + hd)*embeddingSize*w + wd;
                        wh := w * h;
                        if detection then
                            begin
                                grad_contrastive_loss_positive_f(z_index, classIds.data, labels.data, step, z, embeddingSize, temperature, delta.Data + delta_index, wh, contrast_p, contr_size);
                                grad_contrastive_loss_negative_f(z_index, classIds.data, labels.data, step, z, embeddingSize, temperature, delta.Data + delta_index, wh, contrast_p, contr_size, contrastive_neg_max)
                            end
                        else
                            begin
                                grad_contrastive_loss_positive(z_index, labels.data, step, z, embeddingSize, temperature, cos_sim, p_contrastive, delta.Data + delta_index, wh);
                                grad_contrastive_loss_negative(z_index, labels.data, step, z, embeddingSize, temperature, cos_sim, p_contrastive, delta.Data + delta_index, wh)
                            end
                    end;

    delta.multiply(classNormalizer);
    //for i := 0 to l.inputs * l.batch -1 do
    //    l.delta[i] := clip_value(l.delta[i], l.maxDelta);
    delta.clamp(-maxDelta, maxDelta);
    //cost[0] := sqr(mag_array(l.delta, l.inputs * l.batch){, 2});
    cost[0] := delta.sumSquares();
    //if state.adversarial then
    //    writeln(format(' adversarial contrastive loss = %f '#10'', [l.cost[0]]))
    //else
    //    writeln(format(' contrastive loss = %f '#10'', [l.cost[0]]));
    //for b := 0 to batch -1 do
    //    for _n := 0 to n -1 do
    //        for _h := 0 to h -1 do
    //            for _w := 0 to w -1 do
    //                begin
    //                    z_index := ((b*n + _n)*h + _h)*w + _w;
    //                    //if z[z_index] then
    //                        //free(z[z_index])
    //                end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(layerType);
    {$endif}

    //free(contrast_p);
    //free(z)
end;

procedure TContrastiveLayer.backward(var state: TNNetState);
begin
    state.delta.add(delta)
end;

end.

