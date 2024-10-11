unit nMaxPoolLayer;

{$ifdef FPC}
{$mode Delphi}
{$endif}
{$pointermath on}

interface

uses
  SysUtils, math, nTypes, nTensors, nConvolutionLayer {$ifdef USE_MULTITHREADING}, steroids{$endif};

type

  { TMaxPoolLayer }

  TMaxPoolLayer=class(TBaseConvolutionalLayer)
    maxPoolDepth, outChannels   : SizeInt;
    avgPool                     : boolean;
    indexes                     : TArray<SizeInt>;
    constructor Create(const aBatch, aHeight, aWidth, aChannels,
      aKernelSize: SizeInt; aStride_x:SizeInt=0; aStride_y: SizeInt=0; const aPadding:sizeInt=-1;
      aMaxpool_depth:SizeInt = 0; aOutChannels : SizeInt = 0; const aAntialiasing: SizeInt=0; const isAvgPool :boolean = false;
      const ATrain: boolean = false);
    procedure prepareAntialiase(const aStrideX, AStrideY: SizeInt);
    procedure setBatch(ABatch: SizeInt); override;
    procedure setTrain(ATrain: boolean); override;
    procedure forwardMaxPool(var state : TNNetState);
    procedure forwardAvgPool(var state : TNNetState);
    procedure backwardMaxPool(var state : TNNetState);
    procedure backwardAvgPool(var state : TNNetState);
    procedure forward(var state: TNNetState); override;
    procedure backward(var state: TNNetState); override;
  end;

  { TLocalAvgPoolLayer }

  TLocalAvgPoolLayer=class(TMaxPoolLayer)
    constructor Create(const aBatch, aHeight, aWidth, aChannels,
      aKernelSize: SizeInt; aStride_x: SizeInt=0; aStride_y: SizeInt=0;
      const aPadding: sizeInt=-1; aMaxpool_depth: SizeInt=0;
      aOutChannels: SizeInt=0; const aAntialiasing: SizeInt=0;
      const isAvgPool: boolean=true; const ATrain: boolean=false);
  end;

implementation

{$ifdef USE_TELEMETRY}
  uses nBaseLayer;
{$endif}

{ TMaxPoolLayer }

constructor TMaxPoolLayer.Create(const aBatch, aHeight, aWidth, aChannels,
  aKernelSize: SizeInt; aStride_x: SizeInt; aStride_y: SizeInt;
  const aPadding: sizeInt; aMaxpool_depth: SizeInt; aOutChannels: SizeInt;
  const aAntialiasing: SizeInt; const isAvgPool: boolean; const ATrain: boolean);
var
    blur_stride_x, blur_stride_y: SizeInt;
begin
    inherited create;
    avgPool := isAvgPool;
    if avgPool then
        LayerType := ltLOCAL_AVGPOOL
    else
        layerType := ltMAXPOOL;
    FTrain := ATrain;
    batch := Abatch;
    h := aHeight;
    w := aWidth;
    c := aChannels;
    kernelSize := aKernelSize;

    if aStride_x=0 then
        aStride_x:=kernelSize;
    if aStride_y=0 then
        aStride_y:=kernelSize;

    blur_stride_x := aStride_x;
    blur_stride_y := aStride_y;
    antialiasing := aAntialiasing;
    if antialiasing<>0 then begin
        aStride_x := 1;
        aStride_y := 1;
        //stride := 1;
        stride_x := 1;
        stride_y := 1
    end;
    if aPadding>-1 then
        Padding := aPadding
    else
        Padding := kernelSize div 2;
    maxPoolDepth := aMaxpool_depth;
    outChannels := aOutchannels;
    if maxpoolDepth<>0 then
        begin
            assert(OutChannels>0, '[TMaxPpool.Create] : [out-channels] must be greater than Zero');
            outC := OutChannels;
            outH := aHeight;
            outW := aWidth
        end
    else
        begin
            outC := c;
            outH := (h+padding-kernelSize) div aStride_y+1;
            outW := (w+padding-kernelSize) div aStride_x+1
        end;
    outputs := outH * outW * outC;
    inputs := h * w * c;
    inputShape := [batch, c, h, w];
    //stride := stride_x;
    stride_x := aStride_x;
    stride_y := aStride_y;
    if train then
        begin
            if not AvgPool then
                setLength(indexes, outH * outW * outC * batch);
            delta := TSingleTensor.Create([batch, outC, outH, outW], batch);
        end;
    output := TSingleTensor.Create([batch, outC, outH, outW], batch);

    //result.bflops := (result.size * result.size * result.c * result.out_h * result.out_w) / 1000000000.0;
    //if avgpool then
    //    begin
    //        if stride_x = stride_y then
    //            writeln(ErrOutput, format('avg               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF', [size, size, stride_x, w, h, c, result.out_w, result.out_h, result.out_c, result.bflops]))
    //        else
    //            writeln(ErrOutput, format('avg              %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF', [size, size, stride_x, stride_y, w, h, c, result.out_w, result.out_h, result.out_c, result.bflops]))
    //    end
    //else
    //    begin
    //        if maxpool_depth<>0 then
    //            writeln(ErrOutput, format('max-depth         %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF', [size, size, stride_x, w, h, c, result.out_w, result.out_h, result.out_c, result.bflops]))
    //        else
    //            if stride_x = stride_y then
    //                writeln(ErrOutput, format('max               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF', [size, size, stride_x, w, h, c, result.out_w, result.out_h, result.out_c, result.bflops]))
    //        else
    //            writeln(ErrOutput, format('max              %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF', [size, size, stride_x, stride_y, w, h, c, result.out_w, result.out_h, result.out_c, result.bflops]))
    //    end;
    //if antialiasing<>0 then
    //    begin
    //        blur_size := 3;
    //        blur_pad := blur_size div 2;
    //        if antialiasing = 2 then
    //            begin
    //                blur_size := 2;
    //                blur_pad := 0
    //            end;
    //        inputLayer := TConvolutionLayer.Create(batch, 1, outH, outW, outC, outC, outC, blur_size, blur_stride_x, blur_stride_y, 1, blur_pad, acLINEAR, false, false, 1, 0, nil, 0, false, train);
    //        blur_nweights := outC * blur_size * blur_size;
    //        if blur_size = 2 then begin
    //            i := 0;
    //            while i < blur_nweights do begin
    //                inputLayer.weights.Data[i+0] := 1 / 4.0;
    //                inputLayer.weights.Data[i+1] := 1 / 4.0;
    //                inputLayer.weights.Data[i+2] := 1 / 4.0;
    //                inputLayer.weights.Data[i+3] := 1 / 4.0;
    //                i := i + (blur_size * blur_size)
    //            end
    //        end else begin
    //            i := 0;
    //            while i < blur_nweights do begin
    //                inputLayer.weights.Data[i+0] := 1 / 16.0;
    //                inputLayer.weights.Data[i+1] := 2 / 16.0;
    //                inputLayer.weights.Data[i+2] := 1 / 16.0;
    //                inputLayer.weights.Data[i+3] := 2 / 16.0;
    //                inputLayer.weights.Data[i+4] := 4 / 16.0;
    //                inputLayer.weights.Data[i+5] := 2 / 16.0;
    //                inputLayer.weights.Data[i+6] := 1 / 16.0;
    //                inputLayer.weights.Data[i+7] := 2 / 16.0;
    //                inputLayer.weights.Data[i+8] := 1 / 16.0;
    //                i := i + (blur_size * blur_size)
    //            end;
    //        end;
    //        inputLayer.biases.fill(0);
    //    end;
    prepareAntialiase(blur_stride_x, blur_stride_y);
end;

procedure TMaxPoolLayer.prepareAntialiase(const aStrideX, AStrideY: SizeInt);
var
  i, blur_size, blur_pad, blur_nweights: SizeInt;
begin
    if assigned(inputLayer) then
        freeAndNil(inputLayer);
    if antialiasing<>0 then
        begin
            blur_size := 3;
            blur_pad := blur_size div 2;
            if antialiasing = 2 then
                begin
                    blur_size := 2;
                    blur_pad := 0
                end;
            inputLayer := TConvolutionalLayer.Create(batch, {1,} outH, outW, outC, outC, outC, blur_size, aStrideX, aStrideX, 1, blur_pad, acLINEAR, false, false, 1, 0, nil, 0, false, train);
            blur_nweights := outC * blur_size * blur_size;
            if blur_size = 2 then begin
                i := 0;
                while i < blur_nweights do begin
                    inputLayer.weights.Data[i+0] := 1 / 4.0;
                    inputLayer.weights.Data[i+1] := 1 / 4.0;
                    inputLayer.weights.Data[i+2] := 1 / 4.0;
                    inputLayer.weights.Data[i+3] := 1 / 4.0;
                    i := i + (blur_size * blur_size)
                end
            end else begin
                i := 0;
                while i < blur_nweights do begin
                    inputLayer.weights.Data[i+0] := 1 / 16.0;
                    inputLayer.weights.Data[i+1] := 2 / 16.0;
                    inputLayer.weights.Data[i+2] := 1 / 16.0;
                    inputLayer.weights.Data[i+3] := 2 / 16.0;
                    inputLayer.weights.Data[i+4] := 4 / 16.0;
                    inputLayer.weights.Data[i+5] := 2 / 16.0;
                    inputLayer.weights.Data[i+6] := 1 / 16.0;
                    inputLayer.weights.Data[i+7] := 2 / 16.0;
                    inputLayer.weights.Data[i+8] := 1 / 16.0;
                    i := i + (blur_size * blur_size)
                end;
            end;
            inputLayer.biases.fill(0);
        end;

end;

procedure TMaxPoolLayer.setBatch(ABatch: SizeInt);
begin
  if ABatch=Batch then exit();
  Batch := ABatch;
  inputShape[0]  := batch;
  if train then
      begin
          if not AvgPool then
              setLength(indexes, batch * outC * outH * outW);
          delta.resize([batch, outC, outH, outW], batch);
      end;

  output.resize([batch, outC, outH, outW], batch);
  if antialiasing<>0 then
      inputLayer.setBatch(batch);

end;

procedure TMaxPoolLayer.setTrain(ATrain: boolean);
begin
  if ATrain = FTrain then exit;
  FTrain := ATrain;
  if FTrain then begin
      if not AvgPool then
          setLength(indexes, batch * outC * outH * outW );
      delta := TSingleTensor.Create([batch, outC, outH, outW], batch);
  end else begin
      if not AvgPool then
          setLength(indexes,0);
      delta.free
  end;
  if assigned(inputLayer) then
      prepareAntialiase(inputLayer.stride_x, inputLayer.stride_y);

end;

procedure maxpoolMP(const f,t:IntPtr; const p:pointer=nil);
var
  k, i, j, m, n, out_index, max_i, cur_h, cur_w, index
  , out_w, out_h, c, b, kernelSize, stride, w_offset, h_offset, w, h :SizeInt;
  src,dst:PSingle;
  indexes:PSizeInt;
  max, val: single;
  a:PMPParams absolute p;
begin
  out_w     := PSizeInt(a.A)^;
  out_h     := PSizeInt(a.B)^;
  c         := PSizeInt(a.C)^;
  b         := PSizeInt(a.D)^;
  kernelSize      := PSizeInt(a.E)^;
  stride    := PSizeInt(a.F)^;
  w_offset  := PSizeInt(a.G)^;
  h_offset  := PSizeInt(a.H)^;
  w         := PSizeInt(a.I)^;
  h         := PSizeInt(a.J)^;
  src       := a.K;
  dst       :=a.L;
  indexes  := a.M;

  for k := f to t do
      begin
          for i := 0 to out_h -1 do
              for j := 0 to out_w -1 do
                  begin
                      out_index := j+out_w * (i+out_h * (k+c * b));
                      max := -MaxSingle;
                      max_i := -1;
                      for n := 0 to kernelSize -1 do
                          for m := 0 to kernelSize -1 do
                              begin
                                  cur_h := h_offset+i * stride+n;
                                  cur_w := w_offset+j * stride+m;
                                  index := cur_w + w * (cur_h + h * (k+b * c));
                                  if (cur_h >= 0) and (cur_h < h) and (cur_w >= 0) and (cur_w < w) then
                                      val := src[index]
                                  else
                                      val := -MaxSingle;
                                  if (val > max) then begin
                                      max_i := index;
                                      max := val
                                   end;
                                  //if (val > max) then
                                  //    max_i := index;
                                  //else
                                  //    max_i := max_i;
                                  //if (val > max) then
                                  //    max := val
                                  //else
                                  //    max := max
                              end;
                      dst[out_index] := max;
                      if assigned(indexes) then
                          indexes[out_index] := max_i
                  end
      end
end;

procedure forward_maxpool_layer_avx(const src,dst: Psingle; const indexes: PSizeInt; const size, w, h, out_w, out_h, c, pad, stride, batch: SizeInt);
var
  b, w_offset, h_offset:SizeInt;
  a:TMPParams;
begin
    w_offset := -pad div 2;
    h_offset := -pad div 2;
    a.A := @out_w      ;
    a.B := @out_h      ;
    a.C := @c          ;
    a.D := @b          ;
    a.E := @size       ;
    a.F := @stride     ;
    a.G := @w_offset   ;
    a.H := @h_offset   ;
    a.I := @w          ;
    a.J := @h          ;
    a.K := src         ;
    a.L := dst         ;
    a.M := indexes     ;
    for b := 0 to batch -1 do begin
    {$if defined(USE_MULTITHREADING)}
        mp.&for(maxPoolMP, 0, c-1,@a)
    {$else}
        maxPoolMP(0, c-1, @a)
    {$endif}
    end;
end;

procedure TMaxPoolLayer.forwardMaxPool(var state: TNNetState);
var
    b, i, j, k, g, out_index, max_i, in_index: SizeInt;
    max, val: single;
    m, w_offset, h_offset, _n,
      // _h, _w, _c,
    cur_h, cur_w, index: SizeInt;
    s: TNNetState;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(layerType);
    {$endif}

    if maxpoolDepth<>0 then
        begin
            for b := 0 to batch -1 do
                for i := 0 to h -1 do
                    for j := 0 to w -1 do
                        for g := 0 to outC -1 do
                            begin
                                out_index := j+w * (i+h * (g+outC * b));
                                max := -MaxSingle;
                                max_i := -1;
                                k := g;
                                while k < c do begin
                                    in_index := j+w * (i+h * (k+c * b));
                                    val := state.input.Data[in_index];
                                    if (val > max) then begin
                                        max_i := in_index;
                                        max := val
                                    end;
                                    //if (val > max) then
                                    //    max_i := in_index;
                                    //else
                                    //    max_i := max_i;
                                    //if (val > max) then
                                    //    max := val;
                                    //else
                                    //    max := max;
                                    k := k + outC
                                end;
                                output.Data[out_index] := max;
                                if assigned(indexes) then
                                    indexes[out_index] := max_i
                            end;
            {$ifdef USE_TELEMETRY}
            if benchmark then metrics.forward.start(layerType);
            {$endif}
            exit()
        end;
    if not state.isTraining and (stride_x = stride_y) then
        forward_maxpool_layer_avx(state.input, output, Pointer(indexes), kernelSize, w, h, outW, outH, outC, Padding, stride, batch)
    else
        begin
            w_offset := -Padding div 2;
            h_offset := -Padding div 2;
            //_h := out_h;
            //_w := out_w;
            //_c := c;
            for b := 0 to batch -1 do
                for k := 0 to outC -1 do
                    for i := 0 to outH -1 do
                        for j := 0 to outW -1 do
                            begin
                                out_index := j+outW * (i+outH * (k+outC * b));
                                max := -MaxSingle;
                                max_i := -1;
                                for _n := 0 to kernelSize -1 do
                                    for m := 0 to kernelSize -1 do
                                        begin
                                            cur_h := h_offset+i * stride_y+_n;
                                            cur_w := w_offset+j * stride_x+m;
                                            index := cur_w+w * (cur_h+h * (k+b * outC));
                                            if (cur_h >= 0) and (cur_h < h) and (cur_w >= 0) and (cur_w < w) then
                                                val := state.input.Data[index]
                                            else
                                                val := -MaxSingle;
                                            if val > max then begin
                                              max_i := index;
                                              max := val
                                            end;
                                            //if (val > max) then
                                            //    max_i := index;
                                            //else
                                            //    max_i := max_i;
                                            //if (val > max) then
                                            //    max := val;
                                            //else
                                            //    max := max
                                        end;
                                output.Data[out_index] := max;
                                if assigned(indexes) then
                                    indexes[out_index] := max_i
                            end
        end;
    if antialiasing<>0 then
        begin
            s := default(TNNetState);
            s.isTraining := state.isTraining;
            s.workspace := state.workspace;
            s.net := state.net;
            s.input := output;
            inputLayer.forward(s);
            move(inputLayer.output.Data[0], output.Data[0], inputLayer.outputs * inputLayer.batch * sizeof(single))
        end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(layerType);
    {$endif}

end;

procedure TMaxPoolLayer.forwardAvgPool(var state: TNNetState);
var
    b, i, j, k, m, _n, w_offset, h_offset,
      //h, w, c,
      out_index, counter, cur_h, cur_w, index: SizeInt;
    avg: single;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(layerType);
    {$endif}
    w_offset := -Padding div 2;
    h_offset := -Padding div 2;
    //h := l.out_h;
    //w := l.out_w;
    //c := l.c;
    for b := 0 to batch -1 do
        for k := 0 to c -1 do
            for i := 0 to outH -1 do
                for j := 0 to outW -1 do
                    begin
                        out_index := j+outW * (i+outH * (k+c * b));
                        avg := 0;
                        counter := 0;
                        for _n := 0 to kernelSize -1 do
                            for m := 0 to kernelSize -1 do
                                begin
                                    cur_h := h_offset + i * stride_y +_n;
                                    cur_w := w_offset + j * stride_x + m;
                                    index := cur_w + w * (cur_h + h * (k+b * c));
                                    if (cur_h >= 0) and (cur_h < h) and (cur_w >= 0) and (cur_w < w) then
                                        begin
                                            inc(counter);
                                            avg := avg + state.input.data[index]
                                        end
                                end;
                        output.Data[out_index] := avg / counter
                    end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(layerType);
    {$endif}
end;

procedure TMaxPoolLayer.backwardMaxPool(var state: TNNetState);
var
    i, index: SizeInt;
begin
    //h := outH;
    //w := outW;
    //c := outC;
    for i := 0 to outC * outH * outW * Batch -1 do
        begin
            index := indexes[i];
            state.delta.Data[index] := state.delta.Data[index] + delta.Data[i]
        end
end;

procedure TMaxPoolLayer.backwardAvgPool(var state: TNNetState);
var
    b, i, j, k, m, _n, w_offset, h_offset,
      //h, w, c,
      out_index, cur_h, cur_w, index: SizeInt;
begin
    w_offset := -padding div 2;
    h_offset := -padding div 2;
    //h := l.out_h;
    //w := l.out_w;
    //c := l.c;
    for b := 0 to batch -1 do
        for k := 0 to c -1 do
            for i := 0 to outH -1 do
                for j := 0 to outW -1 do
                    begin
                        out_index := j+outW * (i+outH * (k+c * b));
                        for _n := 0 to kernelSize -1 do
                            for m := 0 to kernelSize -1 do
                                begin
                                    cur_h := h_offset+i * stride_y +_n;
                                    cur_w := w_offset+j * stride_x + m;
                                    index := cur_w + w * (cur_h + h * (k + b * c));
                                    if (cur_h >= 0) and (cur_h < h) and (cur_w >= 0) and (cur_w < w) then
                                        state.delta.data[index] := state.delta.data[index] + (delta.data[out_index] / (kernelSize * kernelSize))
                                end
                    end
end;

procedure TMaxPoolLayer.forward(var state: TNNetState);
begin
  if avgPool then
      forwardAvgPool(state)
  else
      forwardMaxPool(state);
end;

procedure TMaxPoolLayer.backward(var state: TNNetState);
begin
    if avgPool then
        backwardAvgPool(state)
    else
        backwardMaxPool(state);
end;

{ TLocalAvgPoolLayer }

constructor TLocalAvgPoolLayer.Create(const aBatch, aHeight, aWidth, aChannels,
  aKernelSize: SizeInt; aStride_x: SizeInt; aStride_y: SizeInt;
  const aPadding: sizeInt; aMaxpool_depth: SizeInt; aOutChannels: SizeInt;
  const aAntialiasing: SizeInt; const isAvgPool: boolean; const ATrain: boolean
  );
begin
  inherited;
end;

end.

