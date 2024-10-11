program MSCOCOYolo;
{$ifdef FPC}
{$mode Delphi}{$H+}
{$ModeSwitch typehelpers}
{$endif}
uses
  {$IFDEF UNIX}
  cthreads,
  {$ENDIF}
  SysUtils, math, ntensors, ntypes, nDatasets, nBaseLayer, nConnectedlayer,
  nLogisticLayer, nSoftmaxLayer, nCostLayer, nnet, nChrono, nConvolutionLayer,
  nModels, Keyboard, nparser
  , OpenCL, OpenCLHelper
  , ShellApi
  { you can add units after this };


const
    cfgFile = '../../../../../cfg/yolov3.cfg';
    weightFile = '../../../../../yolov3.weights';
    images :TStringArray = ['dog.jpg', 'person.jpg', 'eagle.jpg', 'giraffe.jpg', 'horses.jpg', 'kite.jpg', 'startrek1.jpg'];
    imageRoot = '../../../../../data/';
    classNamesFile = 'coco.names';
    scaleDownSteps = 4.0 ;

    colors: array [0..5,0..2] of single = ( (1,0,1), (0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,0,0) );

type
    TDetectionWithClass = record
        det:TDetection;
        // The most probable class id: the best class index in this->prob.
        // Is filled temporary when processing results, otherwise not initialized
        best_class: longint;
    end;

var

  darknet : TDarknetParser;
  i: SizeInt;
  cfg, c : string;
  t : clock_t;
  img , sized: TImageData;
  ImageTensor : TSingleTensor;
  detections : TDetections;
  classNames :  TArray<string>;
  l: TBaseLayer;
  //gDriver, gMode: SmallInt;

function get_color(const c, x, _max: SizeInt):single;
var
    ratio: single;
    i,j: SizeInt;
begin
    ratio := (x / _max) * 5;
    i := floor(ratio);
    j := ceil(ratio);
    ratio := ratio - i;
    result := (1-ratio) * colors[i][c]+ratio * colors[j][c];
end;

procedure swap(var a,b:SizeInt); overload;
var s:SizeInt;
begin
  s:=a;
  a:=b;
  b:=s
end;

procedure draw_line(const a: TImageData; x1, y1, x2, y2: SizeInt; const r, g, b: single; const alpha: single);
var x, y, w, h, idx:  SizeInt;
    aspect, v: single;
    ver, hor:boolean;
    c: Integer;
    rgb:array[0..2] of single;
begin
  rgb[0]:=r;rgb[1]:=g;rgb[2]:=b;
  if x1>x2 then swap(x1,x2);
  if y1>y2 then swap(x1,x2);
  x1:=EnsureRange(x1,0, a.w-1);
  y1:=EnsureRange(y1,0, a.h-1);
  x2:=EnsureRange(x2,0, a.w-1);
  y2:=EnsureRange(y2,0, a.h-1);

  w := x2 - x1;
  h := y2 - y1;
  ver := w=0;
  hor := h=0;


  if  hor then begin
    for c :=0 to 2 do
      for x:=x1 to x2 do begin
          idx :=c*a.h*a.w + y1*a.w + x;
          v:= a.data[idx];
          a.data[idx] := (rgb[c]-v)*alpha +v;
      end;
    exit
  end;

  if ver then begin
    for c :=0 to 2 do
      for y:=y1 to y2 do begin
          idx :=c*a.h*a.w + y*a.w + x1;
          v:= a.data[idx];
          a.data[idx] := (rgb[c]-v)*alpha +v;
      end;
    exit
  end;

  aspect := h/w;
  for c :=0 to 2 do
    for x:=x1 to x2 do begin
        y:= y1 + round((x-x1)*aspect);
        idx :=c*a.h*a.h + y*a.w + x;
        v:= a.data[idx];
        a.data[idx] := (rgb[c]-v)*alpha +v;
    end;

end;

procedure draw_box(const a: TImageData; x1, y1, x2, y2: SizeInt; const r, g, b: single; const alpha: single);
var
    i, c: SizeInt;
    r1, g1, b1, r2, g2, b2:single;
begin
    if alpha =0 then exit;

    draw_line(a, x1,y1,x2,y1, r, g, b, alpha);
    draw_line(a, x1,y2,x2,y2, r, g, b, alpha);

    draw_line(a, x1,y1,x1,y2, r, g, b, alpha);
    draw_line(a, x2,y1,x2,y2, r, g, b, alpha);


    //if (x1 < 0) then
    //    x1 := 0;
    //if (x1 >= a.w) then
    //    x1 := a.w-1;
    //if x2 < 0 then
    //    x2 := 0;
    //if x2 >= a.w then
    //    x2 := a.w-1;
    //if y1 < 0 then
    //    y1 := 0;
    //if y1 >= a.h then
    //    y1 := a.h-1;
    //if y2 < 0 then
    //    y2 := 0;
    //if y2 >= a.h then
    //    y2 := a.h-1;
    //for i := x1 to x2 do
    //    begin
    //        r1:=a.data[i+y1 * a.w+0 * a.w * a.h];
    //        g1:=a.data[i+y1 * a.w+1 * a.w * a.h];
    //        b1:=a.data[i+y1 * a.w+2 * a.w * a.h];
    //        a.data[i+y1 * a.w+0 * a.w * a.h] := (r - r1)* alpha + r1 ;
    //        a.data[i+y1 * a.w+1 * a.w * a.h] := (g - g1)* alpha + g1 ;
    //        a.data[i+y1 * a.w+2 * a.w * a.h] := (b - b1)* alpha + b1 ;
    //
    //        r2:=a.data[i+y2 * a.w+0 * a.w * a.h];
    //        g2:=a.data[i+y2 * a.w+1 * a.w * a.h];
    //        b2:=a.data[i+y2 * a.w+2 * a.w * a.h];
    //        a.data[i+y2 * a.w+0 * a.w * a.h] := (r - r2)* alpha + r2 ;
    //        a.data[i+y2 * a.w+1 * a.w * a.h] := (g - g2)* alpha + g2 ;
    //        a.data[i+y2 * a.w+2 * a.w * a.h] := (b - b2)* alpha + b2 ;
    //    end;
    //for i := y1 to y2 do
    //    begin
    //        r1:=a.data[x1+i * a.w+0 * a.w * a.h];
    //        r2:=a.data[x2+i * a.w+0 * a.w * a.h];
    //        g1:=a.data[x1+i * a.w+1 * a.w * a.h];
    //        g2:=a.data[x2+i * a.w+1 * a.w * a.h];
    //        b1:=a.data[x1+i * a.w+2 * a.w * a.h];
    //        b2:=a.data[x2+i * a.w+2 * a.w * a.h];
    //        a.data[x1+i * a.w+0 * a.w * a.h] := (r - r1)* alpha + r1 ;
    //        a.data[x2+i * a.w+0 * a.w * a.h] := (r - r2)* alpha + r2 ;
    //        a.data[x1+i * a.w+1 * a.w * a.h] := (g - g1)* alpha + g1 ;
    //        a.data[x2+i * a.w+1 * a.w * a.h] := (g - g2)* alpha + g2 ;
    //        a.data[x1+i * a.w+2 * a.w * a.h] := (b - b1)* alpha + b1 ;
    //        a.data[x2+i * a.w+2 * a.w * a.h] := (b - b2)* alpha + b2 ;
    //    end
end;

procedure draw_box_width(const a: TImageData; const x1, y1, x2, y2, w: SizeInt;
  const r, g, b: single; const alpha: single);
var
    i: SizeInt;
begin
    if alpha =0 then exit;
    for i := 0 to w -1 do
        draw_box(a, x1+i, y1+i, x2-i, y2-i, r, g, b, alpha)
end;

procedure draw_box_bw(const a: TImageData; x1, y1, x2, y2: SizeInt; const brightness: single);
var
    i: SizeInt;
begin
    if (x1 < 0) then
        x1 := 0;
    if (x1 >= a.w) then
        x1 := a.w-1;
    if x2 < 0 then
        x2 := 0;
    if x2 >= a.w then
        x2 := a.w-1;
    if y1 < 0 then
        y1 := 0;
    if y1 >= a.h then
        y1 := a.h-1;
    if y2 < 0 then
        y2 := 0;
    if y2 >= a.h then
        y2 := a.h-1;
    for i := x1 to x2 do
        begin
            a.data[i+y1 * a.w+0 * a.w * a.h] := brightness;
            a.data[i+y2 * a.w+0 * a.w * a.h] := brightness
        end;
    for i := y1 to y2 do
        begin
            a.data[x1+i * a.w+0 * a.w * a.h] := brightness;
            a.data[x2+i * a.w+0 * a.w * a.h] := brightness
        end
end;

procedure draw_box_width_bw(const a: TImageData; const x1, y1, x2, y2, w: SizeInt; const brightness: single);
var
    i: SizeInt;
    alternate_color: single;
begin
    for i := 0 to w -1 do
        begin
            if (w mod 2)<>0 then
                alternate_color := (brightness)
            else
                alternate_color := (1.0-brightness);
            draw_box_bw(a, x1+i, y1+i, x2-i, y2-i, alternate_color)
        end
end;

function get_actual_detections(const dets: TArray<TDetection>; const dets_num: SizeInt; const thresh: single; const selected_detections_num: PSizeInt; const names: TArray<string>):TArray<TDetectionWithClass>;
var
    selected_num: SizeInt;
    i: SizeInt;
    best_class: SizeInt;
    best_class_prob: single;
    j: SizeInt;
    show: boolean;
begin
    selected_num := 0;
    setLength(result, dets_num);
    for i := 0 to dets_num -1 do
        begin
            best_class := -1;
            best_class_prob := thresh;
            for j := 0 to dets[i].classes -1 do
                begin
                    show := names[j] <> 'dont_show';
                    if (dets[i].prob[j] > best_class_prob) and show then
                        begin
                            best_class := j;
                            best_class_prob := dets[i].prob[j]
                        end
                end;
            if best_class >= 0 then
                begin
                    result[selected_num].det := dets[i];
                    result[selected_num].best_class := best_class;
                    inc(selected_num)
                end
        end;
    if assigned(selected_detections_num) then
        selected_detections_num[0] := selected_num;
end;

function compare_by_lefts(const a, b: TDetectionWithClass):SizeInt;
var delta: single;
begin
    delta := (a.det.bbox.x-a.det.bbox.w / 2)-(b.det.bbox.x-b.det.bbox.w / 2);
    //exit(ifthen(delta < 0, -1, ifthen(delta > 0, 1, 0)))
end;

function compare_by_probs(const a, b: TDetectionWithClass):SizeInt;
var
    delta: single;
begin
    delta := a.det.prob[a.best_class]-b.det.prob[b.best_class];
    exit(ifthen(delta < 0, -1, ifthen(delta > 0, 1, 0)))
end;

procedure draw_detections_v3(const im: TImageData; const dets: TDetections; const num: SizeInt; const thresh: single; const names: TArray<string>;
  const alphabet: TArray<TArray<TImageData>>; const classes: SizeInt;
  labelAlpha: single; const ext_output: boolean);
var
    frame_id, selected_detections_num, i, best_class, j, width, offset: SizeInt;
    red, green, blue: single;
    rgb : array[0..2] of single;
    b: TBox;
    left, right, top, bot: SizeInt;
    labelstr, prob_str: string;
    &label, mask, resized_mask, tmask: TImageData;
    selected_detections : TArray<TDetectionWithClass>;
begin
    frame_id := 0;
    inc(frame_id);
    selected_detections := get_actual_detections(dets, num, thresh, @selected_detections_num, names);
    if selected_detections_num=0 then exit();
    //TTools<TDetectionWithClass>.QuickSort(pointer(selected_detections), 0, selected_detections_num-1, compare_by_lefts);
    //for i := 0 to selected_detections_num -1 do
    //    begin
    //        best_class := selected_detections[i].best_class;
    //        write(format('%s: %.0f%%', [names[best_class], selected_detections[i].det.prob[best_class] * 100]));
    //        if ext_output then
    //            writeln(format(#9'(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)', [
    //               (selected_detections[i].det.bbox.x-selected_detections[i].det.bbox.w / 2) * im.w,
    //               (selected_detections[i].det.bbox.y-selected_detections[i].det.bbox.h / 2) * im.h,
    //               selected_detections[i].det.bbox.w * im.w,
    //               selected_detections[i].det.bbox.h * im.h]))
    //        else
    //            writeln('');
    //        for j := 0 to classes -1 do
    //            if (selected_detections[i].det.prob[j] > thresh) and (j <> best_class) then
    //                begin
    //                    write(format('%s: %.0f%%', [names[j], selected_detections[i].det.prob[j] * 100]));
    //                    if ext_output then
    //                        writeln(format(#9'(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)',[
    //                             (selected_detections[i].det.bbox.x-selected_detections[i].det.bbox.w / 2) * im.w,
    //                             (selected_detections[i].det.bbox.y-selected_detections[i].det.bbox.h / 2) * im.h,
    //                             selected_detections[i].det.bbox.w * im.w,
    //                             selected_detections[i].det.bbox.h * im.h]))
    //                    else
    //                        writeln('')
    //                end
    //    end;
    TTools<TDetectionWithClass>.QuickSort(pointer(selected_detections), 0, selected_detections_num-1, compare_by_probs);
    for i := 0 to selected_detections_num -1 do
        begin
            width := trunc(im.h * 0.004);
            if width < 1 then
                width := 1;
            offset := selected_detections[i].best_class * 123457 mod classes;
            red := get_color(2, offset, classes);
            green := get_color(1, offset, classes);
            blue := get_color(0, offset, classes);
            rgb[0] := red;
            rgb[1] := green;
            rgb[2] := blue;
            b := selected_detections[i].det.bbox;
            left  := trunc((b.x-b.w / 2) * im.w);
            right := trunc((b.x+b.w / 2) * im.w);
            top   := trunc((b.y-b.h / 2) * im.h);
            bot   := trunc((b.y+b.h / 2) * im.h);
            if left < 0 then
                left := 0;
            if right > im.w-1 then
                right := im.w-1;
            if top < 0 then
                top := 0;
            if bot > im.h-1 then
                bot := im.h-1;
            if im.c = 1 then
                draw_box_width_bw(im, left, top, right, bot, width, 0.8)
            else
                draw_box_width(im, left, top, right, bot, width, red, green, blue, labelAlpha);
            //if assigned(alphabet) then
            //    begin
            //        labelstr:='';
            //        labelstr:= labelstr + names[selected_detections[i].best_class];
            //        prob_str := format(': %.2f', [selected_detections[i].det.prob[selected_detections[i].best_class]]);
            //        labelstr := labelstr + prob_str;
            //        for j := 0 to classes -1 do
            //            if (selected_detections[i].det.prob[j] > thresh) and (j <> selected_detections[i].best_class) then
            //                    labelstr := labelstr + ', ' + names[j];
            //        &label := get_label_v3(alphabet, labelstr, trunc(im.h * 0.02));
            //        draw_weighted_label(im, top + width, left, &label, @rgb[0], labelAlpha);
            //        free_image(&label)
            //    end;
            //if assigned(selected_detections[i].det.mask) then
            //    begin
            //        mask := float_to_image(14, 14, 1, @selected_detections[i].det.mask[0]);
            //        resized_mask := resize_image(mask, trunc(b.w * im.w), trunc(b.h * im.h));
            //        tmask := threshold_image(resized_mask, 0.5);
            //        embed_image(tmask, im, left, top);
            //        free_image(mask);
            //        free_image(resized_mask);
            //        free_image(tmask)
            //    end
        end;
    //free(selected_detections)
end;


procedure doForward(var state :TNNetState);
var
  img: TSingleTensor;
  l:TBaseLayer;
  c:string;
  i:sizeInt;
begin
  //write(#$1B'[1J'#$1B'[1H');
  l := TNNet(state.net).layers[state.index];
  writeln(state.index:3, ' : ', 100*state.index/darknet.Neural.layerCount():1:1,'%', ' ', l.LayerName);
  //l.output.printStat();
  //repeat
  //  writeln('Enter index to Interrogate Output, or [Enter] for next layer:');
  //  readln(C);
  //  if TryStrToInt64(c, i) then
  //      writeln(l.output.Data[i]:1:4);
  //
  //
  //until C='';

end;

const thresh = 0.5;
    NMS =0.45;
    M = 10;
    N = 20;
    K =30;
var
  ocl  : TOpenCL;
  a, b: TSingleTensor;
  kernel : cl_kernel;
  off, gws, lws : TArray<size_t>;
  AA, BB, CC: cl_mem;
  NN ,R: Integer;

  conv : TConvolutionalLayer;
  state:TNNetState;

begin
  write(#$1B'[1J');

  img.loadFromFile(imageRoot+images[0]);
  a := img.toTensor();


  //a.SaveToImage(GetCurrentDir+PathDelim+'tmp.jpg');
  //ShellExecute(0,'open',PChar( GetCurrentDir+PathDelim+'tmp.jpg'),'','', 5);
  //readln;


  //conv:= TConvolutionalLayer.Create(1, a.h, a.w, a.c, 6, 1, 3);
  //conv.biases.fill(0);
  //state.input := a;
  //conv.ActivationType:=acLINEAR;
  //state.workspace.resize([conv.workspaceSize*10]);
  //
  //conv.forward(state);
  ////writeln('A');a.printStat;
  //DeleteFile(GetCurrentDir+PathDelim+'tmp.jpg');
  //conv.output.SaveToImage(GetCurrentDir+PathDelim+'tmp.jpg');
  //ShellExecute(0,'open',PChar( GetCurrentDir+PathDelim+'tmp.jpg'),'','', 5);
  //writeln('outputs :'); conv.output.printStat();
  //readln;
  //
  //b.reSize([1, 9*3, a.h(), a.w()]);
  //a.Conv2D(conv.weights, b);
  //writeLn('outputs 2');b.printStat();
  //DeleteFile(GetCurrentDir+PathDelim+'tmp.jpg');
  //b.SaveToImage(GetCurrentDir+PathDelim+'tmp.jpg');
  //ShellExecute(0,'open',PChar( GetCurrentDir+PathDelim+'tmp.jpg'),'','', 5);
  //readln;

  //a.im2Col(conv.weights.w(), conv.weights.w(), 1, 1, 1, 1, 1, 1, b.data);
  //b.SaveToImage(GetCurrentDir+PathDelim+'tmp.jpg');
  //ShellExecute(0,'open',PChar( GetCurrentDir+PathDelim+'tmp.jpg'),'','', 5);
  //readln;
  //exit;

  cfg := GetCurrentDir + PathDelim + cfgfile;
  if not FileExists(cfg) then begin
    writeln('File [',cfg,'] doesn''t exist!');
    readln();
  end;






  t := clock;
  darknet := TDarknetParser.Create(cfg, 1, 1);
  writeln('Model : ',cfg,' [',(clock()-t)/CLOCKS_PER_SEC:1:3,'] Seconds.');

  t := clock;
  darknet.loadWeights(weightFile);
  writeln('Weights : ', weightFile,' [',(clock()-t)/CLOCKS_PER_SEC:1:3,'] Seconds.');
  readln();

  classNames := fromFile(imageRoot+classNamesFile);



  darknet.Neural.OnForward := doForward();
  darknet.Neural.fuseBatchNorm;
  i:=0;
  benchmark := true;
  repeat
    //write(#$1B'[1J');

    img.loadFromFile(imageRoot+images[i]);
    sized := img.letterBox(darknet.Neural.input.w(), darknet.Neural.input.h());

    metrics.reset;
    t := clock;
    darknet.Neural.predict(sized.toTensor);
    writeln('Inference : [',(clock()-t)/CLOCKS_PER_SEC:1:3,'] Seconds.');
    t := clock;
    detections := darknet.Neural.Detections(img.w, img.h, thresh);
    writeln('Detection : [', length(detections),'] took [', (clock()-t)/CLOCKS_PER_SEC:1:3,'] Seconds.');
    t := clock;
    detections.doNMSSort(darknet.Neural.classCount(), NMS);
    writeln('Sorting : [', length(detections), '] took [',(clock()-t)/CLOCKS_PER_SEC:1:3,'] Seconds.');

    t := clock;
    draw_detections_v3(img, detections, length(detections), thresh, classNames, nil, length(classNames), 0.6, false);
    writeln('Drawing : [',(clock()-t)/CLOCKS_PER_SEC:1:3,'] Seconds.');

    writeln('Metrics :', sLineBreak, metrics.print);

    //img.toTensor.print(0.3);
    if not DeleteFile(GetCurrentDir+PathDelim+'tmp.jpg') then
        writeln('No result image, Saving...');
    img.saveToFile(GetCurrentDir+PathDelim+'tmp.jpg');
    ShellExecute(0,'open',PChar( GetCurrentDir+PathDelim+'tmp.jpg'),'','', 5);
    inc(i);
    if i>high(images) then i:=0;
    readLn(c)
  until lowerCase(c) ='Q';

  //writeln('workspace : [', length(darknet.Neural.workspace)*4/1000000:1:1,'] MB');
  writeln(sLineBreak);
  //ImageTensor.print(0.5);
  //canv := TFPImageCanvas.create(bmp);



(*  //*********************************
  gDriver  := vga;
  gMode    := VGAHi;
  SetDirectVideo(true);
  InitGraph(gDriver, gMode, '');
  //PutImage(1,1,ImageTensor.Data, 0);
  for i:=0 to $ff do
    for j :=0 to $fff do begin
      //SetColor(j div 4);
      //DirectPutPixel(j, i);
      PutPixel(j, i, j div 4);
    end;
  Closegraph;

*) //************************************
  darknet.free;

end.

