program CIFAR10Conv;
{$ifdef FPC}
{$mode Delphi}{$H+}
{$endif}
uses
  {$IFDEF UNIX}
  cthreads,
  {$ENDIF}
  SysUtils, ntensors, ntypes, nDatasets, nBaseLayer, nConnectedlayer,
  nLogisticLayer, nSoftmaxLayer, nCostLayer, nnet, nChrono, nConvolutionLayer,
  nModels, Keyboard, nNormalizationLayer, steroids
  {$if defined(MSWINDOWS)}
  , ShellApi, cudnn_graph, cudnn_adv, cudnn_ops, cudnn_cnn
  {$endif}
  {$if defined(USE_OPENCL)}
  {$endif}
  { you can add units after this };

const
  READ_BATCH   : SizeInt = 32;
  READ_MEASURE : SizeInt = 32;
  READ_TEST    : SizeInt = 3;
var
  Neural:TNNet;
  CF10 : TCIFAR10Data;
  i, j, k, l :SizeInt;
  Data : TData;
  cost :single;
  costDelta : single;
  s : clock_t;
  Predicted, Truth : TInt64Tensor;
  output  : PSingleTensor;
  sampled : TSingleTensor;
  c : shortstring;


procedure _conv2d(const src: PSingle; ker: PSingle; var dest: PSingle;
  const wSrc, hSrc, wKernel, hKernel, wPad, hPad, xStr, yStr, xDil,
  yDil: SizeInt);
var
  {kx, kw, }ky {,kh}, wp, hp, wDst, hDst, i, j: SizeInt;
  ker2, srcIM, dstIM:PSingle;
  acc:Single;
begin

  //kw := wKernel div 2;
  //kh := hKernel div 2;
  //kSize := wKernel * hKernel;
  wDst := wSrc div xStr + wPad*2 - wKernel + 1;
  hDst := hSrc div yStr + hPad*2 - hKernel + 1;
  wP := {kw} - wPad;
  hP := {kh} - hPad;
  ker := ker {+ kh*wKernel}{ + kw};
  for i := hPad to hDst - hPad -1 do begin
    dstIM := dest + i*wDst;
    for j := wPad to wDst - wPad-1 do begin
      acc := dstIM[j];
      for ky := 0{-kh} to hKernel-1{kh} do begin
        srcIM := src + (i*yStr + ky*yDil)*wSrc + j*xStr + hP*wSrc + wp;
        ker2 := ker + ky*wKernel;
        acc := acc + cblas_sdot(wKernel, ker2, 1, srcIm, xDil);
        //for kx := 0{-kw} to wKernel-1{kw} do
        //  acc :=  plus(acc , ker2[kx]*srcIM[kx*xDil]);
      end;
      dstIM[j] := acc
    end;
  end
end;

var
  img : TImageData;
  l1, t1, t2, t3, t4 : TSingleTensor;
  coor : TArray<SizeInt>;
  trainingHistory : TSingleTensor;
begin
  //write(#$1B'[1J');
{$ifdef USE_OPENCL}
  TSingleTensor.defaultDevice := cdOpenCL;
  initOpenCL(1);
  writeln('Using : ',  ocl.PlatformName(ocl.ActivePlatformId));
  writeln('  - Device            : ',  ocl.DeviceName(ocl.ActiveDeviceId));

  ocl.queueInOrder:=true;
  writeln('  - out of Order mode : ', not ocl.queueInOrder);
{$endif}
  sDigits := 6;

  //sleep(500);
  ////img.loadFromFile(['../../../../../data/dog.jpg', '../../../../../data/eagle.jpg'], 416, 416);
  //img.loadFromFile('../../../../../data/dog.jpg');
  //t1 := img.toTensor();
  //t1.printStat;
  //t1.im2Col(5, 5, 2, 2, 1, 1, 1, 1, t2);
  //t2.SaveToImage('tmp.bmp');
  //ShellExecute(0, 'open', 'tmp.bmp', '', '', 0);
  //readln;
  //t3.col2Im(5, 5, 2, 2, 1, 1, 1, 1, t2);
  //t3.printStat;
  //t3.SaveToImage('tmp.bmp');
  //ShellExecute(0, 'open', 'tmp.bmp', '', '', 0);
  //readln;
  //t2.pushToDevice;
  //ocl.fill(t3.size(), t3.devData, 0, 1);
  //ocl.col2im(3, t3.h, t3.w, 5, 5, 2, 2, 1, 1, 1, 1, t2.devData, 0, t3.devData, 0);
  //t3.pullFromDevice();
  //t3.printStat;
  //t3.SaveToImage('tmp.bmp');
  //ShellExecute(0, 'open', 'tmp.bmp', '', '', 0);
  //readln;
  //exit;

  {$ifdef USE_TELEMETRY}
  benchmark:=true;
  {$endif}
  CF10 := TCIFAR10Data.Create('');

  Neural:=TNNet.Create(leNetCIFAR10);
  Neural.setTraining(true);
  Neural.batch       := READ_BATCH;
  Neural.learningRate:= 0.001;
  Neural.momentum    := 0.9;
  neural.decay       := 0.0001;
  neural.policy      := lrpCOST;

  CF10.load(Neural.batch);

  t1.resize([Neural.batch, CF10.CHANNELS, CF10.IMAGE_RES, CF10.IMAGE_RES], Neural.batch);
  l1.reSize([Neural.batch, CF10.CLASS_COUNT], Neural.batch);
  Data.X.reShape([Neural.batch, CF10.IMAGE_SIZE], Neural.batch);
  Data.Y.reShape([Neural.batch, CF10.CLASS_COUNT], Neural.batch);
  sampled.reShape(Data.Y.Shape, Data.Y.groups);
  setLength(Neural.truth, Data.Y.Size());

  i         := 0;
  j         := 0;
  l         := 0;
  costDelta := 0;
  cost :=0 ;
  Randomize;

  s := clock();
  predicted.resize([Neural.batch]);
  truth.resize([Neural.batch]);
  InitKeyboard;
  while true do begin

    i := random(CF10.DATA_COUNT div Neural.batch)-1;

    if not CF10.read(i) then break;

    CF10.TrainingData.toSingles(t1.Data);
    CF10.TrainingLabels.toSingles(l1.Data);

    t1.maxNormalize(1);//FusedMultiplyAdd(1/128, -1);

    data.x.Data :=t1.Data;
    data.y.Data :=l1.Data;



    cost := cost + Neural.trainEpoch(Data);

    output := Neural.output();
    sampled.ShallowCopy(Neural.truth);

    k := PollKeyEvent;
    if keyPressed then
      Break;
    //writeln(#$1B'[4;0H', 'Press [ESC] to stop training...');

    if j mod READ_MEASURE = READ_MEASURE-1 then begin
      cost := cost / READ_MEASURE ;
      costDelta := costDelta - cost;
      s :=  READ_MEASURE * CLOCKS_PER_SEC div (clock - s);
      inc(l);

      output.pullFromDevice;
      output.argMax(Predicted.data);
      sampled.argMax(Truth.data);

      trainingHistory.resize([l]);
      trainingHistory.Data[l-1] := cost;
      write(#$1B'[1H');
      writeln('Batch [',j:4,'], epoch[',j*Neural.batch div CF10.DATA_COUNT:5,'], Cost [',cost:1:8,']',widechar($2191 +2*ord(costDelta>0)),' speed [', s*Neural.batch :5,'] Sample per second, '
        ,'Accuracy [', 100*truth.similarity(predicted.Data):3:2,'%], learningRate [',Neural.computeCurrentLearningRate:1:3,']', sLineBreak);
      //writeln('Conv[1] ');
      //Neural.layers[0].weights.print(true, 18);
      //Neural.layers[0].biases.print(true);
      //Neural.layers[0].output.print(true, 4);
      //writeln('Conv[3] ');
      //Neural.layers[3].weights.print(true, 3);
      //Neural.layers[2].output.print(psGray24, 16);

      //writeln(sLineBreak,'prediction:');
      //output.print(psGray24);
      //writeln(sLineBreak, 'truth:');
      //Sampled.print(psGray24);

      //write('Predicted:',#$1B'[1J');
      write('Predicted:',#$1B'[10D',#$1B'[B');
      coor := output.print(psGray24);
      write(#$1B'[',coor[1]+1,'A',#$1B'[',40,'C');

      write('Actual:',#$1B'[7D',#$1B'[B');
      coor := sampled.print(psGray24);
      write(#$1B'[',coor[1]+1,'A',#$1B'[',40,'C');

      coor := trainingHistory.plot;

      write(#$1B'[',24,';',1,'H');
      writeln(sLineBreak, metrics.print({TELEMETRY_OPS or }TELEMETRY_FWD or TELEMETRY_BWD or TELEMETRY_UPD));


      metrics.reset;
      //writeln(sLineBreak, 'Predicted :');
      //Predicted.print();
      //writeln(sLineBreak, 'Truth :');
      //truth.print();
      //t1.print(true, -1, 1);
      costDelta := cost;
      cost := 0;
      truth.fill(0);
      predicted.fill(0);

      //if j > CF10.DATA_COUNT * Neural.batch then
      //begin
      //  readln(c);
      //  if c = 'b' then break;
      //end;
      //inc(i);
      s := clock();
      inc(k)
    end;
    inc(j);
  end;
  DoneKeyboard;
  // test prediction
  writeln(sLineBreak,' press [Enter] to test:');
  readln;

  CF10.load(0, READ_TEST);

  t1.reSize([READ_TEST, CF10.CHANNELS, CF10.IMAGE_RES, CF10.IMAGE_RES], READ_TEST);
  l1.reSize([READ_TEST, CF10.CLASS_COUNT], READ_TEST);

  Predicted := TInt64Tensor.create([READ_TEST]);
  truth     := TInt64Tensor.create([READ_TEST]);
  Neural.Batch := READ_TEST;


  write(#$1B'[1J'#$1B'[1H');
  while true do try
    write(#$1B'[1H');
    i := random(CF10.TEST_COUNT div READ_TEST);
    CF10.read(-1, i);
    CF10.TestingData.toSingles(t1.Data);
    CF10.TestingLabels.toSingles(l1.Data);
    t1.normalize();//t1.FusedMultiplyAdd(1/127, -1);

    t1.print(psColor24, READ_TEST);

    writeln('truth');
    l1.argMax(truth.Data);
    truth.print;
    writeln(sLineBreak, 'Predicted');
    Neural.Input := t1;
    //Neural.input.reshape([READ_TEST, CF10.IMAGE_SIZE], READ_TEST);
    Neural.predict(Neural.input).argMax(predicted.Data);
    predicted.print();

    writeln('Press [Enter] for next random digit, [Ctrl-C] to exit ...');
    readln(c);
    if LowerCase(c) = 'q' then break;
  except on E:Exception do
    writeln(E.Message)
  end;

  CF10.free;
  Neural.free;

end.

