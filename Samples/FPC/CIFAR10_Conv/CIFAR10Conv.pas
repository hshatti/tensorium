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
  nModels, Keyboard, nNormalizationLayer

  { you can add units after this };

const
  READ_BATCH   : SizeInt = 32;
  READ_MEASURE : SizeInt = 32;
  READ_TEST    : SizeInt = 3;
var
  Neural:TNNet;
  CF10 : TCIFAR10Data;
  t1, l1:TSingleTensor;
  i, j, k, l :SizeInt;
  Data : TData;
  cost :single;
  costDelta : single;
  s : clock_t;
  Predicted, Truth : TInt64Tensor;
  output, sampled : TSingleTensor;
  c : shortstring;


begin
  write(#$1B'[1J');
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
  costDelta := 0;
  cost :=0 ;
  Randomize;

  s := clock();
  predicted.resize([ READ_MEASURE * Neural.batch]);
  truth.resize([ READ_MEASURE * Neural.batch]);
  InitKeyboard;
  while true do begin

    i := random(CF10.DATA_COUNT div Neural.batch)-1;

    if not CF10.read(i) then break;

    CF10.TrainingData.toSingles(t1.Data);
    CF10.TrainingLabels.toSingles(l1.Data);

    t1.Normalize();//FusedMultiplyAdd(1/128, -1);

    data.x.Data :=t1.Data;
    data.y.Data :=l1.Data;



    cost := cost + Neural.trainEpoch(Data);

    output := Neural.output();
    sampled.ShallowCopy(Neural.truth);

    output.argMax(Predicted.data + (j mod READ_MEASURE) * Neural.batch);
    sampled.argMax(Truth.data + (j mod READ_MEASURE) * Neural.batch);

    k := PollKeyEvent;
    if keyPressed then
      Break;
    //writeln(#$1B'[4;0H', 'Press [ESC] to stop training...');

    if j mod READ_MEASURE = READ_MEASURE-1 then begin
      cost := cost / READ_MEASURE ;
      costDelta := costDelta - cost;
      s :=  READ_MEASURE * CLOCKS_PER_SEC div (clock - s);
      write(#$1B'[1H');
      writeln('Batch [',j:4,'], epoch[',j*Neural.batch div CF10.DATA_COUNT:5,'], Cost [',cost:1:8,']',widechar($2191 +2*ord(costDelta>0)),' speed [', s*Neural.batch :5,'] Sample per second, '
        ,'Accuracy [', 100*truth.similarity(predicted.Data):3:2,'%], learningRate [',Neural.computeCurrentLearningRate:1:3,']');

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

      writeln('Predicted:'#$1B'[0J');
      output.print(psGray24);
      writeln('Actual:');
      sampled.print(psGray24);

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
      s := clock()
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

