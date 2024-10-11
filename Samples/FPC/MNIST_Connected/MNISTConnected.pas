program MNISTConnected;
{$ifdef FPC}
{$mode Delphi}{$H+}
{$endif}
uses
  {$IFDEF UNIX}
  cthreads,
  {$ENDIF}
  SysUtils, ntensors, ntypes, nDatasets, nConnectedlayer, nLogisticLayer,
  nSoftmaxLayer, nCostLayer, nnet, nChrono, nConvolutionLayer, nModels, nActivation, keyboard
  { you can add units after this };

const
  READ_BATCH   : SizeInt = 32;
  READ_MEASURE : SizeInt = 128;
  READ_TEST    : SizeInt = 4;
var
  Neural:TNNet;
  MNIST : TMNISTData;
  t1, l1:TSingleTensor;
  i, j, k, l, epoch :SizeInt;
  Data : TData;
  cost :single;
  costDelta : single;
  s : clock_t;
  Predicted, Truth : TTensor<SizeInt>;
  output, actual : TSingleTensor;
  c : shortstring;


begin
  write(#$1B'[1J'#$1B'[0H');
  writeln('Press [ESC] when the accuracy is enough to test.');
  MNIST := TMNISTData.Create('');
  Neural:=TNNet.Create(
    //leNetMNIST
    simpleDenseMNIST
  );
  Neural.batch:= READ_BATCH;
  Neural.setTraining(true);
  Neural.learningRate:=0.001;
  Neural.momentum:=0.9;
  neural.decay:=0.0001;

  MNIST.load(READ_BATCH, 0);

  t1.resize([READ_BATCH, MNIST.IMAGE_RES, MNIST.IMAGE_RES], READ_BATCH);
  l1.reSize([READ_BATCH, MNIST.CLASS_COUNT], READ_BATCH);
  Data.X.reShape([READ_BATCH, MNIST.IMAGE_SIZE], READ_BATCH);
  Data.Y.reShape([READ_BATCH, MNIST.CLASS_COUNT], READ_BATCH);
  actual.reShape(Data.Y.Shape, Data.Y.groups);

  i         := 0;
  j         := 0;
  epoch     := 0;
  costDelta := 0;
  cost :=0 ;
  Randomize;
  //for i := 0 to MNIST.DATA_COUNT div READ_BATCH -1 do begin
  s := clock();
  predicted.resize([ READ_MEASURE * READ_BATCH]);
  truth.resize([ READ_MEASURE * READ_BATCH]);
  output := Neural.output();
  initKeyboard();
  while true do begin

    //i := random(MNIST.DATA_COUNT div READ_BATCH);

    //if not MNIST.read(i) then begin
    //  writeln('i = ',i);
    //  break;
    //end;
    MNIST.read(i);

    MNIST.TrainingData.toSingles(t1.Data);
    MNIST.TrainingLabels.toSingles(l1.Data);

    t1.Normalize();

    data.x.Data :=t1.Data;
    data.y.Data :=l1.Data;

    cost := cost + Neural.trainEpoch(Data);
    actual.Data := Pointer(Neural.truth);


    output.argMax(Predicted.data + (j mod READ_MEASURE) * READ_BATCH);
    actual.argMax(Truth.data + (j mod READ_MEASURE) * READ_BATCH);

    //writeln(#$1B'[4;0H', 'Press [ESC] to stop training...');

    if j mod READ_MEASURE = READ_MEASURE-1 then begin
      cost := cost / READ_MEASURE ;
      costDelta := costDelta - cost;
      s :=  READ_MEASURE * CLOCKS_PER_SEC div (clock - s);
      write(#$1B'[1H','Batch [',j:4,'][',i:5,'], Cost [',cost:1:8,']',widechar($2191 +2*ord(costDelta>0)),' speed [', s*READ_BATCH :5,'] Sample per second '
        ,'Accuracy [', 100*truth.similarity(predicted.Data):3:3,'%] '#13);
      writeln(sLineBreak, 'Predicted :'#$1B'[0J');
      //Predicted.print();
      output.print(psGray);
      writeln(sLineBreak, 'Truth :');
      //truth.print();
      actual.print(psGray);
      costDelta := cost;
      cost := 0;
      truth.fill(0);
      predicted.fill(0);

      if j > MNIST.DATA_COUNT * READ_BATCH then
      begin
        readln(c);
        if c = 'b' then break;
      end;
      s := clock()
    end;
    k := pollKeyEvent();
    if k <>0 then begin
      if GetKeyEventCode(K) = $11B then {escape was pressed}
        break;
    end;

    //{$ifdef MSWINDOWS}
    //if GetKeyState(VK_ESCAPE)<0 then begin
    //    break
    //end;
    //{$endif}
    inc(j);
    inc(i);
    if i = MNIST.DATA_COUNT div READ_BATCH then begin
       //writeln('epoch ', epoch, StringOfChar(' ', 90));
       inc(epoch);
       MNIST.reset;
       i := 0;
    end;
  end;

  doneKeyboard();
  // test prediction
  writeln(sLineBreak,' press [Enter] to test:');
  readln;
  MNIST.load(0, READ_TEST);

  t1.reSize([READ_TEST, MNIST.IMAGE_RES, MNIST.IMAGE_RES], READ_TEST);
  l1.reSize([READ_TEST, MNIST.CLASS_COUNT], READ_TEST);

  Neural.Batch := READ_TEST;
  Predicted.reSize([READ_TEST]);
  truth.reSize([READ_TEST]);


  while true do try
    i := random(MNIST.TEST_COUNT div READ_TEST);
    MNIST.read(0, i);
    MNIST.TestingData.toSingles(t1.Data);
    MNIST.TestingLabels.toSingles(l1.Data);
    t1.Normalize();
    t1.print(psGray, READ_TEST);

    writeln('truth');
    l1.argMax(truth.Data);
    truth.print;
    writeln(sLineBreak, 'Predicted');
    Neural.Input := t1;
//    Neural.input.reshape([READ_TEST, MNIST.IMAGE_SIZE], READ_TEST);
//    Neural.Input.printStat;
    Neural.predict(Neural.input).argMax(predicted.Data);
    predicted.print();

    writeln('Press [Enter] for next random digit, [Ctrl-C] to exit ...');
    readln(c);
    if LowerCase(c) = 'q' then break;
  except on E:Exception do
    writeln(E.Message)
  end;

  MNIST.free;
  Neural.free;

end.

