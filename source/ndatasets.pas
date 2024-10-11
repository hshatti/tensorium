unit nDatasets;
{$ifdef FPC}
{$mode Delphi}{$H+}
{$endif}
interface

uses
  SysUtils, nTensors, nTypes;

type

  { TBaseData }

  TBaseData = class(TInterfacedObject)
  protected
    class function isOpen(var f:file):boolean;static;
  public
    FSources : TArray<string>;
    FPath    : string;
    trainingCount, testingCount, validationCount: SizeInt;
    TrainingData, TrainingLabels     :TByteTensor;
    ValidationData, ValidationLabels :TByteTensor;
    TestingData , TestingLabels      :TByteTensor;
    ClassNames                       : array of string;
    constructor Create(const APath:string; ASources:TArray<string> = nil);
    procedure load(ATrainingCount:SizeInt =-1; ATestCount:SizeInt = -1);virtual; abstract;
    function read(const ATrainingPosition:SizeInt = -1; const ATestingPosition:SizeInt = -1):boolean; virtual; abstract;
    //procedure loadRandom(ATrainingCount:SizeInt =-1; testingPercent:Single = -1);virtual; abstract;
  end;

  { TMNISTData }

  TMNISTData=class(TBaseData)
  const
      CHANNELS = 1;
      IMAGE_RES  = 28;
      IMAGE_SIZE = IMAGE_RES * IMAGE_RES;
      CLASS_COUNT = 10;
      DATA_COUNT = 60000;
      TEST_COUNT = 10000;
  private
    trImg, trLbl, tsImg, TsLbl: file;
    FIMGHeadSize, FLBLHeadSize :SizeInt;
  public
    procedure load(ATrainingCount, ATestCount: SizeInt); override;
    procedure reset();
    function read(const ATrainingPosition : SizeInt = -1; const ATestingPosition : SizeInt =-1):boolean; override;
    destructor Destroy; override;
  end;

  { TCIFAR10Data }

  TCIFAR10Data=class(TBaseData)
  public
    const
      CHANNELS = 3;
      IMAGE_RES  = 32;
      IMAGE_SIZE = IMAGE_RES * IMAGE_RES * CHANNELS;
      CLASS_COUNT = 10;
      DATA_COUNT = 50000;
      TEST_COUNT = 10000;
    type
      TCifar10Rec=packed record
        id:byte;
        Data:array[0..IMAGE_SIZE-1] of byte
      end;
      TCF = file of TCIFAR10Rec;
   protected
     class function isOpen(var f:TCF):boolean;static;
   private
      trImg : array[0..4] of TCF;
      tsImg : TCF;
    public
    procedure load(ATrainingCount: SizeInt=-1; ATestCount: SizeInt=-1); override;
    function read(const ATrainingPosition: SizeInt=-1; const ATestingPosition: SizeInt=-1): boolean; override;
    destructor Destroy; override;
  end;

implementation

{ TBaseData }

class function TBaseData.isOpen(var f: file): boolean;
begin
  {$I-}
  Seek(F,0);
  result := IOResult=0;
  {$I+}
end;

constructor TBaseData.Create(const APath: string; ASources: TArray<string>);
begin
  FPath := Apath;
  if trim(APath)='' then
    FPath := GetCurrentDir + PathDelim
  else
    FPath := APath;
  FSources := ASources;
end;

{ TMNISTData }

procedure TMNISTData.load(ATrainingCount, ATestCount: SizeInt);
var
   buf1:record n_magic, n_count:Int32; w, h:Int32 end;

begin
  if ATrainingCount < 0 then
    ATrainingCount := DATA_COUNT;
  trainingCount := ATrainingCount;;

  if ATestCount < 0 then
    ATestCount := TEST_COUNT;
  testingCount := ATestCount;

  if not Assigned(FSources) then begin
    setLength(FSources, 4);
    FSources[0]:=FPath + 'train-images-idx3-ubyte' ;
    FSources[1]:=FPath + 'train-labels-idx1-ubyte' ;
    FSources[2]:=FPath + 't10k-images-idx3-ubyte' ;
    FSources[3]:=FPath + 't10k-labels-idx1-ubyte' ;
  end;
  FIMGHeadSize    := SizeOf(Buf1);
  FLBLHeadSize    := 8;
  //if assigned(trImg) then begin
  //  trImg.free;
  //  trImg := nil
  //end;
  //if assigned(trLbl) then begin
  //  trLbl.free;
  //  trLbl := nil
  //end;
  //if assigned(tsImg) then begin
  //  tsImg.free;
  //  tsImg := nil
  //end;
  //if assigned(tsLbl) then begin
  //  tsLbl.free;
  //  tsLbl := nil
  //end;

  if isOpen(trImg) then
    closeFile(trImg);

  if isOpen(trLbl) then
    closeFile(trLbl);

  if isOpen(tsImg) then
    closeFile(tsImg);

  if isOpen(tsLbl) then
    closeFile(tsLbl);

  if (trainingCount>0) then begin

    assignFile(trImg, FSources[0]);
    assignFile(trLbl, FSources[1]);

    system.reset(trImg, 1);
    system.reset(trLbl, 1);

    Seek(trImg, FIMGHeadSize);
    Seek(trLbl, FLBLHeadSize);

    TrainingData     := TByteTensor.Create([trainingCount, IMAGE_RES, IMAGE_RES]);
    TrainingLabels   := TByteTensor.Create([trainingCount, CLASS_COUNT]);
  end;

  if (testingCount>0) then begin
    assignFile(tsImg, FSources[2]);
    assignFile(tsLbl, FSources[3]);

    system.reset(tsImg, 1);
    system.reset(tsLbl, 1);

    Seek(tsImg, FIMGHeadSize);
    Seek(tsLbl, FLBLHeadSize);

    TestingData     := TByteTensor.Create([testingCount, IMAGE_RES, IMAGE_RES]);
    TestingLabels   := TByteTensor.Create([testingCount, CLASS_COUNT]);
  end;

end;

procedure TMNISTData.reset();
begin
  {$I-}
  Seek(trImg, FIMGHeadSize);
  Seek(trLbl, FLBLHeadSize);
  Seek(tsImg, FIMGHeadSize);
  Seek(tsLbl, FLBLHeadSize)
  {$I+}
end;

function TMNISTData.read(const ATrainingPosition: SizeInt;
  const ATestingPosition: SizeInt): boolean;
var
   i : SizeInt;
   lbl:byte;
begin
  result := true;
  lbl := $ff;
  //if Assigned(trImg) then begin

  if isOpen(trImg) and (ATrainingPosition >= 0) then begin
    Seek(trImg, FIMGHeadSize + ATrainingPosition * trainingCount * IMAGE_SIZE);
    Seek(TrLbl, FLBLHeadSize + ATrainingPosition * trainingCount);

    blockRead(trImg, TrainingData.Data[0], trainingCount * IMAGE_SIZE);
    result := result and not EOF(trImg);
    TrainingLabels.Fill(0);
    for i:=0 to trainingCount-1 do begin
      blockRead(trLbl, lbl, 1);
      result := result and not EOF(trLbl);
      TrainingLabels.Data[i * CLASS_COUNT + lbl] := 1;
    end;
    if not result then
      exit;
  end;
  //end;

  //if not Assigned(tsImg) then exit;

  if isOpen(tsImg) and (ATestingPosition >= 0) then begin
    Seek(tsImg, FIMGHeadSize + ATestingPosition * testingCount * IMAGE_SIZE);
    Seek(tsLbl, FLBLHeadSize + ATestingPosition * testingCount);

    blockread(tsImg, TestingData.Data[0], testingCount * IMAGE_SIZE);
    result := result and not EOF(tsImg);
    TestingLabels.Fill(0);
    for i:=0 to testingCount -1 do begin
      BlockRead(tsLbl, lbl, 1);
      result := result and not EOF(tsLbl);
      TestingLabels.Data[i * CLASS_COUNT + lbl] := 1;
    end;
  end;

end;

destructor TMNISTData.Destroy;
begin
  if isOpen(trImg) then close(trImg);
  if isOpen(trLbl) then close(trLbl);
  if isOpen(tsImg) then close(tsImg);
  if isOpen(tsLbl) then close(tsLbl);
  inherited Destroy;
end;

{ TCIFAR10Data }


class function TCIFAR10Data.isOpen(var f: TCF): boolean;
begin
  {$I-}
  Seek(F,0);
  result := IOResult=0;
  {$I+}
end;

procedure TCIFAR10Data.load(ATrainingCount: SizeInt; ATestCount: SizeInt);
var batch:integer; CR:TCifar10Rec ;  fclassname: TextFile;
begin
  if ATrainingCount < 0 then
    ATrainingCount := DATA_COUNT;
  trainingCount := ATrainingCount;;

  if ATestCount < 0 then
    ATestCount := TEST_COUNT;
  testingCount := ATestCount;

  if not Assigned(FSources) then begin
    setLength(FSources,7);
    FSources[0]:=FPath+'data_batch_1.bin' ;
    FSources[1]:=FPath+'data_batch_2.bin' ;
    FSources[2]:=FPath+'data_batch_3.bin' ;
    FSources[3]:=FPath+'data_batch_4.bin' ;
    FSources[4]:=FPath+'data_batch_5.bin' ;
    FSources[5]:=FPath+'test_batch.bin' ;
    FSources[6]:=FPath+'batches.meta.txt' ;
  end;

  setLength(ClassNames,0);
  assignFile(fclassname, FSources[6]);
  reset(fclassname);
  while not EOF(fclassname) do begin
    setLength(ClassNames, Length(ClassNames)+1);
    readln(fclassname, ClassNames[high(ClassNames)]);
  end;
  closeFile(fclassname);

  for batch := 0 to 4 do
    if isOpen(trImg[batch]) then
      closeFile(trImg[batch]);

  if isOpen(tsImg) then
    closeFile(tsImg);

  if trainingCount>0 then begin
    for batch:=0 to 4 do begin
      assignFile(trImg[batch], FSources[batch]);
      reset(trImg[batch]);
    end;
    TrainingData    := TByteTensor.Create([trainingCount, CHANNELS, IMAGE_RES, IMAGE_RES]);
    TrainingLabels  := TByteTensor.Create([trainingCount, CLASS_COUNT]);
  end;

  if testingCount>0 then begin
    assignFile(tsImg, FSources[5]);
    reset(tsImg);
    TestingData   := TByteTensor.Create([testingCount, CHANNELS, IMAGE_RES, IMAGE_RES]);
    TestingLabels := TByteTensor.Create([testingCount, CLASS_COUNT]);
  end;

end;

function TCIFAR10Data.read(const ATrainingPosition: SizeInt;
  const ATestingPosition: SizeInt): boolean;
var
  i, index : SizeInt;
  CF:TCifar10Rec;

begin
  result := true;
  //if Assigned(trImg) then begin
  index := ATrainingPosition*trainingCount div 10000;
  if (ATrainingPosition >= 0) and isOpen(trImg[index]) then begin
    Seek(trImg[index], ATrainingPosition * trainingCount mod 10000);
    TrainingLabels.Fill(0);
    for i :=0 to trainingCount -1 do begin
      system.read(trImg[index], CF);
      move(CF.Data[0], TrainingData.Data[i * IMAGE_SIZE], IMAGE_SIZE);
      TrainingLabels.Data[i * CLASS_COUNT + CF.id] := 1;
      result := result and not EOF(trImg[index]) or( index < 4);
      if EOF(trImg[index]) then break
    end;

    if not result then
      exit;
  end;

  if (ATestingPosition >= 0) and isOpen(tsImg) then begin
    Seek(tsImg, ATestingPosition * testingCount mod 10000);
    TestingLabels.Fill(0);
    for i :=0 to testingCount -1 do begin
      system.read(tsImg, CF);
      move(CF.Data[0], TestingData.Data[i * IMAGE_SIZE], IMAGE_SIZE);
      TestingLabels.Data[i * CLASS_COUNT + CF.id] := 1;
      result := result and not EOF(tsImg);
      if EOF(tsImg) then break
    end;

    if not result then
      exit;
  end;

end;

destructor TCIFAR10Data.Destroy;
var i:SizeInt;
begin
  for i:=0 to 4 do
    if isOpen(trImg[i]) then CloseFile(trImg[i]);
  if isOpen(tsImg) then CloseFile(tsImg);
  inherited Destroy;
end;

end.

