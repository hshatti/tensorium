unit NTypes;
{$ifdef FPC}
{$mode Delphi}{$H+}
{$ModeSwitch advancedrecords}
{$ModeSwitch typehelpers}
{$endif}
{$pointermath on}

interface

uses
  SysUtils, nTensors
  {$if defined(FPC)}
  , FPImage, FPImgCanv
  , FPReadBMP, FPReadJPEG, FPReadPNG, FPReadTGA
  , FPWriteBMP, FPWriteJPEG, FPWritePNG, FPWriteTGA
  {$elseif defined(FRAMEWORK_FMX)}
  TypesUI, FMX.Graphics
  {$elseif defined(FRAMEWORK_VCL)}
  Graphics
  {$endif}
  ;

const
  SECRET_NUM = -1234;

{$if not declared(TMPParams)}
type
  PMPParams = ^TMPParams;
  TMPParams = record
     A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q:Pointer;
  end;
{$endif}

type

  PNNetState = ^TNNetState;
  TNNetState = record
      net : TObject;
      truth : TSingleTensor;
      input : PSingleTensor;
      delta : PSingleTensor;
      workspace : TSingleTensor;
      isTraining : boolean;
      index : SizeInt;
      label_smooth_eps : single;
      adversarial : boolean;
  end;

  PActivationType = ^TActivationType;
  TActivationType = (
    acLOGISTIC, acRELU, acRELU6, acRELIE, acLINEAR, acRAMP, acTANH, acPLSE,
    acREVLEAKY, acLEAKY, acELU, acLOGGY, acSTAIR, acHARDTAN, acLHTAN, acSELU, acSOFTMAX,
    acGELU, acSWISH, acMISH, acHARD_MISH, acNORM_CHAN, acNORM_CHAN_SOFTMAX,
    acNORM_CHAN_SOFTMAX_MAXVAL
  );

  PLearningRatePolicy = ^TLearningRatePolicy;
  TLearningRatePolicy = (lrpCONSTANT, lrpSTEP, lrpEXP, lrpPOLY, lrpSTEPS, lrpSIG, lrpRANDOM, lrpSGDR, lrpCOST);

  PUpdateArgs = ^TUpdateArgs;
  TUpdateArgs = record
      batch : SizeInt;
      learningRate : Single;
      momentum : Single;
      decay : Single;
      adam : boolean;
      B1 : Single;
      B2 : Single;
      eps : Single;
      t : SizeInt;
    end;

  PLayerType = ^TLayerType;
  TLayerType = (
    ltNONE,
    ltCONVOLUTIONAL,
    ltDECONVOLUTIONAL,
    ltCONNECTED,
    ltMAXPOOL,
    ltLOCAL_AVGPOOL,
    ltSOFTMAX,
    ltDETECTION,
    ltDROPOUT,
    ltCROP,
    ltROUTE,
    ltCOST,
    ltNORMALIZATION,
    ltAVGPOOL,
    ltLOCAL,
    ltSHORTCUT,
    ltScaleChannels,
    ltSAM,
    ltACTIVE,
    ltRNN,
    ltGRU,
    ltLSTM,
    ltConvLSTM,
    ltHISTORY,
    ltCRNN,
    ltBATCHNORM,
    ltNETWORK,
    ltXNOR,
    ltREGION,
    ltYOLO,
    ltGaussianYOLO,
    ltISEG,
    ltREORG,
    ltREORG_OLD,
    ltUPSAMPLE,
    ltLOGXENT,
    ltL2NORM,
    ltEMPTY,
    ltBLANK,
    ltCONTRASTIVE,
    ltIMPLICIT
  );

  PCostType = ^TCostType;
  TCostType = (ctSSE, ctMASKED, ctL1, ctSEG, ctSMOOTH, ctWGAN);

  TIOULoss = (
      ilIOU, ilGIOU, ilMSE, ilDIOU, ilCIOU
  ) ;

  PAugmentArgs = ^TAugmentArgs;
  TAugmentArgs = record
    w : SizeInt;
    h : SizeInt;
    scale : Single;
    rad : Single;
    dx : Single;
    dy : Single;
    aspect : Single;
  end;

  PPImageData = ^PImageData;
  PImageData = ^TImageData;

  { TImageData }

  TImageData = record
  private
    function Getpixels(x, y, _c: SizeInt): Single;
    procedure Setpixels(x, y, _c: SizeInt; AValue: Single);
    procedure AddPixel(x, y, _c: SizeInt; AValue: Single);
  public
    w : SizeInt;
    h : SizeInt;
    c : SizeInt;
    n : SizeInt;
    data : TArray<Single>;
    procedure fromTensor(const src:TSingleTensor);
    function toTensor():TSingleTensor;
    constructor Create(const aWidth, aHeight, aChannels: SizeInt; const aImageCount : SizeInt=1);
    procedure loadFromFile(const fileName:string);                                                   overload;
    procedure loadFromFile(const fileNames:TArray<string>; const resizeWidth, resizeHeight:SizeInt); overload;
    procedure saveToFile(const filename:string);
    procedure fill(const val:single);
    function resize(const aWidth, aHeight: SizeInt):TImageData;
    procedure Embed(var dst:TImageData ; const dx, dy : SizeInt);
    function letterBox(const aWidth, aHeight:SizeInt):TImageData;
    property pixels[x, y, c:SizeInt]: Single read Getpixels write Setpixels;
  end;

  PBoxAbs = ^TBoxabs;
  TBoxAbs = record
    left : single;
    right : single;
    top : single;
    bot : single;
  end;

  PDxrep = ^TDxrep;
  TDxrep = record
    dt : single;
    db : single;
    dl : single;
    dr : single;
  end;

  PIOUs = ^TIOUs;
  TIOUs = record
    iou : single;
    giou : single;
    diou : single;
    ciou : single;
    dx_iou : TDxrep;
    dx_giou : TDxrep;
  end;

  { TCoord }

  TCoord = record
      x, y :single;
    constructor Create(const aX, aY:single);
    function add(const val:single):TCoord;    overload;
    function add(const p:TCoord):TCoord;      overload;
    function add(const aX, aY:single):TCoord; overload;

    function subtract(const val:single):TCoord;    overload;
    function subtract(const p:TCoord):TCoord;      overload;
    function subtract(const aX, aY:single):TCoord; overload;

    function scale(const ratio:single):TCoord;  overload;
    function scale(const ratios:TCoord):TCoord; overload;
    function scale(const xRatio, yRatio:single):TCoord; overload;
    function max(const b:TCoord):TCoord;
    function min(const b:TCoord):TCoord;
    function lerp(const ratio:single; const b:TCoord):TCoord;
    function centroid(const b:TCoord):TCoord;

  end;

  PPBox= ^PBox;
  PBox = ^TBox;

  { TBox }

  TBox = record
    x : Single;
    y : Single;
    w : Single;
    h : Single;
    procedure regularize();
    function intersect(const b:TBox):Tbox;
    function isIntersecting(const b: TBox):boolean;
    function union(const b:TBox):TBox;
    function scale(const wRatio, hRatio: single):TBox;      overload;
    function scale(const ratio:single):TBox;                overload;
    function inflate(const wRatio, hRatio:single):TBox;     overload;
    function inflate(const ratio:single):TBox;              overload;
    function add(const dw, dh:single):TBox;
    function area():single;
    function iou(const b: TBox): single; overload;
    procedure iou(const b:TBox; var i, u:single);      overload;
    function giou(const b: TBox):single;
    function diou(const b: TBox):single;
    function ciou(const b: TBox):single;
    function centroidBox(const b:TBox):TBoxAbs;
    function centroid():TCoord; overload;
    procedure centroid(var cx, cy:Single); overload;
    function contains(const coord:TCoord):boolean;

    function iouKind(const b:TBox; const iou_kind:TIOULoss):single;
    function dx_iou(const truth: TBox; const iou_loss: TIOULoss):TDxrep;
    function toTblr():TBoxAbs;
    class function fromFloat(const f:PSingle; const stride:SizeInt=1):TBox;overload;static;
  end;


  PDetection = ^TDetection;
  TDetection = record
    bbox : TBox;
    classes, best_class_idx: SizeInt;
    prob, mask : TArray<Single>;
    objectness : Single;
    sort_class : SizeInt;
    uc :TArray<single>;
    points : SizeInt;
    embeddings : TArray<single>;
    embedding_size : SizeInt;
    sim : single;
    track_id : SizeInt;
  end;

  TDetections = TArray<TDetection>;

  { TDetectionsHelper }

  TDetectionsHelper = record helper for TDetections
    //detections : TDetections;
  private
    class function Comparer(const a, b: TDetection):SizeInt; static;
  public
    procedure doNMSObj(const classes: SizeInt; const thresh: single = 0.45);
    procedure doNMSSort(const classes: SizeInt; const thresh: single = 0.45);
    //procedure print();
  end;


  PDetNumPair  = ^TDetNumPair;
  TDetNumPair = record
    num :SizeInt;
    dets :TArray<TDetection>;
  end;

  PNMatrix = ^TNMatrix;
  TNMatrix = record
    rows : SizeInt;
    cols : SizeInt;
    vals : TArray<TArray<Single>>
  end;

  PData = ^TData;

  { TData }

  TData = record
    //w : SizeInt;
    //h : SizeInt;
    X : TSingleTensor;
    y : TSingleTensor;
    shallow : boolean;
    num_boxes : PInteger;
    boxes : PPBox;
    procedure getRandomBatch(const n: SizeInt; const A, B: PSingle);
    procedure getBatch(const n, offset: SizeInt; const A, B: PSingle);
  end;
  PDataType = ^TDatatype;
  TDataType = (
    dtCLASSIFICATION_DATA,      dtDETECTION_DATA, dtCAPTCHA_DATA,
    dtREGION_DATA,     dtIMAGE_DATA,        dtCOMPARE_DATA,   dtWRITING_DATA,
    dtSWAG_DATA,       dtTAG_DATA,          dtOLD_CLASSIFICATION_DATA,
    dtSTUDY_DATA,      dtDET_DATA,          dtSUPER_DATA,     dtLETTERBOX_DATA,
    dtREGRESSION_DATA, dtSEGMENTATION_DATA, dtINSTANCE_DATA,
    dtISEG_DATA);

  PTree = ^TTree;

  { TTree }

  TTree = record
      leaf : TArray<SizeInt>;
      n : SizeInt;
      parent : TArray<SizeInt>;
      child :  TArray<SizeInt>;
      group :  TArray<SizeInt>;
      name : TArray<string>;
      groups : SizeInt;
      group_size :  TArray<SizeInt>;
      group_offset :  TArray<SizeInt>;
      class function loadFromFile(const fileName:string):TTree; static;
  end;

  { TCostTypeHelper }

  TCostTypeHelper = record helper for TCostType
    class function fromString(const s:string):TCostType; static;
    function toString():string;
  end;

  PLoadArgs = ^TLoadArgs;
  TLoadArgs = record
    threads              : SizeInt;
    paths                : TArray<string>;
    path                 : string;
    n                    : SizeInt;
    m                    : SizeInt;
    labels               : TArray<string>;
    h                    : SizeInt;
    w                    : SizeInt;
    c                    : SizeInt;
    out_w                : SizeInt;
    out_h                : SizeInt;
    nh                   : SizeInt;
    nw                   : SizeInt;
    num_boxes            : SizeInt;
    truth_size           : SizeInt;
    min, max, size       : SizeInt;
    classes              : SizeInt;
    background           : SizeInt;
    scale                : SizeInt;
    center               : boolean;
    coords               : SizeInt;
    mini_batch           : SizeInt;
    track                : SizeInt;
    augment_speed        : SizeInt;
    letter_box           : SizeInt;
    mosaic_bound         : SizeInt;
    show_imgs            : SizeInt;
    dontuse_opencv       : SizeInt;
    contrastive          : SizeInt;
    contrastive_jit_flip : SizeInt;
    contrastive_color    : SizeInt;
    jitter               : single;
    resize               : single;
    flip                 : SizeInt;
    gaussian_noise       : SizeInt;
    blur                 : SizeInt;
    mixup                : SizeInt;
    label_smooth_eps     : single;
    angle                : single;
    aspect               : single;
    saturation           : single;
    exposure             : single;
    hue                  : single;
    d                    : PData;
    im                   : PImageData;
    resized              : PImageData;
    &type                : TDataType;
    hierarchy            : TArray<TTree>
  end;

  PBoxlabel = ^TBoxLabel;
  TBoxLabel = record
    id , track_id: SizeInt;
    x : Single;
    y : Single;
    w : Single;
    h : Single;
    left : Single;
    right : Single;
    top : Single;
    bottom : Single;
  end;

  // parser.h
  TNMSKind = (
      nmsDEFAULT_NMS, nmsGREEDY_NMS, nmsDIOU_NMS, nmsCORNERS_NMS
  ) ;

  // parser.h
  TYOLOPoint = (
      ypYOLO_CENTER = 1 shl 0, ypYOLO_LEFT_TOP = 1 shl 1, ypYOLO_RIGHT_BOTTOM = 1 shl 2
  ) ;

  // parser.h
  TWeightsType = (
      wtNO_WEIGHTS, wtPER_FEATURE, wtPER_CHANNEL
  );

  // parser.h
  TWeightsNormalization = (
      wnNO_NORMALIZATION, wnRELU_NORMALIZATION, wnSOFTMAX_NORMALIZATION
  );


const
  acSIGMOID = acLOGISTIC;
  acSiLU    = acSWISH;
  ctL2      = ctSSE;

  ltCONCAT = ltROUTE;
  ltADD    = ltSHORTCUT;

implementation
uses math, typinfo, nChrono;

{ TImageData }

function TImageData.Getpixels(x, y, _c: SizeInt): Single;
begin
  result := data[(_c*h + y)*w + x]
end;

procedure TImageData.Setpixels(x, y, _c: SizeInt; AValue: Single);
begin
  if (x < 0) or (y < 0) or (_c < 0) or (x >= w) or (y >= h) or (_c >= c) then
        exit();
  data[(_c*h + y)*w + x] := aValue
end;

procedure TImageData.AddPixel(x, y, _c: SizeInt; AValue: Single);
var idx : SizeInt;
begin
  if (x < 0) or (y < 0) or (_c < 0) or (x >= w) or (y >= h) or (_c >= c) then
        exit();
  idx := (_c*h + y)*w + x;
  data[idx] := data[idx] + AValue;
end;

procedure TImageData.fromTensor(const src: TSingleTensor);
begin
  w := src.w();
  h := src.h();
  c := src.Size() div src.Area();
  setLength(Data, w*h*c );
  move(src.Data[0], Data[0], src.size()*SizeOf(single));
end;

function TImageData.toTensor(): TSingleTensor;
var imSize : SizeInt;
begin
  imSize := h*w;
  result := TSingleTensor.Create([n, c, h, w]);
  move(Data[0], result.Data[0], length(data)*SizeOf(single));
end;

constructor TImageData.Create(const aWidth, aHeight, aChannels: SizeInt;
  const aImageCount: SizeInt);
begin
  w := aWidth;
  h := aHeight;
  c := aChannels;
  n := aImageCount;
  setLength(Data, w*h*c)
end;

procedure TImageData.loadFromFile(const fileName: string);
{$if defined(FPC)}
var img : TFPMemoryImage;
    P:TFPColor;
    x, y, imSize : SizeInt;
begin
  img := TFPMemoryImage.Create(0, 0);
  try
    img.LoadFromFile(fileName);
    n := 1;
    c := 3;
    w := img.Width;
    h := img.Height;
    imSize := w*h;
    setLength(data, w*h*c);
    for y := 0 to h-1 do
      for x := 0 to w-1 do begin
         p := img.Colors[x, y];
         data[           y*w +x] := p.Red / $ff00;
         data[1*imSize + y*w +x] := p.Green / $ff00;
         data[2*imSize + y*w +x] := p.Blue / $ff00;
      end;
  finally
    freeAndNil(img)
  end;
end;

{$else}
begin

end;
{$endif}

{$if defined(FPC)}
procedure TImageData.loadFromFile(const fileNames: TArray<string>; const resizeWidth, resizeHeight: SizeInt);
var img : TFPMemoryImage;
    P:TFPColor;
    i, x, y, imSize, _h, _w : SizeInt;
begin
  img := TFPMemoryImage.Create(0, 0);
  c := 3;
  n := length(filenames);
  setLength(data, n*resizeHeight*resizeWidth*c);
  try
    for i:=0 to n-1 do begin
      img.LoadFromFile(fileNames[i]);
      _w := img.Width;
      _h := img.Height;
      w := resizeWidth;
      h := resizeHeight;
      imSize := resizeHeight*resizeWidth;
      for y := 0 to resizeHeight-1 do
        for x := 0 to resizeWidth-1 do begin
           p := img.Colors[ round(_w * x / resizeWidth),  round(_h * y / resizeHeight)];  // nearst neighor
           data[i*3*imSize +            y*resizeWidth + x] := p.Red / $ff00;
           data[i*3*imSize + 1*imSize + y*resizeWidth + x] := p.Green / $ff00;
           data[i*3*imSize + 2*imSize + y*resizeWidth + x] := p.Blue / $ff00;
        end;
    end;
  finally
    freeAndNil(img)
  end;
end;
{$else}
begin

end;
{$endif}


procedure TImageData.saveToFile(const filename: string);
{$if defined(fpc)}
var img : TFPMemoryImage;
    P:TFPColor;
    x, y, imSize : SizeInt;
begin
  img := TFPMemoryImage.Create(w, h);
  try
    imSize := w*h;
    if c=1 then
    for y := 0 to h-1 do
      for x := 0 to w-1 do begin
         p.red   := round(data[y*w +x]*$FF00);
         p.green := p.red;
         p.blue  := p.red;
         img.colors[x,y] := p;
      end
    else if c>2 then
      for y := 0 to h-1 do
        for x := 0 to w-1 do begin
           p.red   := round(data[           y*w +x]*$FF00);
           p.green := round(data[1*imSize + y*w +x]*$FF00);
           p.blue  := round(data[2*imSize + y*w +x]*$FF00);
           img.colors[x,y] := p;
        end;
    img.SaveToFile(filename);
  finally
    freeAndNil(img)
  end;
end;
{$else}
begin

end;
{$endif}

procedure TImageData.fill(const val: single);
var i:SizeInt;
begin
  for i:=0 to high(data) do
    data[i] := val
end;

function TImageData.resize(const aWidth, aHeight: SizeInt): TImageData;
var
    part: TImageData;
    r,c,k,ix,iy: SizeInt;
    w_scale, h_scale, val, sx, dx, sy, dy: single;
begin
    result := TImageData.Create(aWidth, aHeight, self.c);
    part := TImageData.Create(aWidth, self.h, self.c);
    w_scale := (self.w-1) / (aWidth-1);
    h_scale := (self.h-1) / (aHeight-1);
    for k := 0 to self.c -1 do
        for r := 0 to self.h -1 do
            for c := 0 to aWidth -1 do
                begin
                    val := 0;
                    if (c = aWidth-1) or (self.w = 1) then
                        val := getPixels(self.w-1, r, k)
                    else
                        begin
                            sx := c * w_scale;
                            ix := trunc(sx);
                            dx := sx-ix;
                            val := (1-dx) * getPixels(ix, r, k)+dx * getPixels(ix+1, r, k)
                        end;
                    part.setPixels(c, r, k, val)
                end;
    for k := 0 to self.c -1 do
        for r := 0 to aHeight -1 do
            begin
                sy := r * h_scale;
                iy := trunc(sy);
                dy := sy-iy;
                for c := 0 to aWidth -1 do
                    begin
                        val := (1-dy) * part.getPixels(c, iy, k);
                        result.setPixels(c, r, k, val)
                    end;
                if (r = aHeight-1) or (self.h = 1) then
                    continue;
                for c := 0 to aWidth -1 do
                    begin
                        val := dy * part.getPixels(c, iy+1, k);
                        result.addPixel(c, r, k, val)
                    end
            end;
end;

procedure TImageData.Embed(var dst: TImageData; const dx, dy: SizeInt);
var
  x, y, k: SizeInt;
  val: single;
begin
  for k := 0 to c -1 do
    for y := 0 to h -1 do
      for x := 0 to w -1 do
        begin
          val := getPixels(x, y, k);
          dst.setPixels(dx+x, dy+y, k, val)
        end
end;

function TImageData.letterBox(const aWidth, aHeight: SizeInt): TImageData;
var
    new_w, new_h: SizeInt;
    resized: TImageData;
begin
    new_w := self.w;
    new_h := self.h;
    if (aWidth / self.w) < (aHeight / self.h) then
        begin
            new_w := aWidth;
            new_h := (self.h * aWidth) div self.w
        end
    else
        begin
            new_h := h;
            new_w := (self.w * aHeight) div self.h
        end;
    resized := resize(new_w, new_h);
    result := TImageData.Create(aWidth, aHeight, self.c);
    result.fill(0.5);
    resized.embed(result, (aWidth-new_w) div 2, (aHeight-new_h) div 2);
end;

{ TCoord }

constructor TCoord.Create(const aX, aY: single);
begin
    x := aX;
    y := aY
end;

function TCoord.add(const val: single): TCoord;
begin
    result.x := x + val;
    result.y := y + val
end;

function TCoord.add(const p: TCoord): TCoord;
begin
    result.x := x + p.x;
    result.y := y + p.y
end;

function TCoord.add(const aX, aY: single): TCoord;
begin
    result.x := x + ax;
    result.y := y + ay
end;

function TCoord.subtract(const val: single): TCoord;
begin
    result.x := x - val;
    result.y := y - val
end;

function TCoord.subtract(const p: TCoord): TCoord;
begin
    result.x := x - p.x;
    result.y := y - p.y
end;

function TCoord.subtract(const aX, aY: single): TCoord;
begin
    result.x := x - ax;
    result.y := y - ay
end;

function TCoord.scale(const ratio: single): TCoord;
begin
    result.x := x * ratio;
    result.y := y * ratio
end;

function TCoord.scale(const ratios: TCoord): TCoord;
begin
    result.x := x * ratios.x;
    result.y := y * ratios.y
end;

function TCoord.scale(const xRatio, yRatio: single): TCoord;
begin
    result.x := x * xRatio;
    result.y := y * yRatio
end;

function TCoord.max(const b: TCoord): TCoord;
begin
    result.x := math.max(x, b.x);
    result.y := math.max(y, b.y)
end;

function TCoord.min(const b: TCoord): TCoord;
begin
    result.x := math.min(x, b.x);
    result.y := math.min(y, b.y)
end;

function TCoord.lerp(const ratio: single; const b: TCoord): TCoord;
begin
  result.x := TensorUtils.lerp(ratio, x, b.x);
  result.y := TensorUtils.lerp(ratio, y, b.y)
end;

function TCoord.centroid(const b: TCoord): TCoord;
begin
  result := lerp(0.5, b)
end;

{ TBox }

procedure TBox.regularize();
begin
  if w < 0 then begin
      x := x + w;
      w := -w
  end;
  if h < 0 then begin
      y := y + h;
      h := -h
  end;
end;

function TBox.intersect(const b: TBox): Tbox;
begin
  if isIntersecting(b) then begin
    result.x := max(x, b.x);
    result.w := min(x + w, b.x + b.w) - result.x;
    result.y := max(y, b.y);
    result.h := min(y + h, b.y + b.h) - result.y
  end else
    exit(default(TBox));
end;

function TBox.isIntersecting(const b: TBox): boolean;
begin
  result := (x+w < b.x) or (b.x+b.w < x)
    or (y+h < b.y) or (b.y+b.h < y);
  result := not result
end;

function TBox.union(const b: TBox): TBox;
begin
  result.x := min(x, b.x);
  result.w := max(x + w, b.x + b.w) - result.x;
  result.x := min(y, b.y);
  result.w := max(y + h, b.y + b.h) - result.y
end;

function TBox.scale(const wRatio, hRatio: single): TBox;
begin
  result.w := w * wRatio;
  result.h := h * hRatio
end;

function TBox.scale(const ratio: single): TBox;
begin
  result := scale(ratio, ratio)
end;

function TBox.inflate(const wRatio, hRatio: single): TBox;
var dw, dh:single;
begin
  dw := wRatio*w;
  dh := hRatio*h;
  result.x := x - (dw - w) / 2;
  result.y := y - (dh - h) / 2;
  w := dw;
  h := dh
end;

function TBox.inflate(const ratio: single): TBox;
begin
  inflate(ratio, ratio)
end;

function TBox.add(const dw, dh: single): TBox;
begin
  result.w := w + dw;
  result.h := h + dh
end;

function TBox.area(): single;
begin
  result := w * h
end;

function TBox.iou(const b: TBox): single;
var I,U:single;
begin
  I := intersect(b).area();
  U := area() + b.area() - I;
  if (I=0) or (U=0) then exit(0);
  exit(I/U)
end;

procedure TBox.iou(const b: TBox; var i, u: single);
begin
  i := intersect(b).area();
  u := area() + b.area() -i
end;

function TBox.giou(const b: TBox): single;
var
    ba: TBoxAbs;
    _w, _h, c, _iou, u, giou_term: single;
begin
    ba := centroidBox(b);
    _w := ba.right - ba.left;
    _h := ba.bot - ba.top;
    c := _w * _h;
    _iou := iou(b);
    if c = 0 then
        exit(_iou);
    u := area() + b.area() - intersect(b).area;
    giou_term := (c-u) / c;
  {$ifdef DEBUG}
    writeln('  c: %f, u: %f, giou_term: %f', c, u, giou_term);
  {$endif}
    exit(_iou - giou_term)
end;

function TBox.diou(const b: TBox): single;
var
    ba: TBoxAbs;
    _w, _h, c, _iou, d, u, diou_term: single;
begin
    ba := centroidBox(b);
    _w := ba.right - ba.left;
    _h := ba.bot - ba.top;
    c := _w*w + _h*_h;
    _iou := iou(b);
    if c = 0 then
        exit(_iou);
    d := (x - b.x) * (x - b.x)+(y - b.y) * (y - b.y);
    u := power(d / c, 0.6);
    diou_term := u;
    {$ifdef DEBUG}
    writeln('  c: %f, u: %f, riou_term: %f', c, u, diou_term);
    {$endif}
    exit(_iou - diou_term)
end;

function TBox.ciou(const b: TBox): single;
var
    ba: TBoxAbs;
    _w, _h, c, _iou, u, d, ar_gt, ar_pred, ar_loss, alpha, ciou_term: single;
begin
    ba := centroidBox(b);
    _w := ba.right-ba.left;
    _h := ba.bot-ba.top;
    c := _w*_w + _h*_h;
    _iou := iou( b);
    if c = 0 then
        exit(_iou);
    u := (x-b.x) * (x-b.x)+(y-b.y) * (y-b.y);
    d := u / c;
    ar_gt := b.w / b.h;
    ar_pred := _w / _h;
    ar_loss := 4 / (PI * PI) * (arctan(ar_gt)-arctan(ar_pred)) * (arctan(ar_gt)-arctan(ar_pred));
    alpha := ar_loss / (1 - _iou+ar_loss + sEPSILON);
    ciou_term := d + alpha * ar_loss;
    {$ifdef DEBUG}
    writeln('  c: %f, u: %f, riou_term: %f', c, u, ciou_term);
    {$endif}
    exit(_iou - ciou_term)
end;

function TBox.centroidBox(const b: TBox): TBoxAbs;
begin
    result.top := min(y-h / 2, b.y-b.h / 2);
    result.bot := max(y+h / 2, b.y+b.h / 2);
    result.left := min(x-w / 2, b.x-b.w / 2);
    result.right := max(x+w / 2, b.x+b.w / 2);
end;

function TBox.centroid(): TCoord;
begin
  result.x := x + w / 2;
  result.y := y + h / 2
end;

procedure TBox.centroid(var cx, cy: Single);
begin
  cx := x + w / 2;
  cy := y + h / 2
end;

function TBox.contains(const coord: TCoord): boolean;
begin
  result := (coord.x >= x) and (coord.x <= x+w) and (coord.y >= y) and (coord.y <= y+h)
end;

function TBox.iouKind(const b: TBox; const iou_kind: TIOULoss): single;
begin
    case iou_kind of
        ilIOU:
            exit(iou(b));
        ilGIOU:
            exit(giou(b));
        ilDIOU:
            exit(diou(b));
        ilCIOU:
            exit(ciou(b))
    end;
    exit(iou(b))
end;

function TBox.dx_iou(const truth: TBox; const iou_loss: TIOULoss): TDxrep;
var
    pred_tblr, truth_tblr: TBoxAbs;
    pred_t, pred_b, pred_l, pred_r, X, Xhat, Ih, Iw, I, U, S, giou_Cw, giou_Ch
      , giou_C, dX_wrt_t, dX_wrt_b, dX_wrt_l, dX_wrt_r, dI_wrt_t, dI_wrt_b, dI_wrt_l
      , dI_wrt_r, dU_wrt_t, dU_wrt_b, dU_wrt_l, dU_wrt_r, dC_wrt_t, dC_wrt_b, dC_wrt_l
      , dC_wrt_r, p_dt, p_db, p_dl, p_dr, Ct, Cb, Cl, Cr, Cw, Ch, C, dCt_dx, dCt_dy
      , dCt_dw, dCt_dh, dCb_dx, dCb_dy, dCb_dw, dCb_dh, dCl_dx, dCl_dy, dCl_dw, dCl_dh
      , dCr_dx, dCr_dy, dCr_dw, dCr_dh, dCw_dx, dCw_dy, dCw_dw, dCw_dh, dCh_dx, dCh_dy
      , dCh_dw, dCh_dh, p_dx, p_dy, p_dw, p_dh, ar_gt, ar_pred, ar_loss, alpha, ar_dw, ar_dh: single;
begin
    pred_tblr := toTblr();
    pred_t := math.min(pred_tblr.top, pred_tblr.bot);
    pred_b := math.max(pred_tblr.top, pred_tblr.bot);
    pred_l := math.min(pred_tblr.left, pred_tblr.right);
    pred_r := math.max(pred_tblr.left, pred_tblr.right);
    truth_tblr := toTblr();
    writeln(#10'iou: %f, giou: %f', self.iou(truth), self.giou(truth));
    writeln('self: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)', self.x, self.y, self.w, self.h, pred_tblr.top, pred_tblr.bot, pred_tblr.left, pred_tblr.right);
    writeln('truth: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)', truth.x, truth.y, truth.w, truth.h, truth_tblr.top, truth_tblr.bot, truth_tblr.left, truth_tblr.right);
    result := Default(TDxrep);
    X := (pred_b-pred_t) * (pred_r-pred_l);
    Xhat := (truth_tblr.bot-truth_tblr.top) * (truth_tblr.right-truth_tblr.left);
    Ih := math.min(pred_b, truth_tblr.bot) - math.max(pred_t, truth_tblr.top);
    Iw := math.min(pred_r, truth_tblr.right) - math.max(pred_l, truth_tblr.left);
    I := Iw * Ih;
    U := X+Xhat-I;
    S := (self.x-truth.x) * (self.x-truth.x)+(self.y-truth.y) * (self.y-truth.y);
    giou_Cw := math.max(pred_r, truth_tblr.right)-math.min(pred_l, truth_tblr.left);
    giou_Ch := math.max(pred_b, truth_tblr.bot)-math.min(pred_t, truth_tblr.top);
    giou_C := giou_Cw * giou_Ch;
    dX_wrt_t := -1 * (pred_r-pred_l);
    dX_wrt_b := pred_r-pred_l;
    dX_wrt_l := -1 * (pred_b-pred_t);
    dX_wrt_r := pred_b-pred_t;
    if (pred_t > truth_tblr.top) then
        dI_wrt_t := (-1 * Iw)
    else
        dI_wrt_t := 0;
    if (pred_b < truth_tblr.bot) then
        dI_wrt_b := Iw
    else
        dI_wrt_b := 0;
    if (pred_l > truth_tblr.left) then
        dI_wrt_l := (-1 * Ih)
    else
        dI_wrt_l := 0;
    if (pred_r < truth_tblr.right) then
        dI_wrt_r := Ih
    else
        dI_wrt_r := 0;
    dU_wrt_t := dX_wrt_t-dI_wrt_t;
    dU_wrt_b := dX_wrt_b-dI_wrt_b;
    dU_wrt_l := dX_wrt_l-dI_wrt_l;
    dU_wrt_r := dX_wrt_r-dI_wrt_r;
    if (pred_t < truth_tblr.top) then
        dC_wrt_t := (-1 * giou_Cw)
    else
        dC_wrt_t := 0;
    if (pred_b > truth_tblr.bot) then
        dC_wrt_b := giou_Cw
    else
        dC_wrt_b := 0;
    if (pred_l < truth_tblr.left) then
        dC_wrt_l := (-1 * giou_Ch)
    else
        dC_wrt_l := 0;
    if (pred_r > truth_tblr.right) then
        dC_wrt_r := giou_Ch
    else
        dC_wrt_r := 0;
    p_dt := 0;
    p_db := 0;
    p_dl := 0;
    p_dr := 0;
    if U > 0 then
        begin
            p_dt := ((U * dI_wrt_t)-(I * dU_wrt_t)) / (U * U);
            p_db := ((U * dI_wrt_b)-(I * dU_wrt_b)) / (U * U);
            p_dl := ((U * dI_wrt_l)-(I * dU_wrt_l)) / (U * U);
            p_dr := ((U * dI_wrt_r)-(I * dU_wrt_r)) / (U * U)
        end;
    if (pred_tblr.top < pred_tblr.bot) then
        p_db := p_db
    else
        p_db := p_dt;
    if (pred_tblr.left < pred_tblr.right) then
        p_dl := p_dl
    else
        p_dl := p_dr;
    if (pred_tblr.left < pred_tblr.right) then
        p_dr := p_dr
    else
        p_dr := p_dl;
    if iou_loss = ilGIOU then
        begin
            if giou_C > 0 then
                begin
                    p_dt := p_dt + (((giou_C * dU_wrt_t)-(U * dC_wrt_t)) / (giou_C * giou_C));
                    p_db := p_db + (((giou_C * dU_wrt_b)-(U * dC_wrt_b)) / (giou_C * giou_C));
                    p_dl := p_dl + (((giou_C * dU_wrt_l)-(U * dC_wrt_l)) / (giou_C * giou_C));
                    p_dr := p_dr + (((giou_C * dU_wrt_r)-(U * dC_wrt_r)) / (giou_C * giou_C))
                end;
            if (Iw <= 0) or (Ih <= 0) then
                begin
                    p_dt := ((giou_C * dU_wrt_t)-(U * dC_wrt_t)) / (giou_C * giou_C);
                    p_db := ((giou_C * dU_wrt_b)-(U * dC_wrt_b)) / (giou_C * giou_C);
                    p_dl := ((giou_C * dU_wrt_l)-(U * dC_wrt_l)) / (giou_C * giou_C);
                    p_dr := ((giou_C * dU_wrt_r)-(U * dC_wrt_r)) / (giou_C * giou_C)
                end
        end;
    Ct := min(self.y-self.h / 2, truth.y-truth.h / 2);
    Cb := max(self.y+self.h / 2, truth.y+truth.h / 2);
    Cl := min(self.x-self.w / 2, truth.x-truth.w / 2);
    Cr := max(self.x+self.w / 2, truth.x+truth.w / 2);
    Cw := Cr-Cl;
    Ch := Cb-Ct;
    C := Cw * Cw+Ch * Ch;
    dCt_dx := 0;
    if (pred_t < truth_tblr.top) then
        dCt_dy := 1
    else
        dCt_dy := 0;
    dCt_dw := 0;
    if (pred_t < truth_tblr.top) then
        dCt_dh := -0.5
    else
        dCt_dh := 0;
    dCb_dx := 0;
    if (pred_b > truth_tblr.bot) then
        dCb_dy := 1
    else
        dCb_dy := 0;
    dCb_dw := 0;
    if (pred_b > truth_tblr.bot) then
        dCb_dh := 0.5
    else
        dCb_dh := 0;
    if (pred_l < truth_tblr.left) then
        dCl_dx := 1
    else
        dCl_dx := 0;
    dCl_dy := 0;
    if (pred_l < truth_tblr.left) then
        dCl_dw := -0.5
    else
        dCl_dw := 0;
    dCl_dh := 0;
    if (pred_r > truth_tblr.right) then
        dCr_dx := 1
    else
        dCr_dx := 0;
    dCr_dy := 0;
    if (pred_r > truth_tblr.right) then
        dCr_dw := 0.5
    else
        dCr_dw := 0;
    dCr_dh := 0;
    dCw_dx := dCr_dx-dCl_dx;
    dCw_dy := dCr_dy-dCl_dy;
    dCw_dw := dCr_dw-dCl_dw;
    dCw_dh := dCr_dh-dCl_dh;
    dCh_dx := dCb_dx-dCt_dx;
    dCh_dy := dCb_dy-dCt_dy;
    dCh_dw := dCb_dw-dCt_dw;
    dCh_dh := dCb_dh-dCt_dh;
    p_dx := 0;
    p_dy := 0;
    p_dw := 0;
    p_dh := 0;
    p_dx := p_dl+p_dr;
    p_dy := p_dt+p_db;
    p_dw := (p_dr-p_dl);
    p_dh := (p_db-p_dt);
    if iou_loss = ilDIOU then
        begin
            if C > 0 then
                begin
                    p_dx := p_dx + ((2 * (truth.x-self.x) * C-(2 * Cw * dCw_dx+2 * Ch * dCh_dx) * S) / (C * C));
                    p_dy := p_dy + ((2 * (truth.y-self.y) * C-(2 * Cw * dCw_dy+2 * Ch * dCh_dy) * S) / (C * C));
                    p_dw := p_dw + ((2 * Cw * dCw_dw+2 * Ch * dCh_dw) * S / (C * C));
                    p_dh := p_dh + ((2 * Cw * dCw_dh+2 * Ch * dCh_dh) * S / (C * C))
                end;
            if (Iw <= 0) or (Ih <= 0) then
                begin
                    p_dx := (2 * (truth.x-self.x) * C-(2 * Cw * dCw_dx+2 * Ch * dCh_dx) * S) / (C * C);
                    p_dy := (2 * (truth.y-self.y) * C-(2 * Cw * dCw_dy+2 * Ch * dCh_dy) * S) / (C * C);
                    p_dw := (2 * Cw * dCw_dw+2 * Ch * dCh_dw) * S / (C * C);
                    p_dh := (2 * Cw * dCw_dh+2 * Ch * dCh_dh) * S / (C * C)
                end
        end;
    if iou_loss = ilCIOU then
        begin
            ar_gt := truth.w / truth.h;
            ar_pred := self.w / self.h;
            ar_loss := 4 / (PI * PI) * (arctan(ar_gt)-arctan(ar_pred)) * (arctan(ar_gt)-arctan(ar_pred));
            alpha := ar_loss / (1-I / U+ar_loss+0.000001);
            ar_dw := 8 / (PI * PI) * (arctan(ar_gt)-arctan(ar_pred)) * self.h;
            ar_dh := -8 / (PI * PI) * (arctan(ar_gt)-arctan(ar_pred)) * self.w;
            if C > 0 then
                begin
                    p_dx := p_dx + ((2 * (truth.x-self.x) * C-(2 * Cw * dCw_dx+2 * Ch * dCh_dx) * S) / (C * C));
                    p_dy := p_dy + ((2 * (truth.y-self.y) * C-(2 * Cw * dCw_dy+2 * Ch * dCh_dy) * S) / (C * C));
                    p_dw := p_dw + ((2 * Cw * dCw_dw+2 * Ch * dCh_dw) * S / (C * C)+alpha * ar_dw);
                    p_dh := p_dh + ((2 * Cw * dCw_dh+2 * Ch * dCh_dh) * S / (C * C)+alpha * ar_dh)
                end;
            if (Iw <= 0) or (Ih <= 0) then
                begin
                    p_dx := (2 * (truth.x-self.x) * C-(2 * Cw * dCw_dx+2 * Ch * dCh_dx) * S) / (C * C);
                    p_dy := (2 * (truth.y-self.y) * C-(2 * Cw * dCw_dy+2 * Ch * dCh_dy) * S) / (C * C);
                    p_dw := (2 * Cw * dCw_dw+2 * Ch * dCh_dw) * S / (C * C)+alpha * ar_dw;
                    p_dh := (2 * Cw * dCw_dh+2 * Ch * dCh_dh) * S / (C * C)+alpha * ar_dh
                end
        end;
    result.dt := p_dx;
    result.db := p_dy;
    result.dl := p_dw;
    result.dr := p_dh;
end;

function TBox.toTblr(): TBoxAbs;
var
    t, b, l, r: single;
begin
    //result := default(TBoxAbs);
    result.top := y-(h / 2);
    result.bot := y+(h / 2);
    result.left := x-(w / 2);
    result.right := x+(w / 2);
end;

class function TBox.fromFloat(const f: PSingle; const stride: SizeInt): TBox;
begin
    result.x := f[0];
    result.y := f[1 * stride];
    result.w := f[2 * stride];
    result.h := f[3 * stride];
end;

{ TDetectionsHelper }

class function TDetectionsHelper.Comparer(const a, b: TDetection): SizeInt;
var diff : single;
begin
    result :=0;
    if b.sort_class >= 0 then
        diff := b.prob[b.sort_class] - a.prob[b.sort_class]
    else
        diff := b.objectness - a.objectness;
   if diff >0 then
       exit(1)
   else if diff <0 then
       exit(-1);

end;

procedure TDetectionsHelper.doNMSObj(const classes: SizeInt; const thresh: single);
var
  i, j, k, total: SizeInt;
  swap: TDetection;
  a, b: TBox;
begin
  total := high(Self);
  k := total;
  i := 0;
  while i<= k do begin
      if self[i].objectness = 0 then
          begin
              swap := self[i];
              self[i] := self[k];
              self[k] := swap;
              dec(k);
          end
      else
        inc(i)
  end;
  total := k;
  for i := 0 to total do
      self[i].sort_class := -1;

  TTools<TDetection>.QuickSort(pointer(self),0, total, Comparer);
  for i := 0 to total do
      begin
          if self[i].objectness = 0 then
              continue;
          a := self[i].bbox;
          for j := i+1 to total do
              begin
                  if self[j].objectness = 0 then
                      continue;
                  b := self[j].bbox;
                  if a.iou(b) > thresh then
                      begin
                          self[j].objectness := 0;
                          for k := 0 to classes -1 do
                              self[j].prob[k] := 0
                      end
              end
      end
end;

procedure TDetectionsHelper.doNMSSort(const classes: SizeInt; const thresh: single);
var
  i,j,k, total: SizeInt;
  swap: TDetection;
  a: TBox;
  b: TBox;
begin
  total := high(self);
  k := total;
  i:=0;
  while i <= k do begin
      if self[i].objectness = 0 then
          begin
              swap := self[i];
              self[i] := self[k];
              self[k] := swap;
              dec(k);
          end
      else
        inc(i)
  end;
  total := k;
  for k := 0 to classes -1 do
      begin
          for i := 0 to total do
              self[i].sort_class := k;
          TTools<TDetection>.QuickSort(pointer(self), 0, total, Comparer);
          for i := 0 to total do
              begin
                  if self[i].prob[k] = 0 then
                      continue;
                  a := self[i].bbox;
                  for j := i+1 to total do
                      begin
                          b := self[j].bbox;
                          if a.iou(b) > thresh then
                              self[j].prob[k] := 0
                      end
              end
      end
end;

{ TData }

procedure TData.getRandomBatch(const n: SizeInt; const A, B: PSingle);
var
    j, i: SizeInt;
    index: SizeInt;
    S, D:PSingle;
begin
    for j := 0 to n -1 do
        begin
            index := random(self.X.h());
            //move(X.data[X.w() * index], A[j * X.w()], X.w() * sizeof(single));
            S := pointer(X.data + X.w() * index);
            D := A + j* X.w();
            for i:=0 to X.w()-1 do
                D[i]  := S[i];
            //move(y.data[y.w() * index], B[j * y.w()], Y.w() * sizeof(single));
            S := pointer(Y.data + Y.w() * index);
            D := B + j* Y.w();
            for i:=0 to Y.w()-1 do
                D[i]  := S[i]
        end
end;

procedure TData.getBatch(const n, offset: SizeInt; const A, B: PSingle);
var
    j, i: SizeInt;
    index: SizeInt;
    S, D:PSingle;
begin
    for j := 0 to n -1 do
        begin
            index := offset+j;
            S := pointer(X.data + X.w() * index);
            D := A + j * X.w();
            move(S[0], D[0], X.w() * sizeof(single));
            //for i:=0 to X.w()-1 do
            //    D[i]  := S[i];
            if assigned(B) then begin
                S := pointer(Y.data + Y.w() * index);
                D := B + j * Y.w();
                move(S[0], D[0], Y.w() * sizeof(single));
                //for i:=0 to Y.w()-1 do
                //  D[i]  := S[i]
            end
        end
end;

{ TTree }

class function TTree.loadFromFile(const fileName: string): TTree;
var
    t: TTree;
    fp: TextFile;
    line, id: string;
    vals:TArray<string>;
    last_parent, group_size, groups, n, parent, i: SizeInt;
begin
    result := default(TTree);
    //fp := fopen(filename, 'r');
    AssignFile(fp,filename);
    reset(fp);
    last_parent := -1;
    group_size := 0;
    groups := 0;
    n := 0;
    while not EOF(fp) do
        begin
            //id := calloc(256, sizeof(char));
            readln(fp,line);
            parent := -1;
            vals:=line.split([' ']);//, '%s %d', id,  and parent);
            id:=vals[0];
            parent := StrToInt(trim(vals[1]) );
            setLength(result.parent,n+1);//.reAllocate(n+1);// := realloc(result.parent, (n+1) * sizeof(int));
            result.parent[n] := parent;
            setLength(result.child, n+1);//.reAllocate(n+1);// := realloc(result.child, (n+1) * sizeof(int));
            result.child[n] := -1;
            insert(id, result.name, n);// := realloc(result.name, (n+1) * sizeof(string));
            result.name[n] := id;
            if parent <> last_parent then
                begin
                    inc(groups);
                    setLength(result.group_offset, groups);//.reAllocate(groups);// := realloc(result.group_offset, groups * sizeof(int));
                    result.group_offset[groups-1] := n-group_size;
                    setLength(result.group_size, groups);//.reAllocate(groups);// := realloc(result.group_size, groups * sizeof(int));
                    result.group_size[groups-1] := group_size;
                    group_size := 0;
                    last_parent := parent
                end;
            setLength(result.group, n+1);//.reAllocate(n+1);// := realloc(result.group, (n+1) * sizeof(int));
            result.group[n] := groups;
            if parent >= 0 then
                result.child[parent] := groups;
            inc(n);
            inc(group_size)
        end;
    inc(groups);
    setLength(result.group_offset, groups);//.reAllocate(groups);// := realloc(result.group_offset, groups * sizeof(int));
    result.group_offset[groups-1] := n-group_size;
    setLength(result.group_size, groups);//.reAllocate(groups) ;//:= realloc(result.group_size, groups * sizeof(int));
    result.group_size[groups-1] := group_size;
    result.n := n;
    result.groups := groups;
    setLength(result.leaf, n);// := TIntegers.Create(n);//, sizeof(int));
    for i := 0 to n -1 do
        result.leaf[i] := 1;
    for i := 0 to n -1 do
        if result.parent[i] >= 0 then
            result.leaf[result.parent[i]] := 0;
    CloseFile(fp);
    //result := AllocMem(1* sizeof(TTree));
    //result[0] := t;
    //exit(tree_ptr)
end;

{ TCostTypeHelper }

class function TCostTypeHelper.fromString(const s: string): TCostType;
begin
    if (CompareStr(s, 'seg') = 0) then
        exit(ctSEG);
    if CompareStr(s, 'sse') = 0 then
        exit(ctSSE);
    if CompareStr(s, 'masked') = 0 then
        exit(ctMASKED);
    if CompareStr(s, 'smooth') = 0 then
        exit(ctSMOOTH);
    if CompareStr(s, 'L1') = 0 then
        exit(ctL1);
    if CompareStr(s, 'wgan') = 0 then
        exit(ctWGAN);
    writeln('Couldn''t find cost type ', s, ' going with SSE'#10'');
    exit(ctSSE)
end;

function TCostTypeHelper.toString(): string;
begin
    result := copy(GetEnumName(TypeInfo(TCostType), ord(Self)), 3, 20)
end;

end.

