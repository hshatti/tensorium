unit quickchart;
{$ifdef FPC}
  {$mode Delphi}{$H+}
  {$ModeSwitch advancedrecords}
  {$ModeSwitch typehelpers}
  {$inline on}
{$endif}
{$pointermath on}
{.$define use_bgra}
{$define USE_FASTBITMAP}
interface
uses
  Classes, SysUtils{$ifdef LCL}, interfaces{$endif}
  {$if defined (FRAMEWORK_FMX)}, FMX.Types, UITypes, FMX.Controls, FMX.Graphics, FMX.forms
  {$elseif defined(LCL) or defined(FRAMEWORK_VCL)}, Controls, Graphics, Forms
  {$else}
  {$define FPC_GRAPH}
  , graph
  {$endif}
  , Types, Math
  {$ifdef USE_BGRA},BGRABitmap, BGRABitmapTypes, BGRACanvas,FPImage{$else}{$ifdef fpc}
  //, FPImage, FPCanvas
  {$endif}{$endif};
  //{$if not declared(TDoubleArrayArray)}
  type
    TDoubleArrayArray = TArray<TDoubleDynArray>;
  //{$endif}
const
  TFluentColors : TLongWordDynArray =[$a0FFB900,$a0E74850,$a00078D,$a00099BC
                                     ,$a0FF8C00,$a0E81120,$a00063B,$a02D7D9A
                                     ,$a0F7630C,$a0EA0050,$a08E8CD,$a000B7C3
                                     ,$a0CA5010,$a0C30050,$a06B69D,$a0038387
                                     ,$a0DA3B01,$a0E30080,$a08764B,$a000B294] ;
  TMaterial1Colors : TLongWordDynArray =[$c0c4ef55,  $c0ffb974 ,$c0ecec81, $c0fe9ba2, $c0dfe6e9
                                        ,$c094b800,  $c0e38409 ,$c0c9ce00, $c0e75c6c, $c0b2bec3
                                        ,$c0a7eaff,  $c07576ff ,$c0a0b1fa, $c0a879fd, $c0636e72
                                        ,$c06ecbfd,  $c03130d6 ,$c05570e1, $c09343e8, $c02d3436];
  TMaterial2Colors : TLongWordDynArray =[$c012C3FF,  $c0C4CB12 ,$c038E5C4, $c0DFA7FD, $c0674CED
                                        ,$c01F9FF7,  $c0A78912 ,$c038CBA3, $c0FA80D9, $c07134B5
                                        ,$c0245AEE,  $c0DD5206 ,$c0329400, $c0FA8099, $c0713483
                                        ,$c02720EA,  $c064141B ,$c0666200, $c0BB5857, $c0511E6F];
type
  {$ifdef FPC_GRAPH}
     TColor = LongWord;
  {$endif}
  { TDoubles2DHelper }
  TDoubles2DHelper = record helper for TDoubleArrayArray
  private
    function getCount: Integer;
    procedure setCount(AValue: Integer);
  public
    function Slice(start,finish:integer):TDoubleArrayArray;
    function Splice(start,deleteCount:integer;Items:TDoubleArrayArray):TDoubleArrayArray;
    function Push(v:TDoubleDynArray):TDoubleDynArray;
    function Pop():TDoubleDynArray;
    function UnShift(v: TDoubleDynArray): TDoubleDynArray;
    function Shift():TDoubleDynArray;
    property Count:Integer read getCount write setCount;
  end;
  { TDoubleDynArrayHelper }
  TDoubleDynArrayHelper = record helper for TDoubleDynArray
  private
    function getCount: Integer;
    procedure setCount(AValue: Integer);
  public
    function indexOf(const val: Double): integer;
    function Slice(start, finish:integer):TDoubleDynArray;
    function Splice(start,deleteCount:integer;Items:TDoubleDynArray):TDoubleDynArray;
    function Push(v:Double):Double;
    function Pop():Double;
    function UnShift(v:Double):Double;
    function Shift():Double;
    function min():Double;
    function max():Double;
    property Count:Integer read getCount write setCount;
    function Round():TIntegerDynArray;
    function Trunc():TIntegerDynArray;
    function Ceil():TIntegerDynArray;
    function Floor():TIntegerDynArray;
    function Map(const a,b,a1,b1:Double):TDoubleDynArray;overload;
    class function fill(const N:NativeInt; const start:Double; const finish:Double=0):TDoubleDynArray; static;
  end;

  TAutoScale=(asBoth,asX,asY,asNone);
  PDataRange=^TDataRange;
  TDataRange=record
    maxVal:Single;
    minVal:Single;
  end;

  PPixel=^Tpixel;
  {$if defined(Darwin) or defined(MACOS)}
     TPixel=record
       case byte of
       0:(n:LongWord);
       1:(a,r,g,b:byte);
     end;

  {$endif}
  {$ifdef MSWindows}
  { TPixel }
  TPixel=record
    class operator Implicit(const v:TColor):TPixel;
    class operator Implicit(const v:TPixel):TColor;
    case byte of
    0:(n:LongWord);
    1:(b,g,r,a:byte);
  end;
  {$endif}
  {$ifdef Linux}
  TPixel=record
    case byte of
    0:(n:LongWord);
    1:(r,g,b,a:byte);
  end;
  {$endif}
  TGradientStyle=(grTop,grLeft,grTopLeft,grCircular) ;
    { TGradient }
  TGradient=record
    Colors:array[0..1] of TPixel;
    style: TGradientStyle;
    class operator Implicit (const b:TPixel):TGradient;
    class operator Implicit (const b:TGradient):TPixel;
    class operator Implicit (const b:longword):TGradient;
    class operator Implicit (const b:TGradient):longword;
  end;
  TDataRanges=array of TDataRange;
  { TStyler }
  TStyler=record
  private
    Fopacity: single;
    function Getx2: smallint; inline;
    function Gety2: smallint; inline;
    procedure Setopacity(AValue: single);
    procedure Setx2(AValue: smallint); inline;
    procedure Sety2(AValue: smallint); inline;
  public
    stroke:TGradient;
    {$ifdef FRAMEWORK_FMX}
    strokeStyle:TStrokeDash;
    {$else}
    strokeStyle:TPenStyle;
    {$endif}
    strokeWidth:single;
    fill:TGradient;
//    strokeIsFill:boolean;
    x,y,w,h:integer;
    textalign:TAlignment;
    r:integer;
    property x2:smallint read Getx2 write Setx2;
    property y2:smallint read Gety2 write Sety2;
    property opacity:single read Fopacity write Setopacity;
  end;

  { TFastBitmap }
  TFastBitmap=class(TBitmap)
  private
    FAlphaBlend: boolean;
    FData:PPixel;
    {$ifdef FRAMEWORK_FMX}
    FBitmapData : TBitmapData;
    procedure beginUpdate;
    procedure endUpdate;
    {$endif}
  private
    FImageWidth:integer;
    FPenPos: TPoint;
    function TextExtent(const txt:string):TSize;
    procedure TextOut(const x, y:integer; const txt:string {$ifdef FRAMEWORK_FMX} ; const TextFlags:TFillTextFlags=[]{$endif});
    procedure SetAlphaBlend(AValue: boolean);
    function setOpacity(const a: TPixel; const alpha: integer): TPixel;
    function GetP(x, y: integer): TPixel;inline;
    procedure SetP(x, y: integer; const AValue: TPixel);inline;
    procedure setPixelInline(Line: PPixel; x: Integer; AValue: TPixel); inline;
    function Blend(const src, dst, alpha:single):single;inline;
    procedure SetPenPos(AValue: TPoint);
  public
    Styler:TStyler;
//    procedure SetSize({$ifdef FRAMEWORK_FMX}const{$endif} AWidth, AHeight: integer); override;
    constructor Create; override;
    procedure Clear(Pixel:TPixel);
//    procedure Clear; overload; override;
    procedure setPixel(const x, y:integer; const pixel:TPixel);inline;
    function xy(const x, y: integer): integer;inline;
    property P[x,y:Integer]:TPixel read GetP write SetP;
    property AlphaBlend:boolean read FAlphaBlend write SetAlphaBlend;
    procedure LineAA(x0, y0, x1, y1: integer);overload;
    procedure LineAA(x0, y0, x1, y1, th: integer); overload;
    procedure Line(x0, y0, x1, y1:integer; thickness: integer =-1);overload;
    procedure LineTo(x, y:integer; thickness:integer=-1);overload;
    procedure LineToAA(x, y: integer);inline;
    procedure Rect(x1, y1, x2, y2: integer);overload;
    procedure FillRect(const x1, y1, x2, y2: integer);
    procedure Ellipse(x0, y0, x1, y1:integer);overload;
    procedure FillEllipse(x0, y0, x1, y1:integer);overload;
    procedure EllipseCAA(cx, cy: integer; rx, ry: single; fill: boolean);overload;//inline;
    procedure EllipseAA(x1, y1, x2, y2: integer);overload; inline;
    procedure EllipseAA(x0, y0, x1, y1: longint; th: single); overload;
    property PenPos:TPoint read FPenPos write SetPenPos;
    procedure MoveTo(x, y: integer);
  end;

  TPlot=class;
  { TPlotAxis }
  TPlotAxis=class(TComponent)
  private
    FColor: TPixel;
    FFont: {$ifdef USE_BGRA}TBGRAFont{$else}TFont{$endif};
    {$ifdef FRAMEWORK_FMX}
    FFontColor : {$ifdef FRAMEWORK_FMX}TAlphaColor{$else}TColor{$endif};
    {$endif}
    FOnUpdate: TNotifyEvent;
    FPlot:TPlot;
    FSpacing: integer;
    FTickWidth: integer;
    Fwidth: integer;
    FMinimumTicks:integer;
    procedure SetColor(AValue: TPixel);
    procedure SetFont(AValue: {$ifdef USE_BGRA}TBGRAFont{$else}TFont{$endif});
    procedure SetMinimumTicks(AValue: integer);
    procedure SetOnUpdate(AValue: TNotifyEvent);
    procedure SetSpacing(AValue: integer);
    procedure SetTickWidth(AValue: integer);
    procedure Setwidth(AValue: integer);
  public
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
    property Font:{$ifdef USE_BGRA}TBGRAFont{$else}TFont{$endif} read FFont write SetFont;
    property Color:TPixel read FColor write SetColor;
    property width:integer read Fwidth write Setwidth;
    property TickWidth:integer read FTickWidth write SetTickWidth;
    property Spacing:integer read FSpacing write SetSpacing;
    property MinimumTicks:integer read FMinimumTicks write SetMinimumTicks;
    property OnUpdate:TNotifyEvent read FOnUpdate write SetOnUpdate ;
  end ;
  TGraphStyle=(gsScatter,gsBar,gsHBar,gsLine,gsArea);
  TGraphOption=record
    Fill:TGradient;
    Stroke:TGradient;
    StrokeWidth:integer;
    Stack:boolean;
    Horizontal:boolean;
    Style:TGraphStyle;
    Size:integer;
  end;
  //TBarOption=class (TGraphOption)
  //  Stack:boolean;
  //end;
  //
  //THBarOption=class (TGraphOption)
  //
  //end;
  //
  //TLineOption=class(TGraphOption)
  //
  //end;
  //
  //TScatterOption=class(TGraphOption)
  //
  //end;
  //
  //TAreaOption=class(TGraphOption)
  //
  //end;
  { TPlot }
  TPlot=class({$ifdef FRAMEWORK_FMX}TControl{$else}TGraphicControl{$endif})
  private
    FMouseDownPos, FMouseUpPos:TPoint;
    FMouseButtonDown : set of TMouseButton;
    FActive: boolean;
    FAutoScale: TAutoScale;
    FAutoTicks: boolean;
    FAxisX: TPlotAxis;
    FAxisY: TPlotAxis;
    FBackground: TGradient;
    FBarsRatio: single;
    FColors: TLongWordDynArray;
    FData:TDoubleArrayArray;
    FMaxs,FMins:TDoubleDynArray;
    FMinY,FMaxY:double;
    FMinX,FMaxX:double;
    yMax,yMin:double;
    FBases:TDoubleArrayArray;
    FPaddingLeft,FPaddingRight,FPaddingTop,FPaddingBottom,FtxtWidthY,FtxtWidthX,FtxtHeightY,FTxtHeightX:smallint;
    FGraphOptions:array of TGraphOption;
    function DoCombineColors(const color1, color2: TPixel): TPixel;
    function GetDataCount: integer;
    function GetGraphOptions(index: integer): TGraphOption;
    function GetPlotHeight: smallint;
    function GetPlotWidth: smallint;
    procedure SetActive(AValue: boolean);
    procedure SetAutoScale(AValue: TAutoScale);
    procedure SetAutoTicks(AValue: boolean);
    procedure SetAxisX(AValue: TPlotAxis);
    procedure SetAxisY(AValue: TPlotAxis);
    procedure SetBackground(AValue: TGradient);
    procedure SetBarsRatio(AValue: single);
    procedure SetColors(AValue: TLongWordDynArray);
    procedure SetDataCount(AValue: integer);
    procedure SetGraphOptions(index: integer; AValue: TGraphOption);
    procedure SetMaxX(AValue: double);
    procedure SetMaxY(AValue: double);
    procedure SetMinX(AValue: double);
    procedure SetMinY(AValue: double);
    procedure SetPaddingBottom(AValue: smallint);
    procedure SetPaddingLeft(AValue: smallint);
    procedure SetPaddingRight(AValue: smallint);
    procedure SetPaddingTop(AValue: smallint);
    procedure SetParent({$ifdef FRAMEWORK_FMX}const NewParent: TFMXObject {$else} NewParent: TWinControl{$endif}); override;
    procedure xScale;
    procedure yScale;
  published
  protected
    procedure Resize; override;
  public
    FBitmap:{$ifdef USE_BGRA}TBGRABitmap{$else}{$ifdef USE_FASTBITMAP}TFastBitmap{$else}TBitmap{$endif}{$endif};
    procedure ShowPlot;
    procedure OnOwnerShow(Sender:TObject);
    function alignMaxTicks(range: extended; minTicks: integer=2): TDoubleDynArray;
//    procedure SetBounds(aLeft, aTop, aWidth, aHeight: integer); override;
    property PlotWidth:smallint read GetPlotWidth;
    property PlotHeight:smallint read GetPlotHeight;
    property PaddingLeft:smallint read FPaddingLeft write SetPaddingLeft;
    property PaddingRight:smallint read FPaddingRight write SetPaddingRight;
    property PaddingTop:smallint read FPaddingTop write SetPaddingTop;
    property PaddingBottom:smallint read FPaddingBottom write SetPaddingBottom;
    property Active:boolean read FActive write SetActive;
    property AutoTicks:boolean read FAutoTicks write SetAutoTicks;
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
    property GraphOptions[index:integer]:TGraphOption read GetGraphOptions write SetGraphOptions;
    function Add(const D: TDoubleDynArray; const GraphStyle: TGraphStyle=gsBar): Integer;        overload;
    function Add(const D: TSingleDynArray; const GraphStyle: TGraphStyle=gsBar): Integer;       overload;
    function Add(const D: TIntegerDynArray; const GraphStyle: TGraphStyle=gsBar ): Integer;     overload;
    //function Add(const D: TExtendedDynArray; const GraphStyle: TGraphStyle=gsBar): Integer;    overload;
    //function Add(const D: TComplexArrayF; const GraphStyle: TGraphStyle=gsBar): Integer;      overload;
    //function Add(const D: TComplexArrayD; const GraphStyle: TGraphStyle=gsBar): Integer;      overload;
    function Add(const D: TDoubleDynArray;  const base:TDoubleDynArray; const GraphStyle: TGraphStyle=gsBar): Integer;   overload;
    function Add(const D: TSingleDynArray;  const base:TSingleDynArray; const GraphStyle: TGraphStyle=gsBar): Integer;   overload;
    function Add(const D: TIntegerDynArray; const base:TIntegerDynArray; const GraphStyle: TGraphStyle=gsBar ): Integer;  overload;
    //function Add(const D: TExtendedDynArray;const base:TExtendedDynArray; const GraphStyle: TGraphStyle=gsBar): Integer; overload;
    //function Add(const D: TComplexArrayF;    const base:TComplexArrayF; const GraphStyle: TGraphStyle=gsBar): Integer;   overload;
    function Remove(const index:integer):TDoubleDynArray;
    function GetData(Index: Integer): TDoubleDynArray;
    function GetRanges(Index: integer): TDataRange;
    procedure SetData(Index: Integer; AValue: TDoubleDynArray);
    procedure SetRanges(Index: integer; AValue: TDataRange);
    procedure Paint; override;
    procedure BeforeDraw(const FBitmap:TFastBitmap); virtual;
    procedure MouseDown(Button: TMouseButton; Shift: TShiftState; X, Y: {$ifdef FRAMEWORK_FMX}single{$else}Integer{$endif}); override;
    procedure MouseUp(Button: TMouseButton; Shift: TShiftState; X, Y: {$ifdef FRAMEWORK_FMX}single{$else}Integer{$endif}); override;
    procedure MouseMove(Shift: TShiftState; X, Y: {$ifdef FRAMEWORK_FMX}single{$else}Integer{$endif}); override;
    property Ranges[Index:integer]:TDataRange read GetRanges write SetRanges;
    property DataCount:integer read GetDataCount write SetDataCount;
    property Data[Index:Integer]:TDoubleDynArray read GetData write SetData;
    property AxisX:TPlotAxis read FAxisX write SetAxisX;
    property AxisY:TPlotAxis read FAxisY write SetAxisY;
    property Background:TGradient read FBackground write SetBackground;
    property BarsRatio:single read FBarsRatio write SetBarsRatio;
    property Colors:TLongWordDynArray read FColors write SetColors;
    property AutoScale:TAutoScale read FAutoScale write SetAutoScale;
    property MaxX:double read FMaxX write SetMaxX;
    property MaxY:double read FMaxY write SetMaxY;
    property MinX:double read FMinX write SetMinX;
    property MinY:double read FMinY write SetMinY;

    //property Scatter
  end;

  TPixelFormat = {$ifdef FRAMEWORK_FMX}FMX.Types.{$else}Graphics.{$endif}TPixelFormat;
{$ifdef USE_BGRA}
//operator := (const p:TPixel):TBGRAPixel;  inline;
//operator := (const p:LongWord):TBGRAPixel; inline;
//operator := (const p:byte):TBGRAPixel; inline;
//operator := (const p:TBGRAPixel):TPixel;  inline;
//operator := (const p:TBGRAPixel):LongWord; inline;
{$endif}
//operator := (const c:TColor):TPixel; inline;
//operator := (const c:TPixel):TColor; inline;
//operator := (const c:TPixel):TPixel; inline;
//operator := (const c:TPixel):TPixel; inline;

function Pixel(code:byte):TPixel; inline;    overload;
function Pixel(const r,g,b,a:Byte):TPixel;   overload;
function Pixel(code:string):TPixel; inline;  overload;
function blend(const p1,p2:TPixel):TPixel;
//function blend(const p1,p2:TPixel):TPixel;
var  Plot:Tplot;
implementation

function blend(const p1,p2:TPixel):TPixel;
var invA:word;
begin
   with p2 do
     if a=$ffff then
       begin
         result:=p2;
       end
     else if a=0 then
       result:=p1
     else begin
       invA:=$ffff-a;
       result.r:=trunc(r*a/$ffff + p1.r*p1.a*invA / $fffffff);
       result.g:=trunc(g*a/$ffff + p1.g*p1.a*invA / $fffffff);
       result.b:=trunc(b*a/$ffff + p1.b*p1.a*invA / $fffffff);
       result.a:=a+trunc(p1.a*invA / $ffff);
     end;
end;
//function blend(const p1,p2:TPixel):TPixel;
//var invA:byte;
//begin
//   with p2 do
//     if Alpha=$ff then
//       begin
//         result:=p2;
//       end
//     else if Alpha=0 then
//       result:=p1
//     else begin
//       invA:=$ff-Alpha;
//       result.Red  :=trunc(Red*Alpha/$ff   + p1.Red*p1.Alpha*invA / $ffff);
//       result.Green:=trunc(Green*Alpha/$ff + p1.Green*p1.Alpha*invA / $ffff);
//       result.Blue :=trunc(Blue*Alpha/$ff  + p1.Blue*p1.Alpha*invA / $ffff);
//       result.Alpha:=Alpha+trunc(p1.Alpha*invA / $ff);
//     end;
//end;

function Pixel(const r,g,b,a:Byte):TPixel;
begin
  result.r:=r;result.g:=g;result.b:=b;result.a:=a;
end;
//function Pixel(const code:LongWord):TPixel; inline;
//begin
//  result.n:=code
//end;
function Pixel(code:byte):TPixel;inline;
begin
  result.r:=code;
  result.g:=code;
  result.b:=code;
  result.a:=$ff;
end;
function Pixel(code:string):TPixel; inline;
begin
  if code='' then exit;
  Assert(code[1]='#');
  code[1]:='$';
  Result.n:=StrToInt(Code);
end;

function lerp(const a,b,x:single):single; inline; overload;
begin
  result:=a+x*(b-a)
end;
function lerp(const a,b:byte;x:single):byte; inline; overload;
begin
  result:=round(a+x*(b-a))
end;
function pixelInterp(const a,b:TPixel;const x:single):TPixel;inline;
begin
  result.r:=lerp(a.r,b.r,x);
  result.g:=lerp(a.g,b.g,x);
  result.b:=lerp(a.b,b.b,x);
  result.a:=lerp(a.a,b.a,x);
end;
{$ifdef USE_BGRA}
//operator :=(const p: TPixel): TBGRAPixel;
//begin
//  result.FromRGB(p.r,p.g,p.b,p.a);
//end;
//
//operator:=(const p: LongWord): TBGRAPixel;
//begin
//  result.FromColor(p shr 8,p and $ff);
//end;
//
//operator:=(const p: TBGRAPixel): TPixel;
//begin
//  with result do begin
//    r:=p.red;
//    g:=p.green;
//    b:=p.blue;
//    a:=p.alpha;
//  end;
//
//end;
//
//operator:=(const p: byte): TBGRAPixel;
//begin
//  result.FromRGB(p,p,p);
//end;
//operator := (const p:TBGRAPixel):LongWord; inline;
//begin
//  result:=(p.blue shl 24) or (p.green shl 16) or (p.blue shl 8) or (p.alpha);
//end;
{$endif}
//operator:=(const c: TColor): TPixel;
//begin
//  result.n:=(c shl 8) or $ff;
//end;
//
//operator:=(const c: TPixel): TColor;
//begin
//  result:=c.n shr 8;
//end;
//
//operator:=(const c: TPixel): TPixel;
//begin
//  with result do begin
//    r:=c.red shr 8;
//    g:=c.green shr 8;
//    b:=c.blue shr 8;
//    a:=c.Alpha shr 8;
//
//  end;
//end;
//operator := (const c:TPixel):TPixel; inline;
//begin
//  with result do
//    begin
//      red:=c.r shl 8;
//      green:=c.g shl 8;
//      blue:=c.b shl 8;
//      alpha:=c.a shl 8;
//    end;
//end;
{ TPlotAxis }
procedure TPlotAxis.SetColor(AValue: TPixel);
begin
  if FColor.n=AValue.n then Exit;
  FColor:=AValue;
  if FPlot.Active then FPlot.repaint;
  if Assigned(FOnUpdate) then FOnUpdate(Self)
end;
procedure TPlotAxis.SetFont(AValue: {$ifdef USE_BGRA}TBGRAFont{$else}TFont{$endif});
begin
  if FFont=AValue then Exit;
  FFont:=AValue;
  if Assigned(FOnUpdate) then FOnUpdate(Self)
end;
procedure TPlotAxis.SetMinimumTicks(AValue: integer);
begin
  if FMinimumTicks=AValue then Exit;
  FMinimumTicks:=AValue;
  FPlot.repaint;
end;
procedure TPlotAxis.SetOnUpdate(AValue: TNotifyEvent);
begin
  //if FOnUpdate=AValue then Exit;
  FOnUpdate:=AValue;
end;
procedure TPlotAxis.SetSpacing(AValue: integer);
begin
  if FSpacing=AValue then Exit;
  FSpacing:=AValue;
  if FPlot.Active then FPlot.repaint ;
  if Assigned(FOnUpdate) then FOnUpdate(Self)
end;
procedure TPlotAxis.SetTickWidth(AValue: integer);
begin
  if FTickWidth=AValue then Exit;
  FTickWidth:=AValue;
  if FPlot.Active then FPlot.repaint ;
  if Assigned(FOnUpdate) then FOnUpdate(Self)
end;
procedure TPlotAxis.Setwidth(AValue: integer);
begin
  if Fwidth=AValue then Exit;
  Fwidth:=AValue;
  if FPlot.Active then FPlot.repaint ;
  if Assigned(FOnUpdate) then FOnUpdate(Self)
end;
constructor TPlotAxis.Create(AOwner: TComponent);
begin
  Assert(AOwner is TPlot);
  inherited Create(AOwner);
  FPlot:=TPlot(AOwner);
  FColor:=Pixel($ff,$ff,$ff,$ff);
  FWidth:=1;
  FTickWidth:=2;
  FSpacing:=4;
  FMinimumTicks:=4;
  {$ifdef USE_BGRA}
  FFont:=TBGRAFont.Create;
  FFont.BGRAColor:=Pixel($ff,$ff,$ff,$aa);
  {$else}
  FFont:=TFont.Create;
  {$ifdef FRAMEWORK_FMX}
  FFontColor := $aaFFFFFF;
  FFont.Family := 'Segoe UI';
  {$else}
  FFont.Color:=Pixel($ff,$ff,$ff,$aa);
  FFont.Name:='Segoe UI';
  FFont.Quality:=fqNonAntialiased;
  {$endif}
  FFont.Size := 10;
  {$endif}
end;
destructor TPlotAxis.Destroy;
begin
  FreeandNil(FFont);
  inherited Destroy;
end;
{ TStyler }
procedure TStyler.Setopacity(AValue: single);
begin
  if Fopacity=AValue then Exit;
  Assert((AValue>=0) and (AValue<=1),'Opacity must be between 0 1nd 1');
  Fopacity:=AValue;
end;
function TStyler.Getx2: smallint;
begin
  result:=x+w
end;
function TStyler.Gety2: smallint;
begin
  result:=y+h
end;
procedure TStyler.Setx2(AValue: smallint);
begin
  w:=AValue-x
end;
procedure TStyler.Sety2(AValue: smallint);
begin
  h:=aValue-y
end;
{ TDoubles2DHelper }
function TDoubles2DHelper.getCount: Integer;
begin
  result := Length(Self);
end;
procedure TDoubles2DHelper.setCount(AValue: Integer);
begin
  setLength(Self, AValue)
end;
function TDoubles2DHelper.Slice(start, finish: integer): TDoubleArrayArray;
var i,C:integer;
begin
  C:=Length(Self);
  if start<0 then start:=C+start;
  if finish<0 then finish:=C + finish;
  result:=copy(Self,start,finish - start)
end;
function TDoubles2DHelper.Splice(start, deleteCount: integer;
  Items: TDoubleArrayArray): TDoubleArrayArray;
var l,n,i,C:integer;
begin
  C:=Length(Self);
  if start<0 then start:=C+start;
  l:=Length(Items);
  if deleteCount>0 then
    result:=copy(Self,start,deleteCount)
  else
    result:=nil;
  n:=l-deleteCount ;
  if n>0 then  begin
    setLength(Self,C+n);
    C:=C+n;
    for i:=C-1 downto start+1 do
      Self[i]:=Self[i-1]
  end;
  if n<0 then begin
    for i:=start to C-1+n do
      Self[i]:=Self[i-n];SetLength(Self,C+n)
  end;
  if l>0 then
    for i:=0 to l-1 do
      Self[start+i]:=Items[i];
end;
function TDoubles2DHelper.Push(v: TDoubleDynArray): TDoubleDynArray;
var C:integer;
begin
  c:=Length(Self)+1;
  SetLength(Self,C);
  Self[C-1]:=v;
  result:=v
end;
function TDoubles2DHelper.Pop(): TDoubleDynArray;
var C:integer;
begin
  C:=Length(Self);
  if C>0 then begin
    result:=Self[C-1];
    SetLength(Self,C-1)
  end
end;
function TDoubles2DHelper.UnShift(v: TDoubleDynArray): TDoubleDynArray;
var i,C:integer;
begin
  C:=Length(Self);
  setLength(Self,C+1);
  for i:=C downto 1 do
    Self[i]:=Self[i-1];
  Self[0]:=v;
  result:=v
end;
function TDoubles2DHelper.Shift(): TDoubleDynArray;
var i,C:integer;
begin
  C:=Length(Self);
  result:=Self[0];
  for i:=1 to C-1 do
    Self[i-1]:=Self[i];
  SetLength(Self,C-1)
end;
{ TDoubleDynArrayHelper }
function TDoubleDynArrayHelper.getCount: Integer;
begin
  result := Length(Self)
end;
procedure TDoubleDynArrayHelper.setCount(AValue: Integer);
begin
  setLength(Self, AValue)
end;
function TDoubleDynArrayHelper.indexOf(const val: Double): integer;
var C:integer;
begin
  C:=Length(Self);
  for result:=0 to C-1 do
    if Self[result]=val then
      exit;
  result:=-1
end;
function TDoubleDynArrayHelper.Slice(start, finish: integer): TDoubleDynArray;
var i,C:integer;
begin
  C:=Length(Self);
  if start<0 then start:=C+start;
  if finish<0 then finish:=C + finish;
  result:=copy(Self,start,finish - start)
end;
function TDoubleDynArrayHelper.Splice(start, deleteCount: integer;
  Items: TDoubleDynArray): TDoubleDynArray;
var l,n,i,C:integer;
begin
  C:=Length(Self);
  if start<0 then start:=C+start;
  l:=Length(Items);
  if deleteCount>0 then
    result:=copy(Self,start,deleteCount)
  else
    result:=nil;
  n:=l-deleteCount ;
  if n>0 then  begin
    setLength(Self,C+n);
    C:=C+n;
    for i:=C-1 downto start+1 do
      Self[i]:=Self[i-1]
  end;
  if n<0 then begin
    for i:=start to C-1+n do
      Self[i]:=Self[i-n];SetLength(Self,C+n)
  end;
  if l>0 then
    for i:=0 to l-1 do
      Self[start+i]:=Items[i];
end;
function TDoubleDynArrayHelper.Push(v: Double): Double;
var C:integer;
begin
  c:=Length(Self)+1;
  SetLength(Self,C);
  Self[C-1]:=v;
  result:=v
end;
function TDoubleDynArrayHelper.Pop(): Double;
var C:integer;
begin
  C:=Length(Self);
  if C>0 then begin
    result:=Self[C-1];
    SetLength(Self,C-1)
  end
end;
function TDoubleDynArrayHelper.UnShift(v: Double): Double;
var i,C:integer;
begin
  C:=Length(Self);
  setLength(Self,C+1);
  for i:=C downto 1 do
    Self[i]:=Self[i-1];
  Self[0]:=v;
  result:=v
end;
function TDoubleDynArrayHelper.Shift(): Double;
var i,C:integer;
begin
  C:=Length(Self);
  result:=Self[0];
  for i:=1 to C-1 do
    Self[i-1]:=Self[i];
  SetLength(Self,C-1)
end;
function TDoubleDynArrayHelper.min(): Double;
begin
  result:=MinValue(self);
end;
function TDoubleDynArrayHelper.max(): Double;
begin
  result:=MaxValue(self);
end;
function TDoubleDynArrayHelper.Round: TIntegerDynArray;
var i:integer;
begin
  setLength(result,Count);
  for i:=0 to Self.Count-1 do
    result[i]:=System.round(Self[i]);
end;
function TDoubleDynArrayHelper.Trunc(): TIntegerDynArray;
var i:integer;
begin
  setLength(result,Count);
  for i:=0 to Self.Count-1 do
    result[i]:=System.trunc(Self[i]);
end;
function TDoubleDynArrayHelper.Ceil(): TIntegerDynArray;
var i:integer;
begin
  setLength(result,Count);
  for i:=0 to Self.Count-1 do
    result[i]:=Math.Ceil(Self[i]);
end;
function TDoubleDynArrayHelper.Floor(): TIntegerDynArray;
var i:integer;
begin
  setLength(result,Count);
  for i:=0 to Self.Count-1 do
    result[i]:=Math.Floor(Self[i]);
end;
function TDoubleDynArrayHelper.Map(const a, b, a1, b1: Double): TDoubleDynArray;
var i:integer;v,v1:Double;
begin
  v:=b-a;
  v1:=b1-a1;
//  writeLn('a:[',v:2:2,'] ,b:[',v1:2:2,']');
  setLength(result,Count);
  if v=0 then
    for i:=0 to Count-1 do
      result[i]:=a1
  else
    for i:=0 To Count-1 do
      Result[i]:=a1+v1*(Self[i]-a)/v;
end;

class function TDoubleDynArrayHelper.fill(const N: NativeInt; const start: Double; const finish: Double): TDoubleDynArray;
var i:NativeInt;
  step:Double;
begin
  setLength(Result, N);
  step:=(finish-start)/N;
  for i:=0 to N-1 do
    result[i]:=start + i * step
end;
{ TPixel }
class operator TPixel.Implicit(const v: TColor): TPixel;
begin
  {$ifdef MSWINDOWS}
  result.n := ((v shr $10) and $ff) or ((v and $ff) shl $10) or $ff000000;
  {$else}
  result.n := v or $ff000000;
  {$endif}
end;
class operator TPixel.Implicit(const v: TPixel): TColor;
begin
  result := v.n and $00ffffff;
  {$ifdef MSWINDOWS}
  result := ((result shr $10) and $ff) or ((result and $ff) shl $10)
  {$else}
  {$endif}
end;
{ TGradient }
class operator TGradient.Implicit(const b: TPixel): TGradient;
begin
  result.Colors[0]:=b;
  result.Colors[1]:=b;
end;
class operator TGradient.Implicit(const b: TGradient): TPixel;
begin
  result:=b.Colors[0]
end;
class operator TGradient.Implicit(const b: longword): TGradient;
begin
  result.Colors[0].n:=b;
  result.Colors[1].n:=b;
end;
class operator TGradient.Implicit(const b: TGradient): longword;
begin
  result:=b.Colors[0].n
end;

{ TFastBitmap }
//procedure TFastBitmap.SetSize({$ifdef FRAMEWORK_FMX}const{$endif} AWidth, AHeight: integer);
//begin
//  inherited SetSize(AWidth, AHeight);
//  FImageWidth:=RawImage.Description.BitsPerLine div RawImage.Description.BitsPerPixel;
//  FData:=ScanLine[0];
//end;
{$ifdef FRAMEWORK_FMX}
procedure TFastBitmap.beginUpdate;
begin
  Map(TMapAccess.ReadWrite, FBitmapData)
end;
procedure TFastBitmap.endUpdate;
begin
  Unmap(FBitmapData);
end;
{$endif}
function TFastBitmap.GetP(x, y: integer): TPixel;
begin
  {$ifdef FRAMEWORK_FMX}
  FData:=FBitmapData.GetScanLine(y);
  {$else}
  FData:=ScanLine[y];
  {$endif}
  result:=FData[x]
end;

procedure TFastBitmap.SetP(x, y: integer; const AValue: TPixel);
var
  c:TPixel;
  invA:byte;
  i:integer;
begin
  if (x<0) or (y<0) or (x>=width) or (y >= height) then exit;
  //assert((x>=0) and (y>=0) and (x<width) and (y<Height), format('Pixel out of bounds! [%d , %d]', [width, height]));
  {$ifdef FRAMEWORK_FMX}
  FData:=FBitmapData.GetScanLine(y);
  {$else}
  FData:=ScanLine[y];
  {$endif}
  if FAlphaBlend then
    try
      if AValue.a=$ff then
        begin
          FData[x]:=Avalue;
          exit;
        end;
      if AValue.a=0 then
        exit;
      c:=FData[x];
      invA:=$ff-AValue.a;
      c.r:=trunc(AValue.r*AValue.a/$ff + c.r*c.a*invA / $ffff);
      c.g:=trunc(AValue.g*AValue.a/$ff + c.g*c.a*invA / $ffff);
      c.b:=trunc(AValue.b*AValue.a/$ff + c.b*c.a*invA / $ffff);
      c.a:=Avalue.a+trunc(c.a*invA / $ff);
      FData[x]:=c;
      exit;
    except
      raise Exception.Create('Error on pixel setting');
    end
  else
    FData[x]:=AValue;
end;
procedure TFastBitmap.setPixelInline(Line: PPixel; x: Integer; AValue: TPixel);
var
  c:TPixel;
  invA:byte;
  i:integer;
begin
  //assert((x>=0) and (y>=0) and (x<width) and (y<Height), 'Pixel out of bounds!');
  if FAlphaBlend then
    begin
      if AValue.a=$ff then
        begin
          Line[x]:=Avalue;
          exit;
        end;
      if AValue.a=0 then
        exit;
      c:=Line[x];
      invA:=$ff-AValue.a;
      c.r:=trunc(AValue.r*AValue.a/$ff + c.r*c.a*invA / $ffff);
      c.g:=trunc(AValue.g*AValue.a/$ff + c.g*c.a*invA / $ffff);
      c.b:=trunc(AValue.b*AValue.a/$ff + c.b*c.a*invA / $ffff);
      c.a:=Avalue.a+trunc(c.a*invA / $ff);
      Line[x]:=c;
      exit;
    //except
    //  raise Exception.Create('Error on pixel setting');
    end
  else
    Line[x]:=AValue;
end;
function TFastBitmap.Blend(const src, dst, alpha: single): single;
var val2:single;
begin
  result := src * alpha + dst * (1-alpha);
end;
procedure TFastBitmap.SetPenPos(AValue: TPoint);
begin
  if FPenPos=AValue then Exit;
  FPenPos:=AValue;
end;
procedure TFastBitmap.Clear(Pixel: TPixel);
var i,j:integer;
begin
    for j:=0 to Height-1 do begin
  {$ifdef FRAMEWORK_FMX}
  FData:=FBitmapData.GetScanLine(j);
  {$else}
  FData:=ScanLine[j];
  {$endif}
    for i:=0 to Width-1 do
      FData[i]:=Pixel
  end;
end;
procedure TFastBitmap.setPixel(const x, y: integer; const pixel: TPixel);
var Data :PPixel;
begin
  {$ifdef FRAMEWORK_FMX}
  FData:=FBitmapData.GetScanLine(y);
  {$else}
  FData:=ScanLine[y];
  {$endif}
  Data[x] := pixel
end;
//procedure TFastBitmap.Clear;
//begin
//  inherited Clear;
//end;
function TFastBitmap.xy(const x, y: integer): integer;
begin
  result:=x + y*FImageWidth;
end;
constructor TFastBitmap.Create;
begin
  inherited Create;
  {$ifdef FRAMEWORK_FMX}
  {$else}
  PixelFormat:=pf32bit;
  {$endif}
  AlphaBlend:=true;
  Styler.strokeWidth := 1;
end;
function TFastBitmap.setOpacity(const a: TPixel; const alpha: integer): TPixel;
begin
  result:=a;
  result.a:=trunc(result.a*alpha div 255);
end;

procedure swap(var a,b:integer); inline;
var tmp:integer;
begin
  tmp:=b;
  b:=a;
  a:=tmp
end;
function ipart(a:single):integer;inline;
begin
  exit(floor(a))
end;
function fpart(a:single):single;inline;
begin
  exit(a - ipart(a))
end;
function rfpart(a:single):single;inline;
begin
  exit(1-frac(a))
end;
procedure TFastBitmap.LineAA(x0, y0, x1, y1: integer);
var
    c1,c2,c:TPixel;
    steep:boolean;
    x:integer;
    xpxl1, xpxl2, ypxl1, ypxl2, w:integer;
    dx, dy, intersectY, gradient , a1, a2:single;
begin
  // https://www.geeksforgeeks.org/anti-aliased-line-xiaolin-wus-algorithm/
  c1:= styler.stroke.colors[0];
  a1 := c1.a;
  c2:=styler.stroke.colors[0];
  a2 := c2.a;
  steep := abs(y1 - y0) > abs(x1 - x0);
  // swap the co-ordinates if slope > 1 or we
  // draw backwards
  if steep then
  begin
      swap(x0, y0);
      swap(x1, y1);
  end;
  if x0 > x1 then
  begin
      swap(x0, x1);
      swap(y0, y1);
  end;
  // compute the slope
   dx := x1 - x0;
   dy := y1 - y0;
  if dx = 0.0 then
    gradient := 1
  else
    gradient := dy / dx;
  xpxl1 :=x0;
  xpxl2 :=x1;
  intersectY := y0;
  // main loop
  if steep then
  begin
      for x := xpxl1 to xpxl2 do
      begin
          // pixel coverage is determined by fractional
          // part of y co-ordinate
          c1.a := trunc(a1 * fPart(intersectY));
          setP(trunc(intersectY), x, c1);
          c1.a := trunc(a1 * rfPart(intersectY));
          setP(trunc(intersectY) - 1, x, c1);
          intersectY := intersectY + gradient;
      end
  end
  else
  begin
      for x := xpxl1 to xpxl2 do
      begin
          // pixel coverage is determined by fractional
          // part of y co-ordinate
          c1.a := trunc(a1 * fPart(intersectY));
          setP(x, trunc(intersectY), c1);
          c1.a := trunc(a1 * rfPart(intersectY));
          setP(x, trunc(intersectY) - 1, c1);
          intersectY := intersectY + gradient;
      end
  end
end;
procedure TFastBitmap.Line(x0, y0, x1, y1: integer; thickness: integer);
var dx, dy, sx, sy:Integer; steep:boolean; slope :single; c1, c2:TPixel;
procedure ln(const x2, y2:Integer; const err:boolean=false);
var
  x, y, xe, ye: integer;
begin
  if steep then begin
    y := 0; xe:=0;
    while y <> dy do begin
      x := x2 + y * dx div dy;
      //if not inRange(x, 0, Width-1) then break;
      //if not inRange(y2 + y, 0, Height-1) then break;
      if err and (x<>xe) then
        setP(x - sx * ord(sx<>sy), y2 + y - sy * ord(sx=sy), c1);
      setP(x, y2 + y, c1);
      xe := x;
      inc(y, sy);
    end
  end else begin
    x := 0; ye:=0;
    while x <> dx do begin
      y := y2 + x * dy div dx;
      //if not inRange(x2 + x, 0, Width-1) then break;
      //if not inRange(y, 0, Height-1) then break;
      if err and (y<>ye) then
        setP(x2 + x - sx * ord(sx<>sy), y - sy * ord(sx=sy), c1);
      setP(x2 + x, y, c1);
      ye := y;
      inc(x, sx);
    end;
  end
end;
//const cs =  0; // rotate 90    cosine(90)
//      sn =  1; // degree         sine(90)
var
  ddx, ddy:integer;
  r:single;
  ssx, ssy, x, y, xerr, yerr:integer;
begin
  c1 := styler.stroke.Colors[0];
  if thickness<0 then
    thickness := trunc(styler.strokeWidth);
  if thickness=0 then exit;
  //steep := abs(y1 - y0) > abs(x1 - x0);
  //if steep then
  //begin
  //    swap(x0, y0);
  //    swap(x1, y1);
  //end;
  //if x0 > x1 then
  //begin
  //    swap(x0, x1);
  //    swap(y0, y1);
  //end;
  dx := x1-x0;
  dy := y1-y0;
  steep := abs(dy) > abs(dx);
  sx := 1 - 2 * integer(x1<x0);
  sy := 1 - 2 * integer(y1<y0);
  if thickness =1 then begin
    ln(x0, y0);
    exit
  end;
  //if steep then
  //  if dy = 0 then slope := 1
  //  else slope := dx / dy
  //else
  //  if dx = 0 then slope := 1
  //  else slope := dy / dx;

  r  := sqrt(dx*dx + dy*dy);
  if r=0 then exit();

  //if dx<>0 then
  //    ystep := dy div dx;
  //if dy<>0 then
  //    xstep := dx div dy;
  ssx := 1 - 2 * ord(dy>0);
  ssy := 1 - 2 * ord(dx<0);
  ddx := round( ssx*0.1 + thickness * ({dx * cs}  -  dy {* sn}) / r); // calculate rotated normal of dx
  ddy := round( ssy*0.1 + thickness * (dx {* sn    - dy  * cs}) / r); // calculate rotated normal of dy
  if abs(ddx) > abs(ddy) then begin
    x := - ddx div 2 ;
    yerr:=trunc(x * ddy / ddx);
    while x <> ddx do begin
      y :=  round(x * ddy / ddx);
      ln(x0 + x, y0 + y, yerr<>y);
      yerr:=y;
      inc(x, ssx)
    end
  end
  else begin
    y := - ddy div 2 ;
    xerr:=trunc(y * ddx / ddy);;
    while y<>ddy do  begin
      x := round(y * ddx / ddy);
      ln(x0 + x, y0 + y, xerr<>x);
      xerr := x;
      inc(y, ssy)
    end;
  end;
end;
procedure TFastBitmap.LineTo(x, y: integer; thickness: integer);
begin
  Line(FPenPos.x, FPenPos.y, x, y, thickness);
  FpenPos.x := x;
  FPenPos.Y := y
end;
procedure TFastBitmap.LineAA(x0, y0, x1, y1, th: integer);
var
  dx, dy , sx, sy: longint;
  steep : boolean;
  a:byte;
  c:TPixel;
  slope : single;
  //xstep, ystep: integer;
procedure edge(const x2, y2:Integer;  top:boolean);
var x, y: integer; g: single;
begin
  if steep then begin
    y := 0;
    while y <> dy do begin
      g := x2 + y * slope;
      x := trunc(g);
      if top xor (sy>0) then
        c. a := round(a * frac(g))
      else
        c.a := a - round(a * frac(g));
      setP(x, y2 + y, c);
      inc(y, sy)
    end
  end else begin
    x := 0;
    while x <> dx do begin
      g := y2 + x * slope;
      y := trunc(g);
      if top xor (sx>0) then
        c. a := round(a * frac(g))
      else
        c.a := a - round(a * frac(g));
      setP(x2 + x, y, c);
      inc(x, sx);
    end;
  end
end;
procedure ln(const x2, y2:Integer; const err:boolean);
var
  x, y, xe, ye: integer;
begin
  if steep then begin
    y := 0; xe:=0;
    while y <> dy do begin
      x := trunc(x2 + y * slope);
      if err and (x<>xe) then
        setP(x - sx * ord(sx<>sy), y2 + y - sy * ord(sx=sy), c);
      setP(x, y2 + y, c);
      xe := x;
      inc(y, sy);
    end
  end else begin
    x := 0; ye:=0;
    while x <> dx do begin
      y := trunc(y2 + x * slope);
      if err and (y<>ye) then
        setP(x2 + x - sx * ord(sx<>sy), y - sy * ord(sx=sy), c);
      setP(x2 + x, y, c);
      ye := y;
      inc(x, sx);
    end;
  end
end;
const cs =  0; // rotate 90    cosine(90)
      sn =  1; // degree         sine(90)
var
  ddx, ddy:integer;
  r:single;
  ssx, ssy, xx, yy, xxx, yyy:integer;
begin
  c := styler.stroke.Colors[0];
  a := c.a;
  if th =1 then begin
    lineAA(x0, y0, x1, y1);
    exit
  end;

  //steep := abs(y1 - y0) > abs(x1 - x0);
  //if steep then
  //begin
  //    swap(x0, y0);
  //    swap(x1, y1);
  //end;
  //if x0 > x1 then
  //begin
  //    swap(x0, x1);
  //    swap(y0, y1);
  //end;
  dx := x1-x0;
  dy := y1-y0;
  steep := abs(dy) > abs(dx);
  if steep then
    if dy = 0 then slope := 1
    else slope := dx / dy
  else
    if dx = 0 then slope := 1
    else slope := dy / dx;
  r  := sqrt(dx*dx + dy*dy);
  sx := 1 - 2 * integer(x1<x0);
  sy := 1 - 2 * integer(y1<y0);
  //if dx<>0 then
  //    ystep := dy div dx;
  //if dy<>0 then
  //    xstep := dx div dy;
  if r=0 then exit;
  ddx := trunc( th * ({dx * cs}  -  dy {* sn}) / r); // calculate rotated normal of dx
  ddy := trunc( th * (dx {* sn    - dy  * cs}) / r); // calculate rotated normal of dy
  ssx := ifthen(ddx>0, 1, -1);
  ssy := ifthen(ddy>0, 1, -1);
  if abs(ddx) > abs(ddy) then begin
    xx := 0;  yyy:=0;
    while xx <> ddx do begin
      yy :=  trunc(xx * ddy / ddx);
      c.a := a;
      ln(x0 + xx, y0 + yy, yyy<>yy);
      if xx = 0 then
        edge(x0 - ssx , y0 - xx, false);
      if xx = ddx - ssx then
        edge(x0 + ddx , y0 + yy, true);
      yyy:=yy;
      inc(xx, ssx)
    end
  end
  else begin
    yy := 0;  xxx:=0;
    while yy<>ddy do  begin
      xx := trunc(yy * ddx / ddy);
      c.a := a;
      ln(x0 + xx, y0 + yy, xxx<>xx);
      if yy = 0 then
        edge(x0 + xx , y0 - ssy, true);
      if yy = ddy - ssy then
        edge(x0 + xx , y0 + ddy, false);
      xxx := xx;
      inc(yy, ssy)
    end;
  end;
end;
//var
//    dx, dy ,err ,e2 ,x2 ,y2, sx, sy: longint;
//    a:byte;
//    ed: single;
//    c:TPixel;
//begin
//    c := styler.stroke.Colors[0];
//    a := c.a;
//    dx := abs(x1-x0);
//    if x0 < x1 then
//        sx := 1
//    else
//        sx := -1;
//    dy := abs(y1-y0);
//    if y0 < y1 then
//        sy := 1
//    else
//        sy := -1;
//    e2 := trunc(sqrt(dx * dx + dy * dy));
//    if (th <= 1) or (e2 = 0) then
//        begin
//            Line(x0, y0, x1, y1);
//            exit()
//        end;
//    dx := dx * 255 div e2;
//    dy := dy * 255 div e2;
//    th := 255 * (th-1);
//    if dx < dy then
//        begin
//            x1 := (e2+th div 2) div dy;
//            err := x1 * dy-th div 2;
//            x0 := x0 - (x1 * sx);
//            while true do begin
//                c.a := a - a * err div $ff;
//                x1 := x0;
//                setP(x1, y0, c);
//                e2 := dy-err-th;
//                while e2+dy < 255 do begin
//                    x1 := x1 + sx;
//                    c.a := a;
//                    setP(x1, y0, c);
//                    e2 := e2 + dy
//                end;
//                c.a := a - a * e2 div $ff;
//                setP(x1+sx, y0, c);
//                if y0 = y1 then
//                    break;
//                err := err + dx;
//                if err > 255 then
//                    begin
//                        err := err - dy;
//                        x0 := x0 + sx
//                    end;
//                y0 := y0 + sy
//            end
//        end
//    else
//        begin
//            y1 := (e2+th div 2) div dx;
//            err := y1 * dx - th div 2;
//            y0 := y0 - (y1 * sy);
//            while true do begin
//                y1 := y0;
//                c.a := a - a * err div $ff;
//                setP(x0, y1, c);
//                e2 := dx-err-th;
//                while e2+dx < 255 do begin
//                    y1 := y1 + sy;
//                    c.a := a;
//                    setP(x0, y1, c);
//                    e2 := e2 + dx
//                end;
//                c.a := a - a * e2 div $ff;
//                setP(x0, y1+sy, c);
//                if x0 = x1 then
//                    break;
//                err := err + dy;
//                if err > 255 then
//                    begin
//                        err := err - dx;
//                        y0 := y0 + sy
//                    end;
//                x0 := x0 + sx
//            end
//        end
//end;
procedure TFastBitmap.MoveTo(x, y: integer);
begin
  FPenPos.x:=x;
  FPenPos.y:=y
end;
procedure TFastBitmap.TextOut(const x, y: integer; const txt: string {$ifdef FRAMEWORK_FMX} ; const TextFlags:TFillTextFlags=[]{$endif});
var s:TSize;
begin
  {$ifdef FRAMEWORK_FMX}
  s := TextExtent(txt);
  canvas.BeginScene();
  canvas.FillText(RectF(x, y, x+s.Width, y+s.Height), txt, false, 1, TextFlags, TTextAlign.Leading);
  canvas.EndScene;
  {$else}
    canvas.TextOut(x, y, txt)
  {$endif}
end;

procedure TFastBitmap.LineToAA(x, y: integer);
begin
  LineAA(FPenPos.x, FPenPos.y, x, y, trunc(Styler.strokeWidth));
  FPenPos.x:=x; FPenPos.y:=y
end;
procedure TFastBitmap.FillRect(const x1, y1, x2, y2: integer);
var x, y, h, w, sx, sy:integer; c, c1,c2 :TPixel;
begin
  c1 := Styler.fill.Colors[0];
  c2 := Styler.fill.Colors[1];
  sx := 1 -2 * ord(x1>x2);
  sy := 1 -2 * ord(y1>y2);
  w := x2 - x1;
  h := y2 - y1;
  if c1.n <> c2.n then
    case Styler.fill.style of
      grTop:
        begin
          y := y1;
          while y <> y2 do begin
            c := pixelInterp(c1, c2,(y - y1)/h);
            x:= x1;
            while x <> x2 do begin
              setP(x, y, c);
              inc(x, sx)
            end;
            inc(y, sy)
          end;
        end;
      grLeft:
        //for x:=s.x to s.x2-1 do begin
        //  s.stroke:=pixelInterp(s.fill.Colors[0],s.fill.Colors[1],(x-s.x)/(s.x2-s.x));
        //  Line(x,s.y,x,s.y2);
        //end;
        begin
          x := x1;
          while x <> x2 do begin
            c := pixelInterp(c1, c2, (x - x1)/w);
            y := y1;
            while y <> y2 do begin
              setP(x, y, c);
              inc(y, sy)
            end;
            inc(x, sx)
          end;
        end;
      grTopLeft:
        begin
          y := y1;
          while y <> y2 do begin
            x := x1;
            while x <> x2 do begin
              setP(x, y, pixelInterp(c1, c2, (x + y -x1 -y1)/(h+w)));
              inc(x, sx)
            end;
            inc(y, sy)
          end;
        end;
    end
  else begin
    y := y1;
    while y <> y2 do begin
      x := x1;
      while x<>x2 do begin
        setP(x,y,c1);
        inc(x, sx)
      end;
      inc(y, sy)
    end;
  end
end;
procedure TFastBitmap.Ellipse(x0, y0, x1, y1: integer);
const br = 3.0;
var
  cx, cy, rx1, ry1, rx2, ry2:integer;
  i, x, y,  d, band:single ;c : TPixel;
begin
  c := styler.stroke.Colors[0];
  cx := round((x0 + x1) / 2);
  cy := round((y0 + y1) / 2);
  i := Styler.strokeWidth / 2;
  rx1 := round(abs(x1 - x0) / 2 + i);
  ry1 := round(abs(y1 - y0) / 2 + i);
  rx2 := round(rx1 - Styler.strokeWidth);
  ry2 := round(ry1 - Styler.strokeWidth);
  //if styler.fill.Colors[0].a>0 then begin
  //  FillEllipse(cx - rx2, cy -ry2, cx + rx2, cy + ry2);
  //end;
  if rx1 * ry1 = 0 then exit;
  if rx1/ry1 < 1 then
  begin
    y := - ry1 ;
    x := sqr(rx1) * (1 - sqr(y)/sqr(ry1));
    d:=0;
    if x >= 0 then
      begin
        x := sqrt(x);
        while true do
          begin
            if abs(y) >= abs(ry2) then begin
              i := -x ;
              while i < x do begin
                setP(trunc(cx + i), trunc(cy + y), c);
                i:= i + 1
              end;
            end else begin
              if (styler.strokeWidth=1) and not InRange(rx1/ry1, 1/br, br) then
                band := x-1
              else
                band := ceil(sqrt(sqr(rx2) * (1 - sqr(y)/sqr(ry2))));
              i := -x;
              while i < -band do begin
                setP(trunc(cx + i), trunc(cy + y), c);
                i := i + 1
              end;
              // fill
              c := styler.fill.colors[0];
              while (c.a>0) and (i< band) do begin
                setP(trunc(cx + i), trunc(cy + y), c);
                i:= i+1
              end;
              c := styler.stroke.colors[0];
              i := band;
              while i < x do begin
                setP(trunc(cx + i), trunc(cy + y), c);
                i := i + 1
              end;
            end;
            y := y + 1;
            d := sqr(rx1) * (1 - sqr(y)/sqr(ry1));
            if d < 0 then
              break;
            d := x - sqrt(d);
            x := x - d
          end;
      end;
    exit;
  end;
  x := - rx1 ;
  //x :=  rx1;
  y := sqr(ry1) * (1 - sqr(x)/sqr(rx1));
  d:=0;
  if y >= 0 then
    begin
      y := sqrt(y);
      while true do
        begin
          if abs(x) >= abs(rx2) then begin
            i := -y  ;
            while i < y do begin
              setP(trunc(cx + x), trunc(cy + i), c);
              i:= i + 1
            end;
          end else begin
            if (styler.strokeWidth=1) and not InRange(ry1/rx1, 1/br, br) then
              band := y-1
            else
              band := ceil(sqrt(sqr(ry2) * (1 - sqr(x)/sqr(rx2))));
            i := -y;
            while i < - band do begin
              setP(trunc(cx + x), trunc(cy + i), c);
              i := i + 1
            end;
            // fill
            c := styler.fill.colors[0];
            while (c.a>0) and (i< band) do begin
              setP(trunc(cx + x), trunc(cy + i), c);
              i:= i+1
            end;
            c := styler.stroke.colors[0];
            i := band;
            while i < y do begin
              setP(trunc(cx + x), trunc(cy + i), c);
              i := i + 1
            end;
          end;
          x := x + 1;
          d := sqr(ry1) * (1 - sqr(x)/sqr(rx1));
          if d < 0 then
            break;
          d := y - sqrt(d);
          y := y - d
        end;
    end;
end;
procedure TFastBitmap.FillEllipse(x0, y0, x1, y1: integer);
var
    cx, cy , rx, ry : integer;
    i, x, y, d:single; c1, c2 :TPixel;
begin
  c1 := Styler.fill.colors[0];
  cx := (x0 + x1) div 2;
  cy := (y0 + y1) div 2;
  rx:= abs(x1 - x0) div 2;
  ry:= trunc(abs(y1 - y0) / 2);
  if rx * ry =0 then exit;
  y := -ry ;
  x := sqr(rx) * (1 - sqr(y)/sqr(ry));
  d:=0;
  if x >= 0 then
    begin
      x := sqrt(x);
      while true do
        begin
            i := -x ;
            while i < ceil(x) do begin
              setP(trunc(cx + i), trunc(cy + y), c1);
              i:= i + 1
            end;
          y := y + 1;
          d := sqr(rx) * (1 - sqr(y)/sqr(ry));
          if d <= 0 then
            break;
          d := x - sqrt(d);
          x := x - d
        end;
    end;
end;


procedure TFastBitmap.Rect(x1, y1, x2, y2: integer);
var sx, sy, i, d:integer;
begin
  //if x2<x1 then swap(x2,x1);
  //if y2<y1 then swap(y2,y1);
  sx := 1 - 2*ord(x1>x2);
  sy := 1 - 2*ord(y1>y2);
  d:=trunc(styler.strokeWidth/2);
  if Styler.fill.Colors[0].a>0 then begin
    FillRect(x1 + (1+d)*sx, y1 + (1+d)*sy, x2 - d*sx, y2 - d*sy);
  end;
  if (styler.strokeWidth>0)
  {$ifndef FRAMEWORK_FMX}
  and (styler.strokeStyle<> psClear)
  {$else}
  {$endif}
  then
    begin
      for i:=-d to d do
        begin
          Moveto(x1+i*sx, y1+i*sy);
          LineTo(x2-i*sx, y1+i*sy, 1);
          LineTo(x2-i*sx, y2-i*sy, 1);
          LineTo(x1+i*sx, y2-i*sy, 1);
          LineTo(x1+i*sx, y1+i*sy, 1);
        end
    end;
end;
procedure TFastBitmap.EllipseCAA(cx, cy: integer; rx, ry: single; fill: boolean
  );
var
    x, y, s, i :single;
    d, band :single; c1, c2 :TPixel; a :byte;
begin
  c1 := Styler.fill.colors[0];
  a := c1.a;;
  //cx := (x0 + x1) div 2;
  //cy := (y0 + y1) div 2;
  //rx:= abs(x1 - x0) div 2;
  //ry:= trunc(abs(y1 - y0) / 2);
  if rx * ry =0 then exit;
  y := 0;
  x := sqr(rx) * (1 - sqr(y)/sqr(ry));
  d:=0;
  if x >= 0 then
    begin
      x := sqrt(x);
      while d < 1 do
        begin
            c1.a := trunc(a * frac(x));
            setP(trunc(cx - x), trunc(cy + y), c1);
            setP(trunc(cx + x), trunc(cy + y), c1);
            setP(trunc(cx - x), trunc(cy - y), c1);
            setP(trunc(cx + x), trunc(cy - y), c1);
            c1.a := a ;
            s := trunc(x);
            i := -s;
            while i < s do begin
              setP(trunc(cx + i), trunc(cy + y), c1);
              if y<>0 then
                setP(trunc(cx + i), trunc(cy - y), c1);
              i:= i + 1.0;
            end;
          y := y + 1.0;
          d := sqr(rx) * (1 - sqr(y)/sqr(ry));
          if d < 0 then
            break;
          d := x - sqrt(d);
          x := x - d
        end;
    end;
  band := y;
  y := sqr(ry) * (1 - sqr(x)/sqr(rx));
  d:=0;
  if y >= 0 then
    begin
      y := sqrt(y);
      while d < 1 do
        begin
          c1.a := trunc(a * frac(y));
          setP(trunc(cx + x), trunc(cy + y), c1);
          setP(trunc(cx + x), trunc(cy - y), c1);
          c1.a := a ;
          s := trunc(y);
          i := -s;
          while i < 1 - band do begin
            setP(trunc(cx + x), trunc(cy + i), c1);
            i := i +1.0
          end;
          i := band ;
          while i < s  do begin
            setP(trunc(cx + x), trunc(cy + i), c1);
            i := i +1.0
          end;
          x := x - 1.0;
          d := sqr(ry) * (1 - sqr(x)/sqr(rx));
          if d <= 0 then
            break;
          d := y - sqrt(d);
          y := y - d
        end;
    end;
end;
function TFastBitmap.TextExtent(const txt:string):TSize;
begin
{$ifdef FRAMEWORK_FMX}
  result.Width := round(Canvas.TextWidth(txt));
  result.Height := round(Canvas.TextHeight(txt));
{$else}
  result := Canvas.TextExtent(txt)
{$endif}
end;

procedure TFastBitmap.SetAlphaBlend(AValue: boolean);
begin
  if FAlphaBlend=AValue then Exit;
  FAlphaBlend:=AValue;
end;
procedure TFastBitmap.EllipseAA(x1, y1, x2, y2: integer);
var rx, ry, cx, cy:integer;
begin
  if x1 > x2 then swap(x1, x2);
  if y1 > y2 then swap(y1, y2);
  rx:=trunc((x2-x1)/2);
  ry:=trunc((y2-y1)/2);
  EllipseCAA(x1+rx, y1+ry, rx, ry, true)
end;
procedure TFastBitmap.EllipseAA(x0, y0, x1, y1: longint; th: single);
var
    a: single;
    b, b1: int64;
    a2, b2: single;
    dx: single;
    dy: single;
    i:single;
    err: single;
    dx2: single;
    dy2: single;
    ed, e2: single;
    c: TPixel;
    alpha : byte;
begin
    c:= styler.stroke.Colors[0];
    alpha := c.a;
    a := abs(x1-x0);
    b:= abs(y1-y0);
    b1:= b and 1;
    a2 := trunc(abs (a-2 * th));
    b2 := trunc(abs (b-2 * th));
    dx := 4 * (a-1) * sqr(b);
    dy := 4 * (b1-1) * sqr(a);
    i := a+b2;
    err := b1 * sqr(a);
    if th < 1.5 then begin
        EllipseAA(x0, y0, x1, y1);
        exit;
    end;
    if (a<>th) and ((th-1) * (2 * b-th) > sqr(a)) then
        b2 := trunc(sqrt(a * (b-a) * i * a2) / (a-th));
    if (b<>th) and ((th-1) * (2 * a-th) > sqr(b)) then
        begin
            a2 := trunc(sqrt(b * (a-b) * i * b2) / (b-th));
            th := (a-a2) / 2
        end;
    if (a = 0) or (b = 0) then begin
        LineAA(x0, y0, x1, y1);
        exit
    end;
    if x0 > x1 then
        begin
            x0 := x1;
            x1 := x1 + trunc(a)
        end;
    if y0 > y1 then
        y0 := y1;
    if b2 <= 0 then
        th := a;
    e2 := th-trunc(th);
    th := trunc(x0+th-e2);
    dx2 := trunc(4 * (a2+2 * e2-1) * sqr(b2));
    dy2 := 4 * (b1-1) * a2 * a2;
    e2 := dx2 * e2;
    y0 := y0 + ((b+1) shr 1);
    y1 := y0-b1;
    a := 8 * sqr(a);
    b1 := 8 * sqr(b);
    a2 := 8 * sqr(a2);
    b2 := 8 * sqr(b2);
    repeat
        while true do begin
            if (err < 0) or (x0 > x1) then
                begin
                    i := x0;
                    break
                end;
            i := min(dx, dy);
            ed := max(dx, dy);
            if (y0 = y1+1) and (2 * err > dx) and (a > b1) then
                ed := a / 4
            else
                ed := ed + (2 * ed * i * i / (4 * ed * ed+i * i+1)+1);
            c.a := alpha - trunc(alpha * err / ed);
            setP(x0, y0, c);
            setP(x0, y1, c);
            setP(x1, y0, c);
            setP(x1, y1, c);
            if err+dy+a < dx then
                begin
                    i := x0+1;
                    break
                end;
            inc(x0);
            dec(x1);
            err := err - dx;
            dx := dx - b1;
        end;
        c.a := alpha;
        while (i < th) and (2 * i <= x0+x1) do begin
            setP(round(i), round(y0), c);
            setP(round(x0+x1-i),round(y0), c);
            setP(round(i),round(y1), c);
            setP(round(x0+x1-i), round(y1), c);
            i := i +1
        end;
        while ((e2 > 0) and (x0+x1 >= 2 * th)) do
            begin
                i := min(dx2, dy2);
                ed := max(dx2, dy2);
                if (y0 = y1+1) and (2 * e2 > dx2) and (a2 > b2) then
                    ed := a2 / 4
                else
                    ed := ed + (2 * ed * i * i / (4 * ed * ed+i * i));
                c.a := round(alpha * e2 / ed);
                setP(round(th), round(y0), c);
                setP(round(x0+x1-th), round(y0), c);
                setP(round(th), round(y1), c);
                setP(round(x0+x1-th), round(y1), c);
                if e2+dy2+a2 < dx2 then
                    break;
                th := th + 1;
                e2 := e2 - dx2;
                dx2 := dx2 - b2
            end;
        dy2 := dy2 + a2;
        e2 := e2 + dy2;
        inc(y0);
        dec(y1);
        dy := dy + a;
        err := err + dy
    until not(x0 < x1);
    if y0-y1 <= b then
        begin
            if err > dy+a then
                begin
                    dec(y0);
                    inc(y1);
                    dy := dy - a;
                    err := err - dy
                end;
            while y0-y1 <= b do begin
                c.a := alpha - trunc(alpha * 4 * err / b1);
                setP(x0, y0, c);
                setP(x1, y0, c); inc(y0);
                setP(x0, y1, c);
                setP(x1, y1, c); dec(y1);
                dy := dy + a;
                err := err + dy
            end
        end
end;
{ TPlot }
function TPlot.Add(const D: TDoubleDynArray;const GraphStyle:TGraphStyle): Integer;
begin
  Add(D,nil,GraphStyle);
end;
//function TPlot.Add(const D: TExtendedDynArray; const GraphStyle: TGraphStyle): Integer;
//var i:integer;DD:TDoubleDynArray;
//begin
//  setLength(DD,Length(D));
//  for i:=0 to Length(D)-1 do
//    DD[i]:=D[i];
//  result:=Add(DD,GraphStyle);
//end;
function TPlot.Add(const D: TSingleDynArray; const GraphStyle: TGraphStyle): Integer;
var i:integer;DD:TDoubleDynArray;
begin
  setLength(DD,Length(D));
  for i:=0 to Length(D)-1 do
    DD[i]:=D[i];
  result:=Add(DD,GraphStyle);
end;
//function TPlot.Add(const D: TComplexArrayF; const GraphStyle: TGraphStyle): Integer;
//begin
//  add(d.Mag(),GraphStyle);
//
//end;
//
//function TPlot.Add(const D: TComplexArrayD; const GraphStyle: TGraphStyle): Integer;
//begin
//  add(d.Mag(),GraphStyle);
//
//end;
function TPlot.Add(const D: TIntegerDynArray; const GraphStyle: TGraphStyle): Integer;
var i:integer;db:TDoubleDynArray;
begin
  setLength(db,Length(D));
  for i:=0 to High(db) do
    db[i]:=d[i];
  add(db,GraphStyle);
end;

function TPlot.Add(const D: TDoubleDynArray; const base: TDoubleDynArray; const GraphStyle: TGraphStyle): Integer;
begin
  insert(D, Self.FData,Length(FData));
  if (FAutoScale in [asBoth,asY]) then begin
    FMaxs.Push(d.Max);
    FMins.push(d.Min);
    FMinY:=FMins.Min();
    FMaxY:=FMaxs.Max();
  end else if FMinY=FMaxY then begin
    FMinY:=0;
    FMaxY:=1;
  end;
  if assigned(base)then begin
    FBases.Push(Base)
  end
  else begin
    //FBases.Push(TDoubleDynArray.fill(D.Count,FMinY, FMinY));
  end;
  setLength(FGraphOptions,Length(FGraphOptions)+1);
  with FGraphOptions[High(FGraphOptions)] do
    begin
      Fill:=FColors[High(FGraphOptions) mod Length(FColors)];
      Style:=GraphStyle;
      case GraphStyle of
        gsLine :
          begin
            Stroke:=Fill;
            strokeWidth:=1;
          end;
        gsBar:
          begin
            BarsRatio:=0.5;
            Stroke:=FBackground;
            StrokeWidth:=0;
          end;
        gsScatter:
          begin
            Size:=3;
            Stroke:=$88ffffff;
            StrokeWidth:=0;
          end;
      end
    end;
  Result:=FData.Count-1;
  if FActive then Repaint
end;
//function TPlot.Add(const D: TExtendedDynArray; const base: TExtendedDynArray; const GraphStyle: TGraphStyle): Integer;
//var i:integer;DD,BB:TDoubleDynArray;
//begin
//  setLength(DD,Length(D));
//  setLength(BB,Length(D));
//  for i:=0 to Length(D)-1 do
//    begin
//      DD[i]:=D[i];
//      BB[i]:=Base[i]
//    end;
//  result:=Add(DD,BB,GraphStyle);
//end;
function TPlot.Add(const D: TSingleDynArray; const base: TSingleDynArray; const GraphStyle: TGraphStyle): Integer;
var i:integer;DD,BB:TDoubleDynArray;
begin
  setLength(DD,Length(D));
  setLength(BB,Length(D));
  for i:=0 to Length(D)-1 do
    begin
      DD[i]:=D[i];
      BB[i]:=Base[i]
    end;
  result:=Add(DD,BB,GraphStyle);
end;
//function TPlot.Add(const D: TComplexArrayF; const base: TComplexArrayF; const GraphStyle: TGraphStyle): Integer;
//begin
//  add(d.Mag(),base.Mag(),GraphStyle);
//
//end;
function TPlot.Add(const D: TIntegerDynArray; const base: TIntegerDynArray; const GraphStyle: TGraphStyle): Integer;
var i:integer;dd,bb:TDoubleDynArray;
begin
  setLength(DD,Length(D));
  setLength(BB,Length(D));
  for i:=0 to Length(D)-1 do
    begin
      DD[i]:=D[i];
      BB[i]:=Base[i]
    end;
  result:=Add(DD,BB,GraphStyle);
end;
function TPlot.Remove(const index: integer): TDoubleDynArray;
begin
  if not Assigned(FData) then exit;
  Result:=FData[index];
  FData.Splice(Index,1,nil);
  FBases.Splice(Index,1,nil);
  if FAutoScale in [asBoth,asY] then
    if DataCount>0 then
      begin
        FMins.Splice(Index,1,nil);
        FMaxs.Splice(Index,1,nil);
        FMinY:=FMins.Min();
        FMaxY:=FMaxs.Max()
      end
    else
      begin
        FMinY:=0;
        FMaxY:=0;
      end;
  delete(FGraphOptions,Index,1);
  if FActive then Repaint
end;
function TPlot.GetData(Index: Integer): TDoubleDynArray;
begin
  if Assigned(FData) then
    Result:=FData[index]
end;
function TPlot.GetRanges(Index: integer): TDataRange;
begin
  begin
    result.minVal:=GetData(index).Min();
    result.maxVal:=GetData(index).Max();
  end;
end;
procedure TPlot.SetData(Index: Integer; AValue: TDoubleDynArray);
begin
  if Length(AValue)=0 then
    FData[Index]:=nil
  else
    begin
      FData[index]:=AValue;
      if FAutoScale in [asBoth,asY] then
        begin
          FMins[Index]:=AValue.Min();
          FMaxs[Index]:=AValue.Max();
          FMinY:=FMins.Min();
          FMaxY:=FMaxs.Max();
        end;
    end;
  if FActive then Repaint
end;
procedure TPlot.SetRanges(Index: integer; AValue: TDataRange);
begin
  FMins[index]:=AValue.minVal;
  FMaxs[index]:=AValue.maxVal;
  FMinY:=FMins.Min();
  FMaxY:=FMaxs.Max();
  if FActive then Repaint
end;
procedure TPlot.SetDataCount(AValue: integer);
var i:integer;
begin
  if FData.Count=AValue then Exit;
  FData.Count:=AValue;
  if FAutoScale in [asBoth,asY] then begin
    FMins.Count:=AValue;
    FMaxs.Count:=AValue;
  end;
  //FColors.Count:=AValue;
  setLength(FGraphOptions,AValue);
  for i:=0 to AValue-1 do begin
    FGraphOptions[i].Fill:=FColors[i mod Length(FColors)];
    FGraphOptions[i].Stroke:=FBackground;
  end;
  if FActive then Repaint;
end;
procedure TPlot.SetGraphOptions(index: integer; AValue: TGraphOption);
begin
  if FActive then Repaint;
end;
procedure TPlot.SetMaxX(AValue: double);
begin
  if FMaxX=AValue then Exit;
  FMaxX:=AValue;
end;
procedure TPlot.SetMaxY(AValue: double);
begin
  if FMaxY=AValue then Exit;
  FMaxY:=AValue;
end;
procedure TPlot.SetMinX(AValue: double);
begin
  if FMinX=AValue then Exit;
  FMinX:=AValue;
end;
procedure TPlot.SetMinY(AValue: double);
begin
  if FMinY=AValue then Exit;
  FMinY:=AValue;
end;
//procedure TPlot.SetBounds(aLeft, aTop, aWidth, aHeight: integer);
//begin
//  //if not Assigned(Parent) then exit;
////
//  //FBitmap.SetSize(Self.Width,Self.Height);
//  //if FActive then Repaint;
//  inherited SetBounds(aLeft, aTop, aWidth, aHeight);
//
//
//end;
procedure TPlot.Resize;
begin
  if (Width>0) and (Height>0) and assigned(FData)
    //and (FBitmap.Width=0) and (FBitmap.Height=0)
    then
      FBitmap.SetSize(trunc(Width), trunc(Height));
  if FActive then Repaint;
  if FAutoTicks then begin
    FAxisY.FMinimumTicks:=max(FBitmap.Height div 100,1);
    FAxisX.FMinimumTicks:=max(FBitmap.Width div 100,1);
  end;
  inherited Resize;
end;
procedure TPlot.SetPaddingBottom(AValue: smallint);
begin
  if FPaddingBottom=AValue then Exit;
  FPaddingBottom:=AValue;
  if FActive then Repaint;
end;
procedure TPlot.SetPaddingLeft(AValue: smallint);
begin
  if FPaddingLeft=AValue then Exit;
  FPaddingLeft:=AValue;
  if FActive then Repaint;
end;
procedure TPlot.SetPaddingRight(AValue: smallint);
begin
  if FPaddingRight=AValue then Exit;
  FPaddingRight:=AValue;
  if FActive then Repaint;
end;
procedure TPlot.SetPaddingTop(AValue: smallint);
begin
  if FPaddingTop=AValue then Exit;
  FPaddingTop:=AValue;
  if FActive then Repaint;
end;
procedure TPlot.SetParent({$ifdef FRAMEWORK_FMX}const NewParent: TFMXObject {$else} NewParent: TWinControl{$endif});
begin
  inherited SetParent(NewParent);
  if not Assigned(Parent) then exit;
  //if ParentFont then begin
  //  Font.Assign(Parent.Font);
  //  FAxisX.FFont.Assign(Self.Font);
  //  FAxisY.FFont.Assign(Self.Font);
  //end;
end;
procedure TPlot.ShowPlot;
begin
  if (Owner is TForm) then
    TForm(Owner).ShowModal
end;
procedure TPlot.OnOwnerShow(Sender: TObject);
begin
//  Application.BringToFront;
end;
function TPlot.DoCombineColors(const color1, color2: TPixel): TPixel;
begin
//  Beep;
  //
end;
function TPlot.GetDataCount: integer;
begin
  result:=FData.Count
end;
function TPlot.GetGraphOptions(index: integer): TGraphOption;
begin
  result:=FGraphOptions[index];
end;
function TPlot.GetPlotHeight: smallint;
begin
  result:=FBitmap.Height-FPaddingTop-FPaddingBottom
end;
function TPlot.GetPlotWidth: smallint;
begin
  result:=FBitmap.Width-FPaddingLeft-FPaddingRight
end;
procedure TPlot.SetActive(AValue: boolean);
begin
  if FActive=AValue then Exit;
  FActive:=AValue;
  if FActive then Repaint;
end;
procedure TPlot.SetAutoScale(AValue: TAutoScale);
begin
  if FAutoScale=AValue then Exit;
  FAutoScale:=AValue;
end;
procedure TPlot.SetAutoTicks(AValue: boolean);
begin
  if FAutoTicks=AValue then Exit;
  FAutoTicks:=AValue;
  if FActive then Repaint;
end;
procedure TPlot.SetAxisX(AValue: TPlotAxis);
begin
  if FAxisX=AValue then Exit;
  FAxisX:=AValue;
end;
procedure TPlot.SetAxisY(AValue: TPlotAxis);
begin
  if FAxisY=AValue then Exit;
  FAxisY:=AValue;
end;
procedure TPlot.SetBackground(AValue: TGradient);
begin
  //if FBackground=AValue then Exit;
  FBackground:=AValue;
end;
procedure TPlot.SetBarsRatio(AValue: single);
begin
  if FBarsRatio=AValue then Exit;
  FBarsRatio:=AValue;
  if FActive then Repaint;
end;
procedure TPlot.SetColors(AValue: TLongWordDynArray);
begin
  if FColors=AValue then Exit;
  FColors:=AValue;
  if FActive then Repaint
end;
constructor TPlot.Create(AOwner: TComponent);
begin
  FAxisX:=TPlotAxis.Create(Self);
  FAxisY:=TPlotAxis.Create(Self);
  FAutoTicks:=true;
  {$if defined(USE_BGRA)}
  FBitmap:=TBGRABitmap.Create;
  {$elseif defined(USE_FASTBITMAP)}
  FBitmap:=TFastBitmap.Create;
  {$ifndef FRAMEWORK_FMX}
  FBitmap.Canvas.AntialiasingMode:=TAntialiasingMode.amOn;
  {$endif}
  {$else}
  FBitmap:=TBitmap.Create;
  //FBitmap.Canvas.DrawingMode:=FBitmap.Canvas.DrawingMode.dmAlphaBlend;
  FBitmap.Canvas.AntialiasingMode:=TAntialiasingMode.amOn;
  {$endif}
  //FBitmap.Canvas.OnCombineColors:=DoCombineColors;
  FPaddingLeft:=24;
  FPaddingRight:=24;
  FPaddingTop:=24;
  FPaddingBottom:=24;
  FBackground:=Pixel($00,$11,$11,$ff);
  FColors:=TMaterial2Colors;
  FData := [];
  inherited Create(AOwner);
  //FBackground.Colors[1]:=Pixel($22,$0,$0,$ff);

end;
destructor TPlot.Destroy;
var i:integer;
begin
  FreeAndNil(FBitmap);
  inherited Destroy;
end;
function TPlot.alignMaxTicks( range:extended;minTicks:integer):TDoubleDynArray;
var i,p,lg,po:integer;vr:TDoubleDynArray;r:extended;
begin
    if range<=0 then begin
      result := [0, 1];
      exit
    end;
    if range<1 then
      lg:=trunc(Power(10, abs(trunc(log10(range))-2)))
    else
      lg:=10;
    setLength(vr,minTicks);
    for i:=0 to minTicks-1 do
      vr[i]:=ceil(range*lg/minticks+i)*(minticks+i)/lg;
    r:=Math.MinValue(vr);
    p:=vr.indexOf(r)+minticks;
    result:= [r,p];
end;
//TMapCallbackVar

procedure TPlot.xScale;
var i, w,h,tick,rounding,ticks,marginX:integer;Spacing,SpanX:single;v:TDoubleDynArray;textSize:TSize;
begin
  w:=PlotWidth;
  h:=PlotHeight;
  setLength(v,FData.Count);
  for i:=0 to High(v) do v[i]:=FData[i].Count;
  if FAutoScale in [asBoth,asX] then begin
    FMaxX:=round(v.Max());
  end;
  with {$ifdef USE_BGRA}FBitmap.CanvasBGRA {$else}FBitmap{$endif}do begin
    BeginUpdate();
  {$ifdef USE_FASTBITMAP}
    styler.stroke:=AxisX.Color;
    Styler.strokeWidth:=Axisx.Fwidth;
    {$ifdef FRAMEWORK_FMX}
    Styler.strokeStyle:= TStrokeDash.Solid;
    {$else}
    Styler.strokeStyle:=psSolid;
    {$endif}
  {$endif}
    MoveTo(FPaddingLeft+FTxtWidthY,FPaddingTop+H);
    LineToAA(FPaddingLeft+W,FPaddingTop+H);
    //TextStyle.Alignment:=taLeftJustify;
    ticks:=round(alignMaxTicks(FMaxX,AxisX.FMinimumTicks)[1]);
    W:=W-FTxtWidthY;
    marginX:=round(W/FMaxX);
    rounding:=floor(log10(FMaxX)/ticks);
    spanx:=FMaxX/ticks;
    Spacing:=(W-marginX)/Ticks;
    for i:=0 to Ticks do begin
      tick:=trunc(FPaddingLeft+FTxtWidthY+Spacing*i+marginX div 2);
      if AxisX.FTickWidth>0 then begin
        MoveTo(tick,FPaddingTop+H-AxisX.Width);
        lineToAA(tick,FPaddingTop+H-AxisX.width-AxisX.FTickWidth);
      end;
    end;
    EndUpdate();
    Canvas.Lock;
    {$ifdef FRAMEWORK_FMX}
    Canvas.Stroke.Color:=AxisX.Color;
    Canvas.Stroke.Thickness:=Axisx.Fwidth;
    Canvas.Stroke.Dash:= TStrokeDash.Solid;
    Canvas.Font.Assign(AxisX.FFont);
//    Canvas.Fill.Kind := TBrushKind.Solid;
    Canvas.fill.Color := AxisX.FFontColor;
    Canvas.BeginScene();
    {$else}
    Canvas.Pen.{$ifdef USE_BGRA}BGRAColor{$else}Color{$endif}:=AxisX.Color;
    Canvas.Pen.Width:=Axisx.Fwidth;
    Canvas.Pen.Style:=psSolid;
    Canvas.Font.Assign(AxisX.FFont);
    Canvas.Brush.Style:=bsClear;
    {$endif}
    for i:=0 to Ticks do begin
      tick := trunc(FPaddingLeft + FTxtWidthY + Spacing * i + marginX div 2);
      textSize:=TextExtent(IntToStr(i));
      TextOut(tick-textSize.Width div 2, FPaddingTop+AxisX.FSpacing+H, FloatToStr(roundto(i*spanX,rounding)) {xLabels go here});
    end;
    {$ifdef FRAMEWORK_FMX}
    Canvas.EndScene;
    {$endif}
    Canvas.Unlock;
  end;
end;

procedure TPlot.yScale;
var w,h,i:integer;tick,rounding:integer;a:TDoubleDynArray;Spacing,Span,lg:Double;txt:string;textSize:TSize;
begin
  h:=PlotHeight;w:=PlotWidth;
  a:=alignMaxTicks(FMaxY-FMinY,AxisY.FMinimumTicks);
  Spacing:=H/a[1];
  yMax:=a[0]+FMinY;
  yMin:=FMinY;
  if ((yMax-yMin)/a[1])>1 then begin
    rounding:=-1;
    yMin:=Floor(FMinY);
  end else
  begin
    lg:=(yMax-yMin)/a[1];
    lg:=abs(log10(lg));
    rounding:=-ceil(lg);
    //yMin:=floor(FMinY*{0.5*}power(10,rounding+1))/(power(10,rounding+1){*0.5});
  end;
  Span:=(yMax-yMin)/a[1];
  //lg:=Log10(span);
//  if lg>0 then rounding:=0;
  with {$ifdef USE_BGRA}FBitmap.CanvasBGRA{$else}FBitmap{$endif} do begin
    BeginUpdate();
    textSize:=TextExtent(RoundTo(yMin,rounding).ToString);
    FtxtWidthY:=max(textSize.Width, TextExtent(RoundTo(yMax,rounding).ToString).Width);
    FTxtHeightY:=max(textSize.Height, TextExtent(RoundTo(yMax,rounding).ToString).Height);
    //TextOut(10,10,a);
    //TextOut(400,30,FMinY.ToString+',  '+FMaxY.ToString);
  {$ifdef USE_FASTBITMAP}
    styler.stroke:=AxisY.Color;;
    Styler.strokeWidth:=AxisY.Fwidth;
    {$ifdef FRAMEWORK_FMX}
    Styler.strokeStyle:= TStrokeDash.Solid;
    {$else}
    Styler.strokeStyle:=psSolid;
    {$endif}
  {$endif}
    MoveTo(FPaddingLeft+FTxtWidthY,FPaddingTop);
    LineToAA(FPaddingLeft+FTxtWidthY,FPaddingTop+h);
    for i:=0 to trunc(a[1]) do
      begin
        tick:=trunc(FPaddingTop+H-i*Spacing-AxisY.width);
        if AxisY.FTickWidth>0 then
          begin
            MoveTo(FPaddingLeft+AxisY.width+FTxtWidthY, Tick);
            lineToAA(FPaddingLeft+FTxtWidthY+AxisY.width+AxisY.FTickWidth, tick);
          end;
      end;
    EndUpdate();
    Canvas.Lock;
  {$ifdef FRAMEWORK_FMX}
    Canvas.stroke.Color := AxisY.Color;
    Canvas.stroke.Thickness := AxisY.Fwidth;
    Canvas.Stroke.Dash := TStrokeDash.Solid;
    Canvas.Font.Assign(AxisY.FFont);
    Canvas.fill.color:= AxisY.FFontColor;

//    TextOut(40,40,Rounding);
//    Canvas.Brush.Style:=bsClear;
  {$else}
    Canvas.Pen.{$ifdef USE_BGRA}BGRAColor{$else}Color{$endif}:=AxisY.Color;
    Canvas.Pen.Width:=AxisY.Fwidth;
    Canvas.Pen.Style:=psSolid;
    Canvas.Font.Assign(AxisY.FFont);
    Canvas.font.color:=AxisY.FFont.Color;
//    TextOut(40,40,Rounding);
    Canvas.Brush.Style:=bsClear;
  {$endif}
    for i:=0 to trunc(a[1]) do begin
        txt:=FloatToStr(RoundTo(yMin+i*span,rounding));
        tick:=trunc(FPaddingTop+H-i*Spacing-AxisY.width);
        textSize := TextExtent(txt);
        TextOut(FPaddingLeft+FTxtWidthY-AxisY.Spacing-textSize.Width, tick-textSize.Height div 2, txt);
    end;

    Canvas.Unlock;
  end;
end;

var  scl:double=1; valSpan:integer=20; tSpan:integer=40 ;

procedure TPlot.Paint;
var i,j,w,h,lPos, barWidth:integer; Spacing:Single ;D,B:TIntegerDynArray;
begin
  if not FActive or not Assigned(FData) then
    begin
      BeforeDraw(FBitmap);
      {$ifdef FRAMEWORK_FMX}
      Canvas.DrawBitmap(FBitmap, rect(0,0, Fbitmap.Width, fBitmap.Height), ClipRect, 1);
      {$else}
      Canvas.Draw(0, 0, FBitmap);
      {$endif}
      exit;
    end;

  {$ifdef USE_BGRA}
  {$else}
  //if FBitmap.Canvas.LockCount>0 then exit;
  //FBitmap.Canvas.TryLock;
  {$endif}
  w:=PlotWidth;
  h:=PlotHeight;
  FBitmap.BeginUpdate();
  FBitmap.Clear(FBackground.Colors[0]);
  FBitmap.EndUpdate();
  if h>0 then begin
    yScale;
    xScale;


  //if true then begin
    FBitmap.BeginUpdate();
    with {$ifdef USE_BGRA}FBitmap.CanvasBGRA{$else}FBitmap{$endif} do begin
        for j:=0 to Self.FData.Count-1 do begin
          D:=Self.FData[j].Map(yMin,yMax,FPaddingTop+h-AxisY.Width,FPaddingTop).Round;
          Spacing:=(w-FTxtWidthY)/length(D);
          barWidth:=round(Spacing*BarsRatio/2);
          if Assigned(FBases) and Assigned(FBases[j]) and (FGraphOptions[j].Style=gsBar) then
            B:=FBases[j].Map(yMin,yMax,0,h).Round();
          Styler.fill        := FGraphOptions[j].Fill;
          Styler.strokeWidth := FGraphOptions[j].StrokeWidth;
          Styler.stroke      := FGraphOptions[j].Stroke;
          for i:=0 to length(D)-1 do begin
            lPos:=FPaddingLeft+FTxtWidthY+trunc(Spacing/2)+trunc(i*Spacing);
            case FGraphOptions[j].Style of
              gsScatter : begin
                EllipseCAA(lPos, d[i], FGraphOptions[j].Size / 2, FGraphOptions[j].Size / 2, true);
              end;
              gsBar:begin
                if Assigned(B) then
                  Rect(lPos-barWidth, d[i], max(lPos+barWidth,1), FPaddingTop + h - - FAxisX.Fwidth - B[i])
                else
                  FillRect(lPos-barWidth, d[i], max(lPos+barWidth,1), FPaddingTop + h - FAxisX.Fwidth);
                  //Rect(lPos-barWidth, d[i], max(lPos+barWidth,1), FPaddingTop+h)
              end;
              gsLine:
                if i=0 then
                  MoveTo(lPos,d[i])
                else
                  LineToAA(lPos,d[i]);
            end;
          end;
        end;
      //      TextOut(60,60,format('yMix[%f] yMax[%f] FMinY[%f] FMaxY[%f]',[yMin, yMax, FMinY, FMaxY]) );
    end;
    {$ifdef USE_BGRA}
    {$else}
    //fBitmap.Canvas.Unlock;
    {$endif}
    BeforeDraw(FBitmap);
    FBitmap.EndUpdate();

  end;
  //Canvas.StretchDraw(ClientRect,{$ifdef USE_BGRA}FBitmap.Bitmap{$else}FBitmap{$endif});
  {$ifdef FRAMEWORK_FMX}
  Canvas.DrawBitmap(FBitmap, rect(0,0, Fbitmap.Width, fBitmap.Height), ClipRect, 1);
  {$else}
  Canvas.Draw(0, 0, FBitmap);
  {$endif}
  inherited Paint;
end;
procedure TPlot.BeforeDraw(const FBitmap: TFastBitmap);
begin
  if (TMouseButton.mbLeft in FMouseButtonDown) and (FMouseDownPos<>FMouseUpPos) then begin
    //FBitmap.line(FMouseDownPos.x, FMouseDownPos.y, FMouseUpPos.x, FMouseUpPos.y, 20);
    FBitmap.Styler.strokeWidth := 4;
    //FBitmap.Rect(FMouseDownPos.x, FMouseDownPos.y, FMouseUpPos.x, FMouseUpPos.y);
    //FBitmap.EllipseAA(FMouseDownPos.x + 10, FMouseDownPos.y + 10, FMouseUpPos.x - 10, FMouseUpPos.y - 10);
    FBitmap.LineAA(FMouseDownPos.x , FMouseDownPos.y , FMouseUpPos.x , FMouseUpPos.y);
    //FBitmap.EllipseAA(100, 100, 200, 200 );
    //FBitmap.line(FMouseDownPos.x + 20, FMouseDownPos.y + 20, FMouseUpPos.x - 20, FMouseUpPos.y - 20);
  end
end;
procedure TPlot.MouseDown(Button: TMouseButton; Shift: TShiftState; X,Y: {$ifdef FRAMEWORK_FMX}single{$else}Integer{$endif});
begin
  inherited MouseDown(Button, Shift, X, Y);
  FMouseDownPos:= Point(trunc(x),trunc(y));
  Include(FMouseButtonDown, button)
end;
procedure TPlot.MouseUp(Button: TMouseButton; Shift: TShiftState; X, Y: {$ifdef FRAMEWORK_FMX}single{$else}Integer{$endif});
begin
  inherited MouseUp(Button, Shift, X, Y);
  FMouseUpPos:=Point(trunc(x),trunc(y));
  Exclude(FMouseButtonDown, button);
  repaint
end;
procedure TPlot.MouseMove(Shift: TShiftState; X, Y: {$ifdef FRAMEWORK_FMX}single{$else}Integer{$endif});
begin
  inherited MouseMove(Shift, X, Y);
  FMouseUpPos:=Point(trunc(x),trunc(y));
  if TMouseButton.mbLeft in FMouseButtonDown then
    repaint;
end;
{ TPlotForm }

var
   frm:TForm;

initialization
{$ifdef FRAMEWORK_FMX}
  if not (ApplicationState = TApplicationState.Running) then begin
    Application.Initialize;
    frm := TForm.CreateNew(nil);
{$else}
  if not (AppInitialized in Application.Flags) then begin
    Application.Initialize;
    frm := TForm.Create(nil);
{$endif}
    Plot:=TPlot.Create(frm);

{$ifdef FRAMEWORK_FMX}
    Plot.Align:= TAlignLayout.Client;
    frm.Width:=trunc(Screen.Width) div 4;
    frm.Height:=trunc(Screen.Height) div 4;
    frm.Position := TFormPosition.ScreenCenter;
{$else}
    Plot.Align:= alClient;
    frm.Width:=Screen.PrimaryMonitor.Width div 4;
    frm.Height:=Screen.PrimaryMonitor.Height div 4;
    frm.Position:=TPosition.poScreenCenter;
{$endif}
    Plot.Parent:=frm;
    Plot.Active:=true;

//    frm:=TForm.Create(nil);
  end ;
finalization
  if assigned(frm) then begin
    FreeAndNil(Plot);
    FreeAndNil(frm);
  end
end.
