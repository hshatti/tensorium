unit OpenCLHelper;
{$H+}
{$ifdef FPC}
  {$ifopt D+}
  {$define DEBUG}
  {$endif}
  {$mode delphi}
{$endif}
{$pointermath on}
{$IFDEF MSWINDOWS}

{$endif}
{.$define debug}
interface

uses
  Classes, SysUtils, {$ifdef DARWIN}CL {$else} OpenCL{$endif} ;

const
  cInfoSize=$7fff;
  MAX_EVENTS_COUNT = $100;

type
{$if not declared(size_t)}
  psize_t = ^size_t;
  size_t  = NativeUInt;
{$endif}

  TCLDeviceType=(
    dtNone = 0 ,
    dtDefault = CL_DEVICE_TYPE_DEFAULT,
    dtCPU = CL_DEVICE_TYPE_CPU,
    dtGPU = CL_DEVICE_TYPE_GPU,
    dtACCELERATOR = CL_DEVICE_TYPE_ACCELERATOR,
    dtALL = CL_DEVICE_TYPE_ALL
  );
  TCLMemAccess=(
    maReadWrite = CL_MEM_READ_WRITE         ,
    maWrite     = CL_MEM_WRITE_ONLY         ,
    maRead      = CL_MEM_READ_ONLY          ,
    maUseHost   = CL_MEM_USE_HOST_PTR       ,
    maAllocHost = CL_MEM_ALLOC_HOST_PTR     ,
    maCopyHost  = CL_MEM_COPY_HOST_PTR
  );

  PWorkSize=^TWorkSizes;
  TWorkSizes=array [0..2] of size_t;

  TCLKernelArgInfo=record
    ArgName:array[0..cInfoSize-1] of ansichar;
    ArgType:array[0..cInfoSize-1] of ansichar;
    ArgAccess:cl_uint;
    ArgAddress:cl_uint;
    ArgTypeQualifier:cl_bitfield;
  end;

  PComplexF=^TComplexF;
  TComplexF=record
    re,im:single
  end;

  PComplexD=^TComplexD;
  TComplexD=record
    re,im:Double
  end;

  TCLKernelInfo=record
    KernelName:array[0..cInfoSize] of ansichar;
    KernelGlobalWorkSize:array[0..2] of size_t;
    KernelWorkGroupSize:size_t;
    KernelSIMDSize : size_t;
    KernelLocalMemSize:cl_ulong;
    KernelPrivateMemSize:cl_ulong;
    KernelArgCount:cl_uint;
    KernelArgs:array of TCLKernelArgInfo;
  end;

  { TOpenCl }

  TOpenCL=class
  type
    TDeviceStr =array[0..cInfoSize-1] of ansichar;
  private
    FSrc:TStringList;
    FActivePlatformId: integer;
    FActiveDeviceId: integer;
    FActiveKernelId: integer;
    FActivePlatform: cl_platform_id;
    FActiveDevice: cl_device_id;
    FActiveKernel: cl_kernel ;
    FActiveKernelInfo: TCLKernelInfo;
    FPlatformCount:cl_uint;
    FDeviceCount:cl_uint;
    FKernelCount:cl_uint;
    FPlatforms:array of cl_platform_id;
    FDevices:array of cl_device_id;  // array of pointer to opaque record
    FDevicesType:array of TCLDeviceType;
    FContext:cl_context;
    FKernels:TArray<cl_kernel>;
    FDeviceType: TCLDeviceType;
    FProgramSource: ansistring;
    cinfo:TDeviceStr;
    FWorkItemDimensions: integer;
    BuffCount:size_t;
    FGlobalOffsets:TWorkSizes;
    FGlobalWorkGroupSizes:TWorkSizes;
    FLocalWorkGroupSizes:TWorkSizes;

    FGlobalMemSize:size_t;
    FLocalMemSize:cl_ulong;
    FGlobalCacheSize:size_t;
    FExecCaps:cl_device_exec_capabilities;
    FMaxWorkItemDimensions:cl_uint;
    FMaxWorkGroupSize:size_t;
    FMaxWorkItemSizes:TWorkSizes;
    FMaxComputeUnits:cl_uint;
    FMaxMemAllocSize:cl_ulong;
    FMaxFrequency:cl_uint;

    FDeviceBuiltInKernels:TDeviceStr;
    FIsBuilt:boolean;
    FProgram:cl_program;
    FBuildStatus:cl_build_status ;
    FBuildLog:ansistring;
    FCallParams:array[0..$ff] of cl_mem;
    FParamSizes:array[0..$ff] of size_t;

    FDevsTypeStr:ansistring;
    FSharedMemory:boolean;
    function GetDevices(index: cl_uint): cl_device_id;
    function getQueueInOrder: boolean;
    procedure SetActiveDeviceId(AValue: integer);
    procedure SetActiveKernelId(AValue: integer);
    procedure SetActivePlatformId(AValue: integer);
    procedure SetDeviceType(AValue: TCLDeviceType);
    function getCL_Device_Type(const dt:TClDeviceType):cl_uint;
    procedure SetGlobalWorkGroupSizes(AValue: TWorkSizes);overload;
    procedure SetProgramSource(AValue: ansistring);
    procedure SetQueueInOrder(AValue: boolean);
    procedure SetWorkItemDimensions(AValue: integer);
  public
    CLDeviceCVersion:TDeviceStr;
    CLDeviceDriver:TDeviceStr;
    FErr:cl_int;
    //events    : array[0..MAX_EVENTS_COUNT-1] of cl_event;
    //eventsCount : cl_uint;
    FQueue:cl_command_queue;
    procedure CheckError(const msg:string=''); inline;
    constructor Create(deviceType:TCLDeviceType=dtGPU);
    destructor Destroy;override;
    procedure SetGlobalWorkGroupSizes(const x: size_t; const y: size_t=0; const z: size_t=0); overload;
    procedure SetLocalWorkGroupSizes(const x: size_t; const y: size_t=0; const z: size_t=0);
    procedure SetParamElementSizes(paramSizes: array of size_t);
    function DevicesTypeStr:ansistring;
    function getQueueSize():cl_uint;
    function getQueueRefCount():cl_uint;
    procedure SetGlobalOffsets(const x: size_t; y: size_t=0; z: size_t=0);
    function CleanUp(const keepContext: boolean=false): boolean;
    function ProcessorsCount:integer;
    function ProcessorsFrequency:integer;
    property DeviceType:TCLDeviceType read FDeviceType write SetDeviceType;
    property Devices[index:cl_uint]:cl_device_id read GetDevices;
    property ActivePlatformId:Integer read FActivePlatformId write SetActivePlatformId;
    property ActiveDeviceId:Integer read FActiveDeviceId write SetActiveDeviceId;
    property ProgramSource:ansistring read FProgramSource write SetProgramSource;

    property LocalMemSize:cl_ulong                 read FLocalMemSize ;
    property ExecCaps:cl_device_exec_capabilities  read FExecCaps;
    property MaxWorkItemDimensions:cl_uint         read FMaxWorkItemDimensions;
    property MaxWorkGroupSize:size_t               read FMaxWorkGroupSize;
    property MaxWorkItemSizes:TWorkSizes           read FMaxWorkItemSizes;
    property MaxComputeUnits:cl_uint               read FMaxComputeUnits;
    property MaxMemAllocSize:cl_ulong              read FMaxMemAllocSize;
    property MaxFrequency:cl_uint                  read FMaxFrequency;
    property ActivePlatform : cl_platform_id read FActivePlatform;
    property ActiveDevice : cl_device_id read FActiveDevice;
    property ActiveContext : cl_context read FContext;
    property ActiveQueue : cl_command_queue read FQueue;
    property ActiveKernel : cl_kernel read FActiveKernel;
    property ActiveKernelInfo : TCLKernelInfo read FActiveKernelInfo;
    property ExecCapabilities : cl_device_exec_capabilities read FExecCaps;

    function PlatformName(Index: integer): ansistring;
    function DeviceName(Index: integer): ansistring;
    function PlatformCount:integer;
    function DeviceCount:integer;
    function Build(const params:ansistring=''):boolean;
    function readLog:ansistring;
    property BuildLog:ansistring read FBuildLog;
    property LastError:cl_int read FErr;
    property Kernels:TArray<cl_kernel> read FKernels;
    function KernelCount:integer;
    function KernelInfo(index:integer):TCLKernelInfo;
    property GlobalWorkGroupSizes:TWorkSizes read FGlobalWorkGroupSizes;
    property LocalWorkGroupSizes:TWorkSizes read FLocalWorkGroupSizes;
    property GlobalOffsets : TWorkSizes read FGlobalOffsets;
    function CanExecuteNative:boolean;
    procedure LoadFromFile(FileName:ansistring);
    //function KernelArgs(index:integer):TCLKernelArgInfo;
    property DeviceBuiltInKernels : TDeviceStr read FDeviceBuiltInKernels;
    property ActiveKernelId:Integer read FActiveKernelId write SetActiveKernelId;
    property WorkItemDimensions:integer read FWorkItemDimensions write SetWorkItemDimensions;
    property isBuilt:boolean read FIsBuilt;
    property queueInOrder: boolean read getQueueInOrder write SetQueueInOrder;
    function createDeviceBuffer(const aByteSize:size_t; const aAccess:TCLMemAccess = maReadWrite; const fromHostMem:Pointer=nil):cl_mem;
    procedure freeDeviceBuffer(aMem:cl_mem);
(*    procedure CallKernel(const Index: integer; const dst: PLongWord;const c: integer);inline;  *)
    procedure CallKernel(const Index: integer; const dst, a, b: PSingle; const bias:single;const c: integer); overload;
    procedure CallKernel(const Index: integer; const params: TArray<Pointer>);    overload;
    procedure CallKernel(const Index: integer; const dst: PLongWord; const a, b: integer);  overload;
    procedure finish();
    procedure waitForEvents(const N :longword; const events:pcl_event);
    procedure freeEvents(const N :longword; const events:pcl_event);
    procedure ActivateArray(const x: cl_mem; const N: SizeInt; const activation: longint; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure activateArraySWISH(const x: cl_mem; const N: SizeInt; const output_sigmoid, output: cl_mem; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure DeriveArray(const x: cl_mem; const N: SizeInt; const activation: longint; delta: cl_mem; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure forwardBias(const N: SizeInt; const a: cl_mem; const blockSize: SizeInt; const b: cl_mem; const incb: SizeInt ; const batch: SizeInt; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure backwardBias(const N: SizeInt; const a: cl_mem; const blockSize: SizeInt; const b: cl_mem; const incb: SizeInt ; const batch: SizeInt; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure gemm(const transA, transB :boolean; const M, N, K:SizeInt; const ALPHA:single;
      const A:cl_mem; const aOffset:SizeInt; const lda:SizeInt;
      const B:cl_mem; const bOffset:SizeInt; const ldb:SizeInt;
      const BETA: single; const C:cl_mem; const cOffset:SizeInt; const ldc:SizeInt;
      const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure addvv(const N:SizeInt; const dst, src:cl_mem; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure subvv(const N:SizeInt; const dst, src:cl_mem; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure axpy(const N:SizeInt; const a:single; const x:cl_mem; const incx:SizeInt; const y:cl_mem; const incy:sizeInt; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure scale(const N:SizeInt; const a:Single; const x:cl_mem; const stride:SizeInt; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure crossEntropyLogistic(const N:SizeInt; const pred, truth: cl_mem; delta, error: cl_mem; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure fill(const N:SizeInt; const x: cl_mem; const val:single; const stride :SizeInt; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure copy(const N:SizeInt; const a:cl_mem; const aOffset, inca:SizeInt; const b:cl_mem; const bOffset, incb:SizeInt; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure softmaxBatch(const input: cl_mem; const iOffset, N: SizeInt;
      const batch, batch_size, groups, group_size, stride: SizeInt;
      const temp: single; const output: cl_mem; const oOffset: SizeInt;
      const eventCount: cl_uint; const events: pcl_event; event: pcl_event);
    procedure crossEntropySoftmax(const N:SizeInt; const pred, truth: cl_mem; delta, error: cl_mem; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure forwardMaxPool(const aBatch, outC, outH, outW: SizeInt; const input: cl_mem; const c, h, w: SizeInt;
      const stride_x, stride_y, padding, kernelSize: SizeInt; indexes, output: cl_mem; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure backwardMaxPool(const aBatch, outC, outH, outW : SizeInt; output:cl_mem; const indexes, delta : cl_mem; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure im2col(const aChannels, aHeight, aWidth
      , kernelHeight, kernelWidth, padHeight, padWidth
      , strideY, strideX, dilationY, dilationX : SizeInt
      ; const im :cl_mem; const imOffset : SizeInt
      ; const col:cl_mem; const colOffset:SizeInt; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    procedure col2im(const aChannels, aHeight, aWidth, kernelHeight, kernelWidth,
      padHeight, padWidth, strideY, strideX, dilationY, dilationX: SizeInt;
      const col: cl_mem; const colOffset: SizeInt;
      const im: cl_mem; const imOffset: SizeInt; const eventCount:cl_uint = 0; const events: pcl_event = nil; event: pcl_event = nil);
    class function Plaforms:cl_uint;static;


  end;

{$if not declared(clGetKernelArgInfo)}
    cl_kernel_arg_info                         = cl_uint;

  const
    CL_KERNEL_ARG_ADDRESS_QUALIFIER = $1196;
    CL_KERNEL_ARG_ACCESS_QUALIFIER  = $1197;
    CL_KERNEL_ARG_TYPE_NAME         = $1198;
    CL_KERNEL_ARG_TYPE_QUALIFIER    = $1199;
    CL_KERNEL_ARG_NAME              = $119A;
    CL_DEVICE_BUILT_IN_KERNELS      = $103f;
    CL_DEVICE_HOST_UNIFIED_MEMORY   = $1035;

  function clGetKernelArgInfo (kernel:cl_kernel;
                     arg_indx:cl_uint;
                     param_name:cl_kernel_arg_info;
                     param_value_size:size_t;
                     param_value:pointer;
                     param_value_size_ret:psize_t):cl_int;winapi;external;
{$endif}

implementation

(*
// CONSTANTS
// The source code of the kernel is represented as a ansistring
// located inside file: "fft1D_1024_kernel_src.cl". For the details see the next listing.

// Looking up the available GPUs
case ComboBox1.ItemIndex of
  0:deviceType:=CL_DEVICE_TYPE_GPU;
  1:deviceType:=CL_DEVICE_TYPE_CPU;
  2:deviceType:=CL_DEVICE_TYPE_DEFAULT;
end;


ret:=clGetDeviceIDs(nil, deviceType, 0, nil, @num);
if ret<>CL_SUCCESS then raise Exception.create('Cannot list Processors');

setLength(devices,num);
       //cl_device_id devices[1];
ret:=clGetDeviceIDs(nil, deviceType, num, @devices[0], nil);
if ret<>CL_SUCCESS then raise Exception.create('Cannot get ALL Device id ');
ListBox1.Items.Clear;
for i:=0 to num -1 do begin
  clGetDeviceInfo(devices[i],CL_DEVICE_NAME,256,@deviceInfo[0],retSize);
  ListBox1.Items.add(deviceInfo);
end;

// create a compute context with GPU device
context := clCreateContextFromType(nil, deviceType, nil, nil, ret);
if ret<>CL_SUCCESS then raise Exception.create('Cannot Create context from GPU Type');

// create a command queue
ret:=clGetDeviceIDs(nil, deviceType, 1, @devices[0], nil);
if ret<>CL_SUCCESS then raise Exception.create('Cannot get Default Device');

ret:=clGetDeviceInfo(devices[0],CL_DEVICE_MAX_COMPUTE_UNITS,256,@deviceInfo[0],retSize);
ListBox1.Items.Add(IntToStr(PLongWord(@deviceInfo)^)+' Units');
ret:=clGetDeviceInfo(devices[0],CL_DEVICE_MAX_CLOCK_FREQUENCY,256,@deviceInfo[0],retSize);
ListBox1.Items[ListBox1.Items.count-1]:=ListBox1.Items[ListBox1.Items.count-1]+'@'+IntToStr(PLongWord(@deviceInfo)^)+'Mhz';



queue := clCreateCommandQueue(context, devices[0], 0{props}, ret);
if ret<>CL_SUCCESS then raise Exception.create('Cannot create command queue');

t:=GetTickCount64;
bmp.BeginUpdate();

// allocate the buffer memory objects
 memobjs[0]:=  clCreateBuffer(context, CL_MEM_WRITE_ONLY , 4*w*h, nil, ret);
 if ret<>CL_SUCCESS then raise Exception.create('Cannot create ReadMem');
 //memobjs[1]:=  clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(single) * 2 * NUM_ENTRIES, nil, ret);
//       if ret<>CL_SUCCESS then raise Exception.create('Cannot create WriteMem');
// cl_mem memobjs[0] = // FIXED, SEE ABOVE
// cl_mem memobjs[1] = // FIXED, SEE ABOVE

// create the compute program
// const ansichar* fft1D_1024_kernel_src[1] = {  };
prog := clCreateProgramWithSource(context, 1, PPAnsiCHAR(@src), nil, ret);

if ret<>CL_SUCCESS then raise Exception.create('Cannot create ProgramWithSource');

// build the compute program executable
ret:=clBuildProgram(prog, 0, nil, nil, nil, nil);

if ret<>CL_BUILD_SUCCESS then begin
  clGetProgramBuildInfo(prog,devices[0],CL_PROGRAM_BUILD_LOG,256,@buildLog[0],retSize);
  raise Exception.CreateFmt('Cannot Build executable message:'#13#10'[%s]',[buildLog]);
end;
// create the compute kernel


kernel := clCreateKernel(prog, 'render', ret);
if ret<>CL_SUCCESS then raise Exception.create('Cannot create Kernel');
// set the args values

//size_t local_work_size[1] = { 256 };

ret:=clSetKernelArg(kernel, 0, sizeof(cl_mem), @memobjs[0]);
if ret<>CL_SUCCESS then raise Exception.create('Cannot set argument[0]');
ret:=clSetKernelArg(kernel, 1, sizeof(max_iteration), @max_iteration);
if ret<>CL_SUCCESS then raise Exception.create('Cannot set argument[1]');
//ret:=clSetKernelArg(kernel, 1, sizeof(cl_mem), @memobjs[1]);
//       if ret<>CL_SUCCESS then raise Exception.create('Cannot set argument[1]');
//ret:=clSetKernelArg(kernel, 2, sizeof(single)*(local_work_size[0] + 1) * 16, nil);
//       if ret<>CL_SUCCESS then raise Exception.create('Cannot set argument[2]');
//ret:=clSetKernelArg(kernel, 3, sizeof(single)*(local_work_size[0] + 1) * 16, nil);
//       if ret<>CL_SUCCESS then raise Exception.create('Cannot set argument[3]');
//
// create N-D range object with work-item dimensions and execute kernel
//size_t global_work_size[1] = { 256 };

//global_work_size[0] := NUM_ENTRIES;
//local_work_size[0] := 64; //Nvidia: 192 or 256

ret:=clEnqueueNDRangeKernel(queue, kernel, 2, global_work_offset, global_work_size, nil, 0, nil, nil);
if ret<>CL_SUCCESS then raise Exception.create('Cannot Enqueue ND Range kernel');

clEnqueueReadBuffer(queue,memobjs[0],cl_false,{offset in byte }0,w*h*4{size in byte},bmp.ScanLine[0],0,nil,nil);
//clFlush(queue);
clFinish(queue);
ListBox1.Items.Add(format(' -Rendering took %d MilliSeconds',[GetTickCount64-t]));
bmp.EndUpdate();
Image1.picture.Graphic:=bmp ;

clReleaseMemObject(memobjs[0]);
clReleaseCommandQueue(queue);
clReleaseContext(context);
clReleaseKernel(kernel);
clReleaseProgram(prog);
*)

{ TOpenCl }

procedure TOpenCL.SetDeviceType(AValue: TCLDeviceType);
var wasBuilt:boolean;
begin
  if FDeviceType=AValue then Exit;
  if FDeviceCount=0 then raise Exception.Create('No Devices found!');
  FDeviceType:=AValue;
  FActiveDeviceId:=-1;
  wasBuilt:=FIsBuilt;
  SetActivePlatformId(FActivePlatformId);
  if wasBuilt then
    Build
end;

procedure TOpenCL.CheckError(const msg: string);
begin
  if FErr<>CL_SUCCESS then
    //writeln(msg, clErrorText(FErr));
    raise Exception.Create(clErrorText(FErr));
end;

function TOpenCL.getCL_Device_Type(const dt: TClDeviceType): cl_uint;
begin
  case dt of
    dtDefault :result:= CL_DEVICE_TYPE_DEFAULT;
    dtCPU :result:=CL_DEVICE_TYPE_CPU;
    dtGPU :result:=CL_DEVICE_TYPE_GPU;
    dtACCELERATOR :result:=CL_DEVICE_TYPE_ACCELERATOR;
    dtALL :result:=CL_DEVICE_TYPE_ALL
  end;
end;

procedure TOpenCL.SetGlobalWorkGroupSizes(AValue: TWorkSizes);
begin
//  if FGlobalWorkGroupSize=AValue then Exit;
  FGlobalWorkGroupSizes:=AValue;
end;

procedure TOpenCL.SetProgramSource(AValue: ansistring);
var i:integer;
begin
  if FProgramSource=AValue then Exit;
  if Assigned(FKernels) then
    for i:=0 to FKernelCount-1 do
      clReleaseKernel(FKernels[i]);
  setLength(FKernels,0);
  if Assigned(FProgram) then clReleaseProgram(FProgram);
  FIsBuilt:=false;
  FProgramSource:=AValue;
end;

procedure TOpenCL.SetQueueInOrder(AValue: boolean);
var oldProp : cl_command_queue_properties;
begin
  clSetCommandQueueProperty(ActiveQueue, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, cl_bool(not AValue), oldProp);
end;

procedure TOpenCL.SetWorkItemDimensions(AValue: integer);
begin
  if FWorkItemDimensions=AValue then Exit;
  FWorkItemDimensions:=AValue;
end;

constructor TOpenCL.Create(deviceType: TCLDeviceType);
var i:integer;
begin
  FDeviceType:=deviceType;
  FPlatformCount:=0;
  FillChar(FParamSizes,sizeof(FParamSizes),0);
  for i:=0 to high(FGlobalOffsets) do FGlobalOffsets[i]:=0;
  buffCount:=Length(cInfo);
  FErr:=clGetPlatformIDs(0,nil,@FPlatformCount); checkError();
  assert(FPlatformCount>0, 'No OpenCL supported platforms found!');
  if FErr=CL_SUCCESS then
    if FPlatformCount>0 then begin
      SetLength(FPlatforms,FPlatformCount);
      FErr:=clGetPlatformIDs(FPlatformCount,@FPlatforms[0],nil);CheckError();
      FSrc:=TStringList.Create;
      FActivePlatformId:=$7fffffff;
      FActiveDeviceId:=$7fffffff;
      FWorkItemDimensions:=1;
      SetActivePlatformId(0);
    end;
end;

destructor TOpenCL.Destroy;
begin
  CleanUp(false);
  FSrc.Free;
  inherited Destroy;
end;

procedure TOpenCL.SetGlobalWorkGroupSizes(const x: size_t; const y: size_t;
  const z: size_t);
begin
  FGlobalWorkGroupSizes[0]:=x;
  FGlobalWorkGroupSizes[1]:=y;
  FGlobalWorkGroupSizes[2]:=z;
  FillChar(FGlobalOffsets,sizeof(FGlobalOffsets),0);
  if z>0 then begin
    FWorkItemDimensions:=3;
    exit
  end;
  if y>0 then begin
    FWorkItemDimensions:=2;
    exit
  end;
  FWorkItemDimensions:=1;
end;

procedure TOpenCL.SetLocalWorkGroupSizes(const x: size_t; const y: size_t;
  const z: size_t);
begin
  FLocalWorkGroupSizes[0]:=x;
  FLocalWorkGroupSizes[1]:=y;
  FLocalWorkGroupSizes[2]:=z;
end;

procedure TOpenCL.SetParamElementSizes(paramSizes: array of size_t);
var i:integer;
begin
  for i:=0 to High(paramSizes) do
    FParamSizes[i]:=paramSizes[i]
end;

function TOpenCL.DevicesTypeStr: ansistring;
begin
  result:=FDevsTypeStr;
end;

function TOpenCL.getQueueSize(): cl_uint;
var r: size_t;
begin
  FErr := clGetCommandQueueInfo(ActiveQueue, CL_QUEUE_REFERENCE_COUNT, sizeOf(cl_uint), @result, r) ; CheckError();
end;

function TOpenCL.getQueueRefCount(): cl_uint;
var r:size_t;
begin
  FErr := clGetCommandQueueInfo(ActiveQueue, CL_QUEUE_SIZE, sizeOf(cl_uint), @result, r) ; CheckError();
end;

procedure TOpenCL.SetGlobalOffsets(const x: size_t; y: size_t; z: size_t);
begin
  FGlobalOffsets[0]:=x;
  FGlobalOffsets[1]:=y;
  FGlobalOffsets[2]:=z;
end;

function TOpenCL.CleanUp(const keepContext:boolean): boolean;
var i:integer;
begin
  try
    for i:=0 to High(FKernels) do begin
      clReleaseKernel(FKernels[i]);  CheckError();
      FKernels[i] := nil
    end;

    if FProgram<>nil then begin
      clReleaseProgram(FProgram);CheckError();
      FProgram := nil
    end;

    if FQueue<>nil then begin
      clReleaseCommandQueue(FQueue);CheckError();
      FQueue := nil
    end;

    if not keepContext then if
      FContext<>nil then begin
        clReleaseContext(FContext);CheckError();
        FContext:=nil;
      end;

    FIsBuilt:=false;
    result:=true
  except on E:Exception do
    begin
      result:=false
    end;

  end;
end;

function TOpenCL.ProcessorsCount: integer;
begin
  result:=FMaxComputeUnits;
end;

function TOpenCL.ProcessorsFrequency: integer;
begin
  result:=FMaxFrequency;
end;

function TOpenCL.PlatformName(Index: integer): ansistring;
begin

  clGetPlatformInfo(FPlatforms[Index],CL_PLATFORM_NAME,cInfoSize,@cinfo[0],buffCount);
  result:=cinfo;
end;

function TOpenCL.DeviceName(Index: integer): ansistring;
begin
  clGetDeviceInfo(FDevices[Index],CL_DEVICE_NAME,cInfoSize,@cinfo[0],buffCount);
  result:=cinfo;
end;

function TOpenCL.PlatformCount: integer;
begin
  result:=FPlatformCount
end;

function TOpenCL.DeviceCount: integer;
begin
  result:=FDeviceCount
end;

function TOpenCL.Build(const params: ansistring): boolean;
var src,par:PAnsiChar; sz:cl_uint;
begin
  result:=False;
  src:=PAnsiChar(FProgramSource);
{$ifdef DEBUG}
  par:=PAnsiCHar('-cl-kernel-arg-info -cl-std=CL2.0 -cl-opt-disable -Werror -g '+params);
{$else}
  par:=PAnsiCHar('-cl-kernel-arg-info -cl-std=CL2.0 -cl-fast-relaxed-math -cl-mad-enable '+params);
{$endif}
  FProgram:=clCreateProgramWithSource(FContext,1,@src,nil,FErr);CheckError();
  FErr:=clBuildProgram(Fprogram,FDeviceCount,@FDevices[0],par,nil,nil);
  FErr:=clGetProgramBuildInfo(FProgram,FActiveDevice,CL_PROGRAM_BUILD_STATUS,cInfoSize,@FBuildStatus,buffCount);CheckError();
  FErr:=clGetProgramBuildInfo(FProgram,FActiveDevice,CL_PROGRAM_BUILD_LOG,cInfoSize,@cinfo[0],buffCount);CheckError();
  FBuildLog:=trim(system.copy(cinfo,0,buffCount));
  assert(FBuildStatus=CL_BUILD_SUCCESS, '[OpenCL] : cannot compile tensor kernels :' + sLineBreak+ FBuildlog);
  if FBuildStatus= CL_BUILD_SUCCESS then begin
    if FBuildlog<>'' then begin
      writeln(StdErr, FBuildLog);
      readln
    end;
    FErr:=clCreateKernelsInProgram(FProgram,0,nil,FKernelCount);CheckError();
    setLength(FKernels,FKernelCount);
    FErr:=clCreateKernelsInProgram(FProgram,FKernelCount,@FKernels[0],sz);CheckError();
    FActiveKernelId:=-1;
    if FKernelCount>0 then
      SetActiveKernelId(0);
    FIsBuilt:=True;
    Result:=True
  end;
//  if cinfo='' then cinfo:='Success';
end;

function TOpenCL.readLog: ansistring;
begin
  FErr:=clGetProgramBuildInfo(FProgram,FActiveDevice,CL_PROGRAM_BUILD_LOG,cInfoSize,@cinfo[0], buffCount);CheckError();
  result := cinfo
end;

function TOpenCL.KernelCount: integer;
begin
  result:=FKernelCount;
end;

function TOpenCL.KernelInfo(index: integer): TCLKernelInfo;
var sz:size_t;i:integer;
begin
    FErr:=clGetKernelInfo(FKernels[Index],CL_KERNEL_FUNCTION_NAME,cInfoSize,@result.KernelName[0], buffCount);                                      CheckError();
    FErr:=clGetKernelInfo(FKernels[Index],CL_KERNEL_NUM_ARGS,cInfoSize,@result.KernelArgCount,buffCount);                                          CheckError();
    //FErr:=clGetKernelWorkGroupInfo(FKernels[Index],FActiveDevice,CL_KERNEL_GLOBAL_WORK_SIZE,cInfoSize,@result.KernelGlobalWorkSize[0],@buffCount);  CheckError();
    FErr:=clGetKernelWorkGroupInfo(FKernels[Index],FActiveDevice,CL_KERNEL_WORK_GROUP_SIZE,sizeOf(result.KernelWorkGroupSize),@result.KernelWorkGroupSize,@buffCount);     CheckError();
    FErr:=clGetKernelWorkGroupInfo(FKernels[Index],FActiveDevice,CL_KERNEL_LOCAL_MEM_SIZE,sizeOf(result.KernelLocalMemSize),@result.KernelLocalMemSize,@buffCount);       CheckError();
    FErr:=clGetKernelWorkGroupInfo(FKernels[Index],FActiveDevice,CL_KERNEL_PRIVATE_MEM_SIZE,sizeOf(result.KernelPrivateMemSize),@result.KernelPrivateMemSize,@buffCount);     CheckError();
    FErr:=clGetKernelWorkGroupInfo(FKernels[Index],FActiveDevice,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeOf(result.KernelSIMDSize),@result.KernelSIMDSize,@buffCount);     CheckError();

    setLength(result.KernelArgs,result.KernelArgCount);
    for i:=0 to result.KernelArgCount-1 do begin
        buffCount:=$ff;
        FErr:=clGetKernelArgInfo(FKernels[Index],i,CL_KERNEL_ARG_NAME,cInfoSize,@result.KernelArgs[i].ArgName[0],@buffCount);                         CheckError();
        FErr:=clGetKernelArgInfo(FKernels[Index],i,CL_KERNEL_ARG_TYPE_NAME,cInfoSize,@result.KernelArgs[i].ArgType[0],@buffCount);                    CheckError();
        FErr:=clGetKernelArgInfo(FKernels[Index],i,CL_KERNEL_ARG_TYPE_QUALIFIER,sizeof(result.KernelArgs[i].ArgTypeQualifier),@result.KernelArgs[i].ArgTypeQualifier,@buffCount);   CheckError();
        FErr:=clGetKernelArgInfo(FKernels[Index],i,CL_KERNEL_ARG_ACCESS_QUALIFIER, sizeof(result.KernelArgs[i].ArgAccess),@result.KernelArgs[i].ArgAccess,@buffCount);              CheckError();
        FErr:=clGetKernelArgInfo(FKernels[Index],i,CL_KERNEL_ARG_ADDRESS_QUALIFIER, sizeof(result.KernelArgs[i].ArgAddress),@result.KernelArgs[i].ArgAddress,@buffCount);           CheckError();
    end;

end;

function TOpenCL.CanExecuteNative: boolean;
begin
  result:=FExecCaps and CL_EXEC_NATIVE_KERNEL>0;
end;

procedure TOpenCL.LoadFromFile(FileName: ansistring);
begin
  FSrc.LoadFromFile(FileName);
  ProgramSource:=FSrc.Text;
end;

function TOpenCL.createDeviceBuffer(const aByteSize: size_t;
  const aAccess: TCLMemAccess; const fromHostMem: Pointer): cl_mem;
begin
  result := clCreateBuffer(FContext, cl_mem_flags(aAccess), aByteSize, fromHostMem, FErr);CheckError();
end;

procedure TOpenCL.freeDeviceBuffer(aMem: cl_mem);
begin
  clReleaseMemObject(aMem);CheckError();
end;

(*
procedure TOpenCL.CallKernel(const Index: integer; const dst:PLongWord;const c: integer);
var ki:TCLKernelInfo;sz:size_t;i:integer;
begin
  sz:=FGlobalWorkGroupSizes[0];
  for i:=1 to FWorkItemDimensions-1 do
    sz:=sz*FGlobalWorkGroupSizes[i];
  FCallParams[0]:=clCreateBuffer(FContext,CL_MEM_READ_WRITE,sz*SizeOf(LongWord),nil,FErr);CheckError();
  //FCallParams[1]:=clCreateBuffer(FContext,CL_MEM_READ_ONLY, c*4,nil,FErr);
  //FCallParams[2]:=clCreateBuffer(FContext,CL_MEM_READ_ONLY, c*8,nil,FErr);
  FErr:=clSetKernelArg(FActiveKernel,0,sizeOf(@FCallParams[0]),@FCallParams[0]);CheckError();
  //FErr:=clSetKernelArg(FActiveKernel,1,sizeOf(cl_mem),FCallParams[1]);
  //FErr:=clSetKernelArg(FActiveKernel,2,sizeOf(cl_mem),FCallParams[2]);
  FErr:=clSetKernelArg(FActiveKernel,1,SizeOf(c),@c);CheckError();
  FErr:=clEnqueueNDRangeKernel(FQueue,FActiveKernel,FWorkItemDimensions,FGlobalOffsets,FGlobalWorkGroupSizes,FLocalWorkGroupSizesPtr,0,nil,nil);CheckError();
  FErr:=clEnqueueReadBuffer(FQueue,FCallParams[0],CL_True,0,sz*SizeOf(LongWord),dst,0,nil,nil);CheckError();
  //FErr:=clFlush(FQueue);
  //FErr:=clFinish(FQueue);

end;
*)
procedure TOpenCL.CallKernel(const Index: integer; const dst, a, b: PSingle;
  const bias: single; const c: integer);
var ki:TCLKernelInfo;sz:size_t;i:integer;
begin
  sz:=FGlobalWorkGroupSizes[0];
  for i:=1 to FWorkItemDimensions-1 do
    sz:=sz*FGlobalWorkGroupSizes[i];

    FCallParams[0]:=clCreateBuffer(FContext,CL_MEM_WRITE_ONLY {or CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR}, {FParamSizes[0]*}sz*SizeOf(dst^),nil,FErr);CheckError();
    FCallParams[1]:=clCreateBuffer(FContext,CL_MEM_USE_HOST_PTR or CL_MEM_READ_ONLY,{FParamSizes[1]*}sz*SizeOf(a^),    a,FErr);CheckError();
    FCallParams[2]:=clCreateBuffer(FContext,CL_MEM_USE_HOST_PTR or CL_MEM_READ_ONLY,{FParamSizes[2]*}c*c*SizeOf(b^),    b,FErr);CheckError();
    FErr:=clSetKernelArg(FKernels[Index],0,sizeof(@FCallParams[0]),@FCallParams[0]);CheckError();
    FErr:=clSetKernelArg(FKernels[Index],1,sizeOf(@FCallParams[1]),@FCallParams[1]);CheckError();
    FErr:=clSetKernelArg(FKernels[Index],2,sizeOf(@FCallParams[2]),@FCallParams[2]);CheckError();
    FErr:=clSetKernelArg(FKernels[Index],3,SizeOf(bias),@bias);CheckError();
    FErr:=clSetKernelArg(FKernels[Index],4,SizeOf(c),@c);CheckError();
    FErr:=clEnqueueNDRangeKernel(FQueue,FKernels[Index], FWorkItemDimensions, @FGlobalOffsets[0], @FGlobalWorkGroupSizes[0], @FLocalWorkGroupSizes[0], 0, nil, nil); CheckError();
//  if not FSharedMemory then
    FErr:=clEnqueueReadBuffer(FQueue,FCallParams[0],CL_FALSE,0,{FParamSizes[0]*}sz*SizeOf(dst^),dst,0,nil,nil);CheckError();
  //FErr:=clFlush(FQueue);
  FErr:=clFinish(FQueue);

end;

procedure TOpenCL.CallKernel(const Index: integer; const params: TArray<Pointer>
  );
var
  ki:TCLKernelInfo;
  //sz:size_t;
  i,j:integer; s:ansistring;
begin
  if FSharedMemory then
    FCallParams[0]:=clCreateBuffer(FContext,CL_MEM_READ_WRITE or CL_MEM_USE_HOST_PTR {or CL_MEM_ALLOC_HOST_PTR}, FParamSizes[0],Params[0],FErr)
  else
    FCallParams[0]:=clCreateBuffer(FContext,CL_MEM_READ_WRITE {or CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR}, FParamSizes[0],nil      ,FErr);CheckError();

  for i:=0 to KernelInfo(Index).KernelArgCount-1 do begin
    j:=KernelInfo(Index).KernelArgs[i].ArgAccess;
    if Pos('*',KernelInfo(Index).KernelArgs[i].ArgType)>0 then
      FCallParams[i]:=clCreateBuffer(FContext,CL_MEM_READ_ONLY or CL_MEM_USE_HOST_PTR ,sizeof(cl_mem),params[i],FErr);CheckError();
  end;

  for i:=0 to KernelInfo(Index).KernelArgCount-1 do begin
    if Pos('*',KernelInfo(Index).KernelArgs[i].ArgType)>0 then
      FErr:=clSetKernelArg(FKernels[Index],i,sizeof(@FCallParams[i]),@FCallParams[i])
    else
      FErr:=clSetKernelArg(FKernels[Index],i,FParamSizes[i],params[i]);CheckError();
  end;
  FErr:=clEnqueueNDRangeKernel(FQueue,FKernels[Index],FWorkItemDimensions, @FGlobalOffsets[0], @FGlobalWorkGroupSizes[0], @FLocalWorkGroupSizes[0] ,0,nil,nil); CheckError();
  if not FSharedMemory then begin
    FErr:=clEnqueueReadBuffer(FQueue,FCallParams[0],CL_FALSE,0,FParamSizes[0],params[0],0,nil,nil);CheckError();
    FErr:=clFinish(FQueue);
  end;

end;

procedure TOpenCL.CallKernel(const Index: integer; const dst: PLongWord; const a, b: integer);
var ki:TCLKernelInfo;sz:size_t;i:integer;
begin
  if Index > high(FKernels) then
    raise ERangeError.CreateFmt('Kernel [%d] out of Bounds : Number of Kernels = %d',[Index, length(FKernels)]);
  sz:=FGlobalWorkGroupSizes[0];
  for i:=1 to FWorkItemDimensions-1 do
    sz:=sz*FGlobalWorkGroupSizes[i];

  FCallParams[0]:=clCreateBuffer(FContext, {CL_MEM_WRITE_ONLY or} CL_MEM_USE_HOST_PTR {or CL_MEM_ALLOC_HOST_PTR}, FParamSizes[0],dst,FErr);CheckError();

  FErr:=clSetKernelArg(FKernels[Index],0,sizeof(cl_mem),@FCallParams);CheckError();
  FErr:=clSetKernelArg(FKernels[Index],1,SizeOf(a),@a);CheckError();
  FErr:=clSetKernelArg(FKernels[Index],2,SizeOf(b),@b);CheckError();
  FErr:=clEnqueueNDRangeKernel(FQueue,FKernels[Index] ,FWorkItemDimensions ,@FGlobalOffsets[0] , @FGlobalWorkGroupSizes[0] ,@FLocalWorkGroupSizes[0] ,0 ,nil ,nil ); CheckError();
  FErr:=clEnqueueReadBuffer(FQueue,FCallParams[0], CL_TRUE,0, FParamSizes[0], dst,0,nil,nil);CheckError();
  //FErr:=clFinish(FQueue);
  FErr:=clReleaseMemObject(FCallParams[0]);CheckError();

end;

procedure TOpenCL.finish();
begin
  FErr := clFinish(ActiveQueue); CheckError();
end;

procedure TOpenCL.waitForEvents(const N: longword; const events: pcl_event);
begin
  if N= 0 then exit;
  FErr := clWaitForEvents(N, events); CheckError();
end;

procedure TOpenCL.freeEvents(const N: longword; const events: pcl_event);
var i:SizeInt;
begin
  for i:=0 to N-1 do begin
      FErr := clReleaseEvent(events[i]); CheckError();
      events[i] := nil;
  end;
end;

function LSize(const aSize:SizeInt):SizeInt;inline;
begin
  if aSize mod 13 = 0 then exit(13);
  if aSize mod 11 = 0 then exit(11);
  if aSize mod 8  = 0 then exit(8);
  if aSize mod 7  = 0 then exit(7);
  if aSize mod 6  = 0 then exit(6);
  if aSize mod 5  = 0 then exit(5);
  if aSize mod 4  = 0 then exit(4);
  if aSize mod 3  = 0 then exit(3);
  if aSize mod 2  = 0 then exit(2);
  result := 1;
end;

procedure TOpenCL.ActivateArray(const x: cl_mem; const N: SizeInt;
  const activation: longint; const eventCount: cl_uint;
  const events: pcl_event; event: pcl_event);
const kernelId = 5;
var NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(x)          , @x);         CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(activation) , @activation);   CheckError();
  FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId]
    , WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
    , eventCount, Events, event); CheckError();
  //inc(eventsCount) ;
  //FErr := clFinish(ActiveQueue); CheckError();
end;

procedure TOpenCL.activateArraySWISH(const x: cl_mem; const N: SizeInt;
  const output_sigmoid, output: cl_mem; const eventCount: cl_uint;
  const events: pcl_event; event: pcl_event);
const kernelId = 6;
var NN:SizeInt;
begin
  //NN:=LSize(N);
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(x)             , @x);               CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(output_sigmoid), @output_sigmoid);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(output)        , @output);          CheckError();
  FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId],
    WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
    , eventCount, Events, event); CheckError();
  //inc(eventsCount);
  //FErr := clFinish(ActiveQueue); CheckError();
end;

procedure TOpenCL.DeriveArray(const x: cl_mem; const N: SizeInt;
  const activation: longint; delta: cl_mem; const eventCount: cl_uint;
  const events: pcl_event; event: pcl_event);
const kernelId = 7;
var NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(x)          , @x);           CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(activation) , @activation);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(delta)      , @delta);       CheckError();
  FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId]
  , WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0]
  , nil{@LocalWorkGroupSizes[0]}
  , eventCount, Events, event); CheckError();
  //inc(eventsCount);
  //FErr := clFinish(ActiveQueue); CheckError();
end;

procedure TOpenCL.forwardBias(const N: SizeInt; const a: cl_mem;
  const blockSize: SizeInt; const b: cl_mem; const incb: SizeInt;
  const batch: SizeInt; const eventCount: cl_uint; const events: pcl_event;
  event: pcl_event);
const kernelId=4;
var
    NN, MM , i, k, aOffset, bOffset:SizeInt;
begin

  //NN:=LSize(N);
  //MM:=LSize(blockSize);
  SetGlobalWorkGroupSizes(N, batch, blockSize);
  SetGlobalOffsets(0);
  //SetLocalWorkGroupSizes(MM, NN);
  //SetGlobalWorkGroupSizes(N, blockSize);
  //SetLocalWorkGroupSizes(NN, MM);
  aOffset :=0;
  bOffset :=0;
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(a)        , @a);         CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(aOffset)  , @aOffset);   CheckError();
  //FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(blockSize), @blockSize); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(b)        , @b);         CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(bOffset)  , @bOffset);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(incb)     , @incb);      CheckError();
  FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId]
  , WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
  , eventCount, Events, event); CheckError();

end;

procedure TOpenCL.backwardBias(const N: SizeInt; const a: cl_mem;
  const blockSize: SizeInt; const b: cl_mem; const incb: SizeInt;
  const batch: SizeInt; const eventCount: cl_uint; const events: pcl_event;
  event: pcl_event);
const kernelId=8;
begin
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //SetLocalWorkGroupSizes(MM, NN);
  //SetGlobalWorkGroupSizes(N, blockSize);
  //SetLocalWorkGroupSizes(NN, MM);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(a)        , @a);         CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(blockSize), @blockSize); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(b)        , @b);         CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(batch)    , @batch);     CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(N)        , @N);         CheckError();
  FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId]
    , WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
    , eventCount, Events, event); CheckError();
  //inc(eventsCount);
  //FErr := clFinish(ActiveQueue); CheckError();
end;

procedure TOpenCL.gemm(const transA, transB: boolean; const M, N, K: SizeInt;
  const ALPHA: single; const A: cl_mem; const aOffset: SizeInt;
  const lda: SizeInt; const B: cl_mem; const bOffset: SizeInt;
  const ldb: SizeInt; const BETA: single; const C: cl_mem;
  const cOffset: SizeInt; const ldc: SizeInt; const eventCount: cl_uint;
  const events: pcl_event; event: pcl_event);
var MM, KK, NN :SizeInt; kernelId:integer;
begin
  //     K          N          N
  //   [...]      [...]      [...]
  // M [.A.]  X K [.B.] => M [.C.]
  //   [...]      [...]      [...]

  //MM := LSize(M);
  //NN := LSize(N);

  if (not transA) and (not transB)then
    if N > M then begin
      SetGlobalWorkGroupSizes(N, M);
      //SetLocalWorkGroupSizes(NN, MM);
      kernelId :=1;
    end else begin
      SetGlobalWorkGroupSizes(M, N);
    //  SetLocalWorkGroupSizes(MM, NN);
      kernelId :=0;
    end
  else if (not transA) and transB then begin
    SetGlobalWorkGroupSizes(M, N);
    kernelId := 2;
  end else if transA and (not transB) then begin
    SetGlobalWorkGroupSizes(M, N);
    kernelId := 3 ;
  end;

  FErr:=clSetKernelArg(FKernels[kernelId], 0, SizeOf(K)       , @K);CheckError();
  FErr:=clSetKernelArg(FKernels[kernelId], 1, SizeOf(ALPHA)   , @ALPHA);CheckError();
  FErr:=clSetKernelArg(FKernels[kernelId], 2, SizeOf(cl_mem)  , @A); CheckError();
  FErr:=clSetKernelArg(FKernels[kernelId], 3, SizeOf(aOffset) , @aOffset); CheckError();
  FErr:=clSetKernelArg(FKernels[kernelId], 4, SizeOf(lda)     , @lda); CheckError();

  FErr:=clSetKernelArg(FKernels[kernelId], 5, SizeOf(cl_mem)  , @B); CheckError();
  FErr:=clSetKernelArg(FKernels[kernelId], 6, SizeOf(bOffset) , @bOffset); CheckError();
  FErr:=clSetKernelArg(FKernels[kernelId], 7, SizeOf(ldb)     , @ldb); CheckError();

  FErr:=clSetKernelArg(FKernels[kernelId], 8, SizeOf(BETA)     , @BETA); CheckError();
  FErr:=clSetKernelArg(FKernels[kernelId], 9, SizeOf(cl_mem)   , @C); CheckError();
  FErr:=clSetKernelArg(FKernels[kernelId], 10, SizeOf(cOffset) , @cOffset); CheckError();
  FErr:=clSetKernelArg(FKernels[kernelId], 11, SizeOf(ldc)     , @ldc); CheckError();

  FErr:=clEnqueueNDRangeKernel(FQueue, FKernels[kernelId] ,FWorkItemDimensions ,@FGlobalOffsets[0]
    , @FGlobalWorkGroupSizes[0] ,nil{@FLocalWorkGroupSizes[0]}
    , eventCount, Events, event); CheckError();
  //inc(eventsCount);

end;

procedure TOpenCL.addvv(const N: SizeInt; const dst, src: cl_mem;
  const eventCount: cl_uint; const events: pcl_event; event: pcl_event);
const kernelId = 9;
var NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(dst) , @dst);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(src) , @src);   CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , eventCount, Events, event); CheckError();
end;

procedure TOpenCL.subvv(const N: SizeInt; const dst, src: cl_mem;
  const eventCount: cl_uint; const events: pcl_event; event: pcl_event);
const kernelId = 10;
var NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(dst) , @dst);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(src) , @src);   CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , eventCount, events, event); CheckError();
end;

procedure TOpenCL.axpy(const N: SizeInt; const a: single; const x: cl_mem;
  const incx: SizeInt; const y: cl_mem; const incy: sizeInt;
  const eventCount: cl_uint; const events: pcl_event; event: pcl_event);
const kernelId = 11;
var NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(a)    , @a);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(x)    , @x);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(incx) , @incx); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(y)    , @y);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(incy) , @incy); CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , eventCount, Events, event); CheckError();
end;

procedure TOpenCL.scale(const N: SizeInt; const a: Single; const x: cl_mem;
  const stride: SizeInt; const eventCount: cl_uint; const events: pcl_event;
  event: pcl_event);
const kernelId = 12;
var NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(a)    , @a);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(x)    , @x);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(stride) , @stride); CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , eventCount, Events, event); CheckError();
end;

procedure TOpenCL.crossEntropyLogistic(const N: SizeInt; const pred,
  truth: cl_mem; delta, error: cl_mem; const eventCount: cl_uint;
  const events: pcl_event; event: pcl_event);
const kernelId = 13;
var NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(pred)    , @pred);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(truth)   , @truth); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(delta)   , @delta); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(error)   , @error); CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , eventCount, Events, event); CheckError();
end;

procedure TOpenCL.fill(const N: SizeInt; const x: cl_mem; const val: single;
  const stride: SizeInt; const eventCount: cl_uint; const events: pcl_event;
  event: pcl_event);
const kernelId = 14;
var NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(x)     , @x);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(val)   , @val); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(stride), @stride); CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , eventCount, Events, event); CheckError();
end;

procedure TOpenCL.copy(const N: SizeInt; const a: cl_mem; const aOffset,
  inca: SizeInt; const b: cl_mem; const bOffset, incb: SizeInt;
  const eventCount: cl_uint; const events: pcl_event; event: pcl_event);
const kernelId = 15;
var NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(a)       , @a);       CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(aOffset) , @aOffset); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(inca)    , @inca);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(b)       , @b);       CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(bOffset) , @bOffset); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5, SizeOf(incb)    , @incb);    CheckError();

  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , eventCount, Events, event); CheckError();
end;

procedure TOpenCL.softmaxBatch(const input: cl_mem; const iOffset, N: SizeInt;
  const batch, batch_size, groups, group_size, stride: SizeInt;
  const temp: single; const output: cl_mem; const oOffset: SizeInt;
  const eventCount: cl_uint; const events: pcl_event; event: pcl_event);
const kernelId = 18;
var NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(batch, groups);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(input)       , @input);      CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(iOffset)     , @iOffset);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(N)           , @N);          CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(batch_size)  , @batch_size); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4, SizeOf(group_size)  , @group_size); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5, SizeOf(stride)      , @stride);     CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 6, SizeOf(temp)        , @temp);       CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 7, SizeOf(output)      , @output);     CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 8, SizeOf(oOffset)     , @oOffset);    CheckError();

  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , eventCount, Events, event); CheckError();
end;

procedure TOpenCL.crossEntropySoftmax(const N: SizeInt; const pred,
  truth: cl_mem; delta, error: cl_mem; const eventCount: cl_uint;
  const events: pcl_event; event: pcl_event);
const kernelId = 19;
var NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(N);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0, SizeOf(pred)    , @pred);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1, SizeOf(truth)   , @truth); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2, SizeOf(delta)   , @delta); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3, SizeOf(error)   , @error); CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , eventCount, Events, event); CheckError();
end;

procedure TOpenCL.forwardMaxPool(const aBatch, outC, outH, outW: SizeInt;
  const input: cl_mem; const c, h, w: SizeInt; const stride_x, stride_y,
  padding, kernelSize: SizeInt; indexes, output: cl_mem;
  const eventCount: cl_uint; const events: pcl_event; event: pcl_event);
const kernelId = 16;
var NN, MM:SizeInt;
begin
  SetGlobalWorkGroupSizes(ABatch * outC, outH, outW);
  SetGlobalOffsets(0);
  //NN := LSize(ABatch * outC);
  //MM := LSize(outH* outW);
  //SetLocalWorkGroupSizes(NN, MM);

  FErr := clSetKernelArg(Kernels[kernelId], 0 , SizeOf(input)      , @input);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1 , SizeOf(c)          , @c);      CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2 , SizeOf(h)          , @h);      CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 3 , SizeOf(w)          , @w);      CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 4 , SizeOf(stride_x)   , @stride_x);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 5 , SizeOf(stride_y)   , @stride_y);   CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 6 , SizeOf(padding)    , @padding);    CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 7 , SizeOf(kernelSize) , @kernelSize); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 8 , SizeOf(indexes)    , @indexes);     CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 9 , SizeOf(output)     , @output);     CheckError();
  FErr := clEnqueueNDRangeKernel(ActiveQueue, Kernels[kernelId], WorkItemDimensions, @GlobalOffsets[0], @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}, eventCount, Events, event); CheckError();
end;

procedure TOpenCL.backwardMaxPool(const aBatch, outC, outH, outW: SizeInt;
  output: cl_mem; const indexes, delta: cl_mem; const eventCount: cl_uint;
  const events: pcl_event; event: pcl_event);
const kernelId = 17;
var
    NN:SizeInt;
begin
  SetGlobalWorkGroupSizes(ABatch*outC, outH*outW);
  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(Kernels[kernelId], 0 , SizeOf(output)  , @output);  CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 1 , SizeOf(indexes) , @indexes); CheckError();
  FErr := clSetKernelArg(Kernels[kernelId], 2 , SizeOf(delta)   , @delta);   CheckError();
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , eventCount, Events, event); CheckError();
end;

//procedure TOpenCL.im2col(const aChannels, aHeight, aWidth, kernelHeight,
//  kernelWidth, padHeight, padWidth, strideY, strideX, dilationY,
//  dilationX: SizeInt; const im: cl_mem; const imOffset: SizeInt;
//  const col: cl_mem; const colOffset: SizeInt; const batch: SizeInt;
//  const eventCount: cl_uint; const events: pcl_event; event: pcl_event);
//const kernelId = 20;
//var
//    NN:SizeInt;
//begin
//  SetGlobalWorkGroupSizes(aChannels, kernelHeight* kernelWidth);
//  SetGlobalOffsets(0);
//  //NN:=LSize(N);
//  //SetLocalWorkGroupSizes(NN);
//  FErr := clSetKernelArg(Kernels[kernelId], 0 , SizeOf(aHeight)         , @aHeight);  CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 1 , SizeOf(aWidth)          , @aWidth); CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 2 , SizeOf(kernelHeight)    , @kernelHeight);   CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 3 , SizeOf(kernelWidth)     , @kernelWidth);  CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 4 , SizeOf(padHeight)       , @padHeight); CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 5 , SizeOf(padWidth)        , @padWidth);   CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 6 , SizeOf(strideY)         , @strideY);  CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 7 , SizeOf(strideX)         , @strideX); CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 8 , SizeOf(dilationY)       , @dilationY);   CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 9 , SizeOf(dilationX)       , @dilationX);  CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 10, SizeOf(im)              , @im); CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 11, SizeOf(imOffset)        , @imOffset);   CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 12, SizeOf(col)             , @col);  CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 13, SizeOf(colOffset)       , @colOffset); CheckError();
//  FErr := clSetKernelArg(Kernels[kernelId], 14, SizeOf(batch)           , @batch);   CheckError();
//  FErr := clEnqueueNDRangeKernel(
//     ActiveQueue, Kernels[kernelId],
//     WorkItemDimensions, @GlobalOffsets[0],
//     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
//     , eventCount, Events, event); CheckError();
//end;

function ceil(const a, b: SizeInt):SizeInt;  overload;
begin
  result := (1 + (a-1) div b)*b
end;

const COPY_DIMX = 8;
      COPY_DIMY = 8;

procedure TOpenCL.im2col(const aChannels, aHeight, aWidth, kernelHeight,
  kernelWidth, padHeight, padWidth, strideY, strideX, dilationY,
  dilationX: SizeInt; const im: cl_mem; const imOffset: SizeInt;
  const col: cl_mem; const colOffset: SizeInt ;
  const eventCount: cl_uint; const events: pcl_event; event: pcl_event);
const kernelId = 22;
var
    NN, size_h, padding_h, col_h, padding_w, size_w, col_w, ceiled_w,
      ceiled_h:SizeInt;
begin
  // Sets the height and width of the 'col' result
  size_h      := aHeight + 2 * padHeight;
  padding_h   := dilationY * (kernelHeight - 1) + 1;
  if size_h >= padding_h then
    col_h       := (size_h - padding_h) div strideY + 1
  else
    col_h       := 1;
  size_w      := aWidth + 2 * padWidth;
  padding_w   := dilationX * (kernelWidth - 1) + 1;
  if size_w >= padding_w then
    col_w       := (size_w - padding_w) div strideX + 1
  else
    col_w       := 1;
  ceiled_w    := Ceil(col_w, COPY_DIMX);
  ceiled_h    := Ceil(col_h, COPY_DIMY);

  SetGlobalWorkGroupSizes(ceiled_w, ceiled_h * aChannels);
  SetGlobalOffsets(0);

  // Sets the kernel arguments
  FErr := clSetKernelArg(FKernels[kernelId], 0,  sizeOf(aHeight    ), @aHeight    ); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 1,  sizeOf(aWidth     ), @aWidth     ); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 2,  sizeOf(aChannels  ), @aChannels  ); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 3,  sizeOf(col_h     ), @col_h     ); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 4,  sizeOf(col_w     ), @col_w     ); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 5,  sizeOf(kernelHeight  ), @kernelHeight  ); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 6,  sizeOf(kernelWidth  ), @kernelWidth  ); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 7,  sizeOf(padHeight     ), @padHeight     ); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 8,  sizeOf(padWidth     ), @padWidth     ); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 9,  sizeOf(strideY  ), @strideY  ); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 10, sizeOf(strideX  ), @strideX  ); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 11, sizeOf(dilationY), @dilationY); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 12, sizeOf(dilationX), @dilationX); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 13, sizeOf(im        ), @im        ); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 14, sizeOf(imOffset  ), @imOffset  ); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 15, sizeOf(col       ), @col       ); checkError();
  FErr := clSetKernelArg(FKernels[kernelId], 16, sizeOf(colOffset ), @colOffset ); checkError();

  // Launches the kernel
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , eventCount, Events, event); CheckError();
end;

//// Solve Bezout's identity
//// a * p + b * q = r = GCD(a, b)
procedure EuclidGCD( a, b: SizeInt; var p, q, r:SizeInt);inline;
var p_1, p_2, q_1, q_2, c:SizeInt;
begin
  p := 0;
  q := 1;
  p_1 := 1;
  q_1 := 0;
  while true do begin
    c := a mod b;
    if c = 0  then
      break;
    p_2 := p_1;
    q_2 := q_1;
    p_1 := p;
    q_1 := q;
    p := p_2 - p_1 * (a div b);
    q := q_2 - q_1 * (a div b);
    a := b;
    b := c;
  end;
  r := b;
end;

procedure TOpenCL.col2im(const aChannels, aHeight, aWidth, kernelHeight,
  kernelWidth, padHeight, padWidth, strideY, strideX, dilationY,
  dilationX: SizeInt; const col: cl_mem; const colOffset: SizeInt;
  const im: cl_mem; const imOffset: SizeInt; const eventCount: cl_uint;
  const events: pcl_event; event: pcl_event);
const kernelId = 24;
var
    NN, size_h, padding_h, col_h, size_w, padding_w, col_w,
      dilation_bez_h, dilation_bez_w, gcd_h, gcd_w, stride_bez_h,
      stride_bez_w, w_ceiled, h_ceiled:SizeInt;

begin
  //SetGlobalWorkGroupSizes(aChannels{, kernelHeight, kernelWidth});

  size_h          := AHeight + 2 * padHeight;
  padding_h       := dilationY * (kernelHeight - 1) + 1;
  if size_h >= padding_h then
    col_h           := (size_h - padding_h) div strideY + 1
  else
    col_h           := 1;
  size_w          := AWidth + 2 * padWidth;
  padding_w       := dilationX * (kernelWidth - 1) + 1;
  if size_w >= padding_w then
    col_w           := (size_w - padding_w) div strideX + 1
  else
    col_w           := 1;
  stride_bez_h    := 0;
  stride_bez_w    := 0;
  dilation_bez_h  := 0;
  dilation_bez_w  := 0;
  gcd_h           := 0;
  gcd_w           := 0;
  EuclidGCD(strideY, dilationY, stride_bez_h, dilation_bez_h, gcd_h);
  EuclidGCD(strideX, dilationX, stride_bez_w, dilation_bez_w, gcd_w);

  w_ceiled := Ceil((aWidth - 1) div gcd_w + 1, COPY_DIMX);
  h_ceiled := Ceil((aHeight - 1) div gcd_h + 1, COPY_DIMY);
  //const auto local = std::vector<size_t>{db_["COPY_DIMX"], db_["COPY_DIMY"]};
  SetGlobalWorkGroupSizes(w_ceiled, h_ceiled * aChannels);

  SetGlobalOffsets(0);
  //NN:=LSize(N);
  //SetLocalWorkGroupSizes(NN);
  FErr := clSetKernelArg(kernels[kernelId], 0,  sizeof(aHeight       ), @aHeight         ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 1,  sizeof(aWidth        ), @aWidth          ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 2,  sizeof(aChannels     ), @aChannels       ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 3,  sizeof(col_h         ), @col_h           ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 4,  sizeof(col_w         ), @col_w           ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 5,  sizeof(kernelHeight  ), @kernelHeight    ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 6,  sizeof(kernelWidth   ), @kernelWidth     ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 7,  sizeof(padHeight     ), @padHeight       ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 8,  sizeof(padWidth      ), @padWidth        ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 9,  sizeof(strideY       ), @strideY         ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 10, sizeof(strideX       ), @strideX         ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 11, sizeof(dilationY     ), @dilationY       ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 12, sizeof(dilationX     ), @dilationX       ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 13, sizeof(stride_bez_h  ), @stride_bez_h    ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 14, sizeof(stride_bez_w  ), @stride_bez_w    ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 15, sizeof(dilation_bez_h), @dilation_bez_h  ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 16, sizeof(dilation_bez_w), @dilation_bez_w  ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 17, sizeof(gcd_h         ), @gcd_h           ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 18, sizeof(gcd_w         ), @gcd_w           ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 19, sizeof(col           ), @col             ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 20, sizeof(colOffset     ), @colOffset       ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 21, sizeof(im            ), @im              ); checkError();
  FErr := clSetKernelArg(kernels[kernelId], 22, sizeof(imOffset      ), @imOffset        ); checkError();

  FErr := clEnqueueNDRangeKernel(
     ActiveQueue, Kernels[kernelId],
     WorkItemDimensions, @GlobalOffsets[0],
     @GlobalWorkGroupSizes[0], nil{@LocalWorkGroupSizes[0]}
     , eventCount, Events, event); CheckError();
end;

class function TOpenCL.Plaforms: cl_uint;
begin
    clGetPlatformIDs(0,nil,@result);
end;

//function TOpenCL.KernelArgs(index: integer): TCLKernelArgInfo;
//begin
//  clGetKernelInfo(FKernels[Index],CL_KERNEL_NUM_ARGS,SizeOf(Result),@result,N);
//end;

function TOpenCL.GetDevices(index: cl_uint): cl_device_id;
begin
  result:=FDevices[index];
end;

function TOpenCL.getQueueInOrder: boolean;
var props:cl_bitfield; sz: size_t;
begin
  FErr := clGetCommandQueueInfo(ActiveQueue, CL_QUEUE_PROPERTIES, SizeOf(props), @props, sz);CheckError();
  result := (props and CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) = 0;
end;

procedure TOpenCL.SetActiveDeviceId(AValue: integer);
var wasBuilt:boolean; isShared:cl_bool;
begin
  if FActiveDevice=FDevices[AValue] then Exit;
  if AValue>High(FDevices) then
    raise Exception.Create('Device index out of bounds!');
  wasBuilt:=FIsBuilt;
  CleanUp(true);
  FQueue:=clCreateCommandQueue(FContext,FDevices[AValue], 0, 0 (* QWord(@FErr) *) ); CheckError();
  FActiveDevice:=FDevices[AValue];
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_EXECUTION_CAPABILITIES,SizeOf(cl_device_exec_capabilities),@FExecCaps, buffCount);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_MAX_WORK_GROUP_SIZE,SizeOf(FMaxWorkGroupSize),@FMaxWorkGroupSize, buffCount);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,SizeOf(FMaxWorkItemDimensions),@FMaxWorkItemDimensions, buffCount);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_MAX_MEM_ALLOC_SIZE,SizeOf(FMaxMemAllocSize),@FMaxMemAllocSize, buffCount);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_MAX_WORK_ITEM_SIZES,SizeOf(size_t)*3,@FMaxWorkItemSizes[0], buffCount);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_MAX_COMPUTE_UNITS,SizeOf(FMaxComputeUnits),@FMaxComputeUnits, buffCount);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_MAX_CLOCK_FREQUENCY,SizeOf(FMaxFrequency),@FMaxFrequency, buffCount);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_GLOBAL_MEM_SIZE,SizeOf(FGlobalMemSize),@FGlobalMemSize, buffCount);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_LOCAL_MEM_SIZE,SizeOf(FLocalMemSize),@FLocalMemSize, buffCount);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,SizeOf(FGlobalCacheSize),@FGlobalCacheSize, buffCount);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_BUILT_IN_KERNELS,cInfoSize,@FDeviceBuiltInKernels[0], buffCount);CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_OPENCL_C_VERSION,cInfoSize,@CLDeviceCVersion[0], buffCount); CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_VENDOR,cInfoSize,@CLDeviceDriver, buffCount);            CheckError();
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_HOST_UNIFIED_MEMORY,SizeOf(isShared),@isShared, buffCount);CheckError();
  FSharedMemory:=isShared=CL_TRUE;
  if wasBuilt then
    Build;
  FActiveDeviceId:=AValue;
end;

procedure TOpenCL.SetActiveKernelId(AValue: integer);
begin
  if FActiveKernelId=AValue then exit;
  FActiveKernel:=FKernels[AValue];
  FActiveKernelId:=AValue;
  FActiveKernelInfo := KernelInfo(AValue)
end;

procedure TOpenCL.SetActivePlatformId(AValue: integer);
var i:integer; dt:cl_device_type;
begin
  assert(AValue<length(FPlatforms), format('Invalid platform id [%d]',[AValue]));
  if (FActivePlatform=FPlatforms[AValue]) then Exit;
  if AValue>High(FPlatforms) then raise Exception.Create('Platform index out of bounds!');
  FActivePlatform:=FPlatforms[AValue];
  FErr:=clGetDeviceIDs(FActivePlatform,getCL_Device_Type(FDeviceType),0,nil,@FDeviceCount);  CheckError();
  if FDeviceCount=0 then raise Exception.Create('No Devices found!');
  setLength(FDevices,FDeviceCount);
  setLength(FDevicesType,FDeviceCount);
  FErr:=clGetDeviceIDs(FActivePlatform,getCL_Device_Type(FDeviceType),FDeviceCount,@FDevices[0],nil);  CheckError();
  FDevsTypeStr:='';
  for i:=0 to FDeviceCount-1 do
    begin
      FErr:=clGetDeviceInfo(FDevices[i],CL_DEVICE_TYPE_INFO,SizeOf(size_t),@dt, buffCount);  CheckError();
      Case dt of
        CL_DEVICE_TYPE_DEFAULT:begin FDevicesType[i]:=dtDefault;FDevsTypeStr:=FDevsTypeStr+#13#10'DEFAULT' end;
        CL_DEVICE_TYPE_CPU:begin FDevicesType[i]:=dtCPU;FDevsTypeStr:=FDevsTypeStr+#13#10'CPU' end;
        CL_DEVICE_TYPE_GPU:begin FDevicesType[i]:=dtGPU;FDevsTypeStr:=FDevsTypeStr+#13#10'GPU' end;
        CL_DEVICE_TYPE_ACCELERATOR:begin FDevicesType[i]:=dtACCELERATOR;FDevsTypeStr:=FDevsTypeStr+#13#10'ACCELERATOR' end;
      end;
    end;
  delete(FDevsTypeStr,1,2);
  if FContext<>nil then begin
    clReleaseContext(FContext);CheckError();
    FContext:=nil
  end;
  FContext:=clCreateContext(nil,FDeviceCount,@FDevices[0],nil,nil,FErr);CheckError();
  FActiveDeviceId:=-1;
  SetActiveDeviceId(0);
  FActivePlatformId:=AValue;
end;

initialization

end.

