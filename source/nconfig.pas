unit nConfig;
{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}

interface

uses
  SysUtils;

type
  { TCFGSection }

  PCFGSection = ^TCFGSection;
  TCFGSection = record
    type
    private
      FOptions : TArray<string>;
      function tryFindKey(key:string; out val:string):boolean;
      procedure insert(const s:string; var Arr:TArray<string>; const position:IntPtr);
      procedure delete(var Arr:TArray<string>;const position, len:IntPtr);
    public
      typeName:string;
      procedure setOptions(str:string);
      function getOptions():string;
      procedure addOption(key,val:string);
      function addOptionLine(str:string):boolean;
      function removeOption(key:string):boolean;
      function keyExists(key:string):boolean;
      function findKey(key:string):SizeInt;
      function getInt(key:string; def: SizeInt; const quite: boolean = true):SizeInt;
      function getFloat(key:string; def: single; const quite: boolean = true):single;
      function getBool(key:string; def: boolean; const quite: boolean = true):boolean;
      function getStr(key: string; def: string; const quite: boolean = true): string;
      procedure setInt(key:string; val:SizeInt);
      procedure setFloat(key:string; val:single);
      procedure setBool(key:string; val:boolean);
      procedure setStr(key: string; val: string);
  end;

  { TCFGList }

  TCFGList = record
    Sections : TArray<TCFGSection>;
    function loadFromFile(const fileName:string):SizeInt;
    procedure saveToFile(const fileName:string);
    procedure Insert(const s:TCFGSection; var Arr:TArray<TCFGSection>; const position:IntPtr);
    function addSection(section:TCFGSection):PCFGSection;
    function isEmpty():boolean;
    function Count():SizeInt;
  end;

  TChars = set of char;


implementation

function stripChars(const s:string; const chars: TChars =[' ',#9]):string;
var j:SizeInt;
begin
  j:=1;
  result := s;
  while j<length(result) do
    if result[j] in chars then delete(result,j,1)
    else inc(j);
end;

function stripComments(const str:string):string;
var
    i:SizeInt;
begin
  result := str;
  for i:=1 to length(result) do
    if result[i] in [';','#'] then
      begin
        System.delete(result, i, length(result));
        break
      end ;
end;

{ TCFGSection }

function TCFGSection.tryFindKey(key: string; out val: string): boolean;
var
  i: LongInt;
begin
  i:=findKey(key);
  result:= i>=0;
  if result then
    val := trim(copy(FOptions[i],pos('=', FOptions[i])+1))
end;


procedure TCFGSection.insert(const s: string; var Arr: TArray<string>;
  const position: IntPtr);
var i:integer;
begin
  setLength(Arr,Length(arr)+1);
  for i:=high(arr) downto position+1 do begin
    Arr[i]:=Arr[i-1]
  end;
  Arr[position]:=s
end;

procedure TCFGSection.delete(var Arr: TArray<string>; const position,
  len: IntPtr);
var
  i: Integer;
begin
  for i:=position to high(Arr)-Len do
    Arr[i]:=Arr[i+len];
  setLength(Arr,Length(arr)-len)
end;

procedure TCFGSection.setOptions(str: string);
var
  i,j: longint;
  opt:TArray<string>;
  s:string;
begin
  opt:=str.Split([#10]);
  FOptions:=nil;
  for i:=0 to high(opt) do begin
    s:=trim(opt[i]);
    for j:=1 to length(s) do
      if s[j] in [';','#'] then
        begin
          System.delete(s,j,length(s));
          break
        end ;
    if (s<>'') then
      insert(s, FOptions, length(FOptions))
  end
end;

function TCFGSection.getOptions(): string;
begin
  result:=string.Join(sLineBreak, FOptions)
end;

procedure TCFGSection.addOption(key, val: string);
var i:longint;
begin
  i:=findKey(key);
  if i<0 then begin
    i:=Length(FOptions);
    insert(key+' = '+val, FOptions,i)
  end
  else
    FOptions[i]:=key+' = '+val;
end;

function TCFGSection.addOptionLine(str: string): boolean;
var
  j: longint;
begin
  result := true;
  for j:=1 to length(str) do
    if str[j] in [';','#'] then
      begin
        System.delete(str,j,length(str));
        break
      end ;
  if pos('=',str)=0 then exit(false);
  insert(str, FOptions,length(FOptions));
end;

function TCFGSection.removeOption(key: string): boolean;
var
  i: Integer;
begin
  i:=findKey(key);
  result := i>=0;
  if result then
    delete(FOptions,i,1)
end;

function TCFGSection.keyExists(key: string): boolean;
begin
  result := findKey(key)>=0
end;

function TCFGSection.findKey(key: string): SizeInt;
var
  i,p: Integer;
begin
  result:=-1;
  for i:=0 to high(FOptions) do begin
    p:=pos('=',FOptions[i]);
    if p>0 then begin
      if trim(copy(FOptions[i],1,p-1))=key then
        exit(i)
    end;
  end;
end;

function TCFGSection.getInt(key: string; def: SizeInt; const quite: boolean
  ): SizeInt;
var
  r:string;code :integer;
begin
  if tryFindKey(key,r) then
    val(r,result, code)
  else begin
    result := def;
    if not quite then
        writeln(ErrOutput, format('%s: Using default [%d]', [key, def]));
  end;
end;

function TCFGSection.getFloat(key: string; def: single; const quite: boolean): single;
var
  r: string; code :integer;
begin
  if tryFindKey(key,r) then
    val(r,result, code)
  else begin
    result := def;
    if not quite then
        writeln(ErrOutput, format('%s: Using default [%f]', [key, def]));
  end;
end;

function TCFGSection.getBool(key: string; def: boolean; const quite: boolean): boolean;
var
  r: string;
  i, code:integer;
begin
  if tryFindKey(key,r) then
    begin
      val(r,i, code);
      result := i<>0;
    end
  else begin
    result := def;
    if not quite then
        writeln(ErrOutput, format('%s: Using default [%d]', [key, longint(def)]))
  end
end;

function TCFGSection.getStr(key: string; def: string; const quite: boolean): string;
begin
  if not tryFindKey(key, result) then begin
    result := def;
      if not quite then
        writeln(ErrOutput, format('%s: Using default [%s]', [key, def]));
  end;

end;

procedure TCFGSection.setInt(key: string; val: SizeInt);
begin
  self.addOption(key, IntToStr(val));
end;

procedure TCFGSection.setFloat(key: string; val: single);
begin
  self.addOption(key, FloatToStr(val));
end;

procedure TCFGSection.setBool(key: string; val: boolean);
begin
  self.addOption(key, IntToStr(longint(val)));
end;

procedure TCFGSection.setStr(key: string; val: string);
begin
  self.addOption(key, val);
end;

{ TCFGList }

function TCFGList.loadFromFile(const fileName: string): SizeInt;
var
    f: TextFile;
    line: string;
    //options: PList;
    current: PCFGSection;
begin
    if not FileExists(filename) then
        raise EFileNotFoundException(filename+': not found.');
    AssignFile(f, filename);
    reset(f);
    //f := fopen(filename, 'r');
    //if file = 0 then
    //    file_error(filename);
    result := 0;
    current := nil;
    while not EOF(f) do
        begin
            readln(f,line);
            inc(result);
            line := stripChars(line);
            line := stripComments(line);
            if line<>'' then
              case line[1] of
                  '[':
                      begin
                          line := stripComments(line);
                          current:=self.addSection(default(TCFGSection));
                          if line[length(line)]=']' then
                            delete(line,length(line),1);
                          delete(line,1,1);
                          current.typeName := line
                      end;
                  '#', ';':
                      //free(line);
              else
                  if not current.addOptionLine(line) then
                      begin
                          writeln(ErrOutput, format('Config file error line %d, could parse: %s', [result, line]));
                          //free(line)
                      end
              end
        end;
    CloseFile(f);
    //exit(options)
end;

procedure TCFGList.saveToFile(const fileName: string);
var i, j:SizeInt;
    f:TextFile;
begin
    //if not FileExists(filename) then
    //    raise EFileNotFoundException(filename+': not found.');
    AssignFile(f, filename);
    try
      rewrite(f);
      for i:=0 to high(Sections) do begin
        writeln(f, Sections[i].typeName);
        for j:=0 to high(sections[i].FOptions) do begin
          writeln(f, sections[i].FOptions[j]);
        end;
      end;

    finally
      closeFile(f);
    end;
end;

procedure TCFGList.Insert(const s: TCFGSection; var Arr: TArray<TCFGSection>;
  const position: IntPtr);
var i:integer;
begin
  setLength(Arr,Length(arr)+1);
  for i:=high(arr) downto position+1 do begin
    Arr[i]:=Arr[i-1]
  end;
  Arr[position]:=s
end;

function TCFGList.addSection(section: TCFGSection): PCFGSection;
begin
  insert(section, Sections, length(Sections));
  result := @Sections[high(Sections)]
end;

function TCFGList.isEmpty(): boolean;
begin
  result:= not assigned(Sections);
end;

function TCFGList.Count: SizeInt;
begin
  result := length(Sections)
end;

end.

