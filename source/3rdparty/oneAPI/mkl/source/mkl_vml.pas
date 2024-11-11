unit mkl_vml;
interface
uses mkl_types;
{$include mkl.inc}

{$IFDEF FPC}
{$PACKRECORDS C}
{$ENDIF}

Type
PDefVmlErrorContext  = ^DefVmlErrorContext;

  _DefVmlErrorContext = record
      iCode : longint;
      iIndex : longint;
      dbA1 : double;
      dbA2 : double;
      dbR1 : double;
      dbR2 : double;
      cFuncName : array[0..63] of char;
      iFuncNameLen : longint;
      dbA1Im : double;
      dbA2Im : double;
      dbR1Im : double;
      dbR2Im : double;
    end;
  DefVmlErrorContext = _DefVmlErrorContext;

  VMLErrorCallBack = function (pdefVmlErrorContext:PDefVmlErrorContext):longint;winapi;

  procedure vsAbs(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdAbs(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsAbs(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdAbs(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcAbs(n:longint; a:PMKL_Complex8; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzAbs(n:longint; a:PMKL_Complex16; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcAbs(n:longint; a:PMKL_Complex8; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzAbs(n:longint; a:PMKL_Complex16; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcArg(n:longint; a:PMKL_Complex8; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzArg(n:longint; a:PMKL_Complex16; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcArg(n:longint; a:PMKL_Complex8; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzArg(n:longint; a:PMKL_Complex16; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsAdd(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdAdd(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsAdd(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdAdd(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vcAdd(n:longint; a:PMKL_Complex8; b:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vzAdd(n:longint; a:PMKL_Complex16; b:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmcAdd(n:longint; a:PMKL_Complex8; b:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmzAdd(n:longint; a:PMKL_Complex16; b:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsSub(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdSub(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsSub(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdSub(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vcSub(n:longint; a:PMKL_Complex8; b:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vzSub(n:longint; a:PMKL_Complex16; b:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmcSub(n:longint; a:PMKL_Complex8; b:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmzSub(n:longint; a:PMKL_Complex16; b:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsInv(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdInv(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsInv(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdInv(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsSqrt(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdSqrt(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsSqrt(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdSqrt(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcSqrt(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzSqrt(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcSqrt(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzSqrt(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsInvSqrt(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdInvSqrt(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsInvSqrt(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdInvSqrt(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsCbrt(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdCbrt(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsCbrt(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdCbrt(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsInvCbrt(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdInvCbrt(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsInvCbrt(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdInvCbrt(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsSqr(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdSqr(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsSqr(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdSqr(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsExp(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdExp(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsExp(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdExp(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcExp(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzExp(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcExp(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzExp(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsExp2(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdExp2(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsExp2(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdExp2(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsExp10(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdExp10(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsExp10(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdExp10(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsExpm1(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdExpm1(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsExpm1(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdExpm1(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsLn(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdLn(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsLn(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdLn(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcLn(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzLn(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcLn(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzLn(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsLog2(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdLog2(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsLog2(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdLog2(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsLog10(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdLog10(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsLog10(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdLog10(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcLog10(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzLog10(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcLog10(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzLog10(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsLog1p(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdLog1p(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsLog1p(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdLog1p(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsLogb(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdLogb(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsLogb(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdLogb(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsCos(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdCos(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsCos(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdCos(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcCos(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzCos(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcCos(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzCos(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsSin(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdSin(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsSin(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdSin(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcSin(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzSin(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcSin(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzSin(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsTan(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdTan(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsTan(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdTan(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcTan(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzTan(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcTan(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzTan(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsCospi(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdCospi(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsCospi(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdCospi(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsSinpi(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdSinpi(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsSinpi(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdSinpi(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsTanpi(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdTanpi(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsTanpi(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdTanpi(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsCosd(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdCosd(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsCosd(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdCosd(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsSind(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdSind(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsSind(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdSind(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsTand(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdTand(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsTand(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdTand(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsCosh(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdCosh(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsCosh(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdCosh(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcCosh(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzCosh(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcCosh(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzCosh(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsSinh(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdSinh(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsSinh(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdSinh(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcSinh(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzSinh(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcSinh(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzSinh(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsTanh(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdTanh(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsTanh(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdTanh(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcTanh(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzTanh(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcTanh(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzTanh(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsAcos(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdAcos(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsAcos(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdAcos(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcAcos(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzAcos(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcAcos(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzAcos(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsAsin(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdAsin(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsAsin(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdAsin(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcAsin(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzAsin(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcAsin(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzAsin(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsAtan(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdAtan(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsAtan(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdAtan(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcAtan(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzAtan(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcAtan(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzAtan(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsAcospi(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdAcospi(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsAcospi(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdAcospi(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsAsinpi(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdAsinpi(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsAsinpi(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdAsinpi(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsAtanpi(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdAtanpi(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsAtanpi(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdAtanpi(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsAcosh(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdAcosh(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsAcosh(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdAcosh(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcAcosh(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzAcosh(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcAcosh(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzAcosh(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsAsinh(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdAsinh(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsAsinh(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdAsinh(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcAsinh(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzAsinh(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcAsinh(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzAsinh(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsAtanh(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdAtanh(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsAtanh(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdAtanh(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcAtanh(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzAtanh(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcAtanh(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzAtanh(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsErf(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdErf(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsErf(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdErf(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsErfInv(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdErfInv(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsErfInv(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdErfInv(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsHypot(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdHypot(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsHypot(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdHypot(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsErfc(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdErfc(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsErfc(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdErfc(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsErfcInv(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdErfcInv(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsErfcInv(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdErfcInv(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsCdfNorm(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdCdfNorm(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsCdfNorm(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdCdfNorm(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsCdfNormInv(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdCdfNormInv(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsCdfNormInv(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdCdfNormInv(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsLGamma(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdLGamma(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsLGamma(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdLGamma(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsTGamma(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdTGamma(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsTGamma(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdTGamma(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsAtan2(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdAtan2(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsAtan2(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdAtan2(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsAtan2pi(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdAtan2pi(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsAtan2pi(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdAtan2pi(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsMul(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdMul(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsMul(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdMul(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vcMul(n:longint; a:PMKL_Complex8; b:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vzMul(n:longint; a:PMKL_Complex16; b:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmcMul(n:longint; a:PMKL_Complex8; b:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmzMul(n:longint; a:PMKL_Complex16; b:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsDiv(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdDiv(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsDiv(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdDiv(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vcDiv(n:longint; a:PMKL_Complex8; b:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vzDiv(n:longint; a:PMKL_Complex16; b:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmcDiv(n:longint; a:PMKL_Complex8; b:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmzDiv(n:longint; a:PMKL_Complex16; b:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsPow(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdPow(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsPow(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdPow(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vcPow(n:longint; a:PMKL_Complex8; b:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vzPow(n:longint; a:PMKL_Complex16; b:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmcPow(n:longint; a:PMKL_Complex8; b:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmzPow(n:longint; a:PMKL_Complex16; b:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsPow3o2(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdPow3o2(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsPow3o2(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdPow3o2(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsPow2o3(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdPow2o3(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsPow2o3(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdPow2o3(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsPowx(n:longint; a:Psingle; b:single; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdPowx(n:longint; a:Pdouble; b:double; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsPowx(n:longint; a:Psingle; b:single; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdPowx(n:longint; a:Pdouble; b:double; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vcPowx(n:longint; a:PMKL_Complex8; b:MKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vzPowx(n:longint; a:PMKL_Complex16; b:MKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmcPowx(n:longint; a:PMKL_Complex8; b:MKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmzPowx(n:longint; a:PMKL_Complex16; b:MKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsPowr(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdPowr(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsPowr(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdPowr(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsSinCos(n:longint; a:Psingle; r1:Psingle; r2:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdSinCos(n:longint; a:Pdouble; r1:Pdouble; r2:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsSinCos(n:longint; a:Psingle; r1:Psingle; r2:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdSinCos(n:longint; a:Pdouble; r1:Pdouble; r2:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

       procedure vsLinearFrac(n:longint; a:Psingle; b:Psingle; scalea:single; shifta:single; 
            scaleb:single; shiftb:single; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

       procedure vdLinearFrac(n:longint; a:Pdouble; b:Pdouble; scalea:double; shifta:double; 
            scaleb:double; shiftb:double; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

       procedure vmsLinearFrac(n:longint; a:Psingle; b:Psingle; scalea:single; shifta:single; 
            scaleb:single; shiftb:single; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

       procedure vmdLinearFrac(n:longint; a:Pdouble; b:Pdouble; scalea:double; shifta:double; 
            scaleb:double; shiftb:double; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsCeil(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdCeil(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsCeil(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdCeil(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsFloor(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdFloor(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsFloor(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdFloor(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsFrac(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdFrac(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsFrac(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdFrac(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsModf(n:longint; a:Psingle; r1:Psingle; r2:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdModf(n:longint; a:Pdouble; r1:Pdouble; r2:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsModf(n:longint; a:Psingle; r1:Psingle; r2:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdModf(n:longint; a:Pdouble; r1:Pdouble; r2:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsFmod(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdFmod(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsFmod(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdFmod(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsRemainder(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdRemainder(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsRemainder(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdRemainder(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsNextAfter(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdNextAfter(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsNextAfter(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdNextAfter(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsCopySign(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdCopySign(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsCopySign(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdCopySign(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsFdim(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdFdim(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsFdim(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdFdim(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsFmax(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdFmax(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsFmax(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdFmax(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsFmin(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdFmin(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsFmin(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdFmin(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsMaxMag(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdMaxMag(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsMaxMag(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdMaxMag(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsMinMag(n:longint; a:Psingle; b:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdMinMag(n:longint; a:Pdouble; b:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmsMinMag(n:longint; a:Psingle; b:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmdMinMag(n:longint; a:Pdouble; b:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsNearbyInt(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdNearbyInt(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsNearbyInt(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdNearbyInt(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsRint(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdRint(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsRint(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdRint(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsRound(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdRound(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsRound(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdRound(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsTrunc(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdTrunc(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsTrunc(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdTrunc(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcConj(n:longint; a:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzConj(n:longint; a:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcConj(n:longint; a:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzConj(n:longint; a:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vcMulByConj(n:longint; a:PMKL_Complex8; b:PMKL_Complex8; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vzMulByConj(n:longint; a:PMKL_Complex16; b:PMKL_Complex16; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmcMulByConj(n:longint; a:PMKL_Complex8; b:PMKL_Complex8; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vmzMulByConj(n:longint; a:PMKL_Complex16; b:PMKL_Complex16; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vcCIS(n:longint; a:Psingle; r:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vzCIS(n:longint; a:Pdouble; r:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmcCIS(n:longint; a:Psingle; r:PMKL_Complex8; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmzCIS(n:longint; a:Pdouble; r:PMKL_Complex16; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vsExpInt1(n:longint; a:Psingle; r:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vdExpInt1(n:longint; a:Pdouble; r:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmsExpInt1(n:longint; a:Psingle; r:Psingle; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

  procedure vmdExpInt1(n:longint; a:Pdouble; r:Pdouble; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsAbsI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdAbsI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsAbsI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdAbsI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcAbsI(n:longint; a:PMKL_Complex8; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzAbsI(n:longint; a:PMKL_Complex16; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcAbsI(n:longint; a:PMKL_Complex8; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzAbsI(n:longint; a:PMKL_Complex16; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcArgI(n:longint; a:PMKL_Complex8; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzArgI(n:longint; a:PMKL_Complex16; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcArgI(n:longint; a:PMKL_Complex8; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzArgI(n:longint; a:PMKL_Complex16; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsAddI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdAddI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsAddI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdAddI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vcAddI(n:longint; a:PMKL_Complex8; inca:longint; b:PMKL_Complex8; incb:longint; 
            r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vzAddI(n:longint; a:PMKL_Complex16; inca:longint; b:PMKL_Complex16; incb:longint; 
            r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmcAddI(n:longint; a:PMKL_Complex8; inca:longint; b:PMKL_Complex8; incb:longint; 
            r:PMKL_Complex8; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmzAddI(n:longint; a:PMKL_Complex16; inca:longint; b:PMKL_Complex16; incb:longint; 
            r:PMKL_Complex16; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsSubI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdSubI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsSubI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdSubI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vcSubI(n:longint; a:PMKL_Complex8; inca:longint; b:PMKL_Complex8; incb:longint; 
            r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vzSubI(n:longint; a:PMKL_Complex16; inca:longint; b:PMKL_Complex16; incb:longint; 
            r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmcSubI(n:longint; a:PMKL_Complex8; inca:longint; b:PMKL_Complex8; incb:longint; 
            r:PMKL_Complex8; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmzSubI(n:longint; a:PMKL_Complex16; inca:longint; b:PMKL_Complex16; incb:longint; 
            r:PMKL_Complex16; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsInvI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdInvI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsInvI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdInvI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsSqrtI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdSqrtI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsSqrtI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdSqrtI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcSqrtI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzSqrtI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcSqrtI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzSqrtI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsInvSqrtI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdInvSqrtI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsInvSqrtI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdInvSqrtI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsCbrtI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdCbrtI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsCbrtI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdCbrtI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsInvCbrtI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdInvCbrtI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsInvCbrtI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdInvCbrtI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsSqrI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdSqrI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsSqrI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdSqrI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsExpI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdExpI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsExpI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdExpI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsExp2I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdExp2I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsExp2I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdExp2I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsExp10I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdExp10I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsExp10I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdExp10I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsExpm1I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdExpm1I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsExpm1I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdExpm1I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcExpI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzExpI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcExpI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzExpI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsLnI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdLnI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsLnI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdLnI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcLnI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzLnI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcLnI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzLnI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsLog10I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdLog10I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsLog10I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdLog10I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcLog10I(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzLog10I(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcLog10I(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzLog10I(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsLog2I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdLog2I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsLog2I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdLog2I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcLog2I(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzLog2I(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcLog2I(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzLog2I(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsLog1pI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdLog1pI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsLog1pI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdLog1pI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsLogbI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdLogbI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsLogbI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdLogbI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsCosI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdCosI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsCosI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdCosI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcCosI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzCosI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcCosI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzCosI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsSinI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdSinI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsSinI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdSinI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcSinI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzSinI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcSinI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzSinI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsTanI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdTanI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsTanI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdTanI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcTanI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzTanI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcTanI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzTanI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsCoshI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdCoshI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsCoshI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdCoshI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcCoshI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzCoshI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcCoshI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzCoshI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsCosdI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdCosdI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsCosdI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdCosdI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsCospiI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdCospiI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsCospiI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdCospiI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsSinhI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdSinhI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsSinhI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdSinhI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcSinhI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzSinhI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcSinhI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzSinhI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsSindI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdSindI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsSindI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdSindI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsSinpiI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdSinpiI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsSinpiI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdSinpiI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsTanhI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdTanhI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsTanhI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdTanhI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcTanhI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzTanhI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcTanhI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzTanhI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsTandI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdTandI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsTandI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdTandI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsTanpiI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdTanpiI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsTanpiI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdTanpiI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsAcosI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdAcosI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsAcosI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdAcosI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcAcosI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzAcosI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcAcosI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzAcosI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsAcospiI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdAcospiI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsAcospiI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdAcospiI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsAsinI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdAsinI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsAsinI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdAsinI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcAsinI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzAsinI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcAsinI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzAsinI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsAsinpiI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdAsinpiI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsAsinpiI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdAsinpiI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsAtanI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdAtanI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsAtanI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdAtanI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcAtanI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzAtanI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcAtanI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzAtanI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsAtanpiI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdAtanpiI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsAtanpiI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdAtanpiI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsAcoshI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdAcoshI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsAcoshI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdAcoshI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcAcoshI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzAcoshI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcAcoshI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzAcoshI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsAsinhI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdAsinhI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsAsinhI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdAsinhI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcAsinhI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzAsinhI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcAsinhI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzAsinhI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsAtanhI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdAtanhI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsAtanhI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdAtanhI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcAtanhI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzAtanhI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcAtanhI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzAtanhI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsErfI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdErfI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsErfI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdErfI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsErfInvI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdErfInvI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsErfInvI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdErfInvI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsHypotI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdHypotI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsHypotI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdHypotI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsErfcI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdErfcI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsErfcI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdErfcI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsErfcInvI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdErfcInvI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsErfcInvI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdErfcInvI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsCdfNormI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdCdfNormI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsCdfNormI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdCdfNormI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsCdfNormInvI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdCdfNormInvI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsCdfNormInvI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdCdfNormInvI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsLGammaI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdLGammaI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsLGammaI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdLGammaI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsTGammaI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdTGammaI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsTGammaI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdTGammaI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsAtan2I(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdAtan2I(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsAtan2I(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdAtan2I(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsAtan2piI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdAtan2piI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsAtan2piI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdAtan2piI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsMulI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdMulI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsMulI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdMulI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vcMulI(n:longint; a:PMKL_Complex8; inca:longint; b:PMKL_Complex8; incb:longint; 
            r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vzMulI(n:longint; a:PMKL_Complex16; inca:longint; b:PMKL_Complex16; incb:longint; 
            r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmcMulI(n:longint; a:PMKL_Complex8; inca:longint; b:PMKL_Complex8; incb:longint; 
            r:PMKL_Complex8; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmzMulI(n:longint; a:PMKL_Complex16; inca:longint; b:PMKL_Complex16; incb:longint; 
            r:PMKL_Complex16; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsDivI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdDivI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsDivI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdDivI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vcDivI(n:longint; a:PMKL_Complex8; inca:longint; b:PMKL_Complex8; incb:longint; 
            r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vzDivI(n:longint; a:PMKL_Complex16; inca:longint; b:PMKL_Complex16; incb:longint; 
            r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmcDivI(n:longint; a:PMKL_Complex8; inca:longint; b:PMKL_Complex8; incb:longint; 
            r:PMKL_Complex8; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmzDivI(n:longint; a:PMKL_Complex16; inca:longint; b:PMKL_Complex16; incb:longint; 
            r:PMKL_Complex16; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsFdimI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdFdimI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsFdimI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdFdimI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsFmodI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdFmodI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsFmodI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdFmodI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsFmaxI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdFmaxI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsFmaxI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdFmaxI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsFminI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdFminI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsFminI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdFminI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsPowI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdPowI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsPowI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdPowI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vcPowI(n:longint; a:PMKL_Complex8; inca:longint; b:PMKL_Complex8; incb:longint; 
            r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vzPowI(n:longint; a:PMKL_Complex16; inca:longint; b:PMKL_Complex16; incb:longint; 
            r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmcPowI(n:longint; a:PMKL_Complex8; inca:longint; b:PMKL_Complex8; incb:longint; 
            r:PMKL_Complex8; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmzPowI(n:longint; a:PMKL_Complex16; inca:longint; b:PMKL_Complex16; incb:longint; 
            r:PMKL_Complex16; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsPowrI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdPowrI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsPowrI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdPowrI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsPow3o2I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdPow3o2I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsPow3o2I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdPow3o2I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsPow2o3I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdPow2o3I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsPow2o3I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdPow2o3I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vsPowxI(n:longint; a:Psingle; inca:longint; _para4:single; r:Psingle; 
            incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vdPowxI(n:longint; a:Pdouble; inca:longint; _para4:double; r:Pdouble; 
            incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vmsPowxI(n:longint; a:Psingle; inca:longint; _para4:single; r:Psingle; 
            incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vmdPowxI(n:longint; a:Pdouble; inca:longint; _para4:double; r:Pdouble; 
            incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vcPowxI(n:longint; a:PMKL_Complex8; inca:longint; b:MKL_Complex8; r:PMKL_Complex8; 
            incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vzPowxI(n:longint; a:PMKL_Complex16; inca:longint; b:MKL_Complex16; r:PMKL_Complex16; 
            incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vmcPowxI(n:longint; a:PMKL_Complex8; inca:longint; b:MKL_Complex8; r:PMKL_Complex8; 
            incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vmzPowxI(n:longint; a:PMKL_Complex16; inca:longint; b:MKL_Complex16; r:PMKL_Complex16; 
            incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vsSinCosI(n:longint; a:Psingle; inca:longint; r1:Psingle; incr1:longint; 
            r2:Psingle; incr2:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vdSinCosI(n:longint; a:Pdouble; inca:longint; r1:Pdouble; incr1:longint; 
            r2:Pdouble; incr2:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vmsSinCosI(n:longint; a:Psingle; inca:longint; r1:Psingle; incr1:longint; 
            r2:Psingle; incr2:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vmdSinCosI(n:longint; a:Pdouble; inca:longint; r1:Pdouble; incr1:longint; 
            r2:Pdouble; incr2:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

          procedure vsLinearFracI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            scalea:single; shifta:single; scaleb:single; shiftb:single; r:Psingle; 
            incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

          procedure vdLinearFracI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            scalea:double; shifta:double; scaleb:double; shiftb:double; r:Pdouble; 
            incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

          procedure vmsLinearFracI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            scalea:single; shifta:single; scaleb:single; shiftb:single; r:Psingle; 
            incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

          procedure vmdLinearFracI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            scalea:double; shifta:double; scaleb:double; shiftb:double; r:Pdouble; 
            incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsCeilI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdCeilI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsCeilI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdCeilI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsFloorI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdFloorI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsFloorI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdFloorI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsFracI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdFracI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsFracI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdFracI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vsModfI(n:longint; a:Psingle; inca:longint; r1:Psingle; incr1:longint; 
            r2:Psingle; incr2:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vdModfI(n:longint; a:Pdouble; inca:longint; r1:Pdouble; incr1:longint; 
            r2:Pdouble; incr2:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vmsModfI(n:longint; a:Psingle; inca:longint; r1:Psingle; incr1:longint; 
            r2:Psingle; incr2:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

     procedure vmdModfI(n:longint; a:Pdouble; inca:longint; r1:Pdouble; incr1:longint; 
            r2:Pdouble; incr2:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsNearbyIntI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdNearbyIntI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsNearbyIntI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdNearbyIntI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsNextAfterI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdNextAfterI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsNextAfterI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdNextAfterI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsMinMagI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdMinMagI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsMinMagI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdMinMagI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsMaxMagI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdMaxMagI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsMaxMagI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdMaxMagI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsRintI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdRintI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsRintI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdRintI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsRoundI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdRoundI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsRoundI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdRoundI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsTruncI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdTruncI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsTruncI(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdTruncI(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcConjI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzConjI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcConjI(n:longint; a:PMKL_Complex8; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzConjI(n:longint; a:PMKL_Complex16; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vcMulByConjI(n:longint; a:PMKL_Complex8; inca:longint; b:PMKL_Complex8; incb:longint; 
            r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vzMulByConjI(n:longint; a:PMKL_Complex16; inca:longint; b:PMKL_Complex16; incb:longint; 
            r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmcMulByConjI(n:longint; a:PMKL_Complex8; inca:longint; b:PMKL_Complex8; incb:longint; 
            r:PMKL_Complex8; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmzMulByConjI(n:longint; a:PMKL_Complex16; inca:longint; b:PMKL_Complex16; incb:longint; 
            r:PMKL_Complex16; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vcCISI(n:longint; a:Psingle; inca:longint; r:PMKL_Complex8; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vzCISI(n:longint; a:Pdouble; inca:longint; r:PMKL_Complex16; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmcCISI(n:longint; a:Psingle; inca:longint; r:PMKL_Complex8; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmzCISI(n:longint; a:Pdouble; inca:longint; r:PMKL_Complex16; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vsExpInt1I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vdExpInt1I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmsExpInt1I(n:longint; a:Psingle; inca:longint; r:Psingle; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

    procedure vmdExpInt1I(n:longint; a:Pdouble; inca:longint; r:Pdouble; incr:longint; 
            mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsCopySignI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdCopySignI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsCopySignI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdCopySignI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vsRemainderI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vdRemainderI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmsRemainderI(n:longint; a:Psingle; inca:longint; b:Psingle; incb:longint; 
            r:Psingle; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

      procedure vmdRemainderI(n:longint; a:Pdouble; inca:longint; b:Pdouble; incb:longint; 
            r:Pdouble; incr:longint; mode:int64);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsPackI(n:longint; a:Psingle; incra:longint; y:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdPackI(n:longint; a:Pdouble; incra:longint; y:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vcPackI(n:longint; a:PMKL_Complex8; incra:longint; y:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vzPackI(n:longint; a:PMKL_Complex16; incra:longint; y:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsPackV(n:longint; a:Psingle; ia:Plongint; y:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdPackV(n:longint; a:Pdouble; ia:Plongint; y:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vcPackV(n:longint; a:PMKL_Complex8; ia:Plongint; y:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vzPackV(n:longint; a:PMKL_Complex16; ia:Plongint; y:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsPackM(n:longint; a:Psingle; ma:Plongint; y:Psingle);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdPackM(n:longint; a:Pdouble; ma:Plongint; y:Pdouble);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vcPackM(n:longint; a:PMKL_Complex8; ma:Plongint; y:PMKL_Complex8);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vzPackM(n:longint; a:PMKL_Complex16; ma:Plongint; y:PMKL_Complex16);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsUnpackI(n:longint; a:Psingle; y:Psingle; incry:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdUnpackI(n:longint; a:Pdouble; y:Pdouble; incry:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vcUnpackI(n:longint; a:PMKL_Complex8; y:PMKL_Complex8; incry:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vzUnpackI(n:longint; a:PMKL_Complex16; y:PMKL_Complex16; incry:longint);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsUnpackV(n:longint; a:Psingle; y:Psingle; iy:Plongint);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdUnpackV(n:longint; a:Pdouble; y:Pdouble; iy:Plongint);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vcUnpackV(n:longint; a:PMKL_Complex8; y:PMKL_Complex8; iy:Plongint);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vzUnpackV(n:longint; a:PMKL_Complex16; y:PMKL_Complex16; iy:Plongint);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vsUnpackM(n:longint; a:Psingle; y:Psingle; my:Plongint);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vdUnpackM(n:longint; a:Pdouble; y:Pdouble; my:Plongint);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vcUnpackM(n:longint; a:PMKL_Complex8; y:PMKL_Complex8; my:Plongint);winapi; external {$ifdef libmkl} libmkl{$endif};

   procedure vzUnpackM(n:longint; a:PMKL_Complex16; y:PMKL_Complex16; my:Plongint);winapi; external {$ifdef libmkl} libmkl{$endif};

 function vmlSetErrStatus(status:longint):longint;winapi; external {$ifdef libmkl} libmkl{$endif};

function vmlGetErrStatus:longint;winapi; external {$ifdef libmkl} libmkl{$endif};

function vmlClearErrStatus:longint;winapi; external {$ifdef libmkl} libmkl{$endif};

 function vmlSetErrorCallBack(func:VMLErrorCallBack):VMLErrorCallBack;winapi; external {$ifdef libmkl} libmkl{$endif};

function vmlGetErrorCallBack:VMLErrorCallBack;winapi; external {$ifdef libmkl} libmkl{$endif};

function vmlClearErrorCallBack:VMLErrorCallBack;winapi; external {$ifdef libmkl} libmkl{$endif};

 function vmlSetMode(newmode:dword):dword;winapi; external {$ifdef libmkl} libmkl{$endif};

function vmlGetMode:dword;winapi; external {$ifdef libmkl} libmkl{$endif};

implementation

end.
