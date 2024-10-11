unit nModels;
{$ifdef fpc}
{$mode Delphi}
{$endif}

interface

uses
  SysUtils
  , ntensors
  , ntypes
  , nBaseLayer
  , nnet
  , nConnectedlayer
  , nLogisticLayer
  , nSoftmaxLayer
  , nCostLayer
  , nConvolutionLayer
  , nMaxPoolLayer
  , nDropOutLayer
  ;

type
  TNetCfg =record
    a: integer;
    b:shortstring
  end;

function simpleDenseMNIST:TArray<TBaseLayer>;
function leNetMNIST:TArray<TBaseLayer>;
function leNetCIFAR10:TArray<TBaseLayer>;

implementation

function simpleDenseMNIST:TArray<TBaseLayer>;
begin
  result := [
        TConnectedLayer.Create(1, 28*28, 64, acRELU{, true})
      , TConnectedLayer.Create(1, 64   , 64, acRELU{, true})
      , TConnectedLayer.Create(1, 64   , 32, acRELU{, true})
      , TConnectedLayer.Create(1, 32   , 32, acRELU{, true})
      , TConnectedLayer.Create(1, 32   , 10, acLINEAR)
      //, TSoftmaxLayer.Create(1,10)
      , TLogisticLayer.Create(1,10)
      //, TCostLayer.Create(1,10,ctSSE,1)
    ]
end;

function leNetMNIST:TArray<TBaseLayer>;
begin
  result := [
        TConvolutionalLayer.Create(1, 28, 28, 1, 6, 1, 5, 1, 1, 1, 2, acReLU)
      , TMaxPoolLayer.Create(1, 28, 28,  6, 2)
      , TConvolutionalLayer.Create(1, 14, 14, 6, 16, 1, 5, 1, 1, 1, 0, acReLU)
      , TMaxPoolLayer.Create(1, 10, 10, 16, 2)
      , TConvolutionalLayer.Create(1, 5, 5, 16, 120, 1, 5, 1, 1, 1, 0, acReLU)
      , TConnectedLayer.Create(1, 120   , 84, acReLU)
      , TConnectedLayer.Create(1, 84   , 10, acLINEAR)
      , TSoftmaxLayer.Create(1,10)
      //, TLogisticLayer.Create(1,10)
      //, TCostLayer.Create(1,10,ctSSE,1)
    ]
end;

function leNetCIFAR10:TArray<TBaseLayer>;
begin
  result := [
        TConvolutionalLayer.Create(1, 32, 32, 3, 6, 1, 5, 1, 1, 1, 0, acReLU)
      , TMaxPoolLayer.Create(1, 28, 28,  6, 2)
      , TConvolutionalLayer.Create(1, 14, 14, 6, 16, 1, 5, 1, 1, 1, 0, acReLU)
      , TMaxPoolLayer.Create(1, 10, 10, 16, 2)
      , TConvolutionalLayer.Create(1, 5, 5, 16, 120, 1, 5, 1, 1, 1, 0, acReLU)
      , TConnectedLayer.Create(1, 120   , 84, acReLU)
      , TConnectedLayer.Create(1, 84   , 10, acLINEAR)
      , TSoftmaxLayer.Create(1,10)
      //, TLogisticLayer.Create(1,10)
      //, TCostLayer.Create(1,10,ctSSE,1)
    ]
end;

initialization

end.

