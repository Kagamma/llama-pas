program llama_cli;

uses
  Classes, SysUtils, Generics.Collections, llama;

const
  N_THREADS = 4;
  N_TOK_PREDICT = 2048;
  TOP_K = 30;
  TOP_P = 0.18;
  TEMP = 2;
  REPEAT_LAST_N = 512;
  REPEAT_PENALTY = 1.15;
  BATCH_SIZE = 8;
  STOP_SEQUENCE: PChar = #10#10;

type
  TTokenList = class(specialize TList<Tllama_token>)
  public
    function Data(const P: Integer = 0): Pointer;
  end;

function TTokenList.Data(const P: Integer = 0): Pointer;
begin
  Result := @FItems[P];
end;

var
  Ctx     : Pllama_context;
  Params  : Tllama_context_params;
  History : TStringList;
  S,
  FullText,
  SessionFName,
  ModelFName: String;
  Pred    : String;
  Prompt  : String;
  Tokens  : array[0..3] of Tllama_token = (0, 1, 2, 3);
  Token,
  MissingToken: Tllama_token;
  SSTokens: TTokenList;
  Embd,
  EmbdInp : TTokenList;
  EmbdCur,
  SSTokenCount,
  BatchDiv,
  SSIndex : Integer;
  I, J,
  C       : Integer;
  P       : Pllama_token;
  TokenStr: String;

procedure SessionSave;
var
  FS: TStringList;
begin
  if SessionFName = '' then
    Exit;
  History.SaveToFile(SessionFName);
end;

procedure SessionLoad;
begin
  if (SessionFName = '') or (not FileExists(SessionFName)) then
    Exit;
  History.LoadFromFile(SessionFName);
end;

procedure ParseParameters;
var
  I: Integer = 1;
  procedure Increase;
  begin
    Inc(I);
    if I > ParamCount then
    begin
      Writeln('Invalid parameter');
      Halt;
    end;
  end;

begin
  if ParamCount = 1 then
  begin
    Writeln('Usage: llama-cli -m <model_name> -p <prompt>');
    Halt;
  end;
  while I <= ParamCount do
  begin
    case ParamStr(I) of
      '-m':
        begin
          Increase;
          ModelFName := ParamStr(I);
        end;
      '-p':
        begin
          Increase;
          Prompt := ParamStr(I);
        end;
      '-s':
        begin
          Increase;
          SessionFName := ParamStr(I);
        end;
      '-h':
        begin
          Writeln(' -h: This help screen');
          Writeln(' -m: Path to model file');
          Writeln(' -p: Prompt');
          Writeln(' -s: Session file (not really work, and time consuming)');
          Halt;
        end;
    end;
    Inc(I);
  end;
  if Prompt = '' then
  begin
    Writeln('No prompt provided.');
    Halt;
  end;
end;

begin
  ParseParameters;
  Params := llama_context_default_params;

  Ctx := llama_init_from_file(PChar(ModelFName), Params);
  if Ctx = nil then
  begin
    Writeln('Failed to load model');
    Halt;
  end;

  SSTokens := TTokenList.Create;
  EmbdInp := TTokenList.Create;
  Embd := TTokenList.Create;
  History := TStringList.Create;
  SessionLoad;

  Writeln;
  Writeln(Prompt);

  S := Prompt;
  if History.Count = 0 then
    Prompt := '### Instruction: '#10#13 + Prompt + #10#10'### Response: '#10#10
  else
    Prompt := History.Text + #10#10'### Instruction: '#10#13 + Prompt + #10#10'### Response: '#10#10;
  SessionLoad;
  History.Add(S);

  llama_eval(ctx, @Tokens[0], Length(Tokens), 0, N_THREADS);

  // Convert stop sequence to token
  SSTokens.Count := Length(STOP_SEQUENCE) + 1;
  SSTokenCount := llama_tokenize(Ctx, STOP_SEQUENCE, SSTokens.Data, SSTokens.Count, False);
  SSTokens.Count := SSTokenCount;

  // Convert prompt to embedings
  EmbdInp.Count := Length(Prompt) + 1;
  C := llama_tokenize(Ctx, PChar(Prompt), EmbdInp.Data, EmbdInp.Count, False);
  EmbdInp.Count := C;

  // Evaluate prompt
  BatchDiv := EmbdInp.Count div BATCH_SIZE;
  P := EmbdInp.Data;

  // Handle full divisions of batch first
  for I := 0 to BatchDiv - 1 do
  begin
    llama_eval(Ctx, P, BATCH_SIZE, I * BATCH_SIZE, N_THREADS);
    P := P + BATCH_SIZE;
  end;

  // Handle remaining batch
  if EmbdInp.Count mod BATCH_SIZE <> 0 then
    llama_eval(Ctx, P, EmbdInp.Count mod BATCH_SIZE, BatchDiv * BATCH_SIZE, N_THREADS);

  for I := 0 to EmbdInp.Count - 1 do
    Embd.Add(EmbdInp[I]);

  SSIndex := -1;
  for I := 0 to N_TOK_PREDICT - 1 do
  begin
    Token := llama_sample_top_p_top_k(Ctx, nil, 0, TOP_K, TOP_P, TEMP, REPEAT_PENALTY);
    // Break is eos
    if Token = llama_token_eos then
    begin
      //Write(' <EOS>');
      break;
    end;

    // Add it to the context (all tokens, prompt + predict)
    Embd.Add(Token);
    if (Length(STOP_SEQUENCE) <> 0) and (Token = SSTokens[SSIndex + 1]) then
    begin
      Inc(SSIndex);
      if SSIndex = SSTokenCount - 1 then
      begin
        //Write(' <STOP SEQUENCE>');
        break;
      end;
    end else
    begin
      if SSIndex <> -1 then
      begin
        // Replay missed string
        EmbdCur := Embd.Count - 2 - SSIndex;
        for J := 0 to SSIndex + 2 - 1 do
        begin
          MissingToken := Embd[EmbdCur];
          Inc(EmbdCur);
          // Add to string
          TokenStr := llama_token_to_str(Ctx, MissingToken);
          Write(TokenStr);
          Pred := Pred + TokenStr;
        end;
        SSIndex := -1;
      end else
      begin
        // Add to string
        TokenStr := llama_token_to_str(Ctx, Token);
        Write(TokenStr);
        Pred := Pred + TokenStr;
      end;
    end;
    // Eval next token
    llama_eval(Ctx, Embd.Data(Embd.Count - 1), 1, Embd.Count - 1, N_THREADS);
  end;
  Writeln;

  History.Add(Pred);
  SessionSave;
  llama_free(Ctx);
  SSTokens.Free;
  EmbdInp.Free;
  Embd.Free;
  History.Free;
end.
