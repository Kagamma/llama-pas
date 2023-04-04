program llama_cli;

uses
  Classes, SysUtils, Generics.Collections, llama;

const
  SESSION_VERSION_NUMBER = $B00B5001;
  N_THREADS = 4;
  N_TOK_PREDICT = 2048;
  TOP_K = 30;
  TOP_P = 0.18;
  TEMP = 2;
  REPEAT_LAST_N = 512;
  REPEAT_PENALTY = 1.15;
  BATCH_SIZE = 8;

type
  TTokenList = class(specialize TList<Tllama_token>)
  public
    function Data(const P: Integer = 0): Pllama_token;
  end;

function TTokenList.Data(const P: Integer = 0): Pllama_token;
begin
  Result := @FItems[P];
end;

var
  Ctx     : Pllama_context;
  Params  : Tllama_context_params;
  S,
  SessionFName,
  ModelFName: String;
  Pred    : String;
  Prompt  : String;
  Token   : Tllama_token;
  SSTokens: TTokenList;
  Embd,
  EmbdInp : TTokenList;
  BatchDiv: Integer;
  I, J,
  C       : Integer;
  P       : Pllama_token;
  TokenStr: String;
  KvCacheSize: QWord = 0;
  KvTokenCount: Cardinal;
  KvCache: array of Byte;
  IsInteractive: Boolean = False;

procedure SessionSave;
var
  FS: TFileStream;
  Q : QWord;
  P: PByte;
begin
  if SessionFName = '' then
    Exit;
  FS := TFileStream.Create(SessionFName, fmCreate);
  try
    // Write version number
    FS.WriteDWord(SESSION_VERSION_NUMBER);
    // Store parameters
    FS.Write(Params, SizeOf(Params));
    // Store kv cache size
    Q := llama_get_kv_cache_size(Ctx);
    FS.Write(Q, SizeOf(Q));
    // Store kv cache
    P := llama_get_kv_cache(Ctx);
    FS.Write(P^, Q);
    // Store kv token count
    FS.WriteDWord(llama_get_kv_cache_token_count(Ctx));
    // Store past token count
    FS.WriteDWord(Embd.Count);
    // Store past token
    FS.Write(Embd.Data^, SizeOf(Tllama_token) * Embd.Count);
  finally
    FS.Free;
  end;
end;

procedure SessionLoad;
var
  FS: TFileStream;
begin
  if (SessionFName = '') or (not FileExists(SessionFName)) then
    Exit;
  FS := TFileStream.Create(SessionFName, fmOpenRead);
  try
    if FS.ReadDWord <> SESSION_VERSION_NUMBER then
      raise Exception.Create('Invalid session format.');
    // Read parameters
    FS.Read(Params, SizeOf(Params));
    // Read kv cache size
    FS.Read(KvCacheSize, SizeOf(KvCacheSize));
    // Read kv cache
    SetLength(KvCache, KvCacheSize);
    FS.Read(KvCache[0], KvCacheSize);
    // Read kv token count
    KvTokenCount := FS.ReadDWord;
    // Read past token count
    Embd.Count := FS.ReadDWord;
    // Read past token
    FS.Read(Embd.Data^, SizeOf(Tllama_token) * Embd.Count);
  finally
    FS.Free;
  end;
end;

procedure ParseParameters;
var
  I: Integer = 1;
  procedure Increase;
  begin
    Inc(I);
    if I > ParamCount then
      raise Exception.Create('Invalid parameter');
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
      '-i':
        begin
          IsInteractive := True;
        end;
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
  if (Prompt = '') and (not IsInteractive) then
    raise Exception.Create('No prompt provided.');
end;

begin
  ParseParameters;
  Params := llama_context_default_params;

  SSTokens := TTokenList.Create;
  EmbdInp := TTokenList.Create;
  Embd := TTokenList.Create;
  SessionLoad;

  Ctx := llama_init_from_file(PChar(ModelFName), Params);
  if Ctx = nil then
    raise Exception.Create('Failed to load model');
  if IsInteractive then
    Writeln(#10'Interactive mode is ON'#10);

  repeat
    if not IsInteractive then
    begin
      Writeln;
      Writeln(Prompt);
    end else
    begin
      Prompt := '';
      Write('>');
      Readln(Prompt);
    end;

    S := Prompt;
    if (KvCacheSize > 0) and (not IsInteractive) then
    begin
      llama_set_kv_cache(Ctx, @KvCache[0], KvCacheSize, KvTokenCount);
      Prompt :=  #10#10#10#10'### Instruction: ' + Prompt + #10#13 + '### Response: ';
    end else
    begin
      Prompt := #10#10#10#10'### Instruction: ' + Prompt + #10#13 + '### Response: ';
    end;

    // Convert prompt to embeddings
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

    for I := 0 to REPEAT_LAST_N - 1 do
      Embd.Add(0);
    for I := 0 to EmbdInp.Count - 1 do
      Embd.Add(EmbdInp[I]);

    for I := 0 to N_TOK_PREDICT - 1 do
    begin
      Token := llama_sample_top_p_top_k(Ctx, Embd.Data + Embd.Count - REPEAT_LAST_N - I, REPEAT_LAST_N, TOP_K, TOP_P, TEMP, REPEAT_PENALTY);
      // Break is eos
      if Token = llama_token_eos then
      begin
        //Write(' <EOS>');
        break;
      end;

      // Add it to the context (all tokens, prompt + predict)
      Embd.Add(Token);
      TokenStr := llama_token_to_str(Ctx, Token);
      Write(TokenStr);
      Pred := Pred + TokenStr;
      // Eval next token
      llama_eval(Ctx, @Token, 1, EmbdInp.Count + I, N_THREADS);
    end;
    Writeln;
  until not IsInteractive;
  SessionSave;
  llama_free(Ctx);
  SSTokens.Free;
  EmbdInp.Free;
  Embd.Free;
end.
