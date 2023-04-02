unit llama;

{.$define LLAMA_STATIC}
{$ifdef LLAMA_STATIC}
  {$linklib libllama.a}
{$endif}
{$mode objfpc}{$H+}
{$macro on}
{$ifdef windows}
  {$define LLAMACALL:=stdcall}
  {$define LLAMALIB:='libllama.dll'}
{$else}
  {$define LLAMACALL:=cdecl}
  {$define LLAMALIB:='libllama.so'}
{$endif}
{$packrecords C}
{$packenum 4}

interface

uses
  ctypes, Math, dynlibs;

type
  Tllama_progress_callback = procedure(progress: Single; ctx: Pointer); LLAMACALL;

  Pllama_context = ^Tllama_context;
  Tllama_context = record
    // Internal C++ struct
  end;

  Pllama_context_params = ^Tllama_context_params;
  Tllama_context_params = record
    n_ctx,
    n_parts,
    seed             : cint;
    f16_kv,
    logits_all,
    vocab_only,
    use_mlock,
    embedding        : cbool;
    progress_callback: Tllama_progress_callback;
    progress_callback_user_data: Pointer;
  end;

  Pllama_token = ^Tllama_token;
  Tllama_token = cint;

function llama_init_from_file(path_model: PChar; params: Tllama_context_params): Pllama_context; LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
procedure llama_free(ctx: Pllama_context); LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
function llama_eval(ctx: Pllama_context; tokens: Pllama_token; n_tokens, n_past, n_threads: cint): cint; LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
function llama_tokenize(ctx: Pllama_context; text: PChar; tokens: Pllama_token; n_max_tokens: cint; add_bos: cbool): cint; LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
function llama_n_vocab(ctx: Pllama_context): cint; LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
function llama_n_ctx(ctx: Pllama_context): cint; LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
function llama_n_embd(ctx: Pllama_context): cint; LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
function llama_get_logits(ctx: Pllama_context): pcfloat; LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
function llama_get_embeddings(ctx: Pllama_context): pcfloat; LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
function llama_token_to_str(ctx: Pllama_context; token: Tllama_token): PChar; LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
function llama_token_bos: Tllama_token; LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
function llama_token_eos: Tllama_token; LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
function llama_sample_top_p_top_k(ctx: Pllama_context; last_n_tokens_data: Pllama_token; last_n_tokens_size, top_k: cint; top_p, temp, repeat_penalty: cfloat): Tllama_token; LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
procedure llama_print_timings(ctx: Pllama_context); LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
procedure llama_reset_timings(ctx: Pllama_context); LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
function llama_print_system_info: PChar; LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};
function llama_context_default_params: Tllama_context_params; LLAMACALL; external {$ifndef LLAMA_STATIC}LLAMALIB{$endif};

implementation

initialization
  SetExceptionMask(GetExceptionMask + [exOverflow, exInvalidOp]);

end.