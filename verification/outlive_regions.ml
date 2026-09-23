(* Standalone smoke executable linked against the real Aeneas library.
   Pass the LLBC generated from verification/outlive_regions.rs as argv[1].
   Main owns execution; this source intentionally does not change src/dune. *)
open Aeneas
open Types
open LlbcAst

let rid = RegionId.of_int
let region i = RVar (Free (rid i))
let ids xs = RegionId.Set.of_list (List.map rid xs)
let scalar = TLiteral (TUInt U32)
let borrow r ty = TRef (r, ty, RMut)
let args regions types = Builtin.Sig.mk_generic_args regions types []
let tuple types = TAdt { id = TTuple; generics = args [] types }
let nested outer inner = borrow outer (borrow inner scalar)
let c = region 10
let d = region 20
let u = region 30

let check decls label selected ty expected =
  let actual =
    TypesAnalysis.compute_outlive_proj_ty None decls (ids selected) ty
  in
  if not (RegionId.Set.equal actual (ids expected)) then
    failwith
      (label ^ ": expected " ^ show_region_id_set (ids expected) ^ ", got "
     ^ show_region_id_set actual);
  Printf.printf "PASS %s\n%!" label

let reject label selected ty =
  match
    TypesAnalysis.compute_outlive_proj_ty None TypeDeclId.Map.empty
      (ids selected) ty
  with
  | _ -> failwith (label ^ ": expected rejection")
  | exception Errors.CFailure _ -> Printf.printf "PASS %s\n%!" label

let () =
  Config.sanity_checks := true;
  Config.fail_hard := false;
  let check_plain = check TypeDeclId.Map.empty in
  check_plain "selected-only" [ 10 ] (borrow c scalar) [];
  check_plain "nested-outliving" [ 10 ] (nested c d) [ 20 ];
  check_plain "reverse-is-not-outliving" [ 20 ] (nested c d) [];
  check_plain "unselected-without-selection" [] (nested c d) [];
  check_plain "unrelated-sibling" [ 10 ]
    (tuple [ nested c d; borrow u scalar ]) [ 20 ];
  check_plain "repeated-unselected" [ 10 ]
    (tuple [ borrow d scalar; nested c d; borrow u scalar ]) [ 20 ];
  check_plain "repeated-selected" [ 10 ]
    (tuple [ borrow c scalar; nested c d ]) [ 20 ];
  check_plain "both-selected" [ 10; 20 ] (nested c d) [];
  check_plain "unselected-in-selected-scc" [ 10 ]
    (tuple [ nested c d; nested d c ]) [ 20 ];
  check_plain "transitive-outlives-across-occurrences" [ 10 ]
    (tuple [ nested c d; nested d u ]) [ 20; 30 ];
  check_plain "static-only" [] (borrow RStatic scalar) [];
  check_plain "static-inner-is-excluded" [ 10 ] (nested c RStatic) [];
  check_plain "unrelated-static-does-not-erase-outlives" [ 10 ]
    (tuple [ nested c d; borrow RStatic scalar ]) [ 20 ];
  check_plain "static-equivalent-is-excluded" [ 10 ]
    (tuple [ nested RStatic d; nested c d ]) [];
  reject "selected-constrained-to-static" [ 20 ] (nested RStatic d);
  reject "selected-in-static-scc" [ 10 ]
    (tuple [ nested c RStatic; nested RStatic c ]);
  Config.sanity_checks := false;
  reject "selected-static-rejected-with-checks-disabled" [ 20 ]
    (nested RStatic d);
  Config.sanity_checks := true;
  reject "erased-remains-unsupported" [ 10 ] (nested c RErased);
  reject "body-region-remains-unsupported" [ 10 ] (nested c (RBody (rid 20)));
  if Array.length Sys.argv <> 2 then
    failwith "usage: outlive_regions <outlive_regions.llbc>";
  let crate =
    match LlbcOfJson.crate_of_json (Yojson.Basic.from_file Sys.argv.(1)) with
    | Ok crate -> crate
    | Error message -> failwith message
  in
  let decls = crate.type_decls in
  let find_decl region_count type_count =
    match
      List.filter
        (fun (_, (decl : type_decl)) ->
          List.length decl.generics.regions = region_count
          && List.length decl.generics.types = type_count)
        (TypeDeclId.Map.bindings decls)
    with
    | [ (id, _) ] -> id
    | _ -> failwith "expected exactly one matching smoke declaration"
  in
  let region_id = find_decl 3 0 in
  let region_adt regions =
    TAdt { id = TAdtId region_id; generics = args regions [] }
  in
  check decls "substituted-region-constraint" [ 10 ]
    (region_adt [ c; d; u ]) [ 20 ];
  check decls "substituted-region-constraint-direction" [ 20 ]
    (region_adt [ c; d; u ]) [];
  check decls "substituted-region-constraint-permutation" [ 20 ]
    (region_adt [ d; c; u ]) [ 10 ];
  check decls "substituted-repeated-region" [ 10 ]
    (region_adt [ c; d; d ]) [ 20 ];
  let type_id = find_decl 1 1 in
  let type_adt ty =
    TAdt { id = TAdtId type_id; generics = args [ c ] [ ty ] }
  in
  check decls "substituted-type-constraint" [ 10 ]
    (tuple [ type_adt (borrow d scalar); borrow u scalar ]) [ 20 ];
  check decls "substituted-type-constraint-all-regions" [ 10 ]
    (type_adt (tuple [ borrow d scalar; borrow u scalar ])) [ 20; 30 ];
  Printf.printf "All outlive-region checks passed.\n%!"
