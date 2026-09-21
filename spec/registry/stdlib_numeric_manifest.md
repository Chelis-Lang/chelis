# Registry: exported stdlib numeric definitions ([05-OP-35])

This file is normative-tier content under the AGENTS.md Documentation
Authority rules: it is incorporated by reference into
`spec/05-risc-primitives.md` [05-OP-35] and is amended only as a
numbered-spec change under the same review discipline. Rows are keyed by
identity; row order is not semantic and no ordinal is part of any identity.

| identity | exact final signature |
|---|---|
| `contracts::normal_cdf` | `(p_float)->p_float` |
| `contracts::normal_cdf_contract_samples` | `()->i64` |
| `contracts::normal_cdf_contract_seed` | `()->i64` |
| `contracts::standard_contract_tolerance` | `()->f32` |
| `decimal::decimal` | `(string)->Decimal` |
| `decimal::decimal_add` | `(Decimal,Decimal)->Decimal` |
| `decimal::decimal_div` | `(Decimal,Decimal,i64,RoundingMode)->Decimal` |
| `decimal::decimal_eq` | `(Decimal,Decimal)->bool` |
| `decimal::decimal_from_int` | `(i64)->Decimal` |
| `decimal::decimal_gt` | `(Decimal,Decimal)->bool` |
| `decimal::decimal_gte` | `(Decimal,Decimal)->bool` |
| `decimal::decimal_lt` | `(Decimal,Decimal)->bool` |
| `decimal::decimal_lte` | `(Decimal,Decimal)->bool` |
| `decimal::decimal_mul` | `(Decimal,Decimal)->Decimal` |
| `decimal::decimal_sub` | `(Decimal,Decimal)->Decimal` |
| `decimal::decimal_to_float` | `(Decimal)->f64` |
| `decimal::decimal_to_string` | `(Decimal)->string` |
| `decimal::try_decimal` | `(string)->Option[Decimal]` |
| `index::list_index` | `(List[T],i64)->T` |
| `index::skip_list` | `(List[T],i64)->List[T]` |
| `index::take_list` | `(List[T],i64)->List[T]` |
| `init/kaiming::kaiming_normal` | `(&tensor[..r,p_float],p_float)->tensor[..r,p_float]!{Random}` |
| `init/kaiming::kaiming_uniform` | `(&tensor[..r,p_float],p_float)->tensor[..r,p_float]!{Random}` |
| `init/random::normal_like` | `(&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]!{Random}` |
| `init/xavierext::trunc_normal` | `(&tensor[..r,p_float],p_float,p_float,p_float,p_float)->tensor[..r,p_float]!{Random}` |
| `init/xavierext::xavier_normal` | `(&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]!{Random}` |
| `init/xavierext::xavier_uniform` | `(&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]!{Random}` |
| `io/json::json_array` | `(Option[Json])->Option[List[Json]]` |
| `io/json::json_bigint` | `(Option[Json])->Option[string]` |
| `io/json::json_bool` | `(Option[Json])->Option[bool]` |
| `io/json::json_float` | `(Option[Json])->Option[f64]` |
| `io/json::json_get` | `(Json,string)->Option[Json]` |
| `io/json::json_int` | `(Option[Json])->Option[i64]` |
| `io/json::json_is_null` | `(Option[Json])->bool` |
| `io/json::json_object` | `(Option[Json])->Option[Dict[string,Json]]` |
| `io/json::json_string` | `(Option[Json])->Option[string]` |
| `io/json::load_json` | `(string)->Json!{IO}` |
| `io/json::parse_json` | `(string)->Json` |
| `io/json::to_json` | `(Json)->string` |
| `io/json::try_load_json` | `(string)->Option[Json]!{IO}` |
| `io/json::try_parse_json` | `(string)->Option[Json]` |
| `io/json::try_to_json` | `(Json)->Option[string]` |
| `io/json::try_write_json` | `(string,Json)->Option[unit]!{IO}` |
| `io/json::write_json` | `(string,Json)->unit!{IO}` |
| `io::mmap_size` | `(string)->i64!{IO}` |
| `io::read_head_bytes` | `(string,i64)->List[i64]!{IO}` |
| `process::run` | `(string,List[string])->(i64,string,string)!{IO}` |
| `process::run_chelis` | `(List[string])->(i64,string,string)!{IO}` |
| `scalar::abs` | `(p_numeric)->p_numeric` |
| `scalar::max` | `(p_numeric,p_numeric)->p_numeric` |
| `scalar::min` | `(p_numeric,p_numeric)->p_numeric` |
| `sort::sort` | `(&tensor[..r,p_numeric],i32)->(tensor[..r,p_numeric],tensor[..r,i64])` |
| `tensor/construct::arange` | `(p_int,p_int)->tensor[n,p_int]` |
| `tensor/construct::linspace` | `(p_float,p_float,i64)->tensor[n,p_float]` |
| `tensor/construct::squeeze` | `(&tensor[..pre,1,..post,p],i32)->tensor[..pre,..post,p]` |
| `tensor/construct::stack` | `(List[tensor[..pre,..post,p]],i32)->tensor[..pre,rows,..post,p]` |
| `tensor/construct::unsqueeze` | `(&tensor[..pre,..post,p],i32)->tensor[..pre,1,..post,p]` |
| `tensor/mask::where_indices` | `(&tensor[..r,bool])->tensor[hits,i64]` |
| `test::assert_close` | `(p_float,p_float,p_float,string)->unit!{Test}` |
| `test::assert_close_tensor` | `(&tensor[..r,p_float],&tensor[..r,p_float],p_float,string)->unit!{Test}` |
| `test::assert_eq` | `(Q,Q,string)->unit!{Test}` |
| `test::assert_eq_tensor` | `(&tensor[..r,p],&tensor[..r,p],string)->unit!{Test}` |
| `test::assert_shape` | `(&tensor[..r,p],List[i64],string)->unit!{Test}` |
| `time::add_days` | `(Date,i64)->Date` |
| `time::date` | `(i64,i64,i64)->Date` |
| `time::date_gt` | `(Date,Date)->bool` |
| `time::date_gte` | `(Date,Date)->bool` |
| `time::date_lt` | `(Date,Date)->bool` |
| `time::date_lte` | `(Date,Date)->bool` |
| `time::date_to_string` | `(Date)->string` |
| `time::day_of_week` | `(Date)->DayOfWeek` |
| `time::day_of_week_name` | `(Date)->string` |
| `time::day_of_year` | `(Date)->i64` |
| `time::days_between` | `(Date,Date)->i64` |
| `time::duration` | `(i64,i64,i64,i64)->Duration` |
| `time::is_leap_year` | `(i64)->bool` |
| `time::parse_date` | `(string)->Option[Date]` |
| `time::sub_days` | `(Date,i64)->Date` |
| `time::try_date` | `(i64,i64,i64)->Option[Date]` |
| `tokenizer::batch_encode` | `(Tokenizer,List[string],i64,i64)->tensor[batch,seq,i64]` |
| `tokenizer::decode` | `(Tokenizer,List[i64])->string` |
| `tokenizer::encode` | `(Tokenizer,string)->List[i64]` |
| `tokenizer::load_tokenizer` | `(string)->Tokenizer!{IO}` |
| `tokenizer::try_load_tokenizer` | `(string)->Option[Tokenizer]!{IO}` |
