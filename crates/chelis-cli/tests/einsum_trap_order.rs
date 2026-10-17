//! An integer `einsum` traps exactly when one of its multiplies or tree
//! additions overflows, with the same [04-NUM-9] line whatever order they
//! evaluate in ([05-OP-33]), identically in `chelis eval` and in compiled C.
//!
//! The cases are generated over a grid: for each accumulator, an output row
//! whose products overflow at a multiply, at a tree addition, at both in
//! either order, or not at all, and rows that overflow in different ways.
//! The compiled lane streams each product into the tree as it is formed,
//! while `chelis eval` forms every product of an output first, so the two
//! lanes evaluate the operations of a doubly overflowing row in different
//! orders. Each case's expected outcome comes from a model that checks every
//! product and tree node, independent of any evaluation order.

#[path = "common/host_effect_parity.rs"]
mod parity;

/// An integer operand dtype and the accumulator its sum defaults to.
struct Width {
    operand: &'static str,
    operand_max: i128,
    accumulator: &'static str,
    min: i128,
    max: i128,
}

const I16: Width = Width {
    operand: "i16",
    operand_max: i16::MAX as i128,
    accumulator: "i32",
    min: i32::MIN as i128,
    max: i32::MAX as i128,
};
const I32: Width = Width {
    operand: "i32",
    operand_max: i32::MAX as i128,
    accumulator: "i32",
    min: i32::MIN as i128,
    max: i32::MAX as i128,
};
const I64: Width = Width {
    operand: "i64",
    operand_max: i64::MAX as i128,
    accumulator: "i64",
    min: i64::MIN as i128,
    max: i64::MAX as i128,
};

/// How one row's products overflow.
#[derive(Clone, Copy, Debug)]
enum Row {
    Clean,
    /// The last product overflows; no addition does.
    Multiply,
    /// The first two products are in range and their sum is not.
    Add,
    /// The first pair's sum overflows, and so does the last product.
    AddThenMultiply,
    /// The first product overflows, and so does the sum of the last pair.
    MultiplyThenAdd,
    /// Every operand is the operand dtype's maximum.
    Saturated,
}

/// The row's operand values `(a, b)` at `width` over `leaves` products.
fn row(width: &Width, leaves: usize, shape: Row) -> (Vec<i128>, Vec<i128>) {
    // A product of `big` and 4 overflows; a product of `big` and 2 is in
    // range, and two of them sum past the accumulator's maximum.
    let big = (width.max + 1) / 4;
    let mut a: Vec<i128> = (0..leaves as i128).map(|i| 3 + i).collect();
    let mut b: Vec<i128> = (0..leaves as i128).map(|i| 5 + 2 * i).collect();
    let overflow_product = |a: &mut Vec<i128>, b: &mut Vec<i128>, at: usize| {
        a[at] = big;
        b[at] = 4;
    };
    let in_range_pair = |a: &mut Vec<i128>, b: &mut Vec<i128>, at: usize| {
        a[at] = big;
        b[at] = 2;
        a[at + 1] = big;
        b[at + 1] = 2;
    };
    match shape {
        Row::Clean => {}
        Row::Multiply => overflow_product(&mut a, &mut b, leaves - 1),
        Row::Add => in_range_pair(&mut a, &mut b, 0),
        Row::AddThenMultiply => {
            in_range_pair(&mut a, &mut b, 0);
            overflow_product(&mut a, &mut b, leaves - 1);
        }
        Row::MultiplyThenAdd => {
            overflow_product(&mut a, &mut b, 0);
            // The last pair the tree's first level adds.
            in_range_pair(&mut a, &mut b, leaves - 2 - leaves % 2);
        }
        Row::Saturated => {
            a = vec![width.operand_max; leaves];
            b = vec![width.operand_max; leaves];
        }
    }
    (a, b)
}

/// The adjacent-pair tree of one row at `width`; `None` when any product or
/// addition leaves the accumulator's range.
fn modeled_sum(width: &Width, a: &[i128], b: &[i128]) -> Option<i128> {
    let in_range = |value: i128| (width.min..=width.max).contains(&value).then_some(value);
    let mut level = a
        .iter()
        .zip(b)
        .map(|(x, y)| in_range(x * y))
        .collect::<Option<Vec<_>>>()?;
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| match pair {
                [x, y] => in_range(x + y),
                [x] => Some(*x),
                _ => unreachable!(),
            })
            .collect::<Option<Vec<_>>>()?;
    }
    Some(level.first().copied().unwrap_or(0))
}

fn literal_list(values: &[i128], dtype: &str) -> String {
    let items = values
        .iter()
        .map(|value| format!("{value}{dtype}"))
        .collect::<Vec<_>>();
    format!("[{}]", items.join(", "))
}

/// One generated program: `einsum("ij,ij->i", a, b)` sums each row of `a`
/// times the same row of `b`, so each row overflows only as it is built.
fn case(width: &Width, leaves: usize, rows: &[Row]) -> (String, Option<Vec<i128>>) {
    let operands = rows
        .iter()
        .map(|&shape| row(width, leaves, shape))
        .collect::<Vec<_>>();
    let expected = operands
        .iter()
        .map(|(a, b)| modeled_sum(width, a, b))
        .collect::<Option<Vec<_>>>();
    let (operand, accumulator) = (width.operand, width.accumulator);
    // Operand `side` (0 for `a`, 1 for `b`) as a tensor literal.
    let matrix = |side: usize| {
        let rows = operands
            .iter()
            .map(|(a, b)| literal_list([a, b][side], operand))
            .collect::<Vec<_>>();
        format!("to_tensor([{}])", rows.join(", "))
    };
    let r = rows.len();
    let source = format!(
        "def contract(a: tensor[{r}, {leaves}, {operand}], b: tensor[{r}, {leaves}, {operand}]) \
         -> tensor[{r}, {accumulator}] = einsum(\"ij,ij->i\", a, b)\n\
         result = contract({}, {})\n",
        matrix(0),
        matrix(1),
    );
    (source, expected)
}

#[test]
fn integer_einsum_traps_identically_whatever_overflows_first() {
    let shapes = [
        Row::Clean,
        Row::Multiply,
        Row::Add,
        Row::AddThenMultiply,
        Row::MultiplyThenAdd,
    ];
    let mut cases = Vec::new();
    for (width, leaves) in [(&I32, 5), (&I64, 4)] {
        for shape in shapes {
            cases.push((width, leaves, vec![Row::Clean, shape]));
        }
        cases.push((width, leaves, vec![Row::Add, Row::Multiply]));
        cases.push((width, leaves, vec![Row::Multiply, Row::Add]));
    }
    // i16 products widen to i32 and cannot overflow there; their sum can.
    cases.push((&I16, 3, vec![Row::Clean, Row::Saturated]));
    cases.push((&I16, 3, vec![Row::Clean, Row::Clean]));
    for (index, (width, leaves, rows)) in cases.into_iter().enumerate() {
        let (source, expected) = case(width, leaves, &rows);
        let name = format!("einsum_trap_{index}");
        let run = parity::assert_lanes_agree(&source, &name);
        match expected {
            None => {
                assert_eq!(run.status, Some(1), "{name} {rows:?}: {run:?}\n{source}");
                assert_eq!(
                    run.failure,
                    format!("numeric trap: overflow in einsum at {}", width.accumulator),
                    "{name} {rows:?}\n{source}"
                );
            }
            Some(sums) => {
                assert_eq!(run.status, Some(0), "{name} {rows:?}: {run:?}\n{source}");
                let data = sums.iter().map(i128::to_string).collect::<Vec<_>>();
                assert!(
                    run.stdout.contains(&format!("data=[{}]", data.join(", "))),
                    "{name} {rows:?}: {run:?}\n{source}"
                );
            }
        }
    }
}
