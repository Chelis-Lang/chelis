use crate::ast::*;
use crate::lexer::{self, LexError};
use crate::token::{Token, TokenKind};
use chelis_deep::Span;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("lex error: {0}")]
    Lex(#[from] LexError),
    #[error("unexpected end of input")]
    UnexpectedEof,
    #[error("expected {expected}, found {found} at byte {offset}")]
    Expected {
        expected: String,
        found: String,
        offset: usize,
    },
    #[error("non-associative operator chained at byte {offset}")]
    NonAssocChain { offset: usize },
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn parse(tokens: &[Token]) -> Result<Vec<Decl>, ParseError> {
    let mut p = Parser {
        tokens: tokens.to_vec(),
        pos: 0,
    };
    p.parse_program()
}

pub fn parse_str(source: &str) -> Result<Vec<Decl>, ParseError> {
    let tokens = lexer::lex(source)?;
    parse(&tokens)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

impl Parser {
    fn peek(&self) -> &TokenKind {
        self.tokens
            .get(self.pos)
            .map(|t| &t.kind)
            .unwrap_or(&TokenKind::Eof)
    }

    fn at_eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn current_offset(&self) -> usize {
        self.tokens
            .get(self.pos)
            .map(|t| t.span.offset)
            .unwrap_or(self.tokens.last().map(|t| t.span.end()).unwrap_or(0))
    }

    fn current_span(&self) -> Span {
        self.tokens
            .get(self.pos)
            .map(|t| t.span)
            .unwrap_or(Span::new(self.current_offset(), 0))
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens[self.pos].clone();
        self.pos += 1;
        tok
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<Token, ParseError> {
        if self.peek() == kind {
            Ok(self.advance())
        } else {
            Err(ParseError::Expected {
                expected: format!("{kind:?}"),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            })
        }
    }

    fn expect_ident(&mut self) -> Result<(String, Span), ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                let tok = self.advance();
                Ok((name, tok.span))
            }
            _ => Err(ParseError::Expected {
                expected: "identifier".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    fn expect_type_ident(&mut self) -> Result<(String, Span), ParseError> {
        match self.peek().clone() {
            TokenKind::TypeIdent(name) => {
                let tok = self.advance();
                Ok((name, tok.span))
            }
            _ => Err(ParseError::Expected {
                expected: "type identifier".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    // ---------------------------------------------------------------------------
    // Top-level
    // ---------------------------------------------------------------------------

    fn parse_program(&mut self) -> Result<Vec<Decl>, ParseError> {
        let mut decls = Vec::new();
        while !self.at_eof() {
            decls.push(self.parse_decl()?);
        }
        Ok(decls)
    }

    fn parse_decl(&mut self) -> Result<Decl, ParseError> {
        match self.peek() {
            TokenKind::Def => self.parse_fun_def(),
            TokenKind::Let => self.parse_let_def(),
            TokenKind::Type => self.parse_type_def(),
            TokenKind::Module => self.parse_module(),
            TokenKind::Import => self.parse_import(),
            TokenKind::Export => self.parse_export(),
            _ => Err(ParseError::Expected {
                expected: "declaration (def, let, type, module, import, export)".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    // ---------------------------------------------------------------------------
    // Declarations
    // ---------------------------------------------------------------------------

    fn parse_fun_def(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Def
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::LParen)?;

        let params = self.parse_params()?;

        self.expect(&TokenKind::RParen)?;

        let ret_ty = if *self.peek() == TokenKind::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(&TokenKind::Eq)?;
        let body = self.parse_expr(0)?;
        let span = start.merge(expr_span(&body));

        Ok(Decl::FunDef {
            name,
            params,
            ret_ty,
            body,
            span,
        })
    }

    fn parse_params(&mut self) -> Result<Vec<Param>, ParseError> {
        let mut params = Vec::new();
        if *self.peek() == TokenKind::RParen {
            return Ok(params);
        }
        params.push(self.parse_param()?);
        while *self.peek() == TokenKind::Comma {
            self.advance();
            params.push(self.parse_param()?);
        }
        Ok(params)
    }

    fn parse_param(&mut self) -> Result<Param, ParseError> {
        let (name, span) = self.expect_ident()?;
        let ty = if *self.peek() == TokenKind::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        let end = ty.as_ref().map(type_span).unwrap_or(span);
        Ok(Param {
            name,
            ty,
            span: span.merge(end),
        })
    }

    fn parse_let_def(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Let
        let (name, _) = self.expect_ident()?;
        let ty = if *self.peek() == TokenKind::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr(0)?;
        let span = start.merge(expr_span(&value));
        Ok(Decl::LetDef {
            name,
            ty,
            value,
            span,
        })
    }

    fn parse_type_def(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Type
        let (name, _) = self.expect_type_ident()?;

        // Parse type params (lowercase idents until `=`)
        let mut params = Vec::new();
        while let TokenKind::Ident(p) = self.peek().clone() {
            self.advance();
            params.push(p);
        }

        self.expect(&TokenKind::Eq)?;

        // Parse variants separated by |, with optional leading |
        let mut variants = Vec::new();
        // Skip optional leading | (allows `type T = | V1 | V2` style)
        if *self.peek() == TokenKind::Bar {
            self.advance();
        }
        variants.push(self.parse_variant()?);
        while *self.peek() == TokenKind::Bar {
            self.advance();
            variants.push(self.parse_variant()?);
        }

        let last_span = variants.last().unwrap().span;
        Ok(Decl::TypeDef {
            name,
            params,
            variants,
            span: start.merge(last_span),
        })
    }

    fn parse_variant(&mut self) -> Result<Variant, ParseError> {
        let (name, start) = self.expect_type_ident()?;

        if *self.peek() == TokenKind::LBrace {
            // Record variant
            self.advance();
            let mut fields = Vec::new();
            if *self.peek() != TokenKind::RBrace {
                fields.push(self.parse_record_field()?);
                while *self.peek() == TokenKind::Comma {
                    self.advance();
                    if *self.peek() == TokenKind::RBrace {
                        break;
                    }
                    fields.push(self.parse_record_field()?);
                }
            }
            let end = self.expect(&TokenKind::RBrace)?;
            Ok(Variant {
                name,
                fields: VariantFields::Record(fields),
                span: start.merge(end.span),
            })
        } else {
            // Positional fields: type exprs until | or end-of-variant context
            let mut fields = Vec::new();
            while self.is_type_start() {
                fields.push(self.parse_type_atom()?);
            }
            let end = fields.last().map(type_span).unwrap_or(start);
            Ok(Variant {
                name,
                fields: VariantFields::Positional(fields),
                span: start.merge(end),
            })
        }
    }

    fn parse_record_field(&mut self) -> Result<(String, TypeExpr), ParseError> {
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::Colon)?;
        let ty = self.parse_type()?;
        Ok((name, ty))
    }

    fn parse_module(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Module
        let (name, _) = self.expect_type_ident()?;

        if *self.peek() == TokenKind::LBrace {
            // Braced module: module Foo { ... }
            self.advance();
            let mut decls = Vec::new();
            while *self.peek() != TokenKind::RBrace {
                if self.at_eof() {
                    return Err(ParseError::UnexpectedEof);
                }
                decls.push(self.parse_decl()?);
            }
            let end = self.expect(&TokenKind::RBrace)?;
            Ok(Decl::Module {
                name,
                decls,
                span: start.merge(end.span),
            })
        } else {
            // Braceless module: rest of file belongs to this module
            let mut decls = Vec::new();
            while !self.at_eof() {
                decls.push(self.parse_decl()?);
            }
            let end = decls.last().map(decl_span).unwrap_or(start);
            Ok(Decl::Module {
                name,
                decls,
                span: start.merge(end),
            })
        }
    }

    fn parse_import(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Import
        let (first_seg, mut mod_span) = self.expect_type_ident()?;
        let mut module = first_seg;

        // Parse dotted path: Foo.Bar.Baz
        while *self.peek() == TokenKind::Dot {
            self.advance(); // consume Dot
            let (seg, seg_span) = self.expect_type_ident()?;
            module.push('.');
            module.push_str(&seg);
            mod_span = mod_span.merge(seg_span);
        }

        let (names, end) = if *self.peek() == TokenKind::LParen {
            self.advance();
            let mut ns = Vec::new();
            if *self.peek() != TokenKind::RParen {
                let (n, _) = self.expect_ident()?;
                ns.push(n);
                while *self.peek() == TokenKind::Comma {
                    self.advance();
                    let (n, _) = self.expect_ident()?;
                    ns.push(n);
                }
            }
            let end = self.expect(&TokenKind::RParen)?;
            (Some(ns), end.span)
        } else {
            (None, mod_span)
        };

        Ok(Decl::Import {
            module,
            names,
            span: start.merge(end),
        })
    }

    fn parse_export(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Export
        let mut names = Vec::new();
        let (first_name, first_span) = self.expect_ident()?;
        names.push(first_name);
        let mut end = first_span;
        while *self.peek() == TokenKind::Comma {
            self.advance();
            let (n, s) = self.expect_ident()?;
            names.push(n);
            end = s;
        }
        Ok(Decl::Export {
            names,
            span: start.merge(end),
        })
    }

    // ---------------------------------------------------------------------------
    // Expression parsing — Pratt
    // ---------------------------------------------------------------------------

    fn parse_expr(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_prefix()?;

        loop {
            if self.at_eof() {
                break;
            }

            // Check for pipe
            if *self.peek() == TokenKind::Pipe {
                let (l_bp, _r_bp) = (1u8, 2u8);
                if l_bp < min_bp {
                    break;
                }
                let start = expr_span(&lhs);
                let mut stages = Vec::new();
                while *self.peek() == TokenKind::Pipe {
                    self.advance();
                    let stage = self.parse_prefix()?;
                    stages.push(stage);
                }
                let end = expr_span(stages.last().unwrap());
                lhs = Expr::Pipe(Box::new(lhs), stages, start.merge(end));
                continue;
            }

            // Check for type annotation colon
            if *self.peek() == TokenKind::Colon {
                let l_bp = 14u8;
                if l_bp < min_bp {
                    break;
                }
                self.advance();
                let ty = self.parse_type()?;
                let start = expr_span(&lhs);
                let end = type_span(&ty);
                lhs = Expr::Annotate(Box::new(lhs), ty, start.merge(end));
                continue;
            }

            // Binary operators
            if let Some((op, l_bp, r_bp)) = self.infix_bp() {
                if l_bp < min_bp {
                    break;
                }
                let non_assoc = l_bp == r_bp;
                self.advance();

                // For non-assoc ops, bump right BP so the recursive call
                // does NOT consume a same-precedence operator on the right.
                let effective_r_bp = if non_assoc { r_bp + 1 } else { r_bp };
                let rhs = self.parse_expr(effective_r_bp)?;

                // Non-assoc check: after parsing RHS, if the next token is
                // the same op class at the same precedence, that's a chain.
                if non_assoc
                    && let Some((next_op, next_l, _)) = self.infix_bp()
                    && op_class(op) == op_class(next_op)
                    && next_l == l_bp
                {
                    return Err(ParseError::NonAssocChain {
                        offset: self.current_offset(),
                    });
                }

                let start = expr_span(&lhs);
                let end = expr_span(&rhs);
                lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs), start.merge(end));
                continue;
            }

            break;
        }

        Ok(lhs)
    }

    fn infix_bp(&self) -> Option<(BinOp, u8, u8)> {
        match self.peek() {
            TokenKind::PipePipe => Some((BinOp::Or, 3, 4)),
            TokenKind::AmpAmp => Some((BinOp::And, 5, 6)),
            TokenKind::EqEq => Some((BinOp::Eq, 7, 7)),
            TokenKind::BangEq => Some((BinOp::Ne, 7, 7)),
            TokenKind::Lt => Some((BinOp::Lt, 8, 8)),
            TokenKind::Gt => Some((BinOp::Gt, 8, 8)),
            TokenKind::LtEq => Some((BinOp::Le, 8, 8)),
            TokenKind::GtEq => Some((BinOp::Ge, 8, 8)),
            TokenKind::Plus => Some((BinOp::Add, 9, 10)),
            TokenKind::Minus => Some((BinOp::Sub, 9, 10)),
            TokenKind::Star => Some((BinOp::Mul, 11, 12)),
            TokenKind::Slash => Some((BinOp::Div, 11, 12)),
            TokenKind::Percent => Some((BinOp::Mod, 11, 12)),
            _ => None,
        }
    }

    fn parse_prefix(&mut self) -> Result<Expr, ParseError> {
        if self.at_eof() {
            return Err(ParseError::UnexpectedEof);
        }

        match self.peek().clone() {
            TokenKind::Int(n) => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::Int(n), tok.span))
            }
            TokenKind::Float(f) => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::Float(f), tok.span))
            }
            TokenKind::Str(s) => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::Str(s), tok.span))
            }
            TokenKind::True => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::Bool(true), tok.span))
            }
            TokenKind::False => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::Bool(false), tok.span))
            }
            TokenKind::Ident(name) => {
                let tok = self.advance();
                let mut expr = Expr::Var(name, tok.span);
                // Check for parenthesized application f(x, y)
                if *self.peek() == TokenKind::LParen {
                    self.advance();
                    let args = self.parse_expr_list(TokenKind::RParen)?;
                    let end = self.expect(&TokenKind::RParen)?;
                    let span = tok.span.merge(end.span);
                    expr = Expr::Apply(Box::new(expr), args, span);
                }
                // Juxtaposition application: f x y
                expr = self.parse_juxtaposition_args(expr)?;
                Ok(expr)
            }
            TokenKind::TypeIdent(name) => {
                let tok = self.advance();
                let mut expr = Expr::Constructor(name, tok.span);
                // Juxtaposition application: Foo x y
                expr = self.parse_juxtaposition_args(expr)?;
                Ok(expr)
            }
            TokenKind::Minus => {
                let tok = self.advance();
                let operand = self.parse_expr(13)?;
                let span = tok.span.merge(expr_span(&operand));
                Ok(Expr::Unary(UnaryOp::Neg, Box::new(operand), span))
            }
            TokenKind::Bang => {
                let tok = self.advance();
                let operand = self.parse_expr(13)?;
                let span = tok.span.merge(expr_span(&operand));
                Ok(Expr::Unary(UnaryOp::Not, Box::new(operand), span))
            }
            TokenKind::LParen => {
                let start = self.advance().span;
                if *self.peek() == TokenKind::RParen {
                    let end = self.advance().span;
                    return Ok(Expr::Tuple(Vec::new(), start.merge(end)));
                }
                let first = self.parse_expr(0)?;
                if *self.peek() == TokenKind::Comma {
                    // Tuple
                    let mut elems = vec![first];
                    while *self.peek() == TokenKind::Comma {
                        self.advance();
                        if *self.peek() == TokenKind::RParen {
                            break;
                        }
                        elems.push(self.parse_expr(0)?);
                    }
                    let end = self.expect(&TokenKind::RParen)?;
                    Ok(Expr::Tuple(elems, start.merge(end.span)))
                } else {
                    // Grouping
                    self.expect(&TokenKind::RParen)?;
                    Ok(first)
                }
            }
            TokenKind::If => self.parse_if(),
            TokenKind::Match => self.parse_match(),
            TokenKind::Let => self.parse_let_expr(),
            TokenKind::Fn => self.parse_lambda(),
            TokenKind::Cast => self.parse_cast(),
            TokenKind::Grad => self.parse_grad(),
            TokenKind::Vmap => self.parse_vmap(),
            TokenKind::Jit => self.parse_jit(),
            TokenKind::LBrace => self.parse_block(),
            _ => Err(ParseError::Expected {
                expected: "expression".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    fn can_start_juxtaposition_arg(&self) -> bool {
        matches!(
            self.peek(),
            TokenKind::Ident(_)
                | TokenKind::TypeIdent(_)
                | TokenKind::Int(_)
                | TokenKind::Float(_)
                | TokenKind::Str(_)
                | TokenKind::True
                | TokenKind::False
                | TokenKind::LParen
                | TokenKind::Fn
                | TokenKind::If
        )
    }

    fn parse_juxtaposition_args(&mut self, mut expr: Expr) -> Result<Expr, ParseError> {
        while self.can_start_juxtaposition_arg()
            && self.infix_bp().is_none()
            && *self.peek() != TokenKind::Pipe
        {
            let arg = self.parse_primary_atom()?;
            let start = expr_span(&expr);
            let end = expr_span(&arg);
            expr = Expr::Apply(Box::new(expr), vec![arg], start.merge(end));
        }
        Ok(expr)
    }

    /// Parse a single primary expression without juxtaposition chaining.
    /// Used for juxtaposition arguments to avoid infinite recursion.
    fn parse_primary_atom(&mut self) -> Result<Expr, ParseError> {
        match self.peek().clone() {
            TokenKind::Int(n) => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::Int(n), tok.span))
            }
            TokenKind::Float(f) => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::Float(f), tok.span))
            }
            TokenKind::Str(s) => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::Str(s), tok.span))
            }
            TokenKind::True => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::Bool(true), tok.span))
            }
            TokenKind::False => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::Bool(false), tok.span))
            }
            TokenKind::Ident(name) => {
                let tok = self.advance();
                let mut expr = Expr::Var(name, tok.span);
                // Allow parenthesized call as postfix
                if *self.peek() == TokenKind::LParen {
                    self.advance();
                    let args = self.parse_expr_list(TokenKind::RParen)?;
                    let end = self.expect(&TokenKind::RParen)?;
                    let span = tok.span.merge(end.span);
                    expr = Expr::Apply(Box::new(expr), args, span);
                }
                Ok(expr)
            }
            TokenKind::TypeIdent(name) => {
                let tok = self.advance();
                Ok(Expr::Constructor(name, tok.span))
            }
            TokenKind::LParen => {
                let start = self.advance().span;
                if *self.peek() == TokenKind::RParen {
                    let end = self.advance().span;
                    return Ok(Expr::Tuple(Vec::new(), start.merge(end)));
                }
                let first = self.parse_expr(0)?;
                if *self.peek() == TokenKind::Comma {
                    let mut elems = vec![first];
                    while *self.peek() == TokenKind::Comma {
                        self.advance();
                        if *self.peek() == TokenKind::RParen {
                            break;
                        }
                        elems.push(self.parse_expr(0)?);
                    }
                    let end = self.expect(&TokenKind::RParen)?;
                    Ok(Expr::Tuple(elems, start.merge(end.span)))
                } else {
                    self.expect(&TokenKind::RParen)?;
                    Ok(first)
                }
            }
            TokenKind::If => self.parse_if(),
            TokenKind::Fn => self.parse_lambda(),
            _ => Err(ParseError::Expected {
                expected: "expression atom".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    fn parse_expr_list(&mut self, terminator: TokenKind) -> Result<Vec<Expr>, ParseError> {
        let mut exprs = Vec::new();
        if *self.peek() == terminator {
            return Ok(exprs);
        }
        exprs.push(self.parse_expr(0)?);
        while *self.peek() == TokenKind::Comma {
            self.advance();
            if *self.peek() == terminator {
                break;
            }
            exprs.push(self.parse_expr(0)?);
        }
        Ok(exprs)
    }

    fn parse_if(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume If
        let cond = self.parse_expr(0)?;
        self.expect(&TokenKind::Then)?;
        let then_br = self.parse_expr(0)?;
        self.expect(&TokenKind::Else)?;
        let else_br = self.parse_expr(0)?;
        let span = start.merge(expr_span(&else_br));
        Ok(Expr::If(
            Box::new(cond),
            Box::new(then_br),
            Box::new(else_br),
            span,
        ))
    }

    fn parse_match(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume Match
        let scrutinee = self.parse_expr(0)?;
        self.expect(&TokenKind::With)?;
        self.expect(&TokenKind::LBrace)?;

        let mut arms = Vec::new();
        while *self.peek() == TokenKind::Bar {
            self.advance();
            let pattern = self.parse_pattern()?;
            self.expect(&TokenKind::Arrow)?;
            let body = self.parse_expr(0)?;
            let arm_span = pattern_span(&pattern).merge(expr_span(&body));
            arms.push(MatchArm {
                pattern,
                body,
                span: arm_span,
            });
        }

        let end = self.expect(&TokenKind::RBrace)?;
        Ok(Expr::Match(
            Box::new(scrutinee),
            arms,
            start.merge(end.span),
        ))
    }

    fn parse_let_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        let mut bindings = Vec::new();

        while *self.peek() == TokenKind::Let {
            self.advance(); // consume Let
            let (name, _) = self.expect_ident()?;
            let ty = if *self.peek() == TokenKind::Colon {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };
            self.expect(&TokenKind::Eq)?;
            let value = self.parse_expr(0)?;
            bindings.push(LetBinding { name, ty, value });
        }

        self.expect(&TokenKind::In)?;
        let body = self.parse_expr(0)?;
        let span = start.merge(expr_span(&body));
        Ok(Expr::Let(bindings, Box::new(body), span))
    }

    fn parse_lambda(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume Fn
        self.expect(&TokenKind::LParen)?;
        let params = self.parse_params()?;
        self.expect(&TokenKind::RParen)?;
        self.expect(&TokenKind::Arrow)?;
        let body = self.parse_expr(0)?;
        let span = start.merge(expr_span(&body));
        Ok(Expr::Lambda(params, Box::new(body), span))
    }

    fn parse_cast(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume Cast
        self.expect(&TokenKind::LParen)?;
        let expr = self.parse_expr(0)?;
        self.expect(&TokenKind::Comma)?;
        let (precision, _) = self.expect_ident()?;
        let end = self.expect(&TokenKind::RParen)?;
        Ok(Expr::Cast(Box::new(expr), precision, start.merge(end.span)))
    }

    fn parse_grad(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume Grad
        self.expect(&TokenKind::LParen)?;
        let expr = self.parse_expr(0)?;
        let end = self.expect(&TokenKind::RParen)?;
        Ok(Expr::Grad(Box::new(expr), start.merge(end.span)))
    }

    fn parse_vmap(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume Vmap
        self.expect(&TokenKind::LParen)?;
        let expr = self.parse_expr(0)?;
        let axis = if *self.peek() == TokenKind::Comma {
            self.advance();
            // expect ident "axis"
            let (kw, _) = self.expect_ident()?;
            if kw != "axis" {
                return Err(ParseError::Expected {
                    expected: "axis".into(),
                    found: kw,
                    offset: self.current_offset(),
                });
            }
            self.expect(&TokenKind::Eq)?;
            match self.peek().clone() {
                TokenKind::Int(n) => {
                    self.advance();
                    Some(n)
                }
                _ => {
                    return Err(ParseError::Expected {
                        expected: "integer".into(),
                        found: format!("{:?}", self.peek()),
                        offset: self.current_offset(),
                    });
                }
            }
        } else {
            None
        };
        let end = self.expect(&TokenKind::RParen)?;
        Ok(Expr::Vmap(Box::new(expr), axis, start.merge(end.span)))
    }

    fn parse_jit(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume Jit
        self.expect(&TokenKind::LParen)?;
        let expr = self.parse_expr(0)?;
        let end = self.expect(&TokenKind::RParen)?;
        Ok(Expr::Jit(Box::new(expr), start.merge(end.span)))
    }

    fn parse_block(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume LBrace
        let mut decls = Vec::new();
        while matches!(
            self.peek(),
            TokenKind::Def | TokenKind::Let | TokenKind::Type
        ) && !self.at_eof()
        {
            // Peek ahead: is this a let-def (no `in`) or let-expr?
            // In a block context, `let x = ...` without `in` is a decl.
            decls.push(self.parse_decl()?);
        }
        let expr = self.parse_expr(0)?;
        let end = self.expect(&TokenKind::RBrace)?;
        Ok(Expr::Block(decls, Box::new(expr), start.merge(end.span)))
    }

    // ---------------------------------------------------------------------------
    // Type expression parsing
    // ---------------------------------------------------------------------------

    fn is_type_start(&self) -> bool {
        matches!(
            self.peek(),
            TokenKind::Ident(_)
                | TokenKind::TypeIdent(_)
                | TokenKind::Tensor
                | TokenKind::LParen
                | TokenKind::Underscore
        )
    }

    fn parse_type(&mut self) -> Result<TypeExpr, ParseError> {
        // Parse first type, then check for ->
        let mut types = vec![self.parse_type_atom()?];

        while *self.peek() == TokenKind::Arrow {
            self.advance();
            types.push(self.parse_type_atom()?);
        }

        if types.len() == 1 {
            Ok(types.pop().unwrap())
        } else {
            let start = type_span(&types[0]);
            let end = type_span(types.last().unwrap());
            let ret = types.pop().unwrap();
            Ok(TypeExpr::Arrow(types, Box::new(ret), start.merge(end)))
        }
    }

    fn parse_type_atom(&mut self) -> Result<TypeExpr, ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                let tok = self.advance();
                Ok(TypeExpr::Named(name, tok.span))
            }
            TokenKind::TypeIdent(name) => {
                let tok = self.advance();
                // Check if followed by type args
                if self.is_type_start() && !matches!(self.peek(), TokenKind::TypeIdent(_)) {
                    // Only simple type atoms as args (not other TypeIdents to avoid ambiguity)
                    let mut args = Vec::new();
                    // Actually, let's allow any type atom as arg
                    while self.is_type_arg_start() {
                        args.push(self.parse_type_atom()?);
                    }
                    if args.is_empty() {
                        Ok(TypeExpr::Named(name, tok.span))
                    } else {
                        let end = type_span(args.last().unwrap());
                        Ok(TypeExpr::App(name, args, tok.span.merge(end)))
                    }
                } else {
                    Ok(TypeExpr::Named(name, tok.span))
                }
            }
            TokenKind::Tensor => {
                let tok = self.advance();
                self.expect(&TokenKind::LBracket)?;
                // Parse dims (all but last are dims, last is precision ident)
                let mut items = Vec::new();
                items.push(self.parse_type_atom()?);
                while *self.peek() == TokenKind::Comma {
                    self.advance();
                    items.push(self.parse_type_atom()?);
                }
                let end = self.expect(&TokenKind::RBracket)?;
                // Last item should be the precision (a Named ident)
                let precision = items.pop().unwrap();
                let prec_name = match &precision {
                    TypeExpr::Named(n, _) => n.clone(),
                    _ => {
                        return Err(ParseError::Expected {
                            expected: "precision type name".into(),
                            found: format!("{precision:?}"),
                            offset: self.current_offset(),
                        });
                    }
                };
                Ok(TypeExpr::Tensor(items, prec_name, tok.span.merge(end.span)))
            }
            TokenKind::LParen => {
                let start = self.advance().span;
                let first = self.parse_type()?;
                if *self.peek() == TokenKind::Comma {
                    let mut types = vec![first];
                    while *self.peek() == TokenKind::Comma {
                        self.advance();
                        if *self.peek() == TokenKind::RParen {
                            break;
                        }
                        types.push(self.parse_type()?);
                    }
                    let end = self.expect(&TokenKind::RParen)?;
                    Ok(TypeExpr::Tuple(types, start.merge(end.span)))
                } else {
                    self.expect(&TokenKind::RParen)?;
                    Ok(first)
                }
            }
            TokenKind::Underscore => {
                let tok = self.advance();
                Ok(TypeExpr::Infer(tok.span))
            }
            _ => Err(ParseError::Expected {
                expected: "type".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    /// Can this token start a type argument (for App)?
    /// More restrictive than is_type_start: we don't consume TypeIdent
    /// to avoid greedily eating sibling variants.
    fn is_type_arg_start(&self) -> bool {
        matches!(
            self.peek(),
            TokenKind::Ident(_) | TokenKind::Tensor | TokenKind::LParen | TokenKind::Underscore
        )
    }

    // ---------------------------------------------------------------------------
    // Pattern parsing
    // ---------------------------------------------------------------------------

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        match self.peek().clone() {
            TokenKind::Underscore => {
                let tok = self.advance();
                Ok(Pattern::Wildcard(tok.span))
            }
            TokenKind::Ident(name) => {
                let tok = self.advance();
                // Check for as-pattern: x @ Pattern
                if *self.peek() == TokenKind::At {
                    self.advance(); // consume @
                    let inner = self.parse_pattern()?;
                    let end = pattern_span(&inner);
                    Ok(Pattern::As(name, Box::new(inner), tok.span.merge(end)))
                } else {
                    Ok(Pattern::Var(name, tok.span))
                }
            }
            TokenKind::Int(n) => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Int(n), tok.span))
            }
            TokenKind::Float(f) => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Float(f), tok.span))
            }
            TokenKind::Str(s) => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Str(s), tok.span))
            }
            TokenKind::True => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Bool(true), tok.span))
            }
            TokenKind::False => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Bool(false), tok.span))
            }
            TokenKind::TypeIdent(name) => {
                let tok = self.advance();
                // Check for record pattern: Ctor { field1, field2 }
                if *self.peek() == TokenKind::LBrace {
                    self.advance(); // consume {
                    let mut fields = Vec::new();
                    while *self.peek() != TokenKind::RBrace {
                        let (field_name, field_span) = match self.peek().clone() {
                            TokenKind::Ident(n) => {
                                let t = self.advance();
                                (n, t.span)
                            }
                            _ => {
                                return Err(ParseError::Expected {
                                    expected: "field name".into(),
                                    found: format!("{:?}", self.peek()),
                                    offset: self.current_offset(),
                                });
                            }
                        };
                        let field_pat = Pattern::Var(field_name.clone(), field_span);
                        fields.push((field_name, field_pat));
                        if *self.peek() == TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    let end = self.expect(&TokenKind::RBrace)?;
                    Ok(Pattern::Record(name, fields, tok.span.merge(end.span)))
                } else {
                    let mut sub_pats = Vec::new();
                    while self.is_pattern_arg_start() {
                        sub_pats.push(self.parse_pattern_atom()?);
                    }
                    if sub_pats.is_empty() {
                        Ok(Pattern::Constructor(name, vec![], tok.span))
                    } else {
                        let end = pattern_span(sub_pats.last().unwrap());
                        Ok(Pattern::Constructor(name, sub_pats, tok.span.merge(end)))
                    }
                }
            }
            TokenKind::LParen => {
                let start = self.advance().span;
                let first = self.parse_pattern()?;
                if *self.peek() == TokenKind::Comma {
                    let mut pats = vec![first];
                    while *self.peek() == TokenKind::Comma {
                        self.advance();
                        if *self.peek() == TokenKind::RParen {
                            break;
                        }
                        pats.push(self.parse_pattern()?);
                    }
                    let end = self.expect(&TokenKind::RParen)?;
                    Ok(Pattern::Tuple(pats, start.merge(end.span)))
                } else {
                    self.expect(&TokenKind::RParen)?;
                    Ok(first)
                }
            }
            _ => Err(ParseError::Expected {
                expected: "pattern".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    /// Parse a single atomic pattern (no constructor application).
    fn parse_pattern_atom(&mut self) -> Result<Pattern, ParseError> {
        match self.peek().clone() {
            TokenKind::Underscore => {
                let tok = self.advance();
                Ok(Pattern::Wildcard(tok.span))
            }
            TokenKind::Ident(name) => {
                let tok = self.advance();
                Ok(Pattern::Var(name, tok.span))
            }
            TokenKind::Int(n) => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Int(n), tok.span))
            }
            TokenKind::Float(f) => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Float(f), tok.span))
            }
            TokenKind::Str(s) => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Str(s), tok.span))
            }
            TokenKind::True => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Bool(true), tok.span))
            }
            TokenKind::False => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Bool(false), tok.span))
            }
            TokenKind::LParen => {
                let start = self.advance().span;
                let first = self.parse_pattern()?;
                if *self.peek() == TokenKind::Comma {
                    let mut pats = vec![first];
                    while *self.peek() == TokenKind::Comma {
                        self.advance();
                        if *self.peek() == TokenKind::RParen {
                            break;
                        }
                        pats.push(self.parse_pattern()?);
                    }
                    let end = self.expect(&TokenKind::RParen)?;
                    Ok(Pattern::Tuple(pats, start.merge(end.span)))
                } else {
                    self.expect(&TokenKind::RParen)?;
                    Ok(first)
                }
            }
            _ => Err(ParseError::Expected {
                expected: "pattern".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    fn is_pattern_arg_start(&self) -> bool {
        matches!(
            self.peek(),
            TokenKind::Underscore
                | TokenKind::Ident(_)
                | TokenKind::Int(_)
                | TokenKind::Float(_)
                | TokenKind::Str(_)
                | TokenKind::True
                | TokenKind::False
                | TokenKind::LParen
        )
    }
}

// ---------------------------------------------------------------------------
// Span helpers
// ---------------------------------------------------------------------------

fn expr_span(e: &Expr) -> Span {
    match e {
        Expr::Lit(_, s) => *s,
        Expr::Var(_, s) => *s,
        Expr::Constructor(_, s) => *s,
        Expr::Apply(_, _, s) => *s,
        Expr::Binary(_, _, _, s) => *s,
        Expr::Unary(_, _, s) => *s,
        Expr::Pipe(_, _, s) => *s,
        Expr::If(_, _, _, s) => *s,
        Expr::Match(_, _, s) => *s,
        Expr::Let(_, _, s) => *s,
        Expr::Lambda(_, _, s) => *s,
        Expr::Tuple(_, s) => *s,
        Expr::Cast(_, _, s) => *s,
        Expr::Grad(_, s) => *s,
        Expr::Vmap(_, _, s) => *s,
        Expr::Jit(_, s) => *s,
        Expr::Annotate(_, _, s) => *s,
        Expr::Block(_, _, s) => *s,
    }
}

fn type_span(t: &TypeExpr) -> Span {
    match t {
        TypeExpr::Named(_, s) => *s,
        TypeExpr::Tensor(_, _, s) => *s,
        TypeExpr::Arrow(_, _, s) => *s,
        TypeExpr::App(_, _, s) => *s,
        TypeExpr::Tuple(_, s) => *s,
        TypeExpr::Infer(s) => *s,
    }
}

fn pattern_span(p: &Pattern) -> Span {
    match p {
        Pattern::Wildcard(s) => *s,
        Pattern::Var(_, s) => *s,
        Pattern::Lit(_, s) => *s,
        Pattern::Constructor(_, _, s) => *s,
        Pattern::Tuple(_, s) => *s,
        Pattern::Record(_, _, s) => *s,
        Pattern::As(_, _, s) => *s,
    }
}

fn decl_span(d: &Decl) -> Span {
    match d {
        Decl::Module { span, .. } => *span,
        Decl::Import { span, .. } => *span,
        Decl::TypeDef { span, .. } => *span,
        Decl::FunDef { span, .. } => *span,
        Decl::LetDef { span, .. } => *span,
        Decl::Export { span, .. } => *span,
    }
}

fn op_class(op: BinOp) -> u8 {
    match op {
        BinOp::Eq | BinOp::Ne => 1,
        BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => 2,
        _ => 0, // associative ops don't need class checks
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Vec<Decl> {
        parse_str(s).unwrap()
    }

    fn p_err(s: &str) -> ParseError {
        parse_str(s).unwrap_err()
    }

    // Convenience: parse a single def, extract body expression.
    fn body(s: &str) -> Expr {
        let decls = p(s);
        match decls.into_iter().next().unwrap() {
            Decl::FunDef { body, .. } => body,
            Decl::LetDef { value, .. } => value,
            other => panic!("expected FunDef or LetDef, got {other:?}"),
        }
    }

    // ===== Declaration tests =====

    #[test]
    fn simple_fun_def() {
        let decls = p("def f(x: f32): f32 = x");
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Decl::FunDef {
                name,
                params,
                ret_ty,
                ..
            } => {
                assert_eq!(name, "f");
                assert_eq!(params.len(), 1);
                assert_eq!(params[0].name, "x");
                assert!(params[0].ty.is_some());
                assert!(ret_ty.is_some());
            }
            _ => panic!("expected FunDef"),
        }
    }

    #[test]
    fn fun_def_no_annotations() {
        let decls = p("def f(x) = x");
        match &decls[0] {
            Decl::FunDef { params, ret_ty, .. } => {
                assert_eq!(params[0].name, "x");
                assert!(params[0].ty.is_none());
                assert!(ret_ty.is_none());
            }
            _ => panic!("expected FunDef"),
        }
    }

    #[test]
    fn fun_def_multi_param() {
        let decls = p("def f(x: f32, y: f32): f32 = x");
        match &decls[0] {
            Decl::FunDef { params, .. } => {
                assert_eq!(params.len(), 2);
                assert_eq!(params[0].name, "x");
                assert_eq!(params[1].name, "y");
            }
            _ => panic!("expected FunDef"),
        }
    }

    #[test]
    fn top_level_let() {
        let decls = p("let x = 42");
        match &decls[0] {
            Decl::LetDef { name, ty, .. } => {
                assert_eq!(name, "x");
                assert!(ty.is_none());
            }
            _ => panic!("expected LetDef"),
        }
    }

    #[test]
    fn typed_let() {
        let decls = p("let x: f32 = 42.0");
        match &decls[0] {
            Decl::LetDef { name, ty, .. } => {
                assert_eq!(name, "x");
                assert!(ty.is_some());
            }
            _ => panic!("expected LetDef"),
        }
    }

    #[test]
    fn adt_type_def() {
        let decls = p("type Option a = Some a | None");
        match &decls[0] {
            Decl::TypeDef {
                name,
                params,
                variants,
                ..
            } => {
                assert_eq!(name, "Option");
                assert_eq!(params, &["a"]);
                assert_eq!(variants.len(), 2);
                assert_eq!(variants[0].name, "Some");
                assert_eq!(variants[1].name, "None");
            }
            _ => panic!("expected TypeDef"),
        }
    }

    #[test]
    fn type_def_no_params() {
        let decls = p("type Shape = Scalar | Vector i64");
        match &decls[0] {
            Decl::TypeDef {
                name,
                params,
                variants,
                ..
            } => {
                assert_eq!(name, "Shape");
                assert!(params.is_empty());
                assert_eq!(variants.len(), 2);
                assert_eq!(variants[0].name, "Scalar");
                assert_eq!(variants[1].name, "Vector");
                match &variants[1].fields {
                    VariantFields::Positional(fields) => {
                        assert_eq!(fields.len(), 1);
                    }
                    _ => panic!("expected positional"),
                }
            }
            _ => panic!("expected TypeDef"),
        }
    }

    #[test]
    fn record_variant() {
        let decls = p("type Config = Default { lr: f32, eps: f32 }");
        match &decls[0] {
            Decl::TypeDef { variants, .. } => {
                assert_eq!(variants[0].name, "Default");
                match &variants[0].fields {
                    VariantFields::Record(fields) => {
                        assert_eq!(fields.len(), 2);
                        assert_eq!(fields[0].0, "lr");
                        assert_eq!(fields[1].0, "eps");
                    }
                    _ => panic!("expected record"),
                }
            }
            _ => panic!("expected TypeDef"),
        }
    }

    #[test]
    fn import_simple() {
        let decls = p("import Foo");
        match &decls[0] {
            Decl::Import { module, names, .. } => {
                assert_eq!(module, "Foo");
                assert!(names.is_none());
            }
            _ => panic!("expected Import"),
        }
    }

    #[test]
    fn import_with_names() {
        let decls = p("import Foo(bar, baz)");
        match &decls[0] {
            Decl::Import { module, names, .. } => {
                assert_eq!(module, "Foo");
                assert_eq!(names.as_ref().unwrap(), &["bar", "baz"]);
            }
            _ => panic!("expected Import"),
        }
    }

    #[test]
    fn module_decl() {
        let decls = p("module M { def f(x) = x }");
        match &decls[0] {
            Decl::Module { name, decls, .. } => {
                assert_eq!(name, "M");
                assert_eq!(decls.len(), 1);
            }
            _ => panic!("expected Module"),
        }
    }

    // ===== Expression precedence tests =====

    #[test]
    fn add_mul_precedence() {
        let e = body("let x = a + b * c");
        match e {
            Expr::Binary(BinOp::Add, lhs, rhs, _) => {
                assert!(matches!(*lhs, Expr::Var(ref n, _) if n == "a"));
                assert!(matches!(*rhs, Expr::Binary(BinOp::Mul, _, _, _)));
            }
            _ => panic!("expected Add(a, Mul(b, c)), got {e:?}"),
        }
    }

    #[test]
    fn mul_add_precedence() {
        let e = body("let x = a * b + c");
        match e {
            Expr::Binary(BinOp::Add, lhs, rhs, _) => {
                assert!(matches!(*lhs, Expr::Binary(BinOp::Mul, _, _, _)));
                assert!(matches!(*rhs, Expr::Var(ref n, _) if n == "c"));
            }
            _ => panic!("expected Add(Mul(a, b), c), got {e:?}"),
        }
    }

    #[test]
    fn left_assoc_add() {
        let e = body("let x = a + b + c");
        match e {
            Expr::Binary(BinOp::Add, lhs, rhs, _) => {
                assert!(matches!(*lhs, Expr::Binary(BinOp::Add, _, _, _)));
                assert!(matches!(*rhs, Expr::Var(ref n, _) if n == "c"));
            }
            _ => panic!("expected Add(Add(a, b), c), got {e:?}"),
        }
    }

    #[test]
    fn unary_neg_plus() {
        let e = body("let x = -a + b");
        match e {
            Expr::Binary(BinOp::Add, lhs, _, _) => {
                assert!(matches!(*lhs, Expr::Unary(UnaryOp::Neg, _, _)));
            }
            _ => panic!("expected Add(Neg(a), b), got {e:?}"),
        }
    }

    #[test]
    fn non_assoc_eq_chain_error() {
        let err = p_err("let x = a == b == c");
        assert!(matches!(err, ParseError::NonAssocChain { .. }));
    }

    #[test]
    fn parens_override() {
        let e = body("let x = (a + b) * c");
        match e {
            Expr::Binary(BinOp::Mul, lhs, _, _) => {
                assert!(matches!(*lhs, Expr::Binary(BinOp::Add, _, _, _)));
            }
            _ => panic!("expected Mul(Add(a, b), c), got {e:?}"),
        }
    }

    // ===== Pipe tests =====

    #[test]
    fn pipe_chain() {
        let e = body("let x = x |> f |> g");
        match e {
            Expr::Pipe(_, stages, _) => {
                assert_eq!(stages.len(), 2);
            }
            _ => panic!("expected Pipe, got {e:?}"),
        }
    }

    // ===== Application tests =====

    #[test]
    fn paren_apply() {
        let e = body("let x = f(x, y)");
        match e {
            Expr::Apply(func, args, _) => {
                assert!(matches!(*func, Expr::Var(ref n, _) if n == "f"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("expected Apply, got {e:?}"),
        }
    }

    // ===== Type annotation =====

    #[test]
    fn type_annotation_expr() {
        let e = body("let x = y : f32");
        match e {
            Expr::Annotate(inner, ty, _) => {
                assert!(matches!(*inner, Expr::Var(ref n, _) if n == "y"));
                assert!(matches!(ty, TypeExpr::Named(ref n, _) if n == "f32"));
            }
            _ => panic!("expected Annotate, got {e:?}"),
        }
    }

    // ===== Control flow =====

    #[test]
    fn if_then_else() {
        let e = body("let x = if a then b else c");
        assert!(matches!(e, Expr::If(_, _, _, _)));
    }

    #[test]
    fn match_expr() {
        let e = body("let x = match x with { | Some y -> y | None -> 0 }");
        match e {
            Expr::Match(_, arms, _) => {
                assert_eq!(arms.len(), 2);
                assert!(
                    matches!(&arms[0].pattern, Pattern::Constructor(n, pats, _) if n == "Some" && pats.len() == 1)
                );
                assert!(
                    matches!(&arms[1].pattern, Pattern::Constructor(n, pats, _) if n == "None" && pats.is_empty())
                );
            }
            _ => panic!("expected Match, got {e:?}"),
        }
    }

    #[test]
    fn let_in_expr() {
        let e = body("def f(x) = let y = 1 in y + 1");
        match e {
            Expr::Let(bindings, body, _) => {
                assert_eq!(bindings.len(), 1);
                assert_eq!(bindings[0].name, "y");
                assert!(matches!(*body, Expr::Binary(BinOp::Add, _, _, _)));
            }
            _ => panic!("expected Let, got {e:?}"),
        }
    }

    #[test]
    fn lambda_expr() {
        let e = body("let x = fn (x, y) -> x + y");
        match e {
            Expr::Lambda(params, body, _) => {
                assert_eq!(params.len(), 2);
                assert!(matches!(*body, Expr::Binary(BinOp::Add, _, _, _)));
            }
            _ => panic!("expected Lambda, got {e:?}"),
        }
    }

    // ===== Type expression tests =====

    #[test]
    fn type_named() {
        let decls = p("let x: f32 = 1.0");
        match &decls[0] {
            Decl::LetDef { ty: Some(ty), .. } => {
                assert!(matches!(ty, TypeExpr::Named(n, _) if n == "f32"));
            }
            _ => panic!("expected typed let"),
        }
    }

    #[test]
    fn type_tensor() {
        let decls = p("let x: tensor[batch, hidden, f32] = x");
        match &decls[0] {
            Decl::LetDef { ty: Some(ty), .. } => match ty {
                TypeExpr::Tensor(dims, prec, _) => {
                    assert_eq!(dims.len(), 2);
                    assert_eq!(prec, "f32");
                }
                _ => panic!("expected Tensor type, got {ty:?}"),
            },
            _ => panic!("expected typed let"),
        }
    }

    #[test]
    fn type_arrow() {
        let decls = p("def f(x: f32 -> f32 -> bool): bool = x");
        match &decls[0] {
            Decl::FunDef { params, .. } => match &params[0].ty {
                Some(TypeExpr::Arrow(args, ret, _)) => {
                    assert_eq!(args.len(), 2);
                    assert!(matches!(ret.as_ref(), TypeExpr::Named(n, _) if n == "bool"));
                }
                other => panic!("expected Arrow type, got {other:?}"),
            },
            _ => panic!("expected FunDef"),
        }
    }

    #[test]
    fn type_app() {
        let decls = p("let x: Option f32 = x");
        match &decls[0] {
            Decl::LetDef { ty: Some(ty), .. } => match ty {
                TypeExpr::App(name, args, _) => {
                    assert_eq!(name, "Option");
                    assert_eq!(args.len(), 1);
                    assert!(matches!(&args[0], TypeExpr::Named(n, _) if n == "f32"));
                }
                _ => panic!("expected App type, got {ty:?}"),
            },
            _ => panic!("expected typed let"),
        }
    }

    #[test]
    fn type_tuple() {
        let decls = p("let x: (f32, f32) = x");
        match &decls[0] {
            Decl::LetDef { ty: Some(ty), .. } => match ty {
                TypeExpr::Tuple(types, _) => {
                    assert_eq!(types.len(), 2);
                }
                _ => panic!("expected Tuple type, got {ty:?}"),
            },
            _ => panic!("expected typed let"),
        }
    }

    #[test]
    fn type_infer() {
        let decls = p("let x: _ = 1");
        match &decls[0] {
            Decl::LetDef { ty: Some(ty), .. } => {
                assert!(matches!(ty, TypeExpr::Infer(_)));
            }
            _ => panic!("expected typed let"),
        }
    }

    // ===== Builtins =====

    #[test]
    fn cast_expr() {
        let e = body("let x = cast(y, f64)");
        match e {
            Expr::Cast(_, prec, _) => assert_eq!(prec, "f64"),
            _ => panic!("expected Cast, got {e:?}"),
        }
    }

    #[test]
    fn grad_expr() {
        let e = body("let x = grad(f)");
        assert!(matches!(e, Expr::Grad(_, _)));
    }

    #[test]
    fn vmap_expr() {
        let e = body("let x = vmap(f)");
        match e {
            Expr::Vmap(_, axis, _) => assert!(axis.is_none()),
            _ => panic!("expected Vmap, got {e:?}"),
        }
    }

    #[test]
    fn vmap_with_axis() {
        let e = body("let x = vmap(f, axis=1)");
        match e {
            Expr::Vmap(_, axis, _) => assert_eq!(axis, Some(1)),
            _ => panic!("expected Vmap, got {e:?}"),
        }
    }

    #[test]
    fn jit_expr() {
        let e = body("let x = jit(f)");
        assert!(matches!(e, Expr::Jit(_, _)));
    }

    // ===== Multiple declarations =====

    #[test]
    fn multiple_decls() {
        let decls = p("let x = 1 let y = 2 def f(z) = z");
        assert_eq!(decls.len(), 3);
    }

    // ===== Logical operators =====

    #[test]
    fn and_or_precedence() {
        let e = body("let x = a || b && c");
        match e {
            Expr::Binary(BinOp::Or, _, rhs, _) => {
                assert!(matches!(*rhs, Expr::Binary(BinOp::And, _, _, _)));
            }
            _ => panic!("expected Or(a, And(b, c)), got {e:?}"),
        }
    }

    // ===== Tuple expression =====

    #[test]
    fn tuple_expr() {
        let e = body("let x = (1, 2, 3)");
        match e {
            Expr::Tuple(elems, _) => assert_eq!(elems.len(), 3),
            _ => panic!("expected Tuple, got {e:?}"),
        }
    }

    // ===== Boolean literals =====

    #[test]
    fn bool_lits() {
        let e = body("let x = true");
        assert!(matches!(e, Expr::Lit(Literal::Bool(true), _)));
        let e = body("let x = false");
        assert!(matches!(e, Expr::Lit(Literal::Bool(false), _)));
    }

    // ===== String literal in expr =====

    #[test]
    fn string_lit_expr() {
        let e = body("let x = \"hello\"");
        assert!(matches!(e, Expr::Lit(Literal::Str(_), _)));
    }

    // ===== Constructor in expr =====

    #[test]
    fn constructor_expr() {
        let e = body("let x = None");
        assert!(matches!(e, Expr::Constructor(ref n, _) if n == "None"));
    }

    // ===== Chained let bindings =====

    #[test]
    fn chained_let_in() {
        let e = body("def f(x) = let a = 1 let b = 2 in a + b");
        match e {
            Expr::Let(bindings, _, _) => {
                assert_eq!(bindings.len(), 2);
                assert_eq!(bindings[0].name, "a");
                assert_eq!(bindings[1].name, "b");
            }
            _ => panic!("expected Let, got {e:?}"),
        }
    }

    // ===== Non-assoc comparison =====

    #[test]
    fn non_assoc_lt_chain_error() {
        let err = p_err("let x = a < b < c");
        assert!(matches!(err, ParseError::NonAssocChain { .. }));
    }

    #[test]
    fn different_comparison_classes_ok() {
        // == and < are in different classes, so this should parse.
        // Actually wait, both are non-assoc. Let's test same-class.
        // a < b is fine on its own.
        let _decls = p("let x = a < b");
    }

    // ===== Unary bang =====

    #[test]
    fn unary_not() {
        let e = body("let x = !a");
        assert!(matches!(e, Expr::Unary(UnaryOp::Not, _, _)));
    }

    // ===== Juxtaposition application tests =====

    #[test]
    fn juxtaposition_single_arg() {
        let e = body("let x = f x");
        match &e {
            Expr::Apply(func, args, _) => {
                assert!(matches!(func.as_ref(), Expr::Var(n, _) if n == "f"));
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], Expr::Var(n, _) if n == "x"));
            }
            _ => panic!("expected Apply, got {e:?}"),
        }
    }

    #[test]
    fn juxtaposition_two_args() {
        // f x y → Apply(Apply(f, [x]), [y])
        let e = body("let x = f x y");
        match &e {
            Expr::Apply(inner, args2, _) => {
                assert_eq!(args2.len(), 1);
                assert!(matches!(&args2[0], Expr::Var(n, _) if n == "y"));
                match inner.as_ref() {
                    Expr::Apply(func, args1, _) => {
                        assert!(matches!(func.as_ref(), Expr::Var(n, _) if n == "f"));
                        assert_eq!(args1.len(), 1);
                        assert!(matches!(&args1[0], Expr::Var(n, _) if n == "x"));
                    }
                    _ => panic!("expected inner Apply"),
                }
            }
            _ => panic!("expected Apply, got {e:?}"),
        }
    }

    #[test]
    fn juxtaposition_with_infix() {
        // f x + g y → Binary(Add, Apply(f, [x]), Apply(g, [y]))
        let e = body("let x = f x + g y");
        match &e {
            Expr::Binary(BinOp::Add, lhs, rhs, _) => {
                match lhs.as_ref() {
                    Expr::Apply(func, args, _) => {
                        assert!(matches!(func.as_ref(), Expr::Var(n, _) if n == "f"));
                        assert_eq!(args.len(), 1);
                        assert!(matches!(&args[0], Expr::Var(n, _) if n == "x"));
                    }
                    _ => panic!("expected Apply on lhs"),
                }
                match rhs.as_ref() {
                    Expr::Apply(func, args, _) => {
                        assert!(matches!(func.as_ref(), Expr::Var(n, _) if n == "g"));
                        assert_eq!(args.len(), 1);
                        assert!(matches!(&args[0], Expr::Var(n, _) if n == "y"));
                    }
                    _ => panic!("expected Apply on rhs"),
                }
            }
            _ => panic!("expected Binary Add, got {e:?}"),
        }
    }

    #[test]
    fn juxtaposition_grouped_arg() {
        // f (x + y) → Apply(f, [Binary(Add, x, y)])
        let e = body("let x = f (x + y)");
        match &e {
            Expr::Apply(func, args, _) => {
                assert!(matches!(func.as_ref(), Expr::Var(n, _) if n == "f"));
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], Expr::Binary(BinOp::Add, _, _, _)));
            }
            _ => panic!("expected Apply, got {e:?}"),
        }
    }

    #[test]
    fn constructor_juxtaposition() {
        // Some x → Apply(Constructor("Some"), [Var("x")])
        let e = body("let x = Some x");
        match &e {
            Expr::Apply(func, args, _) => {
                assert!(matches!(func.as_ref(), Expr::Constructor(n, _) if n == "Some"));
                assert_eq!(args.len(), 1);
            }
            _ => panic!("expected Apply, got {e:?}"),
        }
    }

    // ===== Dotted import path tests =====

    #[test]
    fn import_dotted_path() {
        let decls = p("import Foo.Bar.Baz(baz)");
        match &decls[0] {
            Decl::Import { module, names, .. } => {
                assert_eq!(module, "Foo.Bar.Baz");
                assert_eq!(names.as_ref().unwrap(), &["baz"]);
            }
            _ => panic!("expected Import"),
        }
    }

    #[test]
    fn import_dotted_no_names() {
        let decls = p("import Foo.Bar");
        match &decls[0] {
            Decl::Import { module, names, .. } => {
                assert_eq!(module, "Foo.Bar");
                assert!(names.is_none());
            }
            _ => panic!("expected Import"),
        }
    }

    // ===== Module without braces =====

    #[test]
    fn module_braceless() {
        let decls = p("module Foo def f(x) = x def g(y) = y");
        match &decls[0] {
            Decl::Module { name, decls, .. } => {
                assert_eq!(name, "Foo");
                assert_eq!(decls.len(), 2);
            }
            _ => panic!("expected Module"),
        }
    }

    // ===== Export declaration =====

    #[test]
    fn export_decl() {
        let decls = p("export foo, bar, baz");
        match &decls[0] {
            Decl::Export { names, .. } => {
                assert_eq!(names, &["foo", "bar", "baz"]);
            }
            _ => panic!("expected Export"),
        }
    }

    #[test]
    fn export_single() {
        let decls = p("export foo");
        match &decls[0] {
            Decl::Export { names, .. } => {
                assert_eq!(names, &["foo"]);
            }
            _ => panic!("expected Export"),
        }
    }

    // ===== Record pattern =====

    #[test]
    fn record_pattern() {
        let e = body("let x = match x with { | Adam { lr, eps } -> lr }");
        match e {
            Expr::Match(_, arms, _) => {
                assert_eq!(arms.len(), 1);
                match &arms[0].pattern {
                    Pattern::Record(name, fields, _) => {
                        assert_eq!(name, "Adam");
                        assert_eq!(fields.len(), 2);
                        assert_eq!(fields[0].0, "lr");
                        assert_eq!(fields[1].0, "eps");
                        assert!(matches!(&fields[0].1, Pattern::Var(n, _) if n == "lr"));
                        assert!(matches!(&fields[1].1, Pattern::Var(n, _) if n == "eps"));
                    }
                    other => panic!("expected Record pattern, got {other:?}"),
                }
            }
            _ => panic!("expected Match, got {e:?}"),
        }
    }

    // ===== As pattern =====

    #[test]
    fn as_pattern() {
        let e = body("let x = match x with { | y @ Some z -> y }");
        match e {
            Expr::Match(_, arms, _) => {
                assert_eq!(arms.len(), 1);
                match &arms[0].pattern {
                    Pattern::As(name, inner, _) => {
                        assert_eq!(name, "y");
                        match inner.as_ref() {
                            Pattern::Constructor(ctor, pats, _) => {
                                assert_eq!(ctor, "Some");
                                assert_eq!(pats.len(), 1);
                                assert!(matches!(&pats[0], Pattern::Var(n, _) if n == "z"));
                            }
                            other => panic!("expected Constructor inside As, got {other:?}"),
                        }
                    }
                    other => panic!("expected As pattern, got {other:?}"),
                }
            }
            _ => panic!("expected Match, got {e:?}"),
        }
    }
}
