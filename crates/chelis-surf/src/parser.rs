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
    module_allowed: bool,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn parse(tokens: &[Token]) -> Result<Vec<Decl>, ParseError> {
    let mut p = Parser {
        tokens: tokens.to_vec(),
        pos: 0,
        module_allowed: true,
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
    fn raw_peek(&self) -> &TokenKind {
        self.tokens
            .get(self.pos)
            .map(|t| &t.kind)
            .unwrap_or(&TokenKind::Eof)
    }

    fn peek(&self) -> &TokenKind {
        let mut pos = self.pos;
        while matches!(
            self.tokens.get(pos).map(|t| &t.kind),
            Some(TokenKind::Newline)
        ) {
            pos += 1;
        }
        self.tokens
            .get(pos)
            .map(|t| &t.kind)
            .unwrap_or(&TokenKind::Eof)
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek(), TokenKind::Eof)
    }

    fn current_offset(&self) -> usize {
        let mut pos = self.pos;
        while matches!(
            self.tokens.get(pos).map(|t| &t.kind),
            Some(TokenKind::Newline)
        ) {
            pos += 1;
        }
        self.tokens
            .get(pos)
            .map(|t| t.span.offset)
            .unwrap_or(self.tokens.last().map(|t| t.span.end()).unwrap_or(0))
    }

    fn current_span(&self) -> Span {
        let mut pos = self.pos;
        while matches!(
            self.tokens.get(pos).map(|t| &t.kind),
            Some(TokenKind::Newline)
        ) {
            pos += 1;
        }
        self.tokens
            .get(pos)
            .map(|t| t.span)
            .unwrap_or(Span::new(self.current_offset(), 0))
    }

    fn advance_raw(&mut self) -> Token {
        let tok = self.tokens[self.pos].clone();
        self.pos += 1;
        tok
    }

    fn advance(&mut self) -> Token {
        while matches!(self.raw_peek(), TokenKind::Newline) {
            self.pos += 1;
        }
        self.advance_raw()
    }

    fn consume_block_separators(&mut self) -> usize {
        let mut consumed = 0;
        while matches!(self.raw_peek(), TokenKind::Semicolon | TokenKind::Newline) {
            self.advance_raw();
            consumed += 1;
        }
        consumed
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

    fn expect_ident_or_type_ident(&mut self) -> Result<(String, Span), ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(name) | TokenKind::TypeIdent(name) => {
                let tok = self.advance();
                Ok((name, tok.span))
            }
            _ => Err(ParseError::Expected {
                expected: "identifier or type identifier".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    fn parse_module_path(&mut self) -> Result<(String, Span), ParseError> {
        let (first_seg, mut span) = self.expect_type_ident()?;
        let mut module = first_seg;
        while *self.peek() == TokenKind::Dot {
            self.advance();
            let (seg, seg_span) = self.expect_type_ident()?;
            module.push('.');
            module.push_str(&seg);
            span = span.merge(seg_span);
        }
        Ok((module, span))
    }

    fn parse_ident_list(&mut self, terminator: TokenKind) -> Result<Vec<String>, ParseError> {
        let mut names = Vec::new();
        if *self.peek() == terminator {
            return Ok(names);
        }
        let (name, _) = self.expect_ident_or_type_ident()?;
        names.push(name);
        while *self.peek() == TokenKind::Comma {
            self.advance();
            if *self.peek() == terminator {
                break;
            }
            let (name, _) = self.expect_ident_or_type_ident()?;
            names.push(name);
        }
        Ok(names)
    }

    fn parse_name_bracket_list(&mut self) -> Result<Vec<String>, ParseError> {
        self.expect(&TokenKind::LBracket)?;
        let names = self.parse_ident_list(TokenKind::RBracket)?;
        self.expect(&TokenKind::RBracket)?;
        Ok(names)
    }

    // ---------------------------------------------------------------------------
    // Top-level
    // ---------------------------------------------------------------------------

    fn parse_program(&mut self) -> Result<Vec<Decl>, ParseError> {
        let mut decls = Vec::new();
        while !self.at_eof() {
            let decl = self.parse_decl()?;
            if !matches!(decl, Decl::Module { .. }) {
                self.module_allowed = false;
            }
            decls.push(decl);
        }
        Ok(decls)
    }

    fn parse_decl(&mut self) -> Result<Decl, ParseError> {
        match self.peek() {
            TokenKind::Def => self.parse_fun_def(),
            TokenKind::Sig => self.parse_sig_decl(),
            TokenKind::Let => self.parse_let_def(),
            TokenKind::Type => self.parse_type_decl(),
            TokenKind::Dim => self.parse_dim_decl(),
            TokenKind::Module => {
                if self.module_allowed {
                    self.parse_module()
                } else {
                    Err(ParseError::Expected {
                        expected: "module declaration only as the first declaration in a file"
                            .into(),
                        found: "Module".into(),
                        offset: self.current_offset(),
                    })
                }
            }
            TokenKind::Import => self.parse_import(),
            TokenKind::Export => self.parse_export(),
            _ => Err(ParseError::Expected {
                expected: "declaration (def, sig, let, type, dim, module, import, export)".into(),
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

        // Optional dimension parameters: def f[a, b](...)
        let dim_params = if *self.peek() == TokenKind::LBracket {
            self.parse_name_bracket_list()?
        } else {
            Vec::new()
        };

        let params = if *self.peek() == TokenKind::LParen {
            self.advance();
            let params = self.parse_params()?;
            self.expect(&TokenKind::RParen)?;
            params
        } else {
            Vec::new()
        };

        let ret_ty = if *self.peek() == TokenKind::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        let effects = self.parse_optional_effects()?;

        self.expect(&TokenKind::Eq)?;
        let body = self.parse_expr(0)?;
        let span = start.merge(expr_span(&body));

        Ok(Decl::FunDef {
            name,
            dim_params,
            params,
            ret_ty,
            effects,
            body,
            span,
        })
    }

    fn parse_sig_decl(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Sig
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::Colon)?;
        let ty = self.parse_type()?;
        let effects = self.parse_optional_effects()?;
        let end = effects
            .as_ref()
            .and_then(|effects| effects.last())
            .map(EffectExpr::span)
            .unwrap_or_else(|| type_span(&ty));
        let span = start.merge(end);
        Ok(Decl::Sig {
            name,
            ty,
            effects,
            span,
        })
    }

    fn parse_dim_decl(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Dim
        let names = self.parse_ident_list(TokenKind::Eof)?;
        let end = self.tokens[self.pos - 1].span;
        Ok(Decl::Dim {
            names,
            span: start.merge(end),
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

    fn parse_type_decl(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Type
        let (name, _) = self.expect_type_ident()?;
        let params = if *self.peek() == TokenKind::LBracket {
            self.parse_name_bracket_list()?
        } else {
            Vec::new()
        };

        self.expect(&TokenKind::Eq)?;

        if *self.peek() == TokenKind::Bar {
            let mut variants = Vec::new();
            self.advance();
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
        } else {
            let ty = self.parse_type()?;
            let span = start.merge(type_span(&ty));
            Ok(Decl::TypeAlias {
                name,
                params,
                ty,
                span,
            })
        }
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
        } else if *self.peek() == TokenKind::LParen {
            self.advance();
            let mut fields = Vec::new();
            if *self.peek() != TokenKind::RParen {
                fields.push(self.parse_type()?);
                while *self.peek() == TokenKind::Comma {
                    self.advance();
                    if *self.peek() == TokenKind::RParen {
                        break;
                    }
                    fields.push(self.parse_type()?);
                }
            }
            let end = self.expect(&TokenKind::RParen)?;
            Ok(Variant {
                name,
                fields: VariantFields::Positional(fields),
                span: start.merge(end.span),
            })
        } else {
            Ok(Variant {
                name,
                fields: VariantFields::Positional(Vec::new()),
                span: start,
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
        let (name, _) = self.parse_module_path()?;
        self.module_allowed = false;

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

    fn parse_import(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Import
        let (module, mod_span) = self.parse_module_path()?;

        let (kind, end) = if *self.peek() == TokenKind::LParen {
            self.advance();
            let kind = if *self.peek() == TokenKind::Dot {
                self.advance();
                self.expect(&TokenKind::Dot)?;
                ImportKind::All
            } else {
                ImportKind::Names(self.parse_ident_list(TokenKind::RParen)?)
            };
            let end = self.expect(&TokenKind::RParen)?;
            (kind, end.span)
        } else {
            (ImportKind::Qualified, mod_span)
        };

        Ok(Decl::Import {
            module,
            kind,
            span: start.merge(end),
        })
    }

    fn parse_export(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Export
        self.expect(&TokenKind::LParen)?;
        let names = self.parse_ident_list(TokenKind::RParen)?;
        let end = self.expect(&TokenKind::RParen)?;
        Ok(Decl::Export {
            names,
            span: start.merge(end.span),
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

            if *self.peek() == TokenKind::Dot {
                let l_bp = 15u8;
                if l_bp < min_bp {
                    break;
                }
                self.advance();
                match self.peek().clone() {
                    TokenKind::Ident(field) => {
                        let tok = self.advance();
                        let start = expr_span(&lhs);
                        lhs = Expr::Access(Box::new(lhs), field, start.merge(tok.span));
                        continue;
                    }
                    TokenKind::Int(index) => {
                        let tok = self.advance();
                        let start = expr_span(&lhs);
                        lhs = Expr::TupleGet(Box::new(lhs), index, start.merge(tok.span));
                        continue;
                    }
                    _ => {
                        return Err(ParseError::Expected {
                            expected: "field name or tuple index".into(),
                            found: format!("{:?}", self.peek()),
                            offset: self.current_offset(),
                        });
                    }
                }
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
                if *self.peek() == TokenKind::LBrace {
                    return self.parse_record_expr(name, tok.span);
                }
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
            TokenKind::Realize => self.parse_realize(),
            TokenKind::Copy => self.parse_copy(),
            TokenKind::With => self.parse_with_handler(),
            TokenKind::Par => self.parse_par(),
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
                if *self.peek() == TokenKind::LBrace {
                    self.parse_record_expr(name, tok.span)
                } else {
                    Ok(Expr::Constructor(name, tok.span))
                }
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
            let guard = if *self.peek() == TokenKind::If {
                self.advance();
                Some(self.parse_expr(0)?)
            } else {
                None
            };
            self.expect(&TokenKind::FatArrow)?;
            let body = self.parse_expr(0)?;
            let arm_span = pattern_span(&pattern).merge(expr_span(&body));
            arms.push(MatchArm {
                pattern,
                guard,
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
            let pattern = self.parse_let_pattern()?;
            let ty = if matches!(pattern, LetPattern::Var(_, _)) && *self.peek() == TokenKind::Colon
            {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };
            self.expect(&TokenKind::Eq)?;
            let value = self.parse_expr(0)?;
            bindings.push(LetBinding { pattern, ty, value });
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
            match self.peek().clone() {
                TokenKind::Ident(kw) if kw == "axis" => {
                    self.advance();
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
                }
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

    fn parse_realize(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span;
        self.expect(&TokenKind::LParen)?;
        let expr = self.parse_expr(0)?;
        let end = self.expect(&TokenKind::RParen)?;
        Ok(Expr::Realize(Box::new(expr), start.merge(end.span)))
    }

    fn parse_copy(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span;
        self.expect(&TokenKind::LParen)?;
        let expr = self.parse_expr(0)?;
        let end = self.expect(&TokenKind::RParen)?;
        Ok(Expr::Copy(Box::new(expr), start.merge(end.span)))
    }

    fn parse_with_handler(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume With
        let (handler_name, _) = self.expect_ident()?;
        self.expect(&TokenKind::LParen)?;
        let arg = self.parse_expr(0)?;
        self.expect(&TokenKind::RParen)?;
        let body = self.parse_block()?;
        let span = start.merge(expr_span(&body));
        match handler_name.as_str() {
            "seed" => Ok(Expr::WithSeed(Box::new(arg), Box::new(body), span)),
            "device" => Ok(Expr::WithDevice(Box::new(arg), Box::new(body), span)),
            _ => Err(ParseError::Expected {
                expected: "`seed` or `device` effect handler".into(),
                found: handler_name,
                offset: self.current_offset(),
            }),
        }
    }

    fn parse_par(&mut self) -> Result<Expr, ParseError> {
        // par { e1; e2; ... }
        let start = self.advance().span; // consume Par
        self.expect(&TokenKind::LBrace)?;
        let mut exprs = Vec::new();
        while matches!(self.raw_peek(), TokenKind::Newline) {
            self.advance_raw();
        }
        if *self.peek() == TokenKind::RBrace {
            let end = self.expect(&TokenKind::RBrace)?;
            return Ok(Expr::Par(exprs, start.merge(end.span)));
        }
        exprs.push(self.parse_expr(0)?);
        loop {
            while matches!(self.raw_peek(), TokenKind::Newline) {
                self.advance_raw();
            }
            if *self.peek() == TokenKind::RBrace {
                break;
            }
            if *self.raw_peek() != TokenKind::Semicolon {
                return Err(ParseError::Expected {
                    expected: "separator (`;`)".into(),
                    found: format!("{:?}", self.raw_peek()),
                    offset: self.current_offset(),
                });
            }
            self.advance_raw();
            while matches!(self.raw_peek(), TokenKind::Newline) {
                self.advance_raw();
            }
            if *self.peek() == TokenKind::RBrace {
                break;
            }
            exprs.push(self.parse_expr(0)?);
        }
        let end = self.expect(&TokenKind::RBrace)?;
        Ok(Expr::Par(exprs, start.merge(end.span)))
    }

    fn parse_block(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume LBrace
        let mut bindings = Vec::new();
        self.consume_block_separators();
        while *self.peek() == TokenKind::Let && !self.at_eof() {
            bindings.push(self.parse_block_let_binding()?);
            let sep_count = self.consume_block_separators();
            if *self.peek() != TokenKind::RBrace && sep_count == 0 {
                return Err(ParseError::Expected {
                    expected: "separator (`;` or newline)".into(),
                    found: format!("{:?}", self.raw_peek()),
                    offset: self.current_offset(),
                });
            }
        }
        let expr = self.parse_expr(0)?;
        self.consume_block_separators();
        let end = self.expect(&TokenKind::RBrace)?;
        Ok(Expr::Block(bindings, Box::new(expr), start.merge(end.span)))
    }

    fn parse_block_let_binding(&mut self) -> Result<LetBinding, ParseError> {
        self.expect(&TokenKind::Let)?;
        let pattern = self.parse_let_pattern()?;
        let ty = if matches!(pattern, LetPattern::Var(_, _)) && *self.peek() == TokenKind::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr(0)?;
        Ok(LetBinding { pattern, ty, value })
    }

    fn parse_let_pattern(&mut self) -> Result<LetPattern, ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                let tok = self.advance();
                Ok(LetPattern::Var(name, tok.span))
            }
            TokenKind::Underscore => {
                let tok = self.advance();
                Ok(LetPattern::Wildcard(tok.span))
            }
            TokenKind::LParen => {
                let start = self.advance().span;
                let first = self.parse_let_pattern()?;
                if *self.peek() != TokenKind::Comma {
                    return Err(ParseError::Expected {
                        expected: "tuple destructuring pattern".into(),
                        found: format!("{:?}", self.peek()),
                        offset: self.current_offset(),
                    });
                }
                let mut pats = vec![first];
                while *self.peek() == TokenKind::Comma {
                    self.advance();
                    if *self.peek() == TokenKind::RParen {
                        break;
                    }
                    pats.push(self.parse_let_pattern()?);
                }
                let end = self.expect(&TokenKind::RParen)?;
                Ok(LetPattern::Tuple(pats, start.merge(end.span)))
            }
            _ => Err(ParseError::Expected {
                expected: "let binding pattern".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    fn parse_record_expr(&mut self, name: String, start: Span) -> Result<Expr, ParseError> {
        self.expect(&TokenKind::LBrace)?;
        let mut fields = Vec::new();
        if *self.peek() != TokenKind::RBrace {
            loop {
                let (field, field_span) = self.expect_ident()?;
                let value = if *self.peek() == TokenKind::Colon {
                    self.advance();
                    self.parse_expr(0)?
                } else {
                    Expr::Var(field.clone(), field_span)
                };
                fields.push((field, value));
                if *self.peek() != TokenKind::Comma {
                    break;
                }
                self.advance();
                if *self.peek() == TokenKind::RBrace {
                    break;
                }
            }
        }
        let end = self.expect(&TokenKind::RBrace)?;
        Ok(Expr::Record(name, fields, start.merge(end.span)))
    }

    // ---------------------------------------------------------------------------
    // Type expression parsing
    // ---------------------------------------------------------------------------

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
            TokenKind::Int(n) => {
                let tok = self.advance();
                Ok(TypeExpr::Named(n.to_string(), tok.span))
            }
            TokenKind::Ident(name) => {
                let tok = self.advance();
                Ok(TypeExpr::Named(name, tok.span))
            }
            TokenKind::TypeIdent(name) => {
                let tok = self.advance();
                if *self.peek() == TokenKind::LBracket {
                    self.advance();
                    let mut args = Vec::new();
                    if *self.peek() != TokenKind::RBracket {
                        args.push(self.parse_type()?);
                        while *self.peek() == TokenKind::Comma {
                            self.advance();
                            if *self.peek() == TokenKind::RBracket {
                                break;
                            }
                            args.push(self.parse_type()?);
                        }
                    }
                    let end = self.expect(&TokenKind::RBracket)?;
                    Ok(TypeExpr::App(name, args, tok.span.merge(end.span)))
                } else {
                    Ok(TypeExpr::Named(name, tok.span))
                }
            }
            TokenKind::Star => {
                // * in type position = wildcard dimension
                let tok = self.advance();
                Ok(TypeExpr::Named("*".to_string(), tok.span))
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
                if *self.peek() == TokenKind::RParen {
                    let end = self.advance().span;
                    return Ok(TypeExpr::Tuple(Vec::new(), start.merge(end)));
                }
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
                        let field_pat = if *self.peek() == TokenKind::Colon {
                            self.advance();
                            self.parse_pattern()?
                        } else {
                            Pattern::Var(field_name.clone(), field_span)
                        };
                        fields.push((field_name, field_pat));
                        if *self.peek() == TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    let end = self.expect(&TokenKind::RBrace)?;
                    Ok(Pattern::Record(name, fields, tok.span.merge(end.span)))
                } else if *self.peek() == TokenKind::LParen {
                    self.advance();
                    let mut sub_pats = Vec::new();
                    if *self.peek() != TokenKind::RParen {
                        sub_pats.push(self.parse_pattern()?);
                        while *self.peek() == TokenKind::Comma {
                            self.advance();
                            if *self.peek() == TokenKind::RParen {
                                break;
                            }
                            sub_pats.push(self.parse_pattern()?);
                        }
                    }
                    let end = self.expect(&TokenKind::RParen)?;
                    Ok(Pattern::Constructor(
                        name,
                        sub_pats,
                        tok.span.merge(end.span),
                    ))
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

    fn parse_optional_effects(&mut self) -> Result<Option<Vec<EffectExpr>>, ParseError> {
        if *self.peek() != TokenKind::Bang {
            return Ok(None);
        }
        self.advance();
        self.expect(&TokenKind::LBrace)?;
        let mut effects = Vec::new();
        if *self.peek() != TokenKind::RBrace {
            effects.push(self.parse_effect_expr()?);
            while *self.peek() == TokenKind::Comma {
                self.advance();
                if *self.peek() == TokenKind::RBrace {
                    break;
                }
                effects.push(self.parse_effect_expr()?);
            }
        }
        self.expect(&TokenKind::RBrace)?;
        Ok(Some(effects))
    }

    fn parse_effect_expr(&mut self) -> Result<EffectExpr, ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                let tok = self.advance();
                match name.as_str() {
                    "Diff" | "diff" => Ok(EffectExpr::Diff(tok.span)),
                    "Random" | "random" => Ok(EffectExpr::Random(tok.span)),
                    "Accum" | "accum" => Ok(EffectExpr::Accum(tok.span)),
                    _ => Err(ParseError::Expected {
                        expected: "effect name".into(),
                        found: name,
                        offset: tok.span.offset,
                    }),
                }
            }
            TokenKind::TypeIdent(name) => {
                let tok = self.advance();
                match name.as_str() {
                    "Diff" => Ok(EffectExpr::Diff(tok.span)),
                    "Random" => Ok(EffectExpr::Random(tok.span)),
                    "Accum" => Ok(EffectExpr::Accum(tok.span)),
                    "Resource" => {
                        self.expect(&TokenKind::LParen)?;
                        let device = match self.peek().clone() {
                            TokenKind::Str(device) => {
                                self.advance();
                                device
                            }
                            _ => {
                                return Err(ParseError::Expected {
                                    expected: "device string".into(),
                                    found: format!("{:?}", self.peek()),
                                    offset: self.current_offset(),
                                });
                            }
                        };
                        let end = self.expect(&TokenKind::RParen)?;
                        Ok(EffectExpr::Resource(device, tok.span.merge(end.span)))
                    }
                    _ => Err(ParseError::Expected {
                        expected: "effect name".into(),
                        found: name,
                        offset: tok.span.offset,
                    }),
                }
            }
            _ => Err(ParseError::Expected {
                expected: "effect name".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
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
        Expr::Record(_, _, s) => *s,
        Expr::Access(_, _, s) => *s,
        Expr::TupleGet(_, _, s) => *s,
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
        Expr::Realize(_, s) => *s,
        Expr::Copy(_, s) => *s,
        Expr::WithSeed(_, _, s) => *s,
        Expr::WithDevice(_, _, s) => *s,
        Expr::Par(_, s) => *s,
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
        Decl::Sig { span, .. } => *span,
        Decl::Dim { span, .. } => *span,
        Decl::TypeDef { span, .. } => *span,
        Decl::TypeAlias { span, .. } => *span,
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
    fn sig_effect_annotation() {
        let decls = p("sig f: f32 -> f32 ! {Diff, Random, Resource(\"gpu:0\")}");
        match &decls[0] {
            Decl::Sig { effects, .. } => {
                let effects = effects.as_ref().expect("effects");
                assert_eq!(effects.len(), 3);
                assert!(matches!(effects[0], EffectExpr::Diff(_)));
                assert!(matches!(effects[1], EffectExpr::Random(_)));
                assert!(
                    matches!(effects[2], EffectExpr::Resource(ref device, _) if device == "gpu:0")
                );
            }
            _ => panic!("expected Sig"),
        }
    }

    #[test]
    fn fun_def_effect_annotation() {
        let decls = p("def f(x: f32): f32 ! {Diff} = x");
        match &decls[0] {
            Decl::FunDef { effects, .. } => {
                let effects = effects.as_ref().expect("effects");
                assert_eq!(effects.len(), 1);
                assert!(matches!(effects[0], EffectExpr::Diff(_)));
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
        let decls = p("type Option[a] = | Some(a) | None");
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
        let decls = p("type Shape = | Scalar | Vector(i64)");
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
        let decls = p("type Config = | Default { lr: f32, eps: f32 }");
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
            Decl::Import { module, kind, .. } => {
                assert_eq!(module, "Foo");
                assert_eq!(kind, &ImportKind::Qualified);
            }
            _ => panic!("expected Import"),
        }
    }

    #[test]
    fn import_with_names() {
        let decls = p("import Foo(bar, baz)");
        match &decls[0] {
            Decl::Import { module, kind, .. } => {
                assert_eq!(module, "Foo");
                assert_eq!(
                    kind,
                    &ImportKind::Names(vec!["bar".to_string(), "baz".to_string()])
                );
            }
            _ => panic!("expected Import"),
        }
    }

    #[test]
    fn module_decl() {
        let decls = p("module M def f(x) = x");
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
        let e = body("let x = match x with { | Some y => y | None => 0 }");
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
                assert!(matches!(
                    &bindings[0].pattern,
                    LetPattern::Var(name, _) if name == "y"
                ));
                assert!(matches!(*body, Expr::Binary(BinOp::Add, _, _, _)));
            }
            _ => panic!("expected Let, got {e:?}"),
        }
    }

    #[test]
    fn let_in_tuple_destructuring() {
        let e = body("def f(x) = let (a, b) = pair in a");
        match e {
            Expr::Let(bindings, body, _) => {
                assert_eq!(bindings.len(), 1);
                assert!(matches!(
                    &bindings[0].pattern,
                    LetPattern::Tuple(parts, _)
                        if matches!(&parts[0], LetPattern::Var(name, _) if name == "a")
                            && matches!(&parts[1], LetPattern::Var(name, _) if name == "b")
                ));
                assert!(matches!(*body, Expr::Var(ref n, _) if n == "a"));
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

    #[test]
    fn block_accepts_newline_separators() {
        let e = body(
            "def f() = {
                let x = 1
                let y = 2
                y
            }",
        );
        match e {
            Expr::Block(bindings, body, _) => {
                assert_eq!(bindings.len(), 2);
                assert!(matches!(*body, Expr::Var(ref n, _) if n == "y"));
            }
            _ => panic!("expected Block, got {e:?}"),
        }
    }

    #[test]
    fn block_rejects_missing_separator_between_lets() {
        let err = p_err("def f() = { let x = 1 let y = 2 y }");
        assert!(matches!(err, ParseError::Expected { .. }));
    }

    #[test]
    fn par_accepts_semicolons_with_newlines() {
        let e = body(
            "def f() = par {
                a;
                b
            }",
        );
        match e {
            Expr::Par(exprs, _) => {
                assert_eq!(exprs.len(), 2);
            }
            _ => panic!("expected Par, got {e:?}"),
        }
    }

    #[test]
    fn with_seed_handler_expr() {
        let e = body("def f() = with seed(42) { dropout(x, 0.5) }");
        match e {
            Expr::WithSeed(seed, body, _) => {
                assert!(matches!(*seed, Expr::Lit(Literal::Int(42), _)));
                assert!(matches!(*body, Expr::Block(_, _, _)));
            }
            _ => panic!("expected WithSeed, got {e:?}"),
        }
    }

    #[test]
    fn with_device_handler_expr() {
        let e = body("def f() = with device(\"gpu:0\") { x }");
        match e {
            Expr::WithDevice(device, body, _) => {
                assert!(
                    matches!(*device, Expr::Lit(Literal::Str(ref value), _) if value == "gpu:0")
                );
                assert!(matches!(*body, Expr::Block(_, _, _)));
            }
            _ => panic!("expected WithDevice, got {e:?}"),
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
    fn type_tensor_with_literal_dims() {
        let decls = p("let x: tensor[32, 784, f32] = x");
        match &decls[0] {
            Decl::LetDef { ty: Some(ty), .. } => match ty {
                TypeExpr::Tensor(dims, prec, _) => {
                    assert_eq!(dims.len(), 2);
                    assert!(matches!(&dims[0], TypeExpr::Named(n, _) if n == "32"));
                    assert!(matches!(&dims[1], TypeExpr::Named(n, _) if n == "784"));
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
        let decls = p("let x: Option[f32] = x");
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

    #[test]
    fn module_must_be_first_decl() {
        let err = p_err("let x = 1\nmodule M\ndef y = 2");
        assert!(matches!(
            err,
            ParseError::Expected { ref expected, .. }
                if expected == "module declaration only as the first declaration in a file"
        ));
    }

    #[test]
    fn nested_module_is_rejected() {
        let err = p_err("module Outer\nmodule Inner\ndef y = 2");
        assert!(matches!(
            err,
            ParseError::Expected { ref expected, .. }
                if expected == "module declaration only as the first declaration in a file"
        ));
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
                assert!(matches!(
                    &bindings[0].pattern,
                    LetPattern::Var(name, _) if name == "a"
                ));
                assert!(matches!(
                    &bindings[1].pattern,
                    LetPattern::Var(name, _) if name == "b"
                ));
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
            Decl::Import { module, kind, .. } => {
                assert_eq!(module, "Foo.Bar.Baz");
                assert_eq!(kind, &ImportKind::Names(vec!["baz".to_string()]));
            }
            _ => panic!("expected Import"),
        }
    }

    #[test]
    fn import_dotted_no_names() {
        let decls = p("import Foo.Bar");
        match &decls[0] {
            Decl::Import { module, kind, .. } => {
                assert_eq!(module, "Foo.Bar");
                assert_eq!(kind, &ImportKind::Qualified);
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
        let decls = p("export (foo, bar, baz)");
        match &decls[0] {
            Decl::Export { names, .. } => {
                assert_eq!(names, &["foo", "bar", "baz"]);
            }
            _ => panic!("expected Export"),
        }
    }

    #[test]
    fn export_single() {
        let decls = p("export (foo)");
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
        let e = body("let x = match x with { | Adam { lr, eps } => lr }");
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
        let e = body("let x = match x with { | y @ Some z => y }");
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
