//! Datalog syntax parser for TensorLogic
//!
//! Supports parsing Datalog syntax for facts, rules, and queries:
//! - Facts: `parent(alice, bob).`
//! - Rules: `grandparent(X, Z) :- parent(X, Y), parent(Y, Z).`
//! - Queries: `?- parent(alice, X).`

use crate::ir::{Constant, Predicate, Rule, Term};
use std::fmt;

/// Datalog parse error
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub position: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Parse error at position {}: {}",
            self.position, self.message
        )
    }
}

impl std::error::Error for ParseError {}

type ParseResult<T> = Result<T, ParseError>;

/// Datalog parser
pub struct DatalogParser {
    input: String,
    position: usize,
}

impl DatalogParser {
    /// Create a new parser for the given input
    pub fn new(input: String) -> Self {
        Self { input, position: 0 }
    }

    /// Parse a fact or rule
    pub fn parse_statement(&mut self) -> ParseResult<Statement> {
        self.skip_whitespace();

        if self.peek_char() == Some('?') {
            // Query
            self.advance(); // skip '?'
            self.expect_char('-')?;
            self.skip_whitespace();
            let predicate = self.parse_predicate()?;
            self.skip_whitespace();
            self.expect_char('.')?;
            Ok(Statement::Query(predicate))
        } else {
            // Fact or Rule
            let head = self.parse_predicate()?;
            self.skip_whitespace();

            if self.peek_char() == Some('.') {
                // Fact
                self.advance();
                Ok(Statement::Fact(head))
            } else if self.peek_str(2) == Some(":-") {
                // Rule
                self.advance();
                self.advance();
                self.skip_whitespace();

                let body = self.parse_predicate_list()?;
                self.skip_whitespace();
                self.expect_char('.')?;

                Ok(Statement::Rule(Rule::new(head, body)))
            } else {
                Err(ParseError {
                    message: "Expected '.' or ':-'".to_string(),
                    position: self.position,
                })
            }
        }
    }

    /// Parse a predicate like `parent(alice, bob)`
    fn parse_predicate(&mut self) -> ParseResult<Predicate> {
        let name = self.parse_identifier()?;
        self.skip_whitespace();
        self.expect_char('(')?;
        self.skip_whitespace();

        let args = self.parse_term_list()?;
        self.skip_whitespace();
        self.expect_char(')')?;

        Ok(Predicate::new(name, args))
    }

    /// Parse a comma-separated list of predicates
    fn parse_predicate_list(&mut self) -> ParseResult<Vec<Predicate>> {
        let mut predicates = Vec::new();

        loop {
            predicates.push(self.parse_predicate()?);
            self.skip_whitespace();

            if self.peek_char() == Some(',') {
                self.advance();
                self.skip_whitespace();
            } else {
                break;
            }
        }

        Ok(predicates)
    }

    /// Parse a comma-separated list of terms
    fn parse_term_list(&mut self) -> ParseResult<Vec<Term>> {
        let mut terms = Vec::new();

        if self.peek_char() == Some(')') {
            return Ok(terms); // Empty list
        }

        loop {
            terms.push(self.parse_term()?);
            self.skip_whitespace();

            if self.peek_char() == Some(',') {
                self.advance();
                self.skip_whitespace();
            } else {
                break;
            }
        }

        Ok(terms)
    }

    /// Parse a term (variable, constant, or function)
    fn parse_term(&mut self) -> ParseResult<Term> {
        self.skip_whitespace();

        let ch = self.peek_char().ok_or_else(|| ParseError {
            message: "Unexpected end of input".to_string(),
            position: self.position,
        })?;

        if ch == '?' || ch.is_uppercase() {
            // Variable
            if ch == '?' {
                self.advance();
            }
            let name = self.parse_identifier()?;
            Ok(Term::Var(name))
        } else if ch == '"' {
            // String constant
            self.advance(); // skip opening quote
            let value = self.parse_string()?;
            self.expect_char('"')?;
            Ok(Term::Const(Constant::String(value)))
        } else if ch.is_ascii_digit() || ch == '-' {
            // Numeric constant
            let value = self.parse_number()?;
            Ok(Term::Const(Constant::Int(value)))
        } else if ch.is_lowercase() {
            // Could be a constant atom or function
            let name = self.parse_identifier()?;
            self.skip_whitespace();

            if self.peek_char() == Some('(') {
                // Function
                self.advance();
                self.skip_whitespace();
                let args = self.parse_term_list()?;
                self.skip_whitespace();
                self.expect_char(')')?;
                Ok(Term::Fun(name, args))
            } else {
                // Atom constant
                Ok(Term::Const(Constant::String(name)))
            }
        } else {
            Err(ParseError {
                message: format!("Unexpected character: '{}'", ch),
                position: self.position,
            })
        }
    }

    /// Parse an identifier
    fn parse_identifier(&mut self) -> ParseResult<String> {
        let start = self.position;
        while let Some(ch) = self.peek_char() {
            if ch.is_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }

        if self.position == start {
            return Err(ParseError {
                message: "Expected identifier".to_string(),
                position: self.position,
            });
        }

        Ok(self.input[start..self.position].to_string())
    }

    /// Parse a string literal
    fn parse_string(&mut self) -> ParseResult<String> {
        let start = self.position;
        while let Some(ch) = self.peek_char() {
            if ch == '"' {
                break;
            }
            self.advance();
        }

        Ok(self.input[start..self.position].to_string())
    }

    /// Parse a number
    fn parse_number(&mut self) -> ParseResult<i64> {
        let start = self.position;

        if self.peek_char() == Some('-') {
            self.advance();
        }

        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }

        self.input[start..self.position]
            .parse()
            .map_err(|_| ParseError {
                message: "Invalid number".to_string(),
                position: start,
            })
    }

    /// Skip whitespace and comments
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch.is_whitespace() {
                self.advance();
            } else if ch == '%' {
                // Comment - skip until end of line
                while let Some(ch) = self.peek_char() {
                    self.advance();
                    if ch == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    /// Peek at the next character
    fn peek_char(&self) -> Option<char> {
        self.input[self.position..].chars().next()
    }

    /// Peek at the next n characters
    fn peek_str(&self, n: usize) -> Option<&str> {
        if self.position + n <= self.input.len() {
            Some(&self.input[self.position..self.position + n])
        } else {
            None
        }
    }

    /// Advance the position by one character
    fn advance(&mut self) {
        if let Some(ch) = self.peek_char() {
            self.position += ch.len_utf8();
        }
    }

    /// Expect a specific character
    fn expect_char(&mut self, expected: char) -> ParseResult<()> {
        self.skip_whitespace();
        let ch = self.peek_char().ok_or_else(|| ParseError {
            message: format!("Expected '{}' but found end of input", expected),
            position: self.position,
        })?;

        if ch == expected {
            self.advance();
            Ok(())
        } else {
            Err(ParseError {
                message: format!("Expected '{}' but found '{}'", expected, ch),
                position: self.position,
            })
        }
    }
}

/// Parsed Datalog statement
#[derive(Debug, Clone)]
pub enum Statement {
    /// A fact
    Fact(Predicate),
    /// A rule
    Rule(Rule),
    /// A query
    Query(Predicate),
}

/// Parse a Datalog fact
pub fn parse_fact(input: &str) -> ParseResult<Predicate> {
    let mut parser = DatalogParser::new(input.to_string());
    match parser.parse_statement()? {
        Statement::Fact(fact) => Ok(fact),
        _ => Err(ParseError {
            message: "Expected a fact".to_string(),
            position: 0,
        }),
    }
}

/// Parse a Datalog rule
pub fn parse_rule(input: &str) -> ParseResult<Rule> {
    let mut parser = DatalogParser::new(input.to_string());
    match parser.parse_statement()? {
        Statement::Rule(rule) => Ok(rule),
        _ => Err(ParseError {
            message: "Expected a rule".to_string(),
            position: 0,
        }),
    }
}

/// Parse a Datalog query
pub fn parse_query(input: &str) -> ParseResult<Predicate> {
    let mut parser = DatalogParser::new(input.to_string());
    match parser.parse_statement()? {
        Statement::Query(query) => Ok(query),
        _ => Err(ParseError {
            message: "Expected a query".to_string(),
            position: 0,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fact() {
        let fact = parse_fact("parent(alice, bob).").expect("test: should succeed");
        assert_eq!(fact.name, "parent");
        assert_eq!(fact.arity(), 2);
    }

    #[test]
    fn test_parse_rule() {
        let rule = parse_rule("grandparent(X, Z) :- parent(X, Y), parent(Y, Z).")
            .expect("test: should succeed");
        assert_eq!(rule.head.name, "grandparent");
        assert_eq!(rule.body.len(), 2);
    }

    #[test]
    fn test_parse_query() {
        let query = parse_query("?- parent(alice, X).").expect("test: should succeed");
        assert_eq!(query.name, "parent");
        assert_eq!(query.arity(), 2);
    }

    #[test]
    fn test_parse_with_comments() {
        let fact = parse_fact("parent(alice, bob). % Alice is parent of Bob")
            .expect("test: should succeed");
        assert_eq!(fact.name, "parent");
    }
}
