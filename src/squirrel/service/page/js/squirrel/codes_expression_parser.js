
const SEP = new RegExp('([(]|[)]|!|&&|[|][|]| +)')
const EMPTY = new RegExp('^ *$')
const NSLCE_OP = new RegExp('^[nslce]+$')

const zip = (rows) => rows[0].map((_, c) => rows.map((row) => row[c]))

const trans = (c) => {
    if (c == '*') {
        return '[^.]*'
    } else if (c == '?') {
        return '[^.]'
    } else if ('.^$*+?()[{\\|'.includes(c)) {
        return '\\' + c
    } else {
        return c
    }
}

const glob_to_re = (s) => '^' + Array.from(s).map(trans).join('') + '$'

export const parse_codes_filter = (str) => {
    const raise = (s) => {
        throw new Error(s + ' Expression: "' + str + '"')
    }

    const codes_matcher = (op, args) => {
        const aop = Array.from(op)
        let f = null
        for (const arg of args) {
            const tokens = arg.split('.')
            if (tokens.length != op.length) {
                raise(
                    'Invalid matcher (need ' +
                        op.length +
                        'codes): ' +
                        op +
                        ' ' +
                        arg
                )
            }

            const m = new Map(zip([aop, tokens]))
            const full = Array.from('nslce')
                .map((x) => (m.has(x) ? m.get(x) : '*'))
                .join('.')

            const reg = new RegExp(glob_to_re(full), 'i')
            const g = (x) => reg.exec(x) !== null
            if (f === null) {
                f = g
            } else {
                const prev = f
                f = (x) => prev(x) || g(x)
            }
        }
        if (f === null) {
            raise('Invalid matcher')
        }
        return f
    }

    const tokenize = (s) => {
        const tokens = s
            .split(SEP)
            .filter((token) => EMPTY.exec(token) === null)
        return tokens
    }

    const is_nslce_selector = (token) => NSLCE_OP.exec(token) !== null

    const tree = (tokens) => {
        const parse_parenthesis = () => {
            const expr = parse_expression(false)
            if (tokens.length == 0) {
                raise('Expected closing parenthesis.')
            }
            const token = tokens.shift()
            if (token != ')') {
                raise('Expected closing parenthesis.')
            }
            return {
                op: '(',
                args: [expr],
            }
        }

        const parse_negation = () => {
            const expr = parse_expression(true)
            return {
                op: '!',
                args: [expr],
            }
        }

        const parse_nslce = (token) => {
            const args = []
            while (
                tokens.length > 0 &&
                tokens[0] != ')' &&
                tokens[0] != '&&' &&
                tokens[0] != '||'
            ) {
                args.push(tokens.shift())
            }

            if (args.length == 0) {
                raise('No arguments for operator "' + token + '".')
            }

            return {
                op: token,
                args: args,
            }
        }

        const parse_expression = (phigh) => {
            if (tokens.length == 0) {
                raise('Incomplete expression.')
            }

            const token = tokens.shift()
            let node
            if (token == '(') {
                node = parse_parenthesis()
            } else if (token == '!') {
                node = parse_negation()
            } else if (is_nslce_selector(token)) {
                node = parse_nslce(token)
            } else {
                raise('Unexpected token: ' + token)
            }
            while (tokens.length > 0 && tokens[0] != ')' && !phigh) {
                const op = tokens.shift()
                if (op != '&&' && op != '||') {
                    raise('Expected operator but found "' + op + '".')
                }

                const node_right = parse_expression(op == '&&')

                node = {
                    op: op,
                    args: [node, node_right],
                }
            }
            return node
        }

        const root = parse_expression(false)
        if (tokens.length != 0) {
            raise('Invalid expression.')
        }
        return root
    }

    const matcher = (node) => {
        if (node.op == '!') {
            const f = matcher(node.args[0])
            return (x) => !f(x)
        } else if (node.op == '(') {
            return matcher(node.args[0])
        } else if (node.op == '&&') {
            const f = matcher(node.args[0])
            const g = matcher(node.args[1])
            return (x) => f(x) && g(x)
        } else if (node.op == '||') {
            const f = matcher(node.args[0])
            const g = matcher(node.args[1])
            return (x) => f(x) || g(x)
        } else {
            return codes_matcher(node.op, node.args)
        }
    }
    return matcher(tree(tokenize(str)))
}
